"""Synthetic parser cases. Publisher observations are audited separately."""

import hashlib
import io
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

from wintellisyt.data import read_scada
from wintellisyt.data.acquire import acquire_all_raw_files, acquire_archive, verify_all_raw_files

_CUBICO_HEADER = (
    "# Date and time,Wind speed (m/s),Power (kW),Nacelle ambient temperature (°C),"
    "Wind direction (°),Rotor speed (RPM),Blade angle (pitch position) A (°)\n"
)


class ScadaTest(unittest.TestCase):
    def test_cubico_keeps_duplicates_and_line_numbers(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "Turbine_Data_Kelmarsh_1_example.csv"
            path.write_text(
                "# Greenbyte synthetic fixture\n# Time zone: UTC\n"
                + _CUBICO_HEADER
                + "2021-01-01 00:00:00,7,100,5,315,9,3\n"
                + "2021-01-01 00:00:00,7,-1,NaN,315,9,3\n",
                encoding="utf-8-sig",
            )
            first, second = read_scada(path, "kelmarsh")
        self.assertEqual(first.timestamp_utc, datetime(2021, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(second.timestamp_utc, first.timestamp_utc)
        self.assertEqual((first.source_row, second.source_row), (4, 5))
        self.assertEqual((first.active_power_kw, second.active_power_kw), (100, -1))
        self.assertIsNone(second.ambient_temperature_c)
        self.assertEqual(second.quality_flags, ("missing_ambient_temperature_c",))
        self.assertIsNone(first.status_code)
        self.assertEqual(first.operational_state, "unknown")

    def test_cubico_archive_reads_only_selected_turbine_telemetry(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.zip"
            with ZipFile(path, "w") as archive:
                for turbine in ("11", "12"):
                    archive.writestr(
                        f"folder/Turbine_Data_Penmanshiel_{turbine}_example.csv",
                        _CUBICO_HEADER + "2021-01-01 00:00:00,7,100,5,315,9,3\n",
                    )
                archive.writestr("Status_Penmanshiel_12_example.csv", "status log")
            [record] = read_scada(path, "penmanshiel", turbine_id="12")
        self.assertEqual(record.turbine_id, "12")
        self.assertTrue(record.source_file.endswith("!folder/Turbine_Data_Penmanshiel_12_example.csv"))

    def test_engie_offset_disambiguates_dst_fold(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "engie.csv"
            path.write_text(
                "Wind_turbine_name;Date_time;Ws_avg;P_avg;Ot_avg;Wa_avg;Ba_avg\n"
                "R80736;2014-10-26T02:00:00+02:00;7,1;100;4;300;0\n"
                "R80736;2014-10-26T02:00:00+01:00;7,2;102;4;300;0\n",
                encoding="utf-8",
            )
            first, second = read_scada(path, "la_haute_borne")
        self.assertEqual(first.timestamp_utc, datetime(2014, 10, 26, tzinfo=timezone.utc))
        self.assertEqual(second.timestamp_utc, datetime(2014, 10, 26, 1, tzinfo=timezone.utc))
        self.assertEqual(first.wind_speed_ms, 7.1)
        self.assertIsNone(first.rotor_speed_rpm)

    def test_ambiguous_naive_engie_timestamp_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "engie.csv"
            path.write_text(
                "Wind_turbine_name,Date_time,Ws_avg,P_avg\n"
                "R80736,2014-10-26T02:00:00,7,100\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "explicit UTC offset"):
                list(read_scada(path, "la_haute_borne"))

    def test_bad_acquisition_hash_does_not_publish_file(self):
        with TemporaryDirectory() as tmp:
            payload = b"test only"
            path = Path(tmp) / "manifest.json"
            archive = {
                "state": "verified",
                "path": "data/raw/example.zip",
                "url": "https://example.invalid/",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "publisher_md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
            }
            path.write_text(json.dumps({"kelmarsh": {"acquisition": archive}}), encoding="utf-8")
            with (
                patch("wintellisyt.data.acquire.MANIFEST_PATH", path),
                patch("wintellisyt.data.acquire.urlopen", return_value=io.BytesIO(b"wrong raw")),
                self.assertRaisesRegex(ValueError, "checksum differs"),
            ):
                acquire_archive("kelmarsh", tmp)
            self.assertEqual(list((Path(tmp) / "data" / "raw").iterdir()), [])

    def test_acquire_and_verify_all_raw_files_from_checksum_index(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_dir = root / "data" / "reference"
            ref_dir.mkdir(parents=True)
            payload = b"scada_static_fixture"
            sha = hashlib.sha256(payload).hexdigest()
            md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
            (ref_dir / "DATA_FILE_SHA256.csv").write_text(
                "path,bytes,sha256,publisher_md5\n"
                f"data/raw/kelmarsh/Kelmarsh_WT_static.csv,{len(payload)},{sha},{md5}\n",
                encoding="utf-8",
            )
            missing = verify_all_raw_files(root)
            self.assertEqual(len(missing), 1)
            self.assertFalse(missing[0]["verified"])

            with patch("wintellisyt.data.acquire.urlopen", return_value=io.BytesIO(payload)):
                acquired = acquire_all_raw_files(root)
            self.assertEqual(len(acquired), 1)
            self.assertTrue(acquired[0].exists())

            verified = verify_all_raw_files(root)
            self.assertTrue(verified[0]["verified"])


if __name__ == "__main__":
    unittest.main()
