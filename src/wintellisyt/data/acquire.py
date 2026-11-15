"""Fetch pinned Cubico SCADA archives and static assets with byte/MD5/SHA-256 verification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.request import urlopen

MANIFEST_PATH = Path(__file__).with_name("source_manifest.json")
CHECKSUM_INDEX_REL = Path("data/reference/DATA_FILE_SHA256.csv")

ZENODO_FILE_BASES: dict[str, str] = {
    "kelmarsh": "https://zenodo.org/records/16807551/files",
    "penmanshiel": "https://zenodo.org/records/16807304/files",
}


def _resolve_checksum_csv(root: str | Path = ".") -> Path:
    candidate = Path(root) / CHECKSUM_INDEX_REL
    if candidate.exists():
        return candidate
    repo_fallback = Path(__file__).resolve().parents[3] / CHECKSUM_INDEX_REL
    if repo_fallback.exists():
        return repo_fallback
    raise FileNotFoundError(f"Checksum manifest not found: {candidate}")


def _load_checksum_records(root: str | Path = ".") -> list[dict[str, Any]]:
    csv_path = _resolve_checksum_csv(root)
    records: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rel = str(row["path"]).strip()
            parts = Path(rel).parts
            farm_key = parts[2].lower() if len(parts) >= 3 else ""
            base_url = ZENODO_FILE_BASES.get(farm_key)
            if not base_url:
                raise ValueError(f"Cannot map upstream URL for raw path: {rel}")
            records.append(
                {
                    "path": rel,
                    "url": f"{base_url}/{Path(rel).name}",
                    "bytes": int(row["bytes"]),
                    "sha256": str(row["sha256"]).strip(),
                    "publisher_md5": str(row["publisher_md5"]).strip(),
                }
            )
    return records


def _download_and_verify(target: Path, spec: dict[str, Any], label: str) -> Path:
    if target.exists():
        _verify(target, spec)
        try:
            target.chmod(0o444)
        except OSError:
            pass
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as handle:
        pending = Path(handle.name)
    try:
        with urlopen(spec["url"], timeout=120) as response, pending.open("wb") as handle:
            remaining = int(spec["bytes"])
            while chunk := response.read(min(1024 * 1024, remaining + 1)):
                remaining -= len(chunk)
                if remaining < 0:
                    raise ValueError(f"{label}: archive exceeds published size")
                handle.write(chunk)
        _verify(pending, spec)
        try:
            os.link(pending, target)
        except FileExistsError:
            _verify(target, spec)
    finally:
        pending.unlink(missing_ok=True)
    try:
        target.chmod(0o444)
    except OSError:
        pass
    return target


def acquire_archive(farm_id: str, root: str | Path = ".") -> Path:
    """Return a verified primary local archive for `farm_id`, downloading only if absent."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if farm_id not in manifest:
        raise ValueError(f"Unsupported SCADA farm: {farm_id}")
    acquisition = manifest[farm_id]["acquisition"]
    if acquisition["state"] != "verified":
        raise ValueError(acquisition["message"])
    target = Path(root) / acquisition["path"]
    return _download_and_verify(target, acquisition, farm_id)


def acquire_all_raw_files(root: str | Path = ".") -> list[Path]:
    """Download (if missing) and verify all 9 raw Cubico files pinned in DATA_FILE_SHA256.csv."""
    base = Path(root)
    verified_paths: list[Path] = []
    for rec in _load_checksum_records(base):
        target = base / rec["path"]
        print(f"Checking {rec['path']} ({rec['bytes']:,} bytes)...", flush=True)
        verified_paths.append(_download_and_verify(target, rec, rec["path"]))
    return verified_paths


def verify_all_raw_files(root: str | Path = ".") -> list[dict[str, Any]]:
    """Verify locally present raw files against DATA_FILE_SHA256.csv without downloading."""
    base = Path(root)
    results: list[dict[str, Any]] = []
    for rec in _load_checksum_records(base):
        target = base / rec["path"]
        status: dict[str, Any] = {
            "path": rec["path"],
            "present": target.exists(),
            "verified": False,
            "expected_bytes": rec["bytes"],
            "actual_bytes": target.stat().st_size if target.exists() else 0,
            "error": "",
        }
        if target.exists():
            try:
                _verify(target, rec)
                status["verified"] = True
            except ValueError as exc:
                status["error"] = str(exc)
        else:
            status["error"] = "missing"
        results.append(status)
    return results


def _verify(path: Path, acquisition: dict[str, Any]) -> None:
    if path.stat().st_size != acquisition["bytes"]:
        raise ValueError(f"{path}: archive byte count differs from manifest")
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            sha256.update(chunk)
            md5.update(chunk)
    if sha256.hexdigest() != acquisition["sha256"] or md5.hexdigest() != acquisition["publisher_md5"]:
        raise ValueError(f"{path}: archive checksum differs from manifest")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "farm_id",
        nargs="?",
        choices=["kelmarsh", "penmanshiel", "la_haute_borne"],
        help="Acquire primary archive for a single wind farm from source_manifest.json.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download and checksum-verify all 9 raw files listed in data/reference/DATA_FILE_SHA256.csv.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify local files in data/raw/ against data/reference/DATA_FILE_SHA256.csv without downloading.",
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    if args.verify_only:
        rows = verify_all_raw_files(args.root)
        missing_or_bad = [r for r in rows if not r["verified"]]
        for r in rows:
            badge = "OK" if r["verified"] else r["error"].upper()
            print(f"[{badge}] {r['path']} ({r['actual_bytes']:,} / {r['expected_bytes']:,} bytes)")
        if missing_or_bad:
            parser.exit(1, f"{len(missing_or_bad)} of {len(rows)} raw files missing or unverified.\n")
        print(f"All {len(rows)} raw files verified cleanly.")
        return

    if args.all:
        try:
            paths = acquire_all_raw_files(args.root)
            print(f"Verified {len(paths)} raw files.")
        except (OSError, ValueError) as exc:
            parser.exit(1, f"Acquisition failed: {exc}\n")
        return

    if not args.farm_id:
        parser.error("Provide a farm_id (kelmarsh|penmanshiel|la_haute_borne), --all, or --verify-only.")

    try:
        print(acquire_archive(args.farm_id, args.root))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Acquisition failed: {exc}\n")


if __name__ == "__main__":
    main()
