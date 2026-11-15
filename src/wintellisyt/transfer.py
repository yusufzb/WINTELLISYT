"""Zero-shot binned baselines for explicitly selected turbine pairs."""

import json
import math
import platform
from statistics import median

from . import __version__
from .pipeline import assess, curve_input, file_hash, load_rows, source_manifest
from .power_curve import fit_power_curve


def transfer(train_path, train_source, train_turbine, target_path, target_source, target_turbine,
             output, min_bin_count=5):
    if not train_turbine or not target_turbine:
        raise ValueError("Training and target turbine IDs are required")
    if (train_source, train_turbine) == (target_source, target_turbine):
        raise ValueError("Training and target must be different turbines")
    if output.exists():
        raise ValueError(f"Output file already exists: {output}")
    provenance = source_manifest()
    inputs = []
    selected = []
    for path, source, turbine in (
        (train_path, train_source, train_turbine), (target_path, target_source, target_turbine)
    ):
        digest = file_hash(path)
        acquisition = provenance[source]["acquisition"]
        if digest != acquisition.get("sha256"):
            raise ValueError(f"Transfer requires a pinned verified archive for {source}")
        rows, _ = assess(list(load_rows(path, source, turbine)))
        if file_hash(path) != digest:
            raise ValueError("Input changed during ingestion")
        if not rows:
            raise ValueError(f"No unique observations for {source}/{turbine}")
        selected.append(rows)
        inputs.append({"source": source, "turbine": turbine, "input_sha256": digest,
                       "source_url": provenance[source]["source_url"],
                       "license": provenance[source]["license"]})

    source_rows, target_rows = selected
    target_rows.sort(key=lambda row: row.timestamp_utc)
    # The cutoff uses target timestamps only, including rows with missing labels.
    cutoff = target_rows[4 * len(target_rows) // 5].timestamp_utc
    train = [row for row in source_rows if row.timestamp_utc < cutoff]
    test = [row for row in target_rows if row.timestamp_utc >= cutoff]
    if not train:
        raise ValueError("No source observations precede the target cutoff")
    curve = fit_power_curve(curve_input(train), min_bin_count=min_bin_count)
    training_power = [row.active_power_kw for row in train
                      if _valid_pair(row)]
    if not training_power:
        raise ValueError("No valid source training pairs before the target cutoff")
    constant = median(training_power)

    eligible = [row for row in test if _valid_pair(row)]
    binned_errors = []
    constant_errors = []
    for row in eligible:
        prediction = curve.predict(row.wind_speed_ms, stratum="raw")
        if prediction is not None:
            binned_errors.append(row.active_power_kw - prediction)
            constant_errors.append(row.active_power_kw - constant)
    result = {
        "schema_version": 1,
        "package_version": __version__,
        "python_version": platform.python_version(),
        "source_evidence": "verified_real",
        "protocol": "ordered zero-shot transfer on screened, state-unknown electrical power",
        "comparison": "cross_turbine" if train_source == target_source else "cross_farm",
        "inputs": inputs,
        "target_cutoff_utc": cutoff.isoformat(),
        "latest_train_utc": max(row.timestamp_utc for row in train).isoformat(),
        "train_rows": len(train),
        "train_valid_pairs": len(training_power),
        "test_rows": len(test),
        "test_valid_pairs": len(eligible),
        "test_supported_pairs": len(binned_errors),
        "min_bin_count": min_bin_count,
        "train_median_kw": constant,
        "binned_shared_support": _metrics(binned_errors),
        "median_shared_support": _metrics(constant_errors),
        "limitations": [
            "Errors apply to one directed turbine pair and one historical holdout.",
            "Absolute electrical kW is used. Cross-farm turbines have different model families.",
            "Unknown operating states include a mixture of operating conditions.",
            "No target power is used for fitting, normalization or tuning.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def _valid_pair(row):
    return (row.wind_speed_ms is not None and math.isfinite(row.wind_speed_ms)
            and row.wind_speed_ms >= 0 and row.active_power_kw is not None
            and math.isfinite(row.active_power_kw))


def _metrics(errors):
    if not errors:
        return {"n": 0, "mae_kw": None, "rmse_kw": None}
    return {
        "n": len(errors),
        "mae_kw": math.fsum(abs(error) for error in errors) / len(errors),
        "rmse_kw": math.sqrt(math.fsum(error**2 for error in errors) / len(errors)),
    }
