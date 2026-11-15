"""Compare the pinned third-party code table with literal Cubico Stop categories."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from wintellisyt.pipeline import file_hash

TABLE_SHA = "6de8a72c6862b38e812a912ec805dae7f36a86f7a8680d10af9c910e080796f1"
LABELS = {
    "Forced outage": "forced_outage",
    "Scheduled Maintenance": "scheduled_maintenance",
    "Technical Standby": "technical_standby",
    "Out of Electrical Specification": "electrical",
    "Out of Environmental Specification": "environmental",
    "Requested Shutdown": "requested",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("state_run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if file_hash(args.table) != TABLE_SHA:
        raise ValueError("Unexpected scada-schema code table")
    with args.table.open(encoding="utf-8", newline="") as handle:
        table = {r["code"]: r["category"] for r in csv.DictReader(handle)}
    source = args.state_run / "summary.json"
    state = json.loads(source.read_text(encoding="utf-8"))
    by_code = defaultdict(Counter)
    for turbine, summary in state["turbines"].items():
        farm, _ = turbine.split("/")
        for row in summary["stop_code_category_counts"]:
            by_code[farm, row["code"]][row["category"]] += row["n"]
    records = defaultdict(dict)
    for (farm, code), categories in sorted(by_code.items()):
        counts = Counter()
        for label, n in categories.items():
            if code not in table:
                counts["code_unseen_in_table"] += n
            elif not table[code] or label not in LABELS:
                counts["uncomparable_category"] += n
            elif LABELS[label] == table[code]:
                counts["agree"] += n
            else:
                counts["disagree"] += n
        records[farm][code] = {"table_category": table.get(code), "logged_categories": dict(categories),
                               "counts": dict(counts)}
    totals = {}
    for farm, codes in records.items():
        total = Counter()
        for record in codes.values():
            total.update(record["counts"])
        totals[farm] = dict(total)
    output = {"year": 2017, "comparison": "known-code category consistency",
              "table_sha256": TABLE_SHA, "state_summary_sha256": file_hash(source),
              "totals": totals, "by_code": records}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
