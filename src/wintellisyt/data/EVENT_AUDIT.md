# Cubico 2016 event alignment audit

Both local archives passed byte-count, publisher MD5 and pinned SHA256
verification. Charlie Plumley and Roberta Takeuchi published the underlying
exports through Cubico Sustainable Investments Ltd under CC BY 4.0.
[Kelmarsh v4](https://doi.org/10.5281/zenodo.16807551) ·
[Penmanshiel v3](https://doi.org/10.5281/zenodo.16807304)

The audit matched all 11 UTC status members to their SCADA turbine IDs.
It counted exact repeats and missing or invalid intervals, tabulated
status/code/IEC combinations, and measured positive overlap between
event intervals and half-open 10-minute SCADA windows. It compared
timestamp-at-start and timestamp-at-end interpretations without choosing one.
Exact repeats are retained in event and overlap multiplicity counts.
Rows with a blank or hyphen endpoint, zero duration, or reversed endpoints
are counted separately and excluded from interval calculations.

| Measure | Kelmarsh | Penmanshiel |
| --- | ---: | ---: |
| SCADA rows | 314,496 | 126,402 |
| Event rows | 14,019 | 5,164 |
| Exact repeat rows | 135 | 1 |
| Missing endpoint rows | 4,749 | 15 |
| Stop rows with missing endpoints | 8 | 0 |
| Zero-length / reversed rows | 68 / 1 | 27 / 0 |
| Event rows with blank IEC category | 533 | 576 |
| SCADA timestamps with Stop overlap under both conventions | 16,217 | 35,686 |
| SCADA timestamps with Stop overlap under exactly one convention | 1,086 | 656 |
| SCADA windows with both Stop and another event, start convention | 12,190 | 28,328 |

Both anchor conventions yielded the same site-level Stop counts, but the
affected timestamps differ. Even agreement across conventions establishes
an event overlap, with no conclusion about the operating state of the
entire ten-minute average. A window with no bounded event can still contain
an open-ended event or occur outside reliable event coverage. The publisher
deposits describe ten-minute exports without specifying timestamp anchoring
or a complete operating-state mapping. The baseline retains unknown
operational state on real rows.

From the repository root with the package installed:

```powershell
python scripts/audit_cubico_events.py artifacts/runs/cubico-event-audit-v2.json
```

Use a new output path for each repeat. The JSON includes per-turbine
counts, code/category frequencies, source links and archive hashes.
A state join needs evidence for interval anchoring, treatment of open-ended
records, precedence for concurrent categories, and verified criteria for
normal operation and curtailment.

## Open-ended Stop sensitivity

All 4,764 incomplete events have a start timestamp, with both end and
duration missing. Eight are Kelmarsh Stop records. Under a deliberately
conservative scenario that extends each of those eight Stop records through
the remaining observed SCADA span, 138,652 Kelmarsh timestamps outside the
bounded Stop overlaps would become candidates under at least one anchor.
Relative to each anchor's own bounded result, the additional counts are
138,927 for timestamp-at-start and 138,924 for timestamp-at-end. Penmanshiel
has zero start-only Stop records in this archive.

This scenario quantifies sensitivity to missing event ends. The report
leaves individual SCADA states unknown. The publisher deposits do not
specify an endpoint recovery rule. A verified event export with complete
end times or publisher documentation about how to resolve open records
would narrow this gap.

Reproduce the extended audit with:

```powershell
python scripts/audit_cubico_events.py artifacts/runs/cubico-event-sensitivity.json
```
