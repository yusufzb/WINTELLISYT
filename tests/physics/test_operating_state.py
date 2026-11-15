import unittest
from datetime import datetime, timedelta

from wintellisyt.operating_state import Event, Signals, fuse, physical_proxy, retrospective_diagnostic, window


class StateEngineTests(unittest.TestCase):
    def setUp(self):
        self.t = datetime(2017, 1, 1)
        self.step = timedelta(minutes=10)

    def event(self, ident, first, last, category="Forced outage"):
        return Event(ident, self.t + timedelta(minutes=first), self.t + timedelta(minutes=last),
                     "Stop", category)

    def test_union_exact_boundary_and_midpoint(self):
        events = [self.event(0, -10, 0), self.event(1, -2, 3), self.event(2, 2, 5),
                  self.event(3, 10, 11)]
        before = window(events, self.t - self.step)
        after = window(events, self.t)
        self.assertEqual(before.ids, (0, 1))
        self.assertEqual(before.stop_seconds, 600)
        self.assertEqual(after.ids, (1, 2))
        self.assertEqual(after.union_seconds, 300)
        self.assertEqual(after.stop_seconds, 300)
        self.assertEqual(after.midpoint_ids, ())

    def test_different_unknown_causes_preserve_category_conflict(self):
        events = [self.event(0, -12, 12), self.event(1, -12, 12, "Technical Standby")]
        proxy = physical_proxy("kelmarsh", Signals(8, 0, 0, None, None))
        before, after = window(events, self.t - self.step), window(events, self.t)
        lookup = {e.id: e for e in events}
        self.assertEqual(fuse(proxy, before, after, lookup, False),
                         ("UNKNOWN", "unknown", "unknown", ("concurrent_event_categories",)))

    def test_absent_open_and_unmapped_event_evidence(self):
        proxy = physical_proxy("penmanshiel", Signals(7, 0, None, None, None))
        empty = window([], self.t)
        self.assertEqual(fuse(proxy, empty, empty, {}, False)[:3],
                         ("INFERRED", "unknown", "unknown"))
        self.assertEqual(fuse(proxy, empty, empty, {}, True)[0], "UNKNOWN")
        event = self.event(0, -12, 12, "Unmapped")
        windows = [window([event], self.t - self.step), window([event], self.t)]
        self.assertEqual(fuse(proxy, *windows, {0: event}, False)[:3],
                         ("CORROBORATED_INFERENCE", "logged_stop_candidate", "unknown"))

    def test_proxy_is_independent_of_power(self):
        signals = Signals(8, 8, 900, None, 1300)
        for site in ("kelmarsh", "penmanshiel"):
            proxy = physical_proxy(site, signals)
            self.assertEqual(proxy, physical_proxy(site, signals))
            self.assertEqual(proxy.regime, "rotating")
            self.assertEqual(proxy.restriction, "possible_setpoint_restriction")
            self.assertEqual(retrospective_diagnostic(proxy, 0, 8).regime, "rotating_low_export")
            self.assertEqual(retrospective_diagnostic(proxy, 1900, 8).regime, "rated_like")

    def test_missing_and_disagreeing_channels(self):
        self.assertEqual(physical_proxy("kelmarsh", Signals(None, None, None, None, None)).regime,
                         "unknown")
        self.assertEqual(physical_proxy("kelmarsh", Signals(8, 0, 1000, None, None)).rule,
                         "rotation_channels_conflict")
        self.assertEqual(physical_proxy("kelmarsh", Signals(8, 0.8, None, None, None)).regime,
                         "transition")
        self.assertEqual(physical_proxy("kelmarsh", Signals(8, 0.8, None, None, None),
                                        "sensitivity").regime, "standstill_like")

    def test_incomplete_and_contradictory_events_stay_in_catalog(self):
        from io import BytesIO
        from zipfile import ZipFile

        from scripts.v2_state_engine import parse_events

        buffer = BytesIO()
        member = "Status_Kelmarsh_1_fixture.csv"
        with ZipFile(buffer, "w") as archive:
            archive.writestr(
                member,
                "# Time zone: UTC\nTimestamp start,Timestamp end,Duration,Status,Code,Message,IEC category\n"
                "2017-01-01 00:00:00,-,-,Stop,99,Unknown,\n"
                "2017-01-01 00:00:00,2017-01-01 00:05:00,00:03:00,Stop,99,Unknown,\n",
            )
        with ZipFile(buffer) as archive:
            catalog, bounded, legacy, unresolved = parse_events(archive, member)
        self.assertEqual([x["interval_quality"] for x in catalog],
                         ["missing_endpoint", "contradictory_duration"])
        self.assertEqual(bounded, [])
        self.assertEqual(len(legacy), 1)
        self.assertEqual(len(unresolved), 2)
