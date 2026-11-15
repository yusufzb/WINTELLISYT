import unittest
from datetime import datetime, timedelta

from scripts.audit_cubico_events import alignment, stop_sensitivity


class EventAlignmentTests(unittest.TestCase):
    def test_half_open_boundaries_and_concurrent_status(self):
        start = datetime(2016, 1, 1)
        minute = timedelta(minutes=1)
        events = [
            (start - minute, start, "Stop"),
            (start + minute, start + 2 * minute, "Stop"),
            (start + minute, start + 2 * minute, "Warning"),
            (start + 10 * minute, start + 11 * minute, "Informational"),
        ]
        forward, forward_stops = alignment([start], events, timedelta())
        backward, backward_stops = alignment([start], events, -10 * minute)
        self.assertEqual(forward_stops, backward_stops)
        self.assertEqual(forward, {
            "any_event": 1, "stop_event": 1, "multiple_events": 1, "stop_and_other": 1,
        })
        self.assertEqual(backward, {
            "any_event": 1, "stop_event": 1, "multiple_events": 0, "stop_and_other": 0,
        })

    def test_persistence_adds_only_previously_unexposed_timestamps(self):
        start = datetime(2016, 1, 1)
        ten = timedelta(minutes=10)
        timestamps = [start, start + ten, start + 2 * ten]
        bounded = [(start, start + ten, "Stop")]
        open_stops = [(start + ten, datetime.max, "Stop")]
        _, forward = alignment(timestamps, bounded, timedelta())
        _, backward = alignment(timestamps, bounded, -ten)
        self.assertEqual(stop_sensitivity(timestamps, bounded, open_stops, forward, backward), {
            "additional_timestamp_at_start": 2,
            "additional_timestamp_at_end": 1,
            "additional_under_either_anchor": 1,
        })
