"""Leakage and support checks for the peer reconstruction experiment."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('peer', Path(__file__).parents[2] / 'scripts/engineering_peer.py')
peer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(peer)


def fixture():
    frame = pd.DataFrame(index=range(40))
    for donor in peer.DONORS:
        frame['w' + donor] = 6.0
        frame['d' + donor] = 10.0
    frame['w1'] = 7.0
    frame['p1'] = 300.0
    return frame


def test_target_excluded_and_training_only():
    train = fixture()
    table = peer.fit(train)
    target = fixture()
    before = peer.predict(target, table, peer.DONORS)
    target['w1'] = 99.0
    target['p1'] = -999.0
    target['d1'] = 270.0
    after = peer.predict(target, table, peer.DONORS)
    pd.testing.assert_frame_equal(before, after)
    assert (before['speed'] == 7).all()
    assert (before['power'] == 300).all()


def test_abstains_without_peer_or_training_support():
    train = fixture()
    table = peer.fit(train)
    target = fixture()
    for donor in peer.DONORS[1:]:
        target['w' + donor] = np.nan
    assert peer.predict(target, table, peer.DONORS)['speed'].isna().all()
    target = fixture()
    for donor in peer.DONORS:
        target['w' + donor] = 20.0
    assert peer.predict(target, table, peer.DONORS)['power'].isna().all()
