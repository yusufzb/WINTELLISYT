"""Retrospective target-excluded peer reconstruction; inputs remain untouched."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wintellisyt.data import read_scada
from wintellisyt.pipeline import file_hash

PIN = json.loads((Path(__file__).parent / 'transfer_2017_inputs.json').read_text())['kelmarsh']
DONORS = ['2', '3', '4', '5', '6']


def features(frame, donors):
    """Only contemporaneous donor channels; never target channels."""
    winds = frame[[f'w{d}' for d in donors]].where(lambda x: x >= 0)
    dirs = np.deg2rad(frame[[f'd{d}' for d in donors]])
    dirs.columns = winds.columns
    valid = winds.notna() & dirs.notna()
    count = valid.sum(axis=1)
    wind = winds.where(valid).median(axis=1)
    angle = np.arctan2(np.sin(dirs).where(valid).mean(axis=1),
                       np.cos(dirs).where(valid).mean(axis=1))
    resultant = np.hypot(np.sin(dirs).where(valid).mean(axis=1),
                         np.cos(dirs).where(valid).mean(axis=1))
    sector = (np.rad2deg(angle) % 360 // 45).fillna(-1).astype(int)
    speed_bin = (wind // 2).fillna(-1).astype(int)
    return pd.DataFrame({'wind': wind, 'sector': sector, 'bin': speed_bin,
                         'count': count, 'resultant': resultant}, index=frame.index)


def fit(frame):
    x = features(frame, DONORS)
    eligible = (x['count'] >= 2) & (x['resultant'] >= .5)
    t = x[eligible].copy()
    t['offset'] = (frame['w1'] - x['wind'])[eligible].where(frame['w1'] >= 0)
    t['power'] = frame['p1'][eligible]
    groups = t.groupby(['sector', 'bin'])
    table = groups[['offset', 'power']].median()
    counts = groups[['offset', 'power']].count()
    return table.where(counts >= 30)


def predict(frame, table, donors):
    x = features(frame, donors)
    keys = pd.MultiIndex.from_arrays([x['sector'], x['bin']])
    vals = table.reindex(keys).set_axis(frame.index)
    ok = (x['count'] >= 2) & (x['resultant'] >= .5)
    return pd.DataFrame({'speed': (x['wind'] + vals['offset']).clip(lower=0).where(ok),
                         'power': vals['power'].where(ok), 'median': x['wind'].where(ok),
                         'count': x['count'], 'sector': x['sector'], 'bin': x['bin']},
                        index=frame.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output directory already exists')
    root = Path(__file__).resolve().parents[1]
    raw_path = Path(PIN['path'])
    path = raw_path if raw_path.is_absolute() else (root / raw_path)
    if file_hash(path) != PIN['sha256']:
        raise ValueError('source checksum mismatch')
    rows = []
    for r in read_scada(path, 'kelmarsh'):
        if r.timestamp_utc.year == 2017 and r.turbine_id in ['1', *DONORS]:
            rows.append((r.timestamp_utc, r.turbine_id, r.wind_speed_ms,
                         r.active_power_kw, r.wind_direction_deg))
    if file_hash(path) != PIN['sha256']:
        raise ValueError('source changed during reading')
    raw = pd.DataFrame(rows, columns=['time', 'turbine', 'w', 'p', 'd'])
    if raw.duplicated(['time', 'turbine']).any():
        raise ValueError('duplicate turbine timestamps; no silent resolution')
    wide = raw.pivot(index='time', columns='turbine', values=['w', 'p', 'd'])
    wide.columns = [a + b for a, b in wide.columns]
    wide = wide.sort_index().reindex(pd.date_range('2017-01-01', '2017-12-31 23:50',
                                                    freq='10min', tz='UTC'))
    train = wide.loc[:'2017-09-30 23:50']
    cal = wide.loc['2017-10-01':'2017-10-31 23:50']
    test = wide.loc['2017-11-01':]
    table = fit(train)
    cp = predict(cal, table, DONORS)
    residual = (cal['w1'] - cp['speed']).where(cal['w1'] >= 0).dropna()
    power_residual = (cal['p1'] - cp['power']).dropna()
    if min(len(residual), len(power_residual)) < 100:
        raise ValueError('insufficient calibration')
    speed_q = residual.abs().quantile(.9)
    power_q = power_residual.abs().quantile(.9)
    threshold = residual.abs().quantile(.99)
    records = []
    # Weekly predeclared gaps, one gap per duration and week; overlap between
    # durations is intentional and metrics are stratified by gap length.
    for length in (1, 6, 36, 144):
        for start in pd.date_range('2017-11-02', '2017-12-24', freq='7D', tz='UTC'):
            block = test.loc[start:start + pd.Timedelta(minutes=10 * (length - 1))]
            prior = test.loc[:start - pd.Timedelta(minutes=10)].tail(1)
            persistence = prior['w1'].iloc[0] if len(prior) else np.nan
            if not np.isfinite(persistence) or persistence < 0:
                persistence = np.nan
            for dropout in (False, True):
                donors = DONORS[1:] if dropout else DONORS
                pred = predict(block, table, donors)
                for timestamp, row in block.iterrows():
                    p = pred.loc[timestamp]
                    records.append({'time_utc': timestamp, 'gap_start_utc': start,
                        'gap_rows': length, 'donor_dropout': dropout,
                        'donor_ids': ','.join(donors), 'valid_donors': p['count'],
                        'sector': p['sector'], 'speed_bin': p['bin'],
                        'observed_speed': row['w1'], 'observed_power': row['p1'],
                        'peer_speed': p['speed'], 'peer_power': p['power'],
                        'median_speed': p['median'], 'first_donor_speed': row['w3'] if dropout else row['w2'],
                        'persistence': persistence})
    out = pd.DataFrame(records)
    out['speed_status'] = np.where(out['peer_speed'].notna(), 'imputed', 'abstained')
    out['power_status'] = np.where(out['peer_power'].notna(), 'model_reference', 'abstained')
    out.loc[out['observed_speed'] < 0, 'observed_speed'] = np.nan
    out.loc[out['first_donor_speed'] < 0, 'first_donor_speed'] = np.nan
    metrics = []
    for (length, dropout), group in out.groupby(['gap_rows', 'donor_dropout']):
        for observed, methods, q in [('observed_speed', ['peer_speed', 'median_speed', 'first_donor_speed', 'persistence'], speed_q),
                                     ('observed_power', ['peer_power'], power_q)]:
            # matched support: all baselines share the same rows per estimand.
            valid = group[[observed, *methods]].notna().all(axis=1)
            for method in methods:
                error = group.loc[valid, method] - group.loc[valid, observed]
                metrics.append({'gap_rows': length, 'donor_dropout': dropout, 'method': method,
                    'eligible_observed': int(group[observed].notna().sum()), 'matched_n': int(valid.sum()),
                    'matched_fraction': float(valid.sum() / max(1, group[observed].notna().sum())),
                    'bias': error.mean(), 'mae': error.abs().mean(), 'rmse': np.sqrt((error ** 2).mean()),
                    'interval_coverage': (error.abs() <= q).mean() if method.startswith('peer_') else np.nan,
                    'interval_width': 2 * q if method.startswith('peer_') else np.nan})
    full_pred = predict(test, table, DONORS)
    e = (test['w1'] - full_pred['speed']).where(test['w1'] >= 0).dropna()
    # Same frozen threshold for unchanged control and a fixed +2 m/s injection.
    injection = {'n': len(e), 'threshold_ms': threshold,
                 'control_exceedance': float((e.abs() > threshold).mean()),
                 'plus_2_ms_exceedance': float(((e + 2).abs() > threshold).mean())}
    args.output.mkdir(parents=True)
    out.to_csv(args.output / 'masked_predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(args.output / 'metrics.csv', index=False)
    table.to_csv(args.output / 'training_cells.csv')
    (args.output / 'manifest.json').write_text(json.dumps({'input': PIN, 'target': '1',
        'donors': DONORS, 'fit': '2017-01 through 2017-09 UTC',
        'calibration': '2017-10 UTC', 'evaluation': '2017-11 through 2017-12 UTC',
        'interval_abs_residual_90pct': {'speed_ms': speed_q, 'power_kw': power_q},
        'injection': injection}, indent=2), encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False))
    print(injection)


if __name__ == '__main__':
    main()


