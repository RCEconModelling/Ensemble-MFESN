"""
Same-Budget Timing Records

Recalculate the historical paired checks and production-block checks.
This does not run new timings or certify the speed of another computer.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
from statistics import median

PATH_WD = Path(__file__).resolve().parents[1]


def timing_valid(item):
    wall, cpu = item['wall_seconds'], item['cpu_seconds']
    return wall > 0 and cpu > 0 and wall <= max(2.5*cpu, cpu+5)


def repeat_spread(items):
    times = [x['wall_seconds'] for x in items]
    return (max(times)-min(times))/median(times)


def audit_saved_timing(history):
    records = history['records']
    units = {name: item for name, item in records.items() if name.endswith('/timing.json')}
    gate_rows = []

    def check_attempt(attempt, accepted=False):
        rows = []
        for entry in attempt['pools']:
            tuning_paths = entry['tuning_units']
            ensemble_paths = entry.get('ensemble_units', [
                str(PurePosixPath(tuning_paths[0]).parent/f'ensemble_repeat_{r}') for r in [1, 2]])
            ensembles = [units[path+'/timing.json'] for path in ensemble_paths]
            tunes = [units[path+'/timing.json'] for path in tuning_paths]
            if len(ensembles) != 2 or len(tunes) != 2:
                raise ValueError('Expected two repeats of each arm.')
            ratio = median(x['wall_seconds'] for x in tunes) / \
                median(x['wall_seconds'] for x in ensembles)
            es, ts = repeat_spread(ensembles), repeat_spread(tunes)
            for actual, field in [(ratio, 'ratio'), (es, 'ensemble_repeat_spread'), (ts, 'tuning_repeat_spread')]:
                if abs(actual-entry[field]) > 1e-12:
                    raise ValueError(f'Saved paired statistic differs: {field}')
            passed = .9 <= ratio <= 1.1 and es <= .2 and ts <= .2 and all(
                timing_valid(x) for x in ensembles+tunes)
            if accepted and not passed:
                raise ValueError('An accepted historical timing gate fails recomputation.')
            rows.append(dict(pool=entry['pool'], ratio=ratio, ensemble_spread=es, tuning_spread=ts,
                             ensemble_seconds=median(x['wall_seconds'] for x in ensembles), passed=passed))
        before, after = attempt.get('power_before', {}), attempt.get('power_after', {})
        if accepted and before.get('available') and after.get('available'):
            if any(before[k] != after[k] for k in ['ac_line_status', 'power_saving']):
                raise ValueError('Power transition in accepted timing epoch.')
        return rows

    targets = {}
    for entry in history['gates']:
        gate = entry['data']
        if gate['status'] != 'passed':
            raise ValueError('Expected a passed historical gate.')
        rows = check_attempt(gate['accepted_attempt'], accepted=True)
        if [r['pool'] for r in rows] != [1, 2, 3]:
            raise ValueError('Expected original pools 1, 2, 3 (zero-based).')
        target = median(r['ensemble_seconds'] for r in rows)
        if abs(target-gate['ensemble_target_seconds']) > 1e-10:
            raise ValueError('Representative target differs.')
        if 'runtime_epoch' in gate:
            targets[gate['runtime_epoch']] = target
        gate_rows.append(dict(stage=entry['stage'], case=entry['case'], pools=rows))

    block_rows = []
    accepted_production = set()
    for case in history['production']:
        rows = case['records']
        if [r['pool'] for r in rows] != list(range(100)):
            raise ValueError('Expected 100 ordered production pools.')
        ratios = []
        for row in rows:
            path = row['timing_directory']+'/timing.json'
            item = units[path]
            if not timing_valid(item):
                raise ValueError('Invalid accepted production timing.')
            ratio = item['wall_seconds']/targets[row['runtime_epoch']]
            if abs(ratio-row['normalized_time_ratio']) > 1e-12:
                raise ValueError('Epoch-normalized production ratio differs.')
            ratios.append(ratio)
            accepted_production.add(path)
        for start in range(0, 100, 10):
            value = median(ratios[start:start+10])
            if not .9 <= value <= 1.1:
                raise ValueError('Accepted production block is outside the tolerance.')
            block_rows.append(
                dict(case=case['case'], start_pool=start, end_pool=start+9, ratio=value))
        if abs(median(ratios)-case['final_normalized_median']) > 1e-12 or not .9 <= median(ratios) <= 1.1:
            raise ValueError('Final normalized median differs.')

    attempts = 0
    failed_attempts = 0
    for name, item in records.items():
        if name.endswith('/attempts.json') or name.endswith('/stability_attempts.json'):
            for attempt in item['attempts']:
                rows = check_attempt(attempt)
                matched = all(.9 <= r['ratio'] <= 1.1 for r in rows)
                stable = all(r['ensemble_spread'] <= .2 and r['tuning_spread'] <= .2 for r in rows)
                if matched != attempt['matched_within_10_percent_each_pool'] or stable != attempt['repeat_spread_within_20_percent']:
                    raise ValueError('Historical attempt outcome differs.')
                attempts += 1
                failed_attempts += int(not (matched and stable))
        if name.endswith('/check.json') and 'block_start' in item:
            if item['passed'] != (.9 <= item['median_epoch_normalized_time_ratio'] <= 1.1):
                raise ValueError('Historical block outcome differs.')
    groups = {}
    for prefix in ['original', 'corrected']:
        group = {p: t for p, t in units.items() if p.startswith(prefix+'/')}
        groups[prefix] = dict(completed_timing_units=len(group),
                              numerical_wall_hours=sum(t['wall_seconds']
                                                       for t in group.values())/3600,
                              numerical_cpu_hours=sum(t['cpu_seconds']
                                                      for t in group.values())/3600,
                              gross_wall_cpu_flags=[p for p, t in group.items() if not timing_valid(t)])
    disclosure = history['original_runtime_disclosure']
    if abs(groups['original']['numerical_wall_hours']-disclosure['saved_numerical_wall_hours']) > 1e-10:
        raise ValueError('Original saved timing total differs from the independent audit.')
    accepted = sum(units[p]['wall_seconds'] for p in accepted_production)/3600
    corrected = history['corrected_runtime_disclosure']
    stage_hours = (corrected['next_stage_epoch']-corrected['started_epoch'])/3600
    if abs(stage_hours-corrected['recorded_stage_hours']) > 1e-10 or stage_hours <= 0:
        raise ValueError('Corrected controller stage duration differs.')
    return dict(status='passed', accepted_gates=gate_rows, accepted_production_blocks=block_rows,
                retained_attempts=attempts, unmatched_or_unstable_attempts=failed_attempts, timing_totals=groups,
                original_accepted_production_hours=accepted,
                original_other_retained_numerical_hours=groups['original']['numerical_wall_hours']-accepted,
                original_gross_logged_span_hours=disclosure['gross_logged_span_hours'],
                corrected_controller_stage_hours=stage_hours,
                gross_span_is_lower_bound=True,
                partial_units_have_no_complete_measured_time=disclosure['partial_arrays_without_timing'],
                representative_matching_only=True, all_400_production_timings_remeasured_after_correction=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, default=PATH_WD /
                        'results_same_budget_rev1/timing_history.json')
    args = parser.parse_args()
    result = audit_saved_timing(json.loads(args.history.read_text()))
    print(json.dumps({k: v for k, v in result.items() if k not in [
          'accepted_gates', 'accepted_production_blocks']}, indent=2))
