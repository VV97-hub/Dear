"""Attribute Nsight SQLite kernels through their launch API's PID/TID + NVTX.

No third-party modules. Normal CUDA launches only; no CUDA Graph reconstruction.
Each kernel receives ONE category; nested NVTX durations are never added together.
"""
import argparse
import bisect
import collections
from contextlib import closing
import csv
import heapq
import json
from pathlib import Path
import re
import sqlite3

ENCODE = {'codec_encode', 'compress_param', 'compress', 'add_residual',
          'orthogonalize', 'project_p', 'project_q', 'factor_copy', 'ef_update',
          'factor_state', 'refresh_encode_first', 'refresh_encode_second', 'refresh_ef_update'}
DECODE = {'codec_decode', 'load_factor', 'factor_load_copy', 'reconstruct',
          'reconstruct_matmul', 'refresh_load'}
PACK = {'pack', 'padding', 'dense_pack', 'dense_unpack', 'refresh_pack', 'layout_prepare'}
COMM = {'rs_submit', 'ag_submit', 'refresh_allreduce_first', 'refresh_allreduce_second'}


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end - start for start, end in union(intervals))


def intersection(left, right):
    a, b = union(left), union(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def parse_label(text):
    parts = text.split(' | ')
    if not parts[0].startswith('helix/'):
        return '', {}
    fields = dict(x.split('=', 1) for x in parts[1:] if '=' in x)
    return parts[0][6:], fields


def classify(ranges, kernel):
    # Earlier ranges precede nested ranges. Custom ownership takes precedence
    # over automatic ATen names, even when ATen is the deepest NVTX range.
    custom = [(parse_label(r['label'])[0], parse_label(r['label'])[1], r)
              for r in ranges if r['label'].startswith('helix/')]
    fields = {}
    for _, values, _ in custom:
        fields.update(values)
    owners = []
    for stage, _, r in custom:
        if stage in COMM:
            category = 'nccl'
        elif stage in ENCODE:
            category = 'powersgd_encode'
        elif stage in DECODE:
            category = 'powersgd_decode'
        elif stage in PACK:
            category = 'packing'
        elif stage.startswith('norm_'):
            category = 'rank_control'
        elif stage == 'optimizer_update':
            category = 'optimizer_update'
        elif stage == 'gradient_average':
            category = 'gradient_average'
        else:
            continue
        owners.append((category, stage, r))
    if owners:
        category, stage, owner = owners[-1]
    elif 'nccl' in kernel.lower():
        category, stage, owner = 'nccl', 'nccl_unattributed', None
    elif any(stage == 'bp_hook' for stage, _, _ in custom):
        category, stage, owner = 'hook_other', 'bp_hook', custom[-1][2]
    elif any(re.search(r'Backward\d*\b|AccumulateGrad|autograd::engine', r['label']) for r in ranges):
        category, stage, owner = 'bp', 'bp_operator', None
    elif any(stage == 'backward' for stage, _, _ in custom):
        category, stage, owner = 'bp', 'bp_phase', custom[-1][2]
    elif any(stage == 'forward' for stage, _, _ in custom):
        category, stage, owner = 'forward', 'forward', custom[-1][2]
    elif any(stage == 'loss' for stage, _, _ in custom):
        category, stage, owner = 'loss', 'loss', custom[-1][2]
    else:
        category, stage, owner = 'unclassified', '', None
    return category, stage, fields, owner


def read_attributed(connection):
    connection.row_factory = sqlite3.Row
    tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
    required = {'StringIds', 'NVTX_EVENTS', 'CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_KERNEL'}
    if not required <= tables:
        raise ValueError('Missing SQLite tables: ' + str(required - tables))
    strings = dict(connection.execute('select id,value from StringIds'))
    by_thread = collections.defaultdict(list)
    phases = collections.defaultdict(list)
    capture_windows = []
    for raw in connection.execute('select * from NVTX_EVENTS where end > start order by start, end desc'):
        row = dict(raw)
        if row.get('endGlobalTid') not in (None, row['globalTid']):
            continue
        label = row.get('text') or strings.get(row.get('textId'), '')
        r = dict(start=row['start'], end=row['end'], label=label)
        by_thread[row['globalTid']].append(r)
        stage, fields = parse_label(label)
        if stage in ('forward', 'backward', 'loss'):
            phases[row['globalTid'] & ~0xFFFFFF].append((row['start'], row['end'], fields))
        if stage == 'capture':
            capture_windows.append((int(fields['source_begin']), int(fields['source_end'])))
    if not any(r['label'].startswith('helix/') for rs in by_thread.values() for r in rs):
        raise ValueError('No Helix NVTX markers. A historical unmarked report cannot recover operator ownership.')
    kernels = [dict(row) for row in connection.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
    if any(k.get('graphId') for k in kernels):
        raise ValueError('CUDA Graph attribution is not implemented; use normal launches for this capture.')
    wanted = {(k['globalPid'], k['correlationId']) for k in kernels}
    apis = collections.defaultdict(list)
    # Prefer runtime launch records; driver is a fallback for driver-only launches.
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:
            continue
        for raw in connection.execute('select * from ' + table + ' order by start'):
            a = dict(raw)
            key = (a['globalTid'] & ~0xFFFFFF, a['correlationId'])
            if key in wanted:
                apis[a['globalTid']].append((key, a))
    launch_ranges = {}
    for tid, calls in apis.items():
        rs = by_thread[tid]
        active, ends, pos = {}, [], 0
        for key, a in sorted(calls, key=lambda item: item[1]['start']):
            while pos < len(rs) and rs[pos]['start'] <= a['start']:
                active[pos] = rs[pos]
                heapq.heappush(ends, (rs[pos]['end'], pos))
                pos += 1
            while ends and ends[0][0] < a['start']:
                _, expired = heapq.heappop(ends)
                active.pop(expired, None)
            matched = sorted((r for r in active.values() if r['end'] >= a['end']),
                             key=lambda r: (r['start'], -r['end']))
            if key not in launch_ranges or len(matched) > len(launch_ranges[key][1]):
                launch_ranges[key] = (a, matched)
    for pid in phases:
        phases[pid].sort()
    phase_starts = {pid: [p[0] for p in ps] for pid, ps in phases.items()}
    result = []
    for k in kernels:
        api, ranges = launch_ranges.get((k['globalPid'], k['correlationId']), (None, []))
        name = strings.get(k.get('demangledName'), strings.get(k.get('shortName'), '<unknown>'))
        category, stage, fields, owner = classify(ranges, name)
        # Autograd may launch on another CPU thread. Recover benchmark step from
        # the same process's forward/backward call interval, never GPU wall time.
        if api is not None and 'source_step' not in fields:
            ps = phases.get(k['globalPid'], [])
            idx = bisect.bisect_right(phase_starts.get(k['globalPid'], []), api['start']) - 1
            if idx >= 0 and ps[idx][1] >= api['end']:
                fields.setdefault('source_step', ps[idx][2].get('step'))
        source = fields.get('source_step', fields.get('step'))
        auto = [r['label'] for r in ranges if not r['label'].startswith('helix/')]
        backward_nodes = [label for label in auto if re.search(r'Backward\d*\b|AccumulateGrad',label)]
        normalize = lambda text: re.sub(r'(?:stashed\s+)?seq\s*=\s*\d+', '', text).strip()
        result.append(dict(globalPid=k['globalPid'], device=k['deviceId'], stream=k['streamId'],
                           correlationId=k['correlationId'], gpu_start_ns=k['start'], gpu_end_ns=k['end'],
                           gpu_duration_ns=k['end']-k['start'], kernel=name, category=category, stage=stage,
                           operator=normalize(auto[-1]) if auto else '',
                           autograd_node=normalize(backward_nodes[-1]) if backward_nodes else '',
                           source_step=int(source) if source is not None else None,
                           consume_step=fields.get('consume_step', ''), group_idx=fields.get('group_idx', ''),
                           rank=fields.get('rank', ''), factor=fields.get('factor', fields.get('active_factor', '')),
                           host_range_ns=owner['end']-owner['start'] if owner else '',
                           api_start_ns=api['start'] if api else '', api_end_ns=api['end'] if api else ''))
    return result, sorted(set(capture_windows))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(database, output, begin=None, end=None):
    if (begin is None) != (end is None):
        raise ValueError('Specify both --source-begin and --source-end, or neither')
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as c:
        rows, windows = read_attributed(c)
    if begin is None or end is None:
        if len(windows) != 1:
            raise ValueError('Specify --source-begin/--source-end, or capture exactly one Helix window')
        begin, end = windows[0]
    if begin >= end:
        raise ValueError('source window must be nonempty')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r['included'] = r['source_step'] is not None and begin <= r['source_step'] < end
    selected = [r for r in rows if r['included']]
    if not selected:
        raise ValueError('No source-attributed kernels in the requested window')
    write_csv(output/'kernel_attribution.csv', rows)
    grouped = collections.defaultdict(list)
    for r in selected:
        key = tuple(r[k] for k in ('globalPid','device','category','stage','rank','factor','group_idx','operator','autograd_node','kernel'))
        grouped[key].append(r)
    summary = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        row = dict(zip(('globalPid','device','category','stage','rank','factor','group_idx','operator','autograd_node','kernel'), key))
        row.update(calls=len(group), kernel_sum_ns=sum(r['gpu_duration_ns'] for r in group),
                   interval_union_ns=duration((r['gpu_start_ns'],r['gpu_end_ns']) for r in group))
        summary.append(row)
    write_csv(output/'operator_summary.csv', summary)
    devices = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in selected:
        devices[(r['globalPid'],r['device'])][r['category']].append((r['gpu_start_ns'],r['gpu_end_ns']))
    overlaps = []
    for (pid, device), categories in devices.items():
        comm = categories.get('nccl', [])
        bp = categories.get('bp', [])
        codec = categories.get('powersgd_encode', []) + categories.get('powersgd_decode', [])
        overlaps.append(dict(globalPid=pid, device=device,
                             category_union_ns={k: duration(v) for k,v in categories.items()},
                             nccl_bp_overlap_ns=intersection(comm,bp),
                             nccl_codec_overlap_ns=intersection(comm,codec),
                             nccl_without_bp_or_codec_ns=duration(comm)-intersection(comm,bp+codec)))
    report = dict(source_begin=begin, source_end=end, captured_kernel_count=len(rows),
                  selected_kernel_count=len(selected),
                  no_source_kernel_count=sum(r['source_step'] is None for r in rows),
                  selected_unclassified_count=sum(r['category'] in ('unclassified','hook_other') for r in selected),
                  devices=overlaps,
                  notes=['All times are ns; kernel times, not SM utilization or wire transfer time.',
                         'Each kernel has one category; interval unions remove concurrency double counting.',
                         'NCCL without BP/codec overlap can overlap forward/update or contain NCCL waiting.',
                         'CPU ranges are launch scopes, not GPU completion timers.',
                         'This file is one report only. Do not combine clocks of separate nodes directly.'])
    counts = collections.defaultdict(collections.Counter)
    for r in selected:
        counts[r['source_step']][r['category']] += 1
    report['per_source_kernel_counts'] = {str(k): dict(v) for k,v in sorted(counts.items())}
    present = {r['category'] for r in selected}
    report['warnings'] = ['Missing category: '+name for name in
                          ('bp','powersgd_encode','powersgd_decode','nccl') if name not in present]
    (output/'overlap_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite')
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-begin', type=int)
    parser.add_argument('--source-end', type=int)
    args = parser.parse_args()
    print(json.dumps(analyze(args.sqlite,args.output,args.source_begin,args.source_end),indent=2))
