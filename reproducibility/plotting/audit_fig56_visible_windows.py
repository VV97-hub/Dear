import collections
import importlib.util
import json
from pathlib import Path
import statistics as st
import sys
sys.dont_write_bytecode = True

root = Path('/mnt/c/Users/sg564/Desktop/Helix技术论文/实验文件/timeline甘特图统计')
path = root / 'job965368_统计结果_RS发起时间可见/overlap_bert_base_halfrankk_bs64_nw4_sl64_job965368.timeline.jsonl'
spec = importlib.util.spec_from_file_location('overall', root / 'generate_overall_gantt_exposed_wait.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
items = [json.loads(line) for line in path.open() if line.strip()]
result = {'file': str(path), 'total_cycles': len(items)}
for name, rows in [('dense', items[:500]), ('compressed', items[-500:])]:
    s = mod.summarize(rows)
    bad = []
    gaps = []
    update_contrib = []
    wait_only = []
    for o in rows:
        fw = mod.forward_union(o)
        wait = mod.merge_intervals((e['start_ms'], e['end_ms']) for e in o['events'] if e['type'] == 'ag_wait')
        waits = mod.segments_duration(mod.subtract_intervals(wait, fw))
        total = mod.segments_duration(mod.subtract_intervals(mod.wait_update_intervals(o), fw))
        wait_only.append(waits)
        update_contrib.append(total - waits)
        for t in o['group_traces']:
            if t['rs_complete_cuda_ms'] < t['rs_issue_ms']:
                bad.append({'cycle': o['cycle'], 'group': t['group_idx'], 'issue_ms': t['rs_issue_ms'], 'end_ms': t['rs_complete_cuda_ms']})
        gaps.append(min(t['rs_complete_ms'] - t['rs_complete_cuda_ms'] for t in o['group_traces']))
    result[name] = {
        'cycles': [rows[0]['cycle'], rows[-1]['cycle']],
        'groups': sum(len(o['group_traces']) for o in rows),
        'rs_end_before_issue': len(bad), 'examples': bad[:4],
        'bad_by_group': dict(collections.Counter(b['group'] for b in bad)),
        'gpu_start_fields_present': any(t.get('rs_start_cuda_ms') is not None for o in rows for t in o['group_traces']),
        'min_host_complete_minus_mapped_gpu_end_ms': {'min': min(gaps), 'median': st.median(gaps), 'max': max(gaps)},
        'rs_overlap_pct': s['rs_ratio'] * 100,
        'ag_overlap_pct': s['ag_ratio'] * 100,
        'exposed_wait_plus_update_ms': s['exposed_wait'],
        'ag_wait_excluding_forward_ms': st.mean(wait_only),
        'extra_update_excluding_forward_ms': st.mean(update_contrib),
    }
print(json.dumps(result, ensure_ascii=False, indent=2))
Path(__file__).with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
