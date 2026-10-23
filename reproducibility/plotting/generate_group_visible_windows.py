#!/usr/bin/env python3
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


COLORS = {
    "bg": "#FFFFFF",
    "panel": "#FAFAFA",
    "grid": "#E2E2E2",
    "border": "#E6E6E6",
    "text": "#222222",
    "muted": "#666666",
    "bw_bg": "#FDE0C5",
    "fw_bg": "#DCE9F6",
    "rs": "#4CAF50",
    "rs_ov": "#1B7F3B",
    "ag": "#9C6FB6",
    "ag_ov": "#6E3F8D",
    "wait": "#E45756",
}


def esc(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, content, size=12, anchor="start", weight="normal", fill=None):
    fill = fill or COLORS["text"]
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" '
        f'fill="{fill}">{esc(content)}</text>'
    )


def rect(x, y, w, h, fill, rx=4, stroke="none", opacity=None):
    opacity_attr = "" if opacity is None else f' opacity="{opacity:.2f}"'
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}"{opacity_attr}/>'
    )


def overlap_len(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(s, e) for s, e in merged]


def subtract_intervals(base_intervals, cut_intervals):
    output = []
    cuts = merge_intervals(cut_intervals)
    for start, end in base_intervals:
        cur = start
        for cut_start, cut_end in cuts:
            if cut_end <= cur or cut_start >= end:
                continue
            if cut_start > cur:
                output.append((cur, min(cut_start, end)))
            cur = max(cur, cut_end)
            if cur >= end:
                break
        if cur < end:
            output.append((cur, end))
    return output


def intersect_segments(window, segments):
    out = []
    for start, end in segments:
        s = max(window[0], start)
        e = min(window[1], end)
        if e > s:
            out.append((s, e))
    return out


def overlap_with_segments(window, segments):
    return sum(overlap_len(window[0], window[1], s, e) for s, e in segments)


def phase_ms(obj, key):
    start = obj["cycle_start_ts"]
    phase = obj[key]
    return ((phase["start_ts"] - start) * 1000.0, (phase["end_ts"] - start) * 1000.0)


def rs_interval(trace):
    end = trace.get("rs_complete_cuda_ms")
    if end is None:
        end = trace.get("rs_complete_ms")
    if end is None:
        end = trace.get("rs_global_release_ms")
    return (trace["rs_issue_ms"], end)


def forward_union(obj):
    return merge_intervals(
        (trace["forward_gate_ms"], trace["forward_end_ms"])
        for trace in obj["group_traces"]
        if trace.get("forward_gate_ms") is not None
        and trace.get("forward_end_ms") is not None
    )


def event_intervals(obj, event_types, group_idx=None):
    out = []
    for event in obj["events"]:
        if event["type"] not in event_types:
            continue
        if group_idx is not None and event.get("group_idx") != group_idx:
            continue
        out.append((event["start_ms"], event["end_ms"]))
    return out


def mean_pair(rows, key):
    return (st.mean(r[key][0] for r in rows), st.mean(r[key][1] for r in rows))


def mean_segments(segment_lists):
    max_len = max((len(segments) for segments in segment_lists), default=0)
    out = []
    for idx in range(max_len):
        starts = [segments[idx][0] for segments in segment_lists if idx < len(segments)]
        ends = [segments[idx][1] for segments in segment_lists if idx < len(segments)]
        if starts and ends:
            start = st.mean(starts)
            end = st.mean(ends)
            if end > start:
                out.append((start, end))
    return out


def label_ratio(value):
    if value < 0.30:
        return "low"
    if value < 0.80:
        return "medium"
    return "high"


def summarize_groups(items, group_count):
    rows_by_group = defaultdict(list)
    bw_rows = []
    for obj in items:
        bw = phase_ms(obj, "backward_phase")
        fw_union = forward_union(obj)
        bw_rows.append({"bw": bw})
        traces = {t["group_idx"]: t for t in obj["group_traces"]}
        for group_idx in range(group_count):
            t = traces[group_idx]
            rs = rs_interval(t)
            ag = (t["ag_issue_ms"], t["ag_wait_end_ms"])
            own_fw = (t["forward_gate_ms"], t["forward_end_ms"])
            blocked = merge_intervals(event_intervals(obj, {"ag_wait", "update"}, group_idx=group_idx))
            exposed_wait = subtract_intervals(blocked, fw_union)
            rs_dur = rs[1] - rs[0]
            ag_dur = ag[1] - ag[0]
            rows_by_group[group_idx].append(
                {
                    "bw": bw,
                    "rs": rs,
                    "ag": ag,
                    "own_fw": own_fw,
                    "exposed_wait_segments": exposed_wait,
                    "rs_ratio": overlap_len(rs[0], rs[1], bw[0], bw[1]) / rs_dur if rs_dur > 0 else 0.0,
                    "ag_ratio": overlap_with_segments(ag, fw_union) / ag_dur if ag_dur > 0 else 0.0,
                    "exposed_wait": sum(e - s for s, e in exposed_wait),
                }
            )

    groups = []
    for group_idx in range(group_count):
        rows = rows_by_group[group_idx]
        groups.append(
            {
                "group_idx": group_idx,
                "count": len(rows),
                "bw": mean_pair(rows, "bw"),
                "rs": mean_pair(rows, "rs"),
                "ag": mean_pair(rows, "ag"),
                "own_fw": mean_pair(rows, "own_fw"),
                "exposed_wait_segments": mean_segments([r["exposed_wait_segments"] for r in rows]),
                "rs_ratio": st.mean(r["rs_ratio"] for r in rows),
                "ag_ratio": st.mean(r["ag_ratio"] for r in rows),
                "exposed_wait": st.mean(r["exposed_wait"] for r in rows),
            }
        )
    return {"bw": mean_pair(bw_rows, "bw"), "groups": groups}


def round_up(value, step):
    return int(math.ceil(value / step) * step)


def build_svg(job_id, first, last):
    width = 1800
    height = 1740
    left = 150
    right = 92
    usable_w = width - left - right
    xmax = round_up(
        max(
            max(g["own_fw"][1] for g in first["groups"]),
            max(g["own_fw"][1] for g in last["groups"]),
            max(g["ag"][1] for g in first["groups"]),
            max(g["ag"][1] for g in last["groups"]),
            max(g["rs"][1] for g in first["groups"]),
            max(g["rs"][1] for g in last["groups"]),
        )
        + 10,
        25,
    )

    def sx(value):
        return left + value * usable_w / xmax

    def draw_panel(parts, y0, title, subtitle, stats):
        parts.append(rect(20, y0, width - 40, 760, COLORS["panel"], rx=10, stroke=COLORS["border"]))
        parts.append(txt(35, y0 + 28, title, size=20, weight="bold"))
        parts.append(txt(width - 35, y0 + 28, subtitle, size=12, anchor="end", fill=COLORS["muted"]))
        for tick in range(0, xmax + 1, 25):
            x = sx(tick)
            parts.append(f'<line x1="{x:.2f}" y1="{y0 + 56:.2f}" x2="{x:.2f}" y2="{y0 + 722:.2f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
            parts.append(txt(x, y0 + 46, tick, size=10, anchor="middle", fill=COLORS["muted"]))
        row_y = y0 + 66
        for group in stats["groups"]:
            cy = row_y + 14
            parts.append(txt(left - 12, cy, f"g{group['group_idx']}", size=11, anchor="end"))
            parts.append(rect(sx(stats["bw"][0]), row_y, sx(stats["bw"][1]) - sx(stats["bw"][0]), 18, COLORS["bw_bg"], rx=2, opacity=0.85))
            parts.append(rect(sx(group["own_fw"][0]), row_y, max(1.2, sx(group["own_fw"][1]) - sx(group["own_fw"][0])), 18, COLORS["fw_bg"], rx=2, opacity=0.95))
            parts.append(rect(sx(group["rs"][0]), row_y + 2, max(1.2, sx(group["rs"][1]) - sx(group["rs"][0])), 7, COLORS["rs"], rx=2))
            rs_ov = (max(group["rs"][0], stats["bw"][0]), min(group["rs"][1], stats["bw"][1]))
            if rs_ov[1] > rs_ov[0]:
                parts.append(rect(sx(rs_ov[0]), row_y + 2, max(1.2, sx(rs_ov[1]) - sx(rs_ov[0])), 7, COLORS["rs_ov"], rx=2))
            parts.append(rect(sx(group["ag"][0]), row_y + 9, max(1.2, sx(group["ag"][1]) - sx(group["ag"][0])), 7, COLORS["ag"], rx=2))
            ag_ov = intersect_segments(group["ag"], [g["own_fw"] for g in stats["groups"]])
            for s, e in ag_ov:
                parts.append(rect(sx(s), row_y + 9, max(1.2, sx(e) - sx(s)), 7, COLORS["ag_ov"], rx=2))
            for s, e in group["exposed_wait_segments"]:
                parts.append(rect(sx(s), row_y + 1, max(1.2, sx(e) - sx(s)), 16, COLORS["wait"], rx=2, opacity=0.55))
            parts.append(txt(width - 345, cy, f"RS {group['rs_ratio'] * 100:.1f}% | AG {group['ag_ratio'] * 100:.1f}% | exposed wait {group['exposed_wait']:.1f}ms", size=10, fill=COLORS["muted"]))
            row_y += 24

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, COLORS["bg"], rx=2),
        txt(20, 30, f"Per-Group Own Forward + Exposed AG Wait for Job {job_id}", size=24, weight="bold"),
        txt(20, 52, "Blue is this row group's own forward. Dark purple is AG vs global forward union. Red is wait/update only when no forward is active.", size=12, fill=COLORS["muted"]),
    ]
    draw_panel(parts, 70, "First 500 cycles: Uncompressed", "cycles 0-499", first)
    draw_panel(parts, 910, "Last 500 cycles: Compressed", "cycles 510-1009", last)
    parts.append("</svg>")
    return "\n".join(parts)


def build_analysis(input_path, first, last):
    lines = [
        f"数据文件: {input_path}",
        "绿色RS: 优先使用每组 rs_issue_ms -> rs_complete_cuda_ms；旧数据缺少rs_complete_cuda_ms时回退到rs_complete_ms/rs_global_release_ms。",
        "淡蓝色: 当前行group自己的 forward_gate_ms -> forward_end_ms。",
        "深紫色: 当前group的AG通信窗口与全局forward compute并集求交集。",
        "红色: ag_wait/update 中扣除全局forward compute并集后的暴露等待；如果任意forward正在执行，则不算红色AG-wait。",
        "",
    ]
    for title, stats in [("前500轮(不压缩)", first), ("后500轮(压缩)", last)]:
        lines.append(title)
        for group in stats["groups"]:
            lines.append(
                f"  g{group['group_idx']}: RS {group['rs_ratio'] * 100:.1f}% ({label_ratio(group['rs_ratio'])}), "
                f"AG-system-forward {group['ag_ratio'] * 100:.1f}% ({label_ratio(group['ag_ratio'])}), "
                f"own-forward {group['own_fw'][0]:.2f}~{group['own_fw'][1]:.2f} ms, "
                f"exposed AG-wait {group['exposed_wait']:.2f} ms"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    job_id = sys.argv[3] if len(sys.argv) > 3 else "960291"
    objs = [json.loads(line) for line in input_path.open(encoding="utf-8") if line.strip()]
    first = summarize_groups(objs[:500], len(objs[0]["group_traces"]))
    last = summarize_groups(objs[-500:], len(objs[0]["group_traces"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"overlap_{job_id}_per_group_analysis_exposed_wait.txt").write_text(build_analysis(input_path, first, last), encoding="utf-8")
    (output_dir / f"overlap_{job_id}_per_group_gantt_exposed_wait.svg").write_text(build_svg(job_id, first, last), encoding="utf-8")
    for path in sorted(output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
