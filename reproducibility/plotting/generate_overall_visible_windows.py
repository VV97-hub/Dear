#!/usr/bin/env python3
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path


COLORS = {
    "bg": "#FFFFFF",
    "panel": "#FAFAFA",
    "grid": "#E2E2E2",
    "border": "#E6E6E6",
    "text": "#222222",
    "muted": "#666666",
    "bw": "#F58518",
    "fw": "#4C78A8",
    "rs": "#54A24B",
    "rs_ov": "#1B7F3B",
    "ag": "#B279A2",
    "ag_ov": "#7A3D8A",
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


def segments_duration(segments):
    return sum(end - start for start, end in segments)


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


def rs_segments(obj):
    segments = rs_group_intervals(obj)
    if segments:
        return merge_intervals(segments)
    return [phase_ms(obj, "rs_phase")]


def rs_group_intervals(obj):
    segments = []
    for trace in obj["group_traces"]:
        if trace.get("rs_issue_ms") is None:
            continue
        if (
            trace.get("rs_complete_cuda_ms") is None
            and trace.get("rs_complete_ms") is None
            and trace.get("rs_global_release_ms") is None
        ):
            continue
        segments.append(rs_interval(trace))
    return segments


def forward_union(obj):
    return merge_intervals(
        (trace["forward_gate_ms"], trace["forward_end_ms"])
        for trace in obj["group_traces"]
        if trace.get("forward_gate_ms") is not None
        and trace.get("forward_end_ms") is not None
    )


def wait_update_intervals(obj):
    return merge_intervals(
        (event["start_ms"], event["end_ms"])
        for event in obj["events"]
        if event["type"] in {"ag_wait", "update"}
    )


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


def ratio_label(value):
    if value < 0.30:
        return "low"
    if value < 0.80:
        return "medium"
    return "high"


def summarize(items):
    rows = []
    for obj in items:
        bw = phase_ms(obj, "backward_phase")
        rs_group = rs_group_intervals(obj)
        rs = merge_intervals(rs_group) if rs_group else [phase_ms(obj, "rs_phase")]
        ag = phase_ms(obj, "ag_phase")
        fw_union = forward_union(obj)
        wait_update = wait_update_intervals(obj)
        exposed_wait = subtract_intervals(wait_update, fw_union)
        rs_dur = segments_duration(rs)
        ag_dur = ag[1] - ag[0]
        rs_ov_segments = intersect_segments(bw, rs)
        rs_ov = segments_duration(rs_ov_segments)
        ag_ov = overlap_with_segments(ag, fw_union)
        rows.append(
            {
                "bw": bw,
                "rs_segments": rs,
                "rs_group_intervals": rs_group,
                "ag": ag,
                "fw_union": fw_union,
                "rs_dur": rs_dur,
                "rs_ov": rs_ov,
                "ag_ov": ag_ov,
                "rs_ratio": rs_ov / rs_dur if rs_dur > 0 else 0.0,
                "ag_ratio": ag_ov / ag_dur if ag_dur > 0 else 0.0,
                "exposed_wait": sum(e - s for s, e in exposed_wait),
                "rs_ov_segments": rs_ov_segments,
                "ag_ov_segments": intersect_segments(ag, fw_union),
                "exposed_wait_segments": exposed_wait,
            }
        )
    combos = Counter((ratio_label(r["rs_ratio"]), ratio_label(r["ag_ratio"])) for r in rows)
    bw_mean = mean_pair(rows, "bw")
    rs_draw_segments = merge_intervals(mean_segments([r["rs_group_intervals"] for r in rows]))
    return {
        "count": len(rows),
        "bw": bw_mean,
        "rs_segments": rs_draw_segments,
        "ag": mean_pair(rows, "ag"),
        "fw_union": mean_segments([r["fw_union"] for r in rows]),
        "rs_dur": st.mean(r["rs_dur"] for r in rows),
        "rs_ov": st.mean(r["rs_ov"] for r in rows),
        "ag_ov": st.mean(r["ag_ov"] for r in rows),
        "rs_ratio": st.mean(r["rs_ratio"] for r in rows),
        "ag_ratio": st.mean(r["ag_ratio"] for r in rows),
        "exposed_wait": st.mean(r["exposed_wait"] for r in rows),
        "rs_ov_segments": intersect_segments(bw_mean, rs_draw_segments),
        "ag_ov_segments": mean_segments([r["ag_ov_segments"] for r in rows]),
        "exposed_wait_segments": mean_segments([r["exposed_wait_segments"] for r in rows]),
        "combos": combos,
    }


def round_up(value, step):
    return int(math.ceil(value / step) * step)


def build_svg(job_id, first, last):
    width, height = 1500, 1040
    left, right = 130, 40
    usable_w = width - left - right
    xmax = round_up(
        max(
            max(e for _, e in first["fw_union"]),
            max(e for _, e in last["fw_union"]),
            first["ag"][1],
            last["ag"][1],
            max(e for _, e in first["rs_segments"]),
            max(e for _, e in last["rs_segments"]),
        )
        + 20,
        10,
    )

    def sx(value):
        return left + value * usable_w / xmax

    def draw_panel(parts, y0, title, subtitle, stats):
        parts.append(rect(20, y0, width - 40, 395, COLORS["panel"], rx=10, stroke=COLORS["border"]))
        parts.append(txt(35, y0 + 28, title, size=19, weight="bold"))
        parts.append(txt(width - 35, y0 + 28, subtitle, size=12, anchor="end", fill=COLORS["muted"]))
        for tick in range(0, xmax + 1, 25):
            x = sx(tick)
            parts.append(f'<line x1="{x:.2f}" y1="{y0 + 68:.2f}" x2="{x:.2f}" y2="{y0 + 318:.2f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
            parts.append(txt(x, y0 + 58, tick, size=10, anchor="middle", fill=COLORS["muted"]))

        y = y0 + 88
        labels = [
            ("Backward compute", stats["bw"], COLORS["bw"]),
            ("RS comm", None, COLORS["rs"]),
            ("AG comm", stats["ag"], COLORS["ag"]),
            ("Exposed AG wait", None, COLORS["wait"]),
            ("Forward compute union", None, COLORS["fw"]),
        ]
        for label, interval, color in labels:
            parts.append(txt(left - 12, y + 21, label, size=12, anchor="end"))
            if interval is not None:
                parts.append(rect(sx(interval[0]), y, max(1.4, sx(interval[1]) - sx(interval[0])), 28, color))
            elif label == "RS comm":
                for s, e in stats["rs_segments"]:
                    parts.append(rect(sx(s), y, max(1.4, sx(e) - sx(s)), 28, color))
            y += 42

        for s, e in stats["rs_ov_segments"]:
            parts.append(rect(sx(s), y0 + 130, max(1.4, sx(e) - sx(s)), 28, COLORS["rs_ov"]))
        for s, e in stats["ag_ov_segments"]:
            parts.append(rect(sx(s), y0 + 172, max(1.4, sx(e) - sx(s)), 28, COLORS["ag_ov"]))
        for s, e in stats["exposed_wait_segments"]:
            parts.append(rect(sx(s), y0 + 214, max(1.4, sx(e) - sx(s)), 28, COLORS["wait"], opacity=0.62))
        for s, e in stats["fw_union"]:
            parts.append(rect(sx(s), y0 + 256, max(1.4, sx(e) - sx(s)), 28, COLORS["fw"]))

        y = y0 + 322
        parts.append(txt(left, y, f"RS overlap: {stats['rs_ov']:.2f} ms / {stats['rs_dur']:.2f} ms = {stats['rs_ratio'] * 100:.2f}%", size=13, weight="bold"))
        parts.append(txt(left, y + 24, f"AG overlap with forward compute union: {stats['ag_ov']:.2f} ms / {stats['ag'][1] - stats['ag'][0]:.2f} ms = {stats['ag_ratio'] * 100:.2f}%", size=13, weight="bold"))
        parts.append(txt(left, y + 48, f"Avg exposed AG wait: {stats['exposed_wait']:.2f} ms", size=12, fill=COLORS["muted"]))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, COLORS["bg"]),
        txt(20, 30, f"Forward-Compute-Union Overlap for Job {job_id}", size=24, weight="bold"),
        txt(20, 52, "AG overlap uses union(all forward_gate -> forward_end). Red is exposed wait after removing any time with forward compute.", size=12, fill=COLORS["muted"]),
    ]
    draw_panel(parts, 70, "First 500 cycles: Uncompressed", "cycles 0-499", first)
    draw_panel(parts, 505, "Last 500 cycles: Compressed", "cycles 510-1009", last)
    parts.append("</svg>")
    return "\n".join(parts)


def build_analysis(input_path, first, last):
    def combo_lines(title, stats):
        lines = [title]
        for (rs_label, ag_label), count in sorted(stats["combos"].items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- RS={rs_label}, AG={ag_label}: {count}/{stats['count']} = {count / stats['count'] * 100:.2f}%")
        return lines

    lines = [
        f"数据文件: {input_path}",
        "RS overlap口径: RS通信窗口优先使用每组 rs_issue -> rs_complete_cuda；旧数据缺少rs_complete_cuda时回退到rs_complete/rs_global_release。",
        "AG overlap口径: AG通信窗口与 union(all forward_gate -> forward_end) 求交集。",
        "红色 exposed AG-wait: ag_wait/update 扣除任意forward正在执行的时间。",
        "",
    ]
    lines.extend(combo_lines("前500轮(不压缩)", first))
    lines.extend(combo_lines("后500轮(压缩)", last))
    lines.extend(
        [
            "",
            f"前500轮: RS overlap {first['rs_ratio'] * 100:.2f}%, AG overlap {first['ag_ratio'] * 100:.2f}%, exposed AG-wait {first['exposed_wait']:.2f} ms。",
            f"后500轮: RS overlap {last['rs_ratio'] * 100:.2f}%, AG overlap {last['ag_ratio'] * 100:.2f}%, exposed AG-wait {last['exposed_wait']:.2f} ms。",
        ]
    )
    return "\n".join(lines)


def main():
    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    job_id = sys.argv[3] if len(sys.argv) > 3 else "960291"
    objs = [json.loads(line) for line in input_path.open(encoding="utf-8") if line.strip()]
    first = summarize(objs[:500])
    last = summarize(objs[-500:])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"overlap_{job_id}_analysis_exposed_wait.txt").write_text(build_analysis(input_path, first, last), encoding="utf-8")
    (output_dir / f"overlap_{job_id}_gantt_exposed_wait.svg").write_text(build_svg(job_id, first, last), encoding="utf-8")
    for path in sorted(output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
