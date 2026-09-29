#!/usr/bin/env python3
"""Aggregates bench/run.sh output (one JSON object per line) into markdown tables.

Each cell is the median across repeats, with (min-max) when there is more than one repeat."""
import json
import statistics
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
env = next((r["result"] for r in rows if r["impl"] == "meta"), {})


def label(r):
    impl = r["impl"]
    if r["result"].get("strip_raw"):
        impl += " strip_raw"
    if impl == "ruby":
        impl = "ruby+yjit" if r.get("yjit") else "ruby"
        if r["result"].get("mode") == "threads":
            impl += " threads"
    return impl


groups = defaultdict(list)
for r in rows:
    if r["impl"] == "meta":
        continue
    res = r["result"]
    key = (r["case"], label(r), res.get("chats"), res.get("requests"))
    groups[key].append(res)


def cell(values, fmt="{:.2f}"):
    if not values:
        return "-"
    med = statistics.median(values)
    if len(values) == 1:
        return fmt.format(med)
    return f"{fmt.format(med)} ({fmt.format(min(values))}-{fmt.format(max(values))})"


def pick(case, field, *, chats=None, requests=None):
    out = {}
    for (c, impl, ch, rq), results in groups.items():
        if c == case and (chats is None or ch == chats) and (requests is None or rq == requests):
            out[impl] = [x[field] for x in results if x.get(field) is not None]
    return out


IMPLS = ["rust", "ruby+yjit", "ruby", "ruby+yjit threads"]
print("## Environment\n")
for k, v in env.items():
    print(f"- **{k}**: {v}")
print()

print("## Latency (ms unless noted)\n")
print("| Case | Metric | " + " | ".join(IMPLS[:3]) + " |")
print("|---|---|" + "---:|" * 3)
for case, field, name, fmt in [
    ("first", "first_answer_ms", "process start -> first answer", "{:.1f}"),
    ("overhead", "p50_ms", "per-request overhead p50", "{:.3f}"),
    ("overhead", "p99_ms", "per-request overhead p99", "{:.3f}"),
    ("stream", "p50_ms", "streaming, us per chunk p50", "{:.2f}"),
    ("tools", "p50_ms", "3-round tool loop p50", "{:.3f}"),
    ("tools", "p99_ms", "3-round tool loop p99", "{:.3f}"),
    ("render", "p50_ms", "render 200-message payload p50", "{:.3f}"),
    ("render", "p99_ms", "render 200-message payload p99", "{:.3f}"),
]:
    if case == "first":
        # Each "first" row is one sample; the repeats are many processes.
        vals = pick(case, field)
        cells = []
        for i in IMPLS[:3]:
            v = sorted(vals.get(i, []))
            cells.append(f"{statistics.median(v):.1f} (p90 {v[int(0.9 * (len(v) - 1))]:.1f})" if v else "-")
        print(f"| {case} | {name} | " + " | ".join(cells) + " |")
    else:
        vals = pick(case, field)
        print(f"| {case} | {name} | " + " | ".join(cell(vals.get(i, []), fmt) for i in IMPLS[:3]) + " |")

print("\n## Throughput: N concurrent chats, 5 asks each, 50 ms mock delay\n")
print("| Chats | ideal req/s | rust | ruby+yjit (Async fibers) | ruby+yjit threads | rust RSS MiB | ruby fibers RSS MiB | ruby threads RSS MiB |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|")
chats_seen = sorted({r["result"]["requests"] // 5 for r in rows if r["case"] == "concurrent" and r["result"]["rounds"] == 5})
for ch in chats_seen:
    rps = pick("concurrent", "req_per_s", requests=ch * 5)
    rss = pick("concurrent", "rss_kib", requests=ch * 5)
    ideal = ch * 1000 / 50
    to_mib = lambda d, i: cell([x / 1024 for x in d.get(i, [])], "{:.0f}")
    print(f"| {ch} | {ideal:,.0f} | " + " | ".join(cell(rps.get(i, []), "{:,.0f}") for i in ["rust", "ruby+yjit", "ruby+yjit threads"])
          + " | " + " | ".join(to_mib(rss, i) for i in ["rust", "ruby+yjit", "ruby+yjit threads"]) + " |")

print("\n## Memory: RSS (MiB) with N chats in flight\n")
print("| Chats | rust baseline | rust in flight | ruby fibers baseline | ruby fibers in flight | ruby threads in flight |")
print("|---:|---:|---:|---:|---:|---:|")
for ch in sorted({r["result"]["chats"] for r in rows if r["case"] == "memory"}):
    base = pick("memory", "baseline_rss_kib", chats=ch)
    fl = pick("memory", "in_flight_rss_kib", chats=ch)
    m = lambda d, i: cell([x / 1024 for x in d.get(i, [])], "{:.1f}")
    print(f"| {ch} | {m(base, 'rust')} | {m(fl, 'rust')} | {m(base, 'ruby+yjit')} | {m(fl, 'ruby+yjit')} | {m(fl, 'ruby+yjit threads')} |")

print("\n## One long chat: sequential asks in one conversation, 0 ms delay\n")
print("| Impl | asks | elapsed s | RSS MiB |")
print("|---|---:|---:|---:|")
for (c, impl, ch, rq), results in sorted(groups.items(), key=lambda kv: kv[0][1]):
    if c == "concurrent" and results[0]["rounds"] > 5:
        print(f"| {impl} | {rq} | {cell([x['elapsed_s'] for x in results])} | {cell([x['rss_kib'] / 1024 for x in results], '{:.0f}')} |")
