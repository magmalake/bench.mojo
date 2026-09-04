#!/usr/bin/env python3
"""Fold one benchmark report into a repository's rolling history.

The harness prints what it measured and what it measured it on; it does not
know which commit it is, when it ran, or what came before. This adds all
three, and is the only place that does.

    python3 publish.py --report r.json --out-dir gh-pages-out

writes two things under ``--out-dir``:

    results/<commit>.json    the full report for this commit, kept verbatim
    results/latest.json      the same file, under a stable name
    benchmarks/data.json     the rolling history the dashboard reads

**History is keyed by host.** A run on an M4 laptop and a run on a GitHub
ubuntu runner are not points on the same line, and silently averaging them
would invent a trend that never happened. Each run records which host it came
from; the dashboard draws one series per benchmark per host.

Runs are identified by (commit, host): re-running the same commit on the same
machine replaces its entry rather than appending a second one, so a retried CI
job does not show up as two data points.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

MAX_RUNS = 200
"""Per host. Enough for a long trend without the file growing without bound."""


def git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def host_key(host: dict) -> str:
    """A short, stable slug for one machine.

    Built from the fields that actually change how fast something runs. The
    CPU model is included when known, because "linux/x86_64, 4 cores" covers
    wildly different machines.
    """
    parts = [host.get("os", "unknown"), host.get("arch", "unknown")]
    cpu = (host.get("cpu") or "").strip()
    if cpu:
        parts.append(re.sub(r"[^a-z0-9]+", "-", cpu.lower()).strip("-"))
    cores = host.get("physical_cores") or 0
    if cores:
        parts.append(f"{cores}c")
    return "-".join(p for p in parts if p)


def load_history(path: Path) -> dict:
    if not path.exists():
        return {"hosts": {}, "runs": [], "benchmarks": []}
    with path.open() as f:
        history = json.load(f)
    history.setdefault("hosts", {})
    history.setdefault("runs", [])
    history.setdefault("benchmarks", [])
    return history


def result_entry(r: dict) -> dict:
    """The fields worth keeping per benchmark.

    `runs_ns` is deliberately dropped here: the per-repetition timings live in
    the per-commit snapshot, and carrying them in the rolling file would make
    it tens of megabytes for no gain on a trend line. `stddev_ns` is what the
    dashboard needs from them.

    `sampling` and the percentiles come through when the report carries them.
    They are absent from a batched result, and the missing-key filter below is
    what preserves that rather than substituting a zero: a history entry with
    no `p90_ns` measured no p90.
    """
    keep = (
        "mean_ns",
        "min_ns",
        "max_ns",
        "median_ns",
        "stddev_ns",
        "p50_ns",
        "p90_ns",
        "p99_ns",
        "sampling",
        "samples",
        "samples_seen",
        "iters",
        "reps",
        "throughput",
        "throughput_unit",
        "throughput_metric",
        "throughput_count",
    )
    return {k: r[k] for k in keep if k in r}


def merge(history: dict, report: dict, commit: str, ref: str, timestamp: str) -> dict:
    host = report["host"]
    key = host_key(host)
    history["hosts"][key] = host

    run = {
        "commit": commit,
        "short_commit": commit[:7],
        "timestamp": timestamp,
        "ref": ref,
        "host": key,
        "config": report.get("config", {}),
        "results": {r["name"]: result_entry(r) for r in report["results"]},
    }

    # (commit, host) identifies a run: a retried job replaces, never appends.
    history["runs"] = [
        r
        for r in history["runs"]
        if not (r.get("commit") == commit and r.get("host") == key)
    ]
    history["runs"].append(run)
    history["runs"].sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    # Cap per host, so a busy machine cannot evict another machine's history.
    kept: list[dict] = []
    seen: dict[str, int] = {}
    for r in history["runs"]:
        h = r.get("host", "unknown")
        seen[h] = seen.get(h, 0) + 1
        if seen[h] <= MAX_RUNS:
            kept.append(r)
    history["runs"] = kept

    names: dict[str, None] = {}
    for r in history["runs"]:
        for name in r.get("results", {}):
            names[name] = None
    history["benchmarks"] = sorted(names)

    # Prune hosts nothing references any more.
    live = {r.get("host") for r in history["runs"]}
    history["hosts"] = {k: v for k, v in history["hosts"].items() if k in live}
    return history


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True, type=Path, help="harness --out JSON")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--repo-root", default=Path("."), type=Path)
    ap.add_argument("--commit", help="defaults to HEAD of --repo-root")
    ap.add_argument("--ref", help="defaults to the current branch")
    ap.add_argument("--timestamp", help="ISO-8601 UTC; defaults to now")
    args = ap.parse_args()

    with args.report.open() as f:
        report = json.load(f)
    if "results" not in report or "host" not in report:
        print(
            f"{args.report} is not a harness report: expected 'host' and"
            " 'results' keys. A bare array means the repo is pinned to a"
            " bench-mojo revision older than the host block.",
            file=sys.stderr,
        )
        return 1

    commit = args.commit or git(args.repo_root, "rev-parse", "HEAD")
    ref = args.ref or git(args.repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if args.timestamp:
        timestamp = args.timestamp
    else:
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    results_dir = args.out_dir / "results"
    bench_dir = args.out_dir / "benchmarks"
    results_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    snapshot = dict(report)
    snapshot.update({"commit": commit, "ref": ref, "timestamp": timestamp})
    for name in (f"{commit}.json", "latest.json"):
        with (results_dir / name).open("w") as f:
            json.dump(snapshot, f, indent=2)
            f.write("\n")

    data_file = bench_dir / "data.json"
    history = merge(load_history(data_file), report, commit, ref, timestamp)
    with data_file.open("w") as f:
        json.dump(history, f, indent=2)
        f.write("\n")

    host = host_key(report["host"])
    runs_here = sum(1 for r in history["runs"] if r.get("host") == host)
    print(
        f"published {len(report['results'])} benchmarks for {commit[:7]}"
        f" on {host}: {runs_here} run(s) in this host's history,"
        f" {len(history['runs'])} total"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
