# bench.mojo

A small self-hosted benchmark harness for Mojo: discovers the benchmarks in a
file, calibrates an iteration count, times several repetitions, and prints a
table or a JSON array.

It exists because `std.benchmark` cannot express a benchmark that reads data
it did not build inside the timed region — not on nightly, and the ways round
that differ per toolchain — and because a mean on its own is not enough to
report from.

```mojo
from bench import Benchmark, BenchSuite, Metric, keep

comptime SIZE = 64 * 1024 * 1024


def bench_crc32(mut b: Benchmark) raises:
    var data = _make_buffer(SIZE)        # setup is outside iter, so untimed
    b.throughput(Metric.bytes(), SIZE)

    @parameter
    def call() raises:
        var h = crc32(Span(data))
        keep(h)                          # stop the optimiser deleting the work

    b.iter[call]()
    keep(data)


def main() raises:
    BenchSuite.run[__functions_in_module()]()
```

```
clock: 1.00 us resolution; per-iteration sampling above 100.00 us per iteration
| benchmark            | p50      | p90      | mean     | min - max           | iters x reps    | rate      |
| -------------------- | -------- | -------- | -------- | ------------------- | --------------- | --------- |
| bench_crc32          | 45.21 ms | 45.88 ms | 45.33 ms | 45.14 ms - 47.02 ms | 26 x 5 per-iter | 1.48 GB/s |
| bench_murmur3_x86_32 | 40.14 ms | 40.55 ms | 40.21 ms | 40.08 ms - 41.83 ms | 30 x 5 per-iter | 1.67 GB/s |
| bench_xxh64          | 52.03 ms | 52.61 ms | 52.17 ms | 51.92 ms - 55.10 ms | 22 x 5 per-iter | 1.29 GB/s |
```

## Why not `std.benchmark`

Two reasons, one hard and one soft.

**The hard one.** `Bencher.iter` has lost its parameter form on nightly, and
the value form that remains will not accept a `@parameter` closure — while a
plain closure cannot infer a capture convention on *either* toolchain
(`Could not infer capture convention of the captured value`). Together those
mean nightly `std.benchmark` cannot express a benchmark that reads data it did
not construct inside the timed region, which is most of them. Owning the sixty
lines that actually do the timing sidesteps the whole problem, and this
harness compiles unchanged on stable 1.0.0 and nightly.

**The soft one.** `Bench.dump_report` gives you a mean. A mean cannot tell you
whether a 4% move is a regression or the machine being busy, so `_run_one`
keeps *every* per-repetition timing and the JSON carries them alongside the
summary. That is what makes trend reporting possible later — and where the
work is cheap enough per iteration to be timed individually, it keeps the
iterations too, which is what makes a **p90** possible rather than a guess.

`keep` is the one piece worth keeping, and `bench` re-exports it, so benchmark
files import from one place.

## Credit

The design follows [marrow](https://github.com/kszucs/marrow)'s `BenchSuite`
(Apache-2.0, Krisztián Szűcs) closely: discovery via `__functions_in_module()`,
a `--list` / `--only` / `--json` CLI on the benchmark binary itself, the
throughput declaration read out of a probe call, and keeping per-repetition
timings rather than a mean. Reimplemented rather than vendored — marrow pins
an older toolchain and several of the pieces it uses have since moved — but
the shape is theirs, and it is the most complete benchmark plumbing in the
Mojo ecosystem.

## How it measures

For each benchmark, in order:

1. **A probe call** with `num_iters = 1`, whose only job is to read the
   `b.throughput(...)` declaration out of the body.
2. **Warmup** — `num_warmup_iters` untimed passes (default 2).
3. **Calibration** — the iteration count grows until one pass exceeds
   `min_runtime_secs` (default 1.0), capped at `max_iters` (default 100
   million). The cap matters: a body the optimiser reduces to almost nothing
   needs a hundred million iterations to fill a second, and without a ceiling
   calibration alone runs for the better part of a minute.
4. **Measurement** — `num_repetitions` timed passes (default 5). Each
   contributes one `runs_ns` entry, the mean nanoseconds per iteration for
   that pass.

The benchmark body runs once per phase, so **anything outside `b.iter` is
re-executed but never timed**. That is the point: build inputs there, and the
timed region stays honest. It also means the body must be cheap to re-enter —
if setup dominates, hoist it or shrink the input.

Statistics are mean, min, max, median, the sample standard deviation (n−1,
matching pytest-benchmark), and — in per-iteration mode, below — p50, p90 and
p99. Throughput is `count / mean_ns`, which lands in units of a billion per
second: `GB/s` for `Metric.bytes()`, `GElems/s` for `Metric.elements()`,
`GFLOPS/s` for `Metric.flops()`. The rate stays against the mean deliberately:
it is the one central value both sampling modes have, so a series stays
comparable across a run that switched modes.

## p50 and p90, and the mode that earns them

`runs_ns` looks like a sample vector and **is not**. Each entry is one
repetition, and a repetition is already the mean of `iters` iterations — a
default run gives you three or five numbers, each an average of hundreds. A
percentile over that would be a percentile over a handful of averages, and the
averaging is precisely what erases the tail a p90 is asked about. Printing one
anyway would be worse than printing nothing, because it would look rigorous.

This is not theoretical. The same parquet.mojo benchmarks, batched (top) and
per-iteration (bottom), on one machine:

```
| benchmark       | mean    | min - max         | iters x reps |
| bench_read_big  | 4.54 ms | 4.53 ms - 4.55 ms | 266 x 3      |
| bench_read_wide | 4.84 ms | 4.76 ms - 4.98 ms | 240 x 3      |

| benchmark       | p50     | p90     | mean    | min - max          |
| bench_read_big  | 4.53 ms | 4.59 ms | 4.54 ms | 4.33 ms - 6.42 ms  |
| bench_read_wide | 5.04 ms | 5.27 ms | 5.13 ms | 4.91 ms - 10.42 ms |
```

The means agree to a fraction of a percent — and the batched row's `min - max`
claims a 0.4% spread on a read whose slowest iteration is **2.2× its
fastest**. Averaging 240 reads into one number per repetition is what erased
that, which is exactly why a percentile over `runs_ns` would have been a
fiction.

So the harness measures the distribution rather than inferring it, in one of
two modes, and **every row and every JSON result says which**:

| mode | how it times | what you get |
|---|---|---|
| **per-iteration** | one `perf_counter_ns` pair per call of the closure | `p50`, `p90`, `p99` over real iterations |
| **batched** | one pair around the whole loop, as before | no percentiles at all — `n/a` in the table, keys absent from the JSON |

**The threshold is measured, not assumed**, at suite construction, and it is
the clock's *tick* rather than its call cost. Those are two very different
numbers: on macOS/arm64 a `perf_counter_ns` read costs about **13 ns** while
the clock only advances in **1000 ns** steps. Judging by the call cost would
have enabled per-iteration timing at around 1.5 µs an iteration, where every
sample carries up to 64% quantisation error — a rigorous-looking number
describing the clock instead of the code, which is the exact failure this
whole mechanism exists to avoid. So `_timer_resolution_ns` measures both (a
tight loop of reads for the cost, a spin until the value changes for the tick)
and takes the larger, and an iteration must cost `resolution_factor` × that —
100 by default, holding the clock's contribution under 1% of a sample.

The header line says what came out and what threshold it produced:

```
clock: 1.00 us resolution; per-iteration sampling above 100.00 us per iteration
```

Below that line, batching is the only way to measure the thing at all, and the
harness says so instead of inventing a distribution. `--batched` forces it
everywhere, which is how you reproduce a number recorded before any of this
existed.

**Memory is bounded.** Retained samples are capped at `max_samples` (20,000 by
default, split evenly across repetitions) by reservoir sampling — Algorithm R,
so the draw is uniform over the whole run rather than the first N iterations,
which are the ones most contaminated by cache warming. When the cap bites, the
`iters x reps` cell says so: `10000 x 5 per-iter (4000 of 50000 kept)`. A
percentile over a subsample is honest only if it is labelled as one.

`min` is kept as a clean floor and is not a headline — quote the p50, with the
p90 beside it.

The **table** then picks the SI prefix that suits the number — 873 thousand
column chunks a second reads as `873.00 KElems/s`, not `0.0009 GElems/s` —
and widens its fraction as values shrink (two decimals at or above 1, three
below, four below 0.1). That is display only. The **JSON** always reports
against the fixed giga unit at full precision, so a stored series stays
comparable even as the displayed prefix changes.

## The benchmark binary is a CLI

| flag | effect |
|---|---|
| `--list` | print benchmark names as a JSON array and exit |
| `--only A B …` | run only these |
| `--skip A B …` | run everything except these |
| `--json` | print results as JSON instead of the table |
| `--out PATH` | also write the JSON to `PATH` |
| `--batched` | force batched timing, giving up the percentiles |

`--only` and `--skip` raise on a name that does not exist, rather than
silently running nothing.

## The JSON

Three parts: what it ran on, how it was run, and what it measured.

```json
{
  "host": {"cpu": "Apple M4", "os": "macos", "arch": "arm64",
           "physical_cores": 10, "logical_cores": 10, "performance_cores": 4,
           "memory_bytes": 25769803776, "accelerator": true},
  "config": {"min_runtime_secs": 1.0, "num_warmup_iters": 2,
             "num_repetitions": 5, "max_iters": 100000000,
             "max_samples": 20000, "resolution_factor": 100,
             "timer_resolution_ns": 1000.0, "force_batched": false},
  "results": [
    {"name": "crc32", "unit": "ns", "iters": 23, "reps": 5,
     "sampling": "per-iteration",
     "mean_ns": 44834000.0, "min_ns": 44573931.0, "max_ns": 45787931.0,
     "median_ns": 44659827.0, "stddev_ns": 512344.1,
     "p50_ns": 44659827.0, "p90_ns": 45378517.0, "p99_ns": 45702118.0,
     "samples": 115, "samples_seen": 115,
     "runs_ns": [44573931.0, 44659827.0, 45378517.0, 45787931.0, 44030066.0],
     "throughput_metric": "bytes", "throughput_unit": "GB/s",
     "throughput_count": 67108864, "throughput": 1.4968}
  ]
}
```

`sampling` is always present and says what every other number in the object
was computed over. `p50_ns`, `p90_ns`, `p99_ns`, `samples` and `samples_seen`
appear **only** on a per-iteration result — omitted rather than null, the same
way the `throughput_*` fields are omitted when no metric was declared, so a
consumer that finds a percentile key knows it was measured. `config` records
the measured clock resolution and the factor, so a reader can check the
threshold rather than take it on trust.

The per-iteration samples themselves are not dumped: twenty thousand floats
per benchmark would dwarf the rest of the report, and the statistics are what
a document or a trend line reads. `runs_ns` keeps its old meaning and its old
length — one entry per repetition — in both modes.

**`host` is there because a timing without its machine is not comparable to
anything.** Cores, OS and architecture come from `std.sys` and always resolve;
the CPU model and memory come from `sysctlbyname` on macOS and `/proc` on
Linux, and are left empty or zero if they cannot be read — unknown beats
invented. `accelerator` records only whether the toolchain can see one: there
is no GPU model, because these are CPU benchmarks and a GPU field would imply
the number depended on it.

The human table prints the same thing as a one-line header:

```
Apple M4 | macos/arm64 | 10 cores (4 performance) | 24 GiB
```

Still no commit and no timestamp: the binary has no business shelling out to
git, and whatever saves a report adds them. The machine is different — the
binary is the only thing that knows it for certain.

`throughput_*` fields are omitted entirely when a benchmark declares no
metric.

## Conventions

**Every `bench_*` function in the module is discovered**, and must have the
signature `def bench_name(mut b: Benchmark) raises`. There is no comptime
type-equality predicate on either toolchain right now, so a `bench_*` helper
with a different signature is a compile error rather than a silent skip —
prefix helpers with `_`.

**`keep` everything the timed closure captures, after `b.iter`.** Mojo
destroys a value at its last *use*, and a capture does not count — so a
schema, a selection list, or a buffer that the closure reads but the body
never mentions again is freed while the timed loop is still running. It
surfaces as a crash or as nonsense inside the library under test, not as a
lifetime error:

```mojo
def bench_read(mut b: Benchmark) raises:
    var file = build()
    var select: List[String] = ["id"]

    @parameter
    def call() raises:
        keep(read(file, select.copy()))

    b.iter[call]()
    keep(file)        # both of these are load-bearing
    keep(select)      # without it, `select` dies mid-benchmark
```

The rule: after `b.iter`, `keep` every variable the closure touched. It costs
nothing and the failure mode is ugly — one of these missing produced
`String span ends on 1 which is not a codepoint boundary` from deep inside a
decoder.

**Benchmark names are the function names minus nothing.** `bench_crc32`
reports as `bench_crc32`; that is what `--only` takes and what lands in the
JSON, so renaming a function renames its history.

## Publishing a history

`--out` gives you one report. `tools/publish.py` turns a stream of them into a
history, and `benchmarks/index.html` draws it:

```sh
pixi run -e bench bench -- --out report.json
python3 tools/publish.py --report report.json --out-dir gh-pages-out
```

That writes `results/<commit>.json` (the report, verbatim, plus commit and
timestamp), `results/latest.json`, and `benchmarks/data.json` — the rolling
file the dashboard reads, capped at 200 runs per machine.

**History is keyed by host.** A run on a laptop and a run on a CI runner are
not points on the same line, and averaging them would invent a trend that
never happened. Each run records which machine it came from and the dashboard
draws one series per benchmark per machine. Re-running the same commit on the
same machine replaces its entry rather than appending, so a retried job is not
two data points.

Regressions are flagged when the last five runs are slower than the previous
five by more than `max(5%, 2 × CV)` of that baseline — so a noisy benchmark
has to move further before it is called out. Where a benchmark declares a
throughput the comparison uses it; otherwise it uses the reciprocal of the
mean time, so a fall is always the bad direction.

Repositories in this org get all of it from one caller workflow — see
`.github/workflows/bench.yml` in any of the tins.

## Install as a mojoshelf tin

```sh
pixi shelf add bench-mojo     # pixi mode (git source dependency)
```

Or as a plain source dependency: `-I ../bench.mojo/src`, no FFI, no link
flags.

Benchmarks are usually a separate environment, so the tin does not weigh down
the environments that run tests:

```toml
[feature.bench.dependencies]
bench-mojo = { git = "https://github.com/magmalake/bench.mojo", rev = "..." }

[feature.bench.tasks]
bench = "mojo run -I src -I $CONDA_PREFIX/lib/mojo bench/bench_mine.mojo"

[environments]
bench = ["stable", "bench"]
```

Consume the **package** on stable, or the **source** on either.

A precompiled Mojo package (`.mojoc` — `.mojopkg` is the deprecated spelling,
as is `mojo package` for `mojo precompile`) is stamped with the exact compiler
version that produced it and refused by any other:

```
error: Mojo precompiled file is incompatible with the current version of the
Mojo compiler. Precompiled file 'bench.mojoc' version 1.0.0 is older than
compiler version 1.1.0.dev2026090105.
```

magmalake tins build with `mojo-compiler 1.0.0`, so a tin consumed as a
package is stable-only — an org-wide constraint that has nothing to do with
this harness. Vendoring the source (`-I ../bench.mojo/src`) works on both, and
that is what this repo's own CI does.

## Test

```sh
pixi run -e stable test         # stable Mojo 1.0.0
pixi run -e default test        # nightly
pixi run -e stable check-cli    # drives every flag against a built binary
pixi run -e stable example      # the worked example in tests/
```

`tests/example_bench.mojo` is the only place `__functions_in_module()` is
exercised end to end: calling `_discover[__functions_in_module()]()` from a
function that lives *in* the module being enumerated makes the parameter-domain
expansion recursive, so the unit tests pass explicit tuples instead and
`scripts/check_cli.sh` covers the real path.

Part of [magmalake](https://magmalake.org) — data workflows on Mojo.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
