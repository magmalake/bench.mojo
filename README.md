# bench.mojo

A small self-hosted benchmark harness for Mojo: discovers the benchmarks in a
file, calibrates an iteration count, times several repetitions, and prints a
table or a JSON array.

It exists because `std.benchmark` cannot currently do this on both toolchains
magmalake targets, and because a mean on its own is not enough to report from.

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
| benchmark      | mean     | min - max           | iters x reps | rate         |
| -------------- | -------- | ------------------- | ------------ | ------------ |
| crc32          | 44.83 ms | 44.57 ms - 45.79 ms | 23 x 5       | 1.50 GB/s    |
| murmur3_x86_32 | 39.91 ms | 39.74 ms - 40.55 ms | 26 x 5       | 1.68 GB/s    |
| xxh64          | 51.94 ms | 51.66 ms - 52.56 ms | 20 x 5       | 1.29 GB/s    |
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
summary. That is what makes trend reporting possible later.

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

Statistics over `runs_ns` are mean, min, max, median, and the sample standard
deviation (n−1, matching pytest-benchmark). Throughput is
`count / mean_ns`, which lands in units of a billion per second: `GB/s` for
`Metric.bytes()`, `GElems/s` for `Metric.elements()`, `GFLOPS/s` for
`Metric.flops()`.

## The benchmark binary is a CLI

| flag | effect |
|---|---|
| `--list` | print benchmark names as a JSON array and exit |
| `--only A B …` | run only these |
| `--skip A B …` | run everything except these |
| `--json` | print results as JSON instead of the table |
| `--out PATH` | also write the JSON to `PATH` |

`--only` and `--skip` raise on a name that does not exist, rather than
silently running nothing.

## The JSON

One array, in the order the benchmarks ran. No commit and no timestamp: the
binary has no business shelling out to git, so whatever saves these wraps them
in that envelope.

```json
[
  {"name": "crc32", "unit": "ns", "iters": 23, "reps": 5,
   "mean_ns": 44834000.0, "min_ns": 44573931.0, "max_ns": 45787931.0,
   "median_ns": 44659827.0, "stddev_ns": 512344.1,
   "runs_ns": [44573931.0, 44659827.0, 45378517.0, 45787931.0, 44030066.0],
   "throughput_metric": "bytes", "throughput_unit": "GB/s",
   "throughput_count": 67108864, "throughput": 1.4968}
]
```

`throughput_*` fields are omitted entirely when a benchmark declares no
metric.

## Conventions

**Every `bench_*` function in the module is discovered**, and must have the
signature `def bench_name(mut b: Benchmark) raises`. There is no comptime
type-equality predicate on either toolchain right now, so a `bench_*` helper
with a different signature is a compile error rather than a silent skip —
prefix helpers with `_`.

**Benchmark names are the function names minus nothing.** `bench_crc32`
reports as `bench_crc32`; that is what `--only` takes and what lands in the
JSON, so renaming a function renames its history.

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
bench = ["nightly", "bench"]
bench-stable = ["stable", "bench"]
```

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
