"""A worked example, and the only place `__functions_in_module()` is exercised
end to end.

`scripts/check_cli.sh` builds this and drives every flag against it. The unit
tests cannot: calling `_discover[__functions_in_module()]()` from a function
that lives in the module being enumerated makes the parameter-domain expansion
recursive, so they pass explicit tuples instead.
"""

from bench import Benchmark, BenchSuite, Metric, keep

# Large enough that one iteration costs comfortably more than a hundred clock
# ticks, so `scripts/check_cli.sh` exercises the per-iteration sampling path
# rather than only the batched one; `--batched` covers the other.
comptime N = 1 << 19


def _make(n: Int) -> List[Int64]:
    var out = List[Int64](capacity=n)
    for i in range(n):
        out.append(Int64(i))
    return out^


def _helper_ignored_by_discovery(x: Int) -> Int:
    return x


def bench_sum(mut b: Benchmark) raises:
    var data = _make(N)
    b.throughput(Metric.elements(), N)

    def call() raises {imm data}:
        var total = Int64(0)
        for i in range(len(data)):
            total += data[i]
        keep(total)

    b.iter(call)
    keep(data)


def bench_max(mut b: Benchmark) raises:
    var data = _make(N)
    b.throughput(Metric.elements(), N)

    def call() raises {imm data}:
        var hi = Int64(0)
        for i in range(len(data)):
            if data[i] > hi:
                hi = data[i]
        keep(hi)

    b.iter(call)
    keep(data)


def main() raises:
    BenchSuite.run[__functions_in_module()](
        min_runtime_secs=0.05, num_warmup_iters=1, num_repetitions=3
    )
