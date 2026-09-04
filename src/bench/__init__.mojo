"""`bench` — a small self-hosted benchmark harness for Mojo.

Discovers `bench_*` functions in a module, calibrates an iteration count,
times several repetitions, and reports a table or JSON. Works on stable Mojo
1.0.0 and nightly, which `std.benchmark` currently does not.

    from bench import Benchmark, BenchSuite, Metric, keep

    def bench_sum(mut b: Benchmark) raises:
        var data = build(1 << 20)
        b.throughput(Metric.elements(), 1 << 20)

        @parameter
        def call() raises:
            keep(total(data))

        b.iter[call]()
        keep(data)

    def main() raises:
        BenchSuite.run[__functions_in_module()]()
"""

from .suite import (
    Benchmark,
    BenchResult,
    BenchSuite,
    Metric,
    Stability,
    keep,
)
