"""Unit tests for the bench harness.

The statistics are checked against values worked out by hand, because the
whole point of keeping per-repetition timings is that the summary numbers are
trustworthy. The timing loop is checked for shape rather than duration: a test
that asserts on wall-clock is a test that fails on a busy CI runner.
"""

from std.testing import TestSuite, assert_equal, assert_true, assert_almost_equal

from bench import Benchmark, BenchResult, BenchSuite, Metric, keep
from bench.suite import (
    Host,
    _format_ns,
    _json_report,
    _json_results,
    _rate_str,
    _scaled_rate,
    _round2,
    _sorted,
    _table,
)


def _result(var runs: List[Float64]) -> BenchResult:
    return BenchResult(String("b"), 10, runs^, None, 0)


def _measured(var runs: List[Float64], count: Int) -> BenchResult:
    return BenchResult(String("b"), 10, runs^, Metric.bytes(), count)


# ── statistics ──────────────────────────────────────────────────────────────


def test_mean_of_known_values() raises:
    var r = _result([2.0, 4.0, 6.0, 8.0])
    assert_equal(r.mean_ns(), 5.0)


def test_min_and_max() raises:
    var r = _result([7.0, 3.0, 9.0, 5.0])
    assert_equal(r.min_ns(), 3.0)
    assert_equal(r.max_ns(), 9.0)


def test_median_odd_count() raises:
    var r = _result([5.0, 1.0, 3.0])
    assert_equal(r.median_ns(), 3.0)


def test_median_even_count_averages_the_middle_pair() raises:
    var r = _result([1.0, 2.0, 3.0, 4.0])
    assert_equal(r.median_ns(), 2.5)


def test_stddev_is_the_sample_deviation() raises:
    # mean 4; deviations -2,-1,0,1,2; sum of squares 10; 10/(5-1) = 2.5.
    var r = _result([2.0, 3.0, 4.0, 5.0, 6.0])
    assert_almost_equal(r.stddev_ns(), Float64(1.5811388300841898))


def test_stddev_of_a_single_run_is_zero() raises:
    var r = _result([42.0])
    assert_equal(r.stddev_ns(), 0.0)


def test_statistics_of_no_runs_are_zero() raises:
    var r = _result(List[Float64]())
    assert_equal(r.mean_ns(), 0.0)
    assert_equal(r.min_ns(), 0.0)
    assert_equal(r.max_ns(), 0.0)
    assert_equal(r.median_ns(), 0.0)
    assert_equal(r.stddev_ns(), 0.0)


def test_sorted_leaves_the_input_alone() raises:
    var original: List[Float64] = [3.0, 1.0, 2.0]
    var sorted = _sorted(original)
    assert_equal(sorted[0], 1.0)
    assert_equal(sorted[2], 3.0)
    assert_equal(original[0], 3.0)


# ── throughput ──────────────────────────────────────────────────────────────


def test_rate_is_count_over_mean_nanoseconds() raises:
    # 1e9 bytes averaging 1e9 ns is exactly 1 GB/s.
    var r = _measured([1.0e9], 1_000_000_000)
    assert_almost_equal(r.rate(), Float64(1.0))


def test_rate_is_zero_when_no_throughput_was_declared() raises:
    var r = _result([100.0])
    assert_equal(r.rate(), 0.0)


def test_metric_units() raises:
    assert_equal(Metric.bytes().unit, String("GB/s"))
    assert_equal(Metric.elements().unit, String("GElems/s"))
    assert_equal(Metric.flops().unit, String("GFLOPS/s"))
    assert_equal(Metric.bytes().base_unit, String("B/s"))
    assert_equal(Metric.elements().base_unit, String("Elems/s"))


def test_scaled_rate_picks_a_readable_prefix() raises:
    # Argument is in giga-per-second, as `BenchResult.rate()` returns.
    assert_equal(_scaled_rate(28.47, String("B/s")), String("28.47 GB/s"))
    assert_equal(_scaled_rate(0.873e-3, String("Elems/s")), String("873.00 KElems/s"))
    assert_equal(_scaled_rate(0.0189, String("Elems/s")), String("18.90 MElems/s"))
    assert_equal(_scaled_rate(0.0000005, String("Elems/s")), String("500.00 Elems/s"))


# ── the Benchmark handle ────────────────────────────────────────────────────


def test_iter_runs_the_closure_exactly_num_iters_times() raises:
    var calls = 0
    var b = Benchmark(7)

    @parameter
    def call() raises:
        calls += 1

    b.iter[call]()
    assert_equal(calls, 7)
    assert_true(b.elapsed >= 0)


def test_throughput_declaration_is_readable_afterwards() raises:
    var b = Benchmark(1)
    assert_true(not b.metric)
    b.throughput(Metric.elements(), 512)
    assert_true(Bool(b.metric))
    assert_equal(b.metric.value().name, String("elements"))
    assert_equal(b.count, 512)


# ── output ──────────────────────────────────────────────────────────────────


def test_json_carries_every_run_and_the_summary() raises:
    var results = List[BenchResult]()
    results.append(_measured([10.0, 20.0, 30.0], 60))
    var out = _json_results(results)
    assert_true('"name": "b"' in out)
    assert_true('"unit": "ns"' in out)
    assert_true('"reps": 3' in out)
    assert_true('"runs_ns": [10.0, 20.0, 30.0]' in out)
    assert_true('"mean_ns": 20.0' in out)
    assert_true('"min_ns": 10.0' in out)
    assert_true('"max_ns": 30.0' in out)
    assert_true('"median_ns": 20.0' in out)
    assert_true('"throughput_metric": "bytes"' in out)
    assert_true('"throughput_count": 60' in out)


def test_json_omits_throughput_when_undeclared() raises:
    var results = List[BenchResult]()
    results.append(_result([1.0]))
    var out = _json_results(results)
    assert_true("throughput" not in out)


def test_host_reports_a_usable_machine() raises:
    """CPU and memory may be unknown on an unfamiliar platform; the rest comes
    from std.sys and always resolves."""
    var h = Host.detect()
    assert_true(h.os == String("macos") or h.os == String("linux"))
    assert_true(h.arch == String("arm64") or h.arch == String("x86_64"))
    assert_true(h.physical_cores > 0)
    assert_true(h.logical_cores >= h.physical_cores)
    assert_true('"physical_cores": ' in h.as_json())
    assert_true(h.os in h.as_json())
    assert_true(len(h.summary().codepoints()) > 0)


def test_report_wraps_host_config_and_results() raises:
    var results = List[BenchResult]()
    results.append(_result([1.0]))
    var out = _json_report(Host.detect(), String('{"k": 1}'), results)
    assert_true(out.startswith('{"host": {'))
    assert_true('"config": {"k": 1}' in out)
    assert_true('"results": [' in out)
    assert_true(out.endswith("}"))


def test_json_separates_multiple_results() raises:
    var results = List[BenchResult]()
    results.append(_result([1.0]))
    results.append(_result([2.0]))
    var out = _json_results(results)
    assert_true(out.startswith("["))
    assert_true(out.endswith("]"))
    assert_true("}," in out)


def test_format_ns_picks_a_readable_unit() raises:
    assert_equal(_format_ns(500.0), String("500.00 ns"))
    assert_equal(_format_ns(1_500.0), String("1.50 us"))
    assert_equal(_format_ns(2_500_000.0), String("2.50 ms"))
    assert_equal(_format_ns(3_000_000_000.0), String("3.00 s"))


def test_rate_widens_the_fraction_as_the_value_shrinks() raises:
    # A fixed two decimals would render the last two as 0.01 and 0.00.
    assert_equal(_rate_str(28.4712), String("28.47"))
    assert_equal(_rate_str(0.8123), String("0.812"))
    assert_equal(_rate_str(0.008123), String("0.0081"))
    assert_equal(_rate_str(1.0), String("1.00"))


def test_round2_always_gives_two_decimals() raises:
    assert_equal(_round2(1.0), String("1.00"))
    assert_equal(_round2(1.5), String("1.50"))
    assert_equal(_round2(1.23456), String("1.23"))


def test_table_has_a_header_and_a_row_per_result() raises:
    var results = List[BenchResult]()
    results.append(_measured([100.0], 200))
    var out = _table(results)
    var lines = out.split("\n")
    assert_equal(len(lines), 3)
    assert_true("benchmark" in lines[0])
    assert_true("rate" in lines[0])
    assert_true("GB/s" in lines[2])


# ── discovery ───────────────────────────────────────────────────────────────
#
# These pass an explicit tuple rather than `__functions_in_module()`: calling
# that from a function *in* the module makes the parameter-domain expansion
# recursive ("function recursively calls itself in the parameter domain").
# `tests/example_bench.mojo` covers the `__functions_in_module()` path for
# real, through the CLI.


def _helper_not_a_bench(x: Int) -> Int:
    return x


def bench_one(mut b: Benchmark) raises:
    @parameter
    def call() raises:
        keep(1)

    b.iter[call]()


def bench_two(mut b: Benchmark) raises:
    @parameter
    def call() raises:
        keep(2)

    b.iter[call]()


def test_discovery_finds_bench_prefixed_functions_only() raises:
    var suite = BenchSuite()
    suite._discover[(_helper_not_a_bench, bench_one, bench_two)]()
    var names = List[String]()
    for b in suite.benches:
        names.append(String(b.name))
    assert_equal(len(names), 2)  # _helper_not_a_bench is filtered out
    assert_true(String("bench_one") in names)
    assert_true(String("bench_two") in names)


def test_registered_function_is_callable() raises:
    var suite = BenchSuite()
    suite._discover[(_helper_not_a_bench, bench_one, bench_two)]()
    var handle = Benchmark(3)
    suite.benches[0].bench_fn(handle)
    assert_equal(handle.num_iters, 3)


def test_selection_honours_only_and_skip() raises:
    var suite = BenchSuite()
    suite._discover[(_helper_not_a_bench, bench_one, bench_two)]()
    suite.only.append(String("bench_one"))
    assert_true(suite._selected(String("bench_one")))
    assert_true(not suite._selected(String("bench_two")))

    var other = BenchSuite()
    other._discover[(_helper_not_a_bench, bench_one, bench_two)]()
    other.skip.append(String("bench_one"))
    assert_true(not other._selected(String("bench_one")))
    assert_true(other._selected(String("bench_two")))


def test_run_one_calibrates_past_the_target() raises:
    var suite = BenchSuite(
        min_runtime_secs=0.01,
        num_warmup_iters=0,
        num_repetitions=3,
        max_iters=1000,
    )
    suite._discover[(_helper_not_a_bench, bench_one, bench_two)]()
    var result = suite._run_one(suite.benches[0])
    assert_equal(len(result.runs_ns), 3)
    assert_true(result.iters >= 1)
    assert_true(result.name == String("bench_one"))


def test_calibration_respects_max_iters() raises:
    # The body here is close to free, so calibration would otherwise climb to
    # the default ceiling and take most of a minute.
    var suite = BenchSuite(
        min_runtime_secs=10.0,
        num_warmup_iters=0,
        num_repetitions=1,
        max_iters=64,
    )
    suite._discover[(_helper_not_a_bench, bench_one, bench_two)]()
    var result = suite._run_one(suite.benches[0])
    assert_equal(result.iters, 64)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
