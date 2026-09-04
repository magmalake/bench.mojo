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
    _mean,
    _percentile,
    _rate_str,
    _sampling_cell,
    _scaled_rate,
    _round2,
    _sorted,
    _table,
    _timer_resolution_ns,
)


def _result(var runs: List[Float64]) -> BenchResult:
    return BenchResult(String("b"), 10, runs^, None, 0)


def _measured(var runs: List[Float64], count: Int) -> BenchResult:
    return BenchResult(String("b"), 10, runs^, Metric.bytes(), count)


def _sampled(var samples: List[Float64]) -> BenchResult:
    """A per-iteration result: three repetition means, N real samples."""
    var seen = len(samples)
    return BenchResult.sampled(
        String("b"), 10, [1.0, 2.0, 3.0], None, 0, samples^, seen
    )


def _ramp(n: Int) -> List[Float64]:
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(Float64(i + 1))
    return out^


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


def test_sorted_handles_the_heapsort_path() raises:
    """Above 48 elements `_sorted` switches algorithm; both must agree."""
    var n = 500
    var descending = List[Float64](capacity=n)
    for i in range(n):
        descending.append(Float64(n - i))
    var sorted = _sorted(descending)
    assert_equal(len(sorted), n)
    for i in range(n):
        assert_equal(sorted[i], Float64(i + 1))
    assert_equal(descending[0], Float64(n))  # input untouched


def test_sorted_heapsort_handles_duplicates_and_an_odd_length() raises:
    var values = List[Float64]()
    for i in range(101):
        values.append(Float64(i % 7))
    var sorted = _sorted(values)
    for i in range(1, len(sorted)):
        assert_true(sorted[i] >= sorted[i - 1])


# ── percentiles ─────────────────────────────────────────────────────────────
#
# The maths, on vectors where the answer is worked out by hand. Linear
# interpolation between closest ranks -- numpy's default -- so these are the
# same numbers numpy would print.


def test_percentile_of_a_ramp() raises:
    # 1..10: p50 sits between 5 and 6; p90 at h = 0.9 * 9 = 8.1, so
    # s[8] + 0.1 * (s[9] - s[8]) = 9 + 0.1.
    var v = _ramp(10)
    assert_almost_equal(_percentile(v, 0.5), Float64(5.5))
    assert_almost_equal(_percentile(v, 0.9), Float64(9.1))
    assert_almost_equal(_percentile(v, 0.0), Float64(1.0))
    assert_almost_equal(_percentile(v, 1.0), Float64(10.0))


def test_percentile_of_one_sample_is_that_sample() raises:
    var v: List[Float64] = [42.0]
    assert_equal(_percentile(v, 0.5), 42.0)
    assert_equal(_percentile(v, 0.9), 42.0)
    assert_equal(_percentile(v, 0.99), 42.0)


def test_percentile_of_two_samples_interpolates() raises:
    # h = q * (n - 1) = q, so p50 is the midpoint and p90 is nine tenths up.
    var v: List[Float64] = [10.0, 20.0]
    assert_almost_equal(_percentile(v, 0.5), Float64(15.0))
    assert_almost_equal(_percentile(v, 0.9), Float64(19.0))


def test_percentile_of_no_samples_is_zero() raises:
    assert_equal(_percentile(List[Float64](), 0.5), 0.0)


def test_percentile_clamps_a_fraction_outside_zero_to_one() raises:
    var v = _ramp(10)
    assert_equal(_percentile(v, -1.0), 1.0)
    assert_equal(_percentile(v, 7.0), 10.0)


def test_percentile_ignores_a_single_huge_outlier_and_the_mean_does_not() raises:
    """The whole argument for reporting a p50 instead of a mean.

    Ninety-nine iterations at 2 ms and one first call at 125 ms -- the shape of
    the pyarrow measurement that started this. The median does not move; the
    mean moves by more than half.
    """
    var v = List[Float64](length=99, fill=2.0)
    v.append(125.0)

    assert_equal(_percentile(v, 0.5), 2.0)
    assert_equal(_percentile(v, 0.9), 2.0)
    assert_almost_equal(_percentile(v, 1.0), Float64(125.0))

    assert_almost_equal(_mean(v), Float64(3.23))
    assert_true(_mean(v) > _percentile(v, 0.5) * 1.5)


def test_percentiles_are_none_in_batched_mode() raises:
    """A p90 over three repetition means would look rigorous and mean nothing,
    so there is no number to print at all."""
    var r = _result([10.0, 20.0, 30.0])
    assert_true(not r.per_iteration)
    assert_equal(r.sampling(), String("batched"))
    assert_true(not r.has_percentiles())
    assert_true(not r.p50_ns())
    assert_true(not r.p90_ns())
    assert_true(not r.p99_ns())
    assert_true(not r.percentile_ns(0.5))
    # The statistics that *were* measured over those means are still there.
    assert_equal(r.mean_ns(), 20.0)
    assert_equal(r.median_ns(), 20.0)


def test_percentiles_are_present_in_per_iteration_mode() raises:
    var r = _sampled(_ramp(10))
    assert_true(r.per_iteration)
    assert_equal(r.sampling(), String("per-iteration"))
    assert_true(r.has_percentiles())
    assert_almost_equal(r.p50_ns().value(), Float64(5.5))
    assert_almost_equal(r.p90_ns().value(), Float64(9.1))


def test_per_iteration_statistics_come_from_the_samples_not_the_runs() raises:
    """`runs_ns` is [1, 2, 3] in the fixture; every statistic must ignore it."""
    var r = _sampled(_ramp(10))
    assert_almost_equal(r.mean_ns(), Float64(5.5))
    assert_equal(r.min_ns(), 1.0)
    assert_equal(r.max_ns(), 10.0)
    assert_almost_equal(r.median_ns(), Float64(5.5))
    assert_equal(len(r.runs_ns), 3)  # kept, unchanged, for the JSON


def test_subsampled_is_true_only_when_the_cap_dropped_something() raises:
    var full = _sampled(_ramp(10))
    assert_true(not full.subsampled())
    assert_equal(full.num_samples(), 10)

    var capped = BenchResult.sampled(
        String("b"), 10, [1.0], None, 0, _ramp(10), 4000
    )
    assert_true(capped.subsampled())
    assert_equal(capped.num_samples(), 10)
    assert_equal(capped.samples_seen, 4000)


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


def test_iter_keeps_one_sample_per_iteration_when_asked() raises:
    var b = Benchmark(7, sample_each=True, max_samples=100)

    @parameter
    def call() raises:
        keep(1)

    b.iter[call]()
    assert_equal(len(b.samples_ns), 7)
    assert_equal(b.samples_seen, 7)
    # `elapsed` is the sum of the samples, so the per-repetition mean the suite
    # derives from it stays consistent with them.
    var total = Float64(0)
    for i in range(len(b.samples_ns)):
        total += b.samples_ns[i]
    assert_equal(Float64(b.elapsed), total)


def test_iter_keeps_no_samples_in_batched_mode() raises:
    var b = Benchmark(7)

    @parameter
    def call() raises:
        keep(1)

    b.iter[call]()
    assert_equal(len(b.samples_ns), 0)
    assert_equal(b.samples_seen, 0)
    assert_true(b.elapsed >= 0)


def test_reservoir_caps_retained_samples_but_counts_every_one() raises:
    """Past the cap the reservoir replaces rather than appends, so the memory
    is bounded and `samples_seen` still says what the percentiles are over."""
    var b = Benchmark(500, sample_each=True, max_samples=16)

    @parameter
    def call() raises:
        keep(1)

    b.iter[call]()
    assert_equal(len(b.samples_ns), 16)
    assert_equal(b.samples_seen, 500)


def test_reservoir_draws_from_the_whole_run_not_just_the_start() raises:
    """Retaining the first N would take the percentiles from the part of the
    run most contaminated by cache warming, so Algorithm R replaces."""
    var b = Benchmark(0, sample_each=True, max_samples=8)
    for i in range(1000):
        b._offer(Float64(i))
    assert_equal(len(b.samples_ns), 8)
    assert_equal(b.samples_seen, 1000)
    var late = 0
    for i in range(8):
        if b.samples_ns[i] >= 8.0:
            late += 1
    assert_true(late > 0)


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


def test_json_labels_the_sampling_mode_and_omits_unmeasured_percentiles() raises:
    var results = List[BenchResult]()
    results.append(_result([10.0, 20.0, 30.0]))
    var out = _json_results(results)
    assert_true('"sampling": "batched"' in out)
    assert_true("p50_ns" not in out)
    assert_true("p90_ns" not in out)
    assert_true("p99_ns" not in out)
    assert_true('"samples"' not in out)


def test_json_carries_the_percentiles_it_measured() raises:
    var results = List[BenchResult]()
    results.append(_sampled(_ramp(10)))
    var out = _json_results(results)
    assert_true('"sampling": "per-iteration"' in out)
    assert_true('"p50_ns": 5.5' in out)
    assert_true('"p90_ns": 9.1' in out)
    assert_true('"samples": 10' in out)
    assert_true('"samples_seen": 10' in out)
    # `runs_ns` keeps its old meaning and length in both modes.
    assert_true('"reps": 3' in out)
    assert_true('"runs_ns": [1.0, 2.0, 3.0]' in out)


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
    assert_true("p50" in lines[0])
    assert_true("p90" in lines[0])
    assert_true("rate" in lines[0])
    assert_true("GB/s" in lines[2])


def test_table_writes_n_a_where_it_measured_no_percentile() raises:
    var results = List[BenchResult]()
    results.append(_result([10.0, 20.0, 30.0]))
    var out = _table(results)
    assert_true("n/a" in out)
    assert_true("batched" in out)


def test_table_prints_the_percentiles_it_measured() raises:
    var results = List[BenchResult]()
    results.append(_sampled(_ramp(10)))
    var out = _table(results)
    assert_true("n/a" not in out)
    assert_true("5.50 ns" in out)  # p50
    assert_true("9.10 ns" in out)  # p90
    assert_true("per-iter" in out)


def test_sampling_cell_says_how_the_row_was_measured() raises:
    assert_equal(
        _sampling_cell(_result([1.0, 2.0, 3.0])), String("10 x 3 batched")
    )
    assert_equal(_sampling_cell(_sampled(_ramp(10))), String("10 x 3 per-iter"))
    var capped = BenchResult.sampled(
        String("b"), 10, [1.0, 2.0, 3.0], None, 0, _ramp(10), 4000
    )
    assert_equal(
        _sampling_cell(capped), String("10 x 3 per-iter (10 of 4000 kept)")
    )


# ── the clock, and the decision it drives ───────────────────────────────────


def test_timer_resolution_is_measured_and_plausible() raises:
    """It is the tick, not the call cost: on macOS/arm64 a `perf_counter_ns`
    read costs about 13 ns while the clock advances in 1000 ns steps, and it is
    the step that bounds what a single sample can say."""
    var r = _timer_resolution_ns()
    assert_true(r > 0.0)
    assert_true(r < 1_000_000.0)  # a clock coarser than a millisecond is a bug


def test_per_iteration_sampling_needs_work_worth_a_hundred_ticks() raises:
    var suite = BenchSuite(resolution_factor=100)
    suite.timer_resolution_ns = 1000.0  # pretend a 1 us tick

    assert_true(suite._samples_per_iteration(100_000.0))
    assert_true(suite._samples_per_iteration(1_000_000.0))
    assert_true(not suite._samples_per_iteration(99_000.0))
    assert_true(not suite._samples_per_iteration(50.0))


def test_batched_is_forced_by_the_flag_and_by_a_zero_budget() raises:
    var suite = BenchSuite()
    suite.timer_resolution_ns = 1.0
    assert_true(suite._samples_per_iteration(1_000_000.0))

    suite.force_batched = True
    assert_true(not suite._samples_per_iteration(1_000_000.0))

    var no_budget = BenchSuite(max_samples=0)
    no_budget.timer_resolution_ns = 1.0
    assert_true(not no_budget._samples_per_iteration(1_000_000.0))


def test_config_json_records_how_the_decision_was_made() raises:
    var suite = BenchSuite()
    var cfg = suite._config_json()
    assert_true('"max_samples": 20000' in cfg)
    assert_true('"resolution_factor": 100' in cfg)
    assert_true('"timer_resolution_ns": ' in cfg)
    assert_true('"force_batched": false' in cfg)


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


def bench_measurable(mut b: Benchmark) raises:
    """A body the clock can actually see one iteration of.

    `bench_one` costs less than a tick, so a single iteration of it times as
    zero however low the threshold is set -- which is the correct answer, and
    useless for exercising the sampling path. The sum runs over a list built
    outside the closure so the optimiser cannot fold it to a constant.
    """
    var data = List[Int64](capacity=50_000)
    for i in range(50_000):
        data.append(Int64(i))

    @parameter
    def call() raises:
        var total = Int64(0)
        for i in range(len(data)):
            total += data[i]
        keep(total)

    b.iter[call]()
    keep(data)


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


def test_run_one_samples_per_iteration_when_the_clock_can_afford_it() raises:
    """Driven by lowering the threshold rather than by finding a slow enough
    benchmark: a test that depends on wall-clock is a test that fails on a busy
    runner."""
    var suite = BenchSuite(
        min_runtime_secs=0.005,
        num_warmup_iters=0,
        num_repetitions=3,
        max_iters=200,
        max_samples=60,
    )
    suite.timer_resolution_ns = 1.0
    suite.resolution_factor = 1
    suite._discover[(_helper_not_a_bench, bench_measurable)]()
    var result = suite._run_one(suite.benches[0])

    assert_true(result.per_iteration)
    assert_equal(result.sampling(), String("per-iteration"))
    assert_true(len(result.samples_ns) > 0)
    assert_true(len(result.samples_ns) <= 60)  # 20 per repetition, capped
    assert_true(result.samples_seen >= len(result.samples_ns))
    assert_true(Bool(result.p50_ns()))
    # `runs_ns` is still one entry per repetition, whatever the mode.
    assert_equal(len(result.runs_ns), 3)


def test_run_one_batches_when_an_iteration_is_too_cheap_to_time() raises:
    var suite = BenchSuite(
        min_runtime_secs=0.005,
        num_warmup_iters=0,
        num_repetitions=3,
        max_iters=200,
    )
    suite.timer_resolution_ns = 1.0e12  # nothing can clear this
    suite._discover[(_helper_not_a_bench, bench_measurable)]()
    var result = suite._run_one(suite.benches[0])

    assert_true(not result.per_iteration)
    assert_equal(len(result.samples_ns), 0)
    assert_true(not result.p90_ns())
    assert_equal(len(result.runs_ns), 3)


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
