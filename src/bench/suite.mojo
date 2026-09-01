"""A small self-hosted benchmark harness: discovery, calibration, JSON output.

Modelled on the `BenchSuite` in [marrow](https://github.com/kszucs/marrow)
(Apache-2.0, Krisztián Szűcs), which is the most complete benchmark plumbing
in the Mojo ecosystem. Reimplemented rather than vendored: marrow pins an
older toolchain, and the pieces it leans on have since moved.

Why not `std.benchmark`? Two reasons, one hard and one soft.

The hard one: `Bencher.iter` has lost its parameter form on nightly, and the
value form that remains will not accept a `@parameter` closure -- while a
plain closure cannot infer a capture convention on either toolchain. So on
nightly, `std.benchmark` cannot express a benchmark that reads data it did not
construct inside the timed region. Owning the ~60 lines that actually do the
timing sidesteps that, and this file compiles on stable 1.0.0 and nightly.

The soft one: `Bench.dump_report` gives a mean and nothing else. Reports need
the spread, so `_run_one` keeps every per-repetition timing and the JSON
carries them alongside the summary statistics.

Writing a benchmark::

    from harness import Benchmark, BenchSuite, Metric, keep

    def bench_crc32(mut b: Benchmark) raises:
        var data = _make_buffer(SIZE)
        b.throughput(Metric.bytes(), SIZE)

        @parameter
        def call() raises:
            var h = crc32(Span(data))
            keep(h)

        b.iter[call]()
        keep(data)

    def main() raises:
        BenchSuite.run[__functions_in_module()]()

Every `bench_*` function in the module is discovered and must have the
signature above; name helpers with a leading underscore so discovery skips
them.

The resulting binary is its own CLI:

    --list            print benchmark names as a JSON array and exit
    --only A B ...    run only these
    --skip A B ...    run everything but these
    --json            print results as JSON instead of a table
    --out PATH        also write the JSON to PATH
"""

from std.benchmark.compiler import keep
from std.math import sqrt
from std.reflection import get_function_name
from std.sys import argv
from std.time import perf_counter_ns


# ── Throughput metrics ──────────────────────────────────────────────────────


@fieldwise_init
struct Metric(Copyable, Movable):
    """What a benchmark's per-iteration count means, and the rate's unit."""

    var name: String
    var unit: String

    @staticmethod
    def bytes() -> Self:
        return Self("bytes", "GB/s")

    @staticmethod
    def elements() -> Self:
        return Self("elements", "GElems/s")

    @staticmethod
    def flops() -> Self:
        return Self("flops", "GFLOPS/s")


# ── The value handed to each benchmark ──────────────────────────────────────


struct Benchmark:
    """Times a closure `num_iters` times and records the total.

    The suite calls a benchmark repeatedly with different `num_iters` -- once
    to read its throughput declaration, then to calibrate, then once per
    repetition -- so the body must be cheap to re-enter. Building the input
    outside `iter` is the point: that work is never timed.
    """

    var num_iters: Int
    var elapsed: Int
    var metric: Optional[Metric]
    var count: Int

    def __init__(out self, num_iters: Int):
        self.num_iters = num_iters
        self.elapsed = 0
        self.metric = None
        self.count = 0

    def throughput(mut self, var metric: Metric, count: Int):
        """Declare how much work one iteration does, for the rate column.

        Args:
            metric: What `count` measures.
            count: Bytes/elements/flops processed per iteration.
        """
        self.metric = metric^
        self.count = count

    def iter[f: def() capturing raises -> None](mut self) raises:
        """Run `f` `num_iters` times and record the elapsed nanoseconds."""
        var t0 = perf_counter_ns()
        for _ in range(self.num_iters):
            f()
        self.elapsed = perf_counter_ns() - t0


# ── Results ─────────────────────────────────────────────────────────────────


struct BenchResult(Copyable, Movable):
    """One benchmark's per-repetition timings, plus statistics over them."""

    var name: String
    var iters: Int
    var runs_ns: List[Float64]
    var metric: Optional[Metric]
    var count: Int

    def __init__(
        out self,
        var name: String,
        iters: Int,
        var runs_ns: List[Float64],
        var metric: Optional[Metric],
        count: Int,
    ):
        self.name = name^
        self.iters = iters
        self.runs_ns = runs_ns^
        self.metric = metric^
        self.count = count

    def mean_ns(self) -> Float64:
        if len(self.runs_ns) == 0:
            return 0.0
        var total = Float64(0)
        for i in range(len(self.runs_ns)):
            total += self.runs_ns[i]
        return total / Float64(len(self.runs_ns))

    def min_ns(self) -> Float64:
        if len(self.runs_ns) == 0:
            return 0.0
        var lo = self.runs_ns[0]
        for i in range(1, len(self.runs_ns)):
            if self.runs_ns[i] < lo:
                lo = self.runs_ns[i]
        return lo

    def max_ns(self) -> Float64:
        if len(self.runs_ns) == 0:
            return 0.0
        var hi = self.runs_ns[0]
        for i in range(1, len(self.runs_ns)):
            if self.runs_ns[i] > hi:
                hi = self.runs_ns[i]
        return hi

    def median_ns(self) -> Float64:
        var n = len(self.runs_ns)
        if n == 0:
            return 0.0
        var s = _sorted(self.runs_ns)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

    def stddev_ns(self) -> Float64:
        """Sample standard deviation (n-1), matching pytest-benchmark."""
        var n = len(self.runs_ns)
        if n < 2:
            return 0.0
        var m = self.mean_ns()
        var acc = Float64(0)
        for i in range(n):
            var d = self.runs_ns[i] - m
            acc += d * d
        return sqrt(acc / Float64(n - 1))

    def rate(self) -> Float64:
        """Throughput in units of a billion per second, or 0 if undeclared."""
        var mean = self.mean_ns()
        if self.count == 0 or mean <= 0.0:
            return 0.0
        return Float64(self.count) / mean


def _sorted(values: List[Float64]) -> List[Float64]:
    """Insertion sort. Repetition counts are single digits; nothing else fits
    better, and it avoids depending on a `List.sort` that keeps moving."""
    var out = values.copy()
    for i in range(1, len(out)):
        var v = out[i]
        var j = i - 1
        while j >= 0 and out[j] > v:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = v
    return out^


# ── Registration ────────────────────────────────────────────────────────────


@fieldwise_init
struct _Bench(Copyable):
    comptime fn_type = def(mut Benchmark) thin raises

    var bench_fn: Self.fn_type
    var name: String


# ── The suite ───────────────────────────────────────────────────────────────


struct BenchSuite(Movable):
    """Discovers `bench_*` functions, filters them, runs them, reports."""

    var benches: List[_Bench]
    var only: List[String]
    var skip: List[String]
    var list_only: Bool
    var json: Bool
    var out_path: String
    var min_runtime_secs: Float64
    var num_warmup_iters: Int
    var num_repetitions: Int
    var max_iters: Int

    def __init__(
        out self,
        *,
        min_runtime_secs: Float64 = 1.0,
        num_warmup_iters: Int = 2,
        num_repetitions: Int = 5,
        max_iters: Int = 100_000_000,
    ):
        self.benches = List[_Bench]()
        self.only = List[String]()
        self.skip = List[String]()
        self.list_only = False
        self.json = False
        self.out_path = String("")
        self.min_runtime_secs = min_runtime_secs
        self.num_warmup_iters = num_warmup_iters
        self.num_repetitions = num_repetitions
        self.max_iters = max_iters

    # -- discovery ----------------------------------------------------------

    def register[f: _Bench.fn_type](mut self):
        """Register one benchmark function."""
        self.benches.append(_Bench(f, String(get_function_name[f]())))

    def _discover[funcs: Tuple, /](mut self) raises:
        comptime for idx in range(len(funcs)):
            comptime func = funcs[idx]

            comptime if get_function_name[func]().startswith("bench_"):
                # No signature check: there is no comptime type-equality
                # predicate on either toolchain right now, so a `bench_*`
                # helper with a different signature is a compile error rather
                # than a silent skip. Prefix helpers with `_`.
                self.register[rebind[_Bench.fn_type](func)]()

    @staticmethod
    def run[
        funcs: Tuple, /
    ](
        *,
        min_runtime_secs: Float64 = 1.0,
        num_warmup_iters: Int = 2,
        num_repetitions: Int = 5,
        max_iters: Int = 100_000_000,
    ) raises:
        """Discover every `bench_*` in the module, then parse argv and run.

        Parameters:
            funcs: Pass `__functions_in_module()`.

        Args:
            min_runtime_secs: Calibration target for one repetition.
            num_warmup_iters: Untimed passes before calibrating.
            num_repetitions: Timed repetitions; each becomes a reported run.
            max_iters: Ceiling on the calibrated iteration count.
        """
        var suite = Self(
            min_runtime_secs=min_runtime_secs,
            num_warmup_iters=num_warmup_iters,
            num_repetitions=num_repetitions,
            max_iters=max_iters,
        )
        suite._discover[funcs]()
        suite.execute()

    # -- CLI ----------------------------------------------------------------

    def _parse_args(mut self) raises:
        var args = List[StaticString](argv())
        var i = 1
        var mode = String("")
        while i < len(args):
            var arg = String(args[i])
            if arg == "--list":
                self.list_only = True
                mode = String("")
            elif arg == "--json":
                self.json = True
                mode = String("")
            elif arg == "--out":
                if i + 1 >= len(args):
                    raise Error("--out needs a path")
                self.out_path = String(args[i + 1])
                i += 1
                mode = String("")
            elif arg == "--only":
                mode = String("only")
            elif arg == "--skip":
                mode = String("skip")
            elif arg.startswith("--"):
                raise Error("unknown flag: ", arg)
            elif mode == "only":
                self.only.append(arg)
            elif mode == "skip":
                self.skip.append(arg)
            else:
                raise Error(
                    "unexpected argument: ", arg, " (expected --only or --skip"
                    " first)"
                )
            i += 1

        # Fail loudly on a typo rather than silently running nothing.
        for name in self.only:
            if not self._known(name):
                raise Error("--only names an unknown benchmark: ", name)
        for name in self.skip:
            if not self._known(name):
                raise Error("--skip names an unknown benchmark: ", name)

    def _known(self, name: String) -> Bool:
        for b in self.benches:
            if b.name == name:
                return True
        return False

    def _selected(self, name: String) -> Bool:
        for s in self.skip:
            if s == name:
                return False
        if len(self.only) == 0:
            return True
        for o in self.only:
            if o == name:
                return True
        return False

    # -- execution ----------------------------------------------------------

    def _run_one(self, b: _Bench) raises -> BenchResult:
        """Warm up, calibrate an iteration count, then time N repetitions."""
        var target_ns = Int(self.min_runtime_secs * 1e9)
        if target_ns <= 0:
            target_ns = 100_000_000

        # The first call also carries the benchmark's throughput declaration,
        # which is just a statement in its body.
        var probe = Benchmark(1)
        b.bench_fn(probe)
        var metric = probe.metric.copy()
        var count = probe.count

        for _ in range(self.num_warmup_iters):
            var warm = Benchmark(1)
            b.bench_fn(warm)

        # Grow the iteration count until one repetition clears the target.
        #
        # `max_iters` is a real ceiling, not a formality: a benchmark whose
        # body the optimiser reduces to almost nothing needs a hundred million
        # iterations to fill a second, and without the cap calibration alone
        # runs for the better part of a minute. Hitting the cap means the
        # repetitions are shorter than `min_runtime_secs`, which is the right
        # trade -- a benchmark that cheap is measuring loop overhead anyway.
        var num_iters = 1
        while True:
            var cal = Benchmark(num_iters)
            b.bench_fn(cal)
            if cal.elapsed >= target_ns or num_iters >= self.max_iters:
                break
            if cal.elapsed <= 0:
                num_iters = min(num_iters * 10, self.max_iters)
            else:
                # 1.2x headroom so the next attempt usually lands past target.
                var scaled = (
                    Float64(target_ns)
                    * Float64(num_iters)
                    / Float64(cal.elapsed)
                    * 1.2
                )
                num_iters = min(max(Int(scaled), num_iters + 1), self.max_iters)

        var runs_ns = List[Float64](capacity=self.num_repetitions)
        for _ in range(self.num_repetitions):
            var rep = Benchmark(num_iters)
            b.bench_fn(rep)
            runs_ns.append(Float64(rep.elapsed) / Float64(num_iters))

        return BenchResult(
            String(b.name), num_iters, runs_ns^, metric^, count
        )

    def execute(mut self) raises:
        """Parse argv, run the selected benchmarks, and report."""
        self._parse_args()

        if self.list_only:
            var names = List[String]()
            for b in self.benches:
                names.append(String(b.name))
            print(_json_string_array(names))
            return

        var results = List[BenchResult]()
        for b in self.benches:
            if not self._selected(b.name):
                continue
            if not self.json:
                print("running", b.name, "...")
            results.append(self._run_one(b))

        var payload = _json_results(results)
        if self.json:
            print(payload)
        else:
            print(_table(results))
        if self.out_path.byte_length() > 0:
            with open(self.out_path, "w") as f:
                f.write(payload)
                f.write("\n")
            if not self.json:
                print("wrote", self.out_path)


# ── Output ──────────────────────────────────────────────────────────────────


def _json_string_array(names: List[String]) -> String:
    var out = String("[")
    for i in range(len(names)):
        if i > 0:
            out += ", "
        out += '"' + names[i] + '"'
    out += "]"
    return out^


def _json_results(results: List[BenchResult]) -> String:
    """One JSON array of objects, in the order the benchmarks ran.

    Deliberately no commit or timestamp: the binary has no business shelling
    out to git. Whatever saves these wraps them in that envelope.
    """
    var out = String("[\n")
    for i in range(len(results)):
        ref r = results[i]
        out += '  {"name": "' + r.name + '"'
        out += ', "unit": "ns"'
        out += ", " + '"iters": ' + String(r.iters)
        out += ", " + '"reps": ' + String(len(r.runs_ns))
        out += ", " + '"mean_ns": ' + String(r.mean_ns())
        out += ", " + '"min_ns": ' + String(r.min_ns())
        out += ", " + '"max_ns": ' + String(r.max_ns())
        out += ", " + '"median_ns": ' + String(r.median_ns())
        out += ", " + '"stddev_ns": ' + String(r.stddev_ns())
        out += ", " + '"runs_ns": ['
        for j in range(len(r.runs_ns)):
            if j > 0:
                out += ", "
            out += String(r.runs_ns[j])
        out += "]"
        if r.metric:
            ref m = r.metric.value()
            out += ", " + '"throughput_metric": "' + m.name + '"'
            out += ", " + '"throughput_unit": "' + m.unit + '"'
            out += ", " + '"throughput_count": ' + String(r.count)
            out += ", " + '"throughput": ' + String(r.rate())
        out += "}"
        if i < len(results) - 1:
            out += ","
        out += "\n"
    out += "]"
    return out^


def _format_ns(ns: Float64) -> String:
    if ns < 1_000.0:
        return _round2(ns) + " ns"
    if ns < 1_000_000.0:
        return _round2(ns / 1_000.0) + " us"
    if ns < 1_000_000_000.0:
        return _round2(ns / 1_000_000.0) + " ms"
    return _round2(ns / 1_000_000_000.0) + " s"


def _rate_str(v: Float64) -> String:
    """Rates span orders of magnitude -- 28 GB/s down to 0.008 GElems/s -- so
    a fixed two decimals rounds the slow end to "0.01" and throws the number
    away. Widen the fraction as the value shrinks."""
    if v >= 1.0:
        return _round(v, 2)
    if v >= 0.1:
        return _round(v, 3)
    return _round(v, 4)


def _round2(v: Float64) -> String:
    return _round(v, 2)


def _round(v: Float64, places: Int) -> String:
    """Fixed decimal places. `String(Float64)` prints far more than a table
    column wants, and there is no format spec to lean on here."""
    var factor = Float64(1)
    for _ in range(places):
        factor *= 10.0
    var scaled = round(v * factor) / factor
    var s = String(scaled)
    var dot = s.find(".")
    if dot < 0:
        s += "."
        dot = s.byte_length() - 1
    var frac = s.byte_length() - dot - 1
    if frac > places:
        return String(s[byte = 0 : dot + places + 1])
    while frac < places:
        s += "0"
        frac += 1
    return s^


def _pad(var s: String, width: Int) -> String:
    while s.byte_length() < width:
        s += " "
    return s^


def _table(results: List[BenchResult]) -> String:
    """A markdown table: mean, spread, repetition count, and the rate."""
    comptime NCOL = 5
    var headers = List[String]()
    headers.append(String("benchmark"))
    headers.append(String("mean"))
    headers.append(String("min - max"))
    headers.append(String("iters x reps"))
    headers.append(String("rate"))

    var rows = List[List[String]]()
    for i in range(len(results)):
        ref r = results[i]
        var row = List[String]()
        row.append(String(r.name))
        row.append(_format_ns(r.mean_ns()))
        row.append(_format_ns(r.min_ns()) + " - " + _format_ns(r.max_ns()))
        row.append(String(r.iters) + " x " + String(len(r.runs_ns)))
        if r.metric:
            row.append(_rate_str(r.rate()) + " " + r.metric.value().unit)
        else:
            row.append(String("-"))
        rows.append(row^)

    var widths = List[Int]()
    for c in range(NCOL):
        var w = headers[c].byte_length()
        for i in range(len(rows)):
            if rows[i][c].byte_length() > w:
                w = rows[i][c].byte_length()
        widths.append(w)

    var out = String("|")
    for c in range(NCOL):
        out += " " + _pad(headers[c].copy(), widths[c]) + " |"
    out += "\n|"
    for c in range(NCOL):
        out += " " + _dashes(widths[c]) + " |"
    for i in range(len(rows)):
        out += "\n|"
        for c in range(NCOL):
            out += " " + _pad(rows[i][c].copy(), widths[c]) + " |"
    return out^


def _dashes(n: Int) -> String:
    var out = String("")
    for _ in range(n):
        out += "-"
    return out^
