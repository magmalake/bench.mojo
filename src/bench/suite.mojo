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

Two sampling modes, and why a row says which one it used
--------------------------------------------------------

`runs_ns` holds one number per *repetition*, and a repetition is already the
mean of `num_iters` iterations. A default run produces three or five of them.
Percentiles over that vector would be percentiles over a handful of averages:
the averaging inside a repetition is exactly what destroys the tail a p90 is
asked to show, and printing one anyway would look rigorous while meaning
nothing. So the harness measures the distribution instead of inferring it.

* **per-iteration** -- each call of the timed closure gets its own
  `perf_counter_ns` pair and the sample is kept. `p50`, `p90` and `p99` are
  real order statistics over real iterations.
* **batched** -- one timer pair wraps the whole loop, as before. Only
  `runs_ns` exists, and the percentile columns read `n/a` in the table and are
  **absent** from the JSON. `mean_ns`, `min_ns`, `max_ns`, `median_ns` and
  `stddev_ns` remain, over the repetition means, as they always were.

The rule for choosing: `_timer_resolution_ns()` measures, at suite
construction, the smallest interval this machine's clock can actually
distinguish -- nothing is hard-coded, and it is the *tick*, not the call cost.
Those differ by two orders of magnitude on macOS/arm64: a `perf_counter_ns`
read costs about 13 ns and the clock advances in 1000 ns steps. Per-iteration
timing is used only when one iteration costs at least `resolution_factor`
times the tick (100 by default), holding quantisation under 1% of every
sample. Below that, batching is the only way to measure the thing at all, and
the harness says so rather than inventing a distribution.

Every row and every JSON result carries its mode, so no number is ambiguous
about what it was computed over. Retained samples are capped at `max_samples`
(20,000 by default) by reservoir sampling; when the cap bites, the count of
kept-versus-seen samples is printed too, because a percentile over a subsample
is honest only if it is labelled as one.

Whether the run is worth believing at all
-----------------------------------------

Everything above measures the code. None of it can tell whether the *machine*
was the same machine for the whole run, and on a shared one it frequently is
not: another process takes a performance core, the scheduler drops the thread
onto an efficiency core, the package throttles. Measured here, one benchmark
that is 31 ms on an idle machine reports 51 ms against eight competing threads
and 89 ms against sixteen -- identical code, identical fixture, identical
binary. Nothing in the timings says which of those a table is showing, and the
longer a run takes, the more likely it is to have caught someone else's build.

So the harness measures the machine as well, and `Stability` is the verdict:

* **A reference kernel** -- a fixed dependent integer chain, no memory, no
  allocation -- is timed either side of every benchmark. Its duration is a
  proxy for how fast this core is running right now, and every result records
  the slower of the two probes that bracketed it. If the machine changed
  speed mid-run, the row that ran while it was slow can be named.
* **A recheck**: the first benchmark is timed again at the end of the run, on
  the iteration count it calibrated. This is the direct test of the property
  a harness owes its readers -- that a printed number does not depend on what
  else was selected or on when in the run it was taken -- and it costs one
  repetition. `--no-recheck` opts out.
* **The load average**, read from the OS. Both checks above compare the run
  against itself, so a machine that was busy for the whole run passes both
  while every number on it is wrong. That is the case that produced the 1.9x
  spread, and only the operating system can report it.

A clean run says so in one line under the table. A run that fails any of the
three says that instead, loudly, and `--strict` makes it a non-zero exit so
whatever publishes benchmark numbers can refuse to publish that one.

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
    --batched         force batched timing, giving up the percentiles
    --no-recheck      skip re-timing the first benchmark at the end
    --strict          exit non-zero if the run cannot be vouched for
"""

from std.benchmark.compiler import keep
from std.ffi import external_call
from std.math import sqrt
from std.reflection import get_function_name
from std.sys import (
    argv,
    exit,
    has_accelerator,
    num_logical_cores,
    num_performance_cores,
    num_physical_cores,
)
from std.sys.info import CompilationTarget
from std.time import perf_counter_ns


# ── Throughput metrics ──────────────────────────────────────────────────────


@fieldwise_init
struct Metric(Copyable, Movable):
    """What a benchmark's per-iteration count means, and the rate's unit.

    `unit` is the giga-scaled label the JSON reports against, kept fixed so a
    stored series stays comparable. `base_unit` is unprefixed, and the human
    table pairs it with whatever SI prefix actually suits the number -- 873
    thousand column chunks a second reads as `873.0 KElems/s`, not
    `0.0009 GElems/s`.
    """

    var name: String
    var unit: String
    var base_unit: String

    @staticmethod
    def bytes() -> Self:
        return Self("bytes", "GB/s", "B/s")

    @staticmethod
    def elements() -> Self:
        return Self("elements", "GElems/s", "Elems/s")

    @staticmethod
    def flops() -> Self:
        return Self("flops", "GFLOPS/s", "FLOPS/s")


# ── the machine ─────────────────────────────────────────────────────────────


def _cstr(s: StringSlice) -> List[UInt8]:
    var out = List[UInt8]()
    for b in String(s).as_bytes():
        out.append(b)
    out.append(0)
    return out^


def _sysctl_str(name: StringSlice) -> String:
    """macOS only. Empty string on any failure -- unknown beats invented."""
    var key = _cstr(name)
    var size = List[UInt64](length=1, fill=UInt64(512))
    var buf = List[UInt8](length=512, fill=UInt8(0))
    var rc = external_call["sysctlbyname", Int32](
        key.unsafe_ptr(), buf.unsafe_ptr(), size.unsafe_ptr(), Int(0), Int(0)
    )
    if rc != 0:
        return String("")
    var out = String("")
    for i in range(Int(size[0])):
        if buf[i] == 0:
            break
        out += chr(Int(buf[i]))
    return out^


def _sysctl_u64(name: StringSlice) -> Int:
    var key = _cstr(name)
    var size = List[UInt64](length=1, fill=UInt64(8))
    var val = List[UInt64](length=1, fill=UInt64(0))
    var rc = external_call["sysctlbyname", Int32](
        key.unsafe_ptr(), val.unsafe_ptr(), size.unsafe_ptr(), Int(0), Int(0)
    )
    return Int(val[0]) if rc == 0 else 0


def _proc_field(path: StringSlice, key: StringSlice) -> String:
    """First `key: value` line of a /proc file. Empty on any failure."""
    try:
        with open(String(path), "r") as f:
            var text = f.read()
            for line in text.split("\n"):
                var colon = line.find(":")
                if colon < 0:
                    continue
                if String(line[byte = 0 : colon]).strip() == String(key):
                    return String(String(line[byte=colon + 1 :]).strip())
    except:
        pass
    return String("")


@fieldwise_init
struct Host(Copyable, Movable):
    """What a run was measured on.

    A timing without its machine is not comparable to anything, so every JSON
    report carries this. Fields that could not be determined are left empty or
    zero rather than guessed.

    There is no GPU model here on purpose: these are CPU benchmarks, and
    `accelerator` records only whether the toolchain sees one at all.
    """

    var cpu: String
    var os: String
    var arch: String
    var physical_cores: Int
    var logical_cores: Int
    var performance_cores: Int
    var memory_bytes: Int
    var accelerator: Bool

    @staticmethod
    def detect() -> Self:
        var cpu = String("")
        var memory = 0

        comptime if CompilationTarget.is_macos():
            cpu = _sysctl_str("machdep.cpu.brand_string")
            memory = _sysctl_u64("hw.memsize")
        elif CompilationTarget.is_linux():
            cpu = _proc_field("/proc/cpuinfo", "model name")
            var kb = _proc_field("/proc/meminfo", "MemTotal")
            # "16316360 kB" -- take the leading integer.
            var digits = String("")
            for ch in kb:
                if ch >= "0" and ch <= "9":
                    digits += ch
                else:
                    break
            if len(digits.codepoints()) > 0:
                try:
                    memory = Int(digits) * 1024
                except:
                    memory = 0

        var os = String("unknown")

        comptime if CompilationTarget.is_macos():
            os = String("macos")
        elif CompilationTarget.is_linux():
            os = String("linux")

        var arch = String("unknown")

        # `has_neon` rather than `is_apple_silicon`: the latter came back false
        # on a GitHub macOS runner, leaving arch "unknown" and poisoning the
        # host key. Every arm64 target has NEON, which is the property we
        # actually mean here.
        comptime if CompilationTarget.is_x86():
            arch = String("x86_64")
        elif CompilationTarget.has_neon():
            arch = String("arm64")

        return Self(
            cpu^,
            os^,
            arch^,
            num_physical_cores(),
            num_logical_cores(),
            num_performance_cores(),
            memory,
            has_accelerator(),
        )

    def summary(self) -> String:
        """One line for the human header."""
        var out = self.cpu if len(self.cpu.codepoints()) > 0 else String("unknown cpu")
        out += String(" | ", self.os, "/", self.arch)
        out += String(" | ", self.physical_cores, " cores")
        if self.performance_cores != self.physical_cores:
            out += String(" (", self.performance_cores, " performance)")
        if self.memory_bytes > 0:
            out += String(" | ", self.memory_bytes // (1024 * 1024 * 1024), " GiB")
        return out^

    def as_json(self) -> String:
        return String(
            '{"cpu": "', self.cpu,
            '", "os": "', self.os,
            '", "arch": "', self.arch,
            '", "physical_cores": ', self.physical_cores,
            ', "logical_cores": ', self.logical_cores,
            ', "performance_cores": ', self.performance_cores,
            ', "memory_bytes": ', self.memory_bytes,
            ', "accelerator": ', "true" if self.accelerator else "false",
            "}",
        )


# ── The clock ───────────────────────────────────────────────────────────────


def _timer_resolution_ns() -> Float64:
    """The smallest interval `perf_counter_ns()` can honestly report here.

    Measured rather than assumed, and it is deliberately *not* the call cost.
    On macOS/arm64 a read costs about 13 ns while the clock advances in 1000 ns
    steps -- a single sample is quantised to the microsecond however cheap the
    read was. Treating the call cost as the limit would have enabled
    per-iteration timing at around 1.5 us an iteration, where every sample
    carries up to 64% quantisation error: a rigorous-looking number that
    describes the clock rather than the code, which is the exact failure this
    mode exists to avoid.

    So both are measured and the larger wins. The amortised cost of a call in a
    tight loop is what a read costs; the tick, found by spinning until the
    value changes, is what a read can *distinguish*. The spin is bounded --
    a clock that never advances gets the call cost and nothing worse than a
    conservative threshold.
    """
    comptime N = 512

    var t0 = perf_counter_ns()
    for _ in range(N):
        keep(perf_counter_ns())
    var call_ns = Float64(perf_counter_ns() - t0) / Float64(N)

    comptime TICKS = 32
    comptime SPIN_LIMIT = 1_000_000
    var ticks = List[Float64](capacity=TICKS)
    for _ in range(TICKS):
        var start = perf_counter_ns()
        var now = start
        var spins = 0
        while now == start and spins < SPIN_LIMIT:
            now = perf_counter_ns()
            spins += 1
        if now != start:
            ticks.append(Float64(now - start))
    var tick_ns = _percentile(ticks, 0.5)

    var floor = call_ns if call_ns > tick_ns else tick_ns
    # A clock that reports nothing about itself is not a licence to treat
    # per-iteration timing as free; one nanosecond is the smallest honest
    # floor.
    return floor if floor > 0.0 else 1.0


def _next_rand(mut state: UInt64) -> UInt64:
    """xorshift64*, for reservoir sampling.

    Seeded deterministically: what is wanted is a draw uniform over iteration
    index, not unpredictability, and a fixed seed means two runs of the same
    binary subsample the same positions.
    """
    state ^= state >> 12
    state ^= state << 25
    state ^= state >> 27
    return state * UInt64(0x2545F4914F6CDD1D)


# ── How fast the machine was ────────────────────────────────────────────────


comptime _REFERENCE_INNER = 1024
"""Trips of the dependent chain in one unit of reference work."""

comptime _REFERENCE_TARGET_NS = 1_000_000.0
"""What one reference measurement is calibrated to cost: about a millisecond,
short enough to take one either side of every benchmark for free."""

comptime _REFERENCE_MAX_REPS = 1 << 22
"""Ceiling on the calibrated repetition count, so a machine whose clock says
nothing cannot spin here forever."""

comptime _REFERENCE_SETTLE_MIN_NS = 250_000_000.0
"""Floor on the settling spin, in wall time. A core coming out of idle runs a
short burst above the clock it can hold: on an M4 the same kernel measures 30%
faster in the first tenth of a second than it does for the rest of the run.
Calibrating inside that window fixes the reference to a speed the machine will
not keep, and every probe afterwards then looks like interference."""

comptime _REFERENCE_SETTLE_CAP_NS = 1_000_000_000.0
"""Ceiling on the settling spin. A machine whose speed never stops moving gets
a second of patience and then a reference measured anyway -- and, since the
drift is real, a warning about it."""


def _reference_kernel(reps: Int) -> UInt64:
    """A fixed amount of work whose duration is a proxy for CPU speed.

    Every step depends on the one before it, so this measures how fast a single
    core executes a serial chain and nothing else: no memory traffic, no
    allocation, no syscalls, nothing a benchmark body might legitimately change
    the cost of. Two measurements of it, taken minutes apart on an idle
    machine, agree to a fraction of a percent.

    Args:
        reps: How many units of `_REFERENCE_INNER` steps to run.

    Returns:
        The accumulator, which the caller must `keep`.
    """
    var acc = UInt64(0x2545F4914F6CDD1D)
    for _ in range(reps):
        for i in range(_REFERENCE_INNER):
            acc = acc * UInt64(6364136223846793005) + UInt64(i) + UInt64(1)
    return acc


def _reference_once(reps: Int) -> Float64:
    """Time one run of the kernel, in nanoseconds."""
    var t0 = perf_counter_ns()
    var acc = _reference_kernel(reps)
    var dt = Float64(perf_counter_ns() - t0)
    keep(acc)
    return dt


def _grow_reference_reps(var reps: Int) -> Int:
    """Grow a repetition count until one run clears the target duration."""
    if reps < 1:
        reps = 1
    while reps < _REFERENCE_MAX_REPS:
        var dt = _reference_once(reps)
        if dt >= _REFERENCE_TARGET_NS:
            break
        if dt <= 0.0:
            reps = min(reps * 8, _REFERENCE_MAX_REPS)
        else:
            var scaled = Float64(reps) * _REFERENCE_TARGET_NS / dt * 1.2
            reps = min(max(Int(scaled), reps + 1), _REFERENCE_MAX_REPS)
    return reps


def _calibrate_reference() -> Int:
    """Repetitions that make one reference measurement cost about a
    millisecond here, measured once the core has come up to speed.

    The settling pass is not a nicety. A core that has been idle starts at a
    low clock and boosts over the first tens of milliseconds of sustained
    work: measured on an M4, the same kernel takes 947 us on the first attempt
    and 370 us twenty attempts later, a 2.6x ramp with nothing else running.
    Calibrating during that would fix the reference at whatever speed the
    machine happened to be at when the process started, and every probe
    afterwards would disagree with it -- a contention warning on every run,
    which is worse than no warning at all.

    So: grow a rough count, spin until the ramp flattens out -- two probes in
    a row that fail to beat the best by a percent -- and only then calibrate
    against the speed the benchmarks themselves will see. In practice that
    costs a tenth of a second, once, at the start of a run.
    """
    var reps = _grow_reference_reps(1)

    var start = perf_counter_ns()
    var best = _reference_ns(reps)
    var flat = 0
    while True:
        var elapsed = Float64(perf_counter_ns() - start)
        if elapsed >= _REFERENCE_SETTLE_CAP_NS:
            break
        if flat >= 2 and elapsed >= _REFERENCE_SETTLE_MIN_NS:
            break
        var now = _reference_ns(reps)
        if now < best:
            best = now
            flat = 0
        elif now <= best * 1.01:
            flat += 1
        else:
            flat = 0

    return _grow_reference_reps(reps)


def _reference_ns(reps: Int) -> Float64:
    """The fastest of three back-to-back kernel runs.

    The fastest, not the mean: what is wanted is the best this core managed in
    this window, so that a probe only reads slow when the machine could not do
    better. Taking the minimum makes the contention warning hard to trigger by
    accident, which is the right bias for something that tells a person their
    numbers are worthless.

    Args:
        reps: The calibrated repetition count.
    """
    var best = _reference_once(reps)
    for _ in range(2):
        var dt = _reference_once(reps)
        if dt < best:
            best = dt
    return best


# ── The value handed to each benchmark ──────────────────────────────────────


struct Benchmark:
    """Times a closure `num_iters` times and records the total.

    The suite calls a benchmark repeatedly with different `num_iters` -- once
    to read its throughput declaration, then to calibrate, then once per
    repetition -- so the body must be cheap to re-enter. Building the input
    outside `iter` is the point: that work is never timed.

    `sample_each` switches `iter` from one timer pair around the loop to one
    per iteration, which is what makes a real p90 possible. The suite decides;
    a benchmark body never has to know which mode it is running under.
    """

    var num_iters: Int
    var elapsed: Int
    var metric: Optional[Metric]
    var count: Int
    var sample_each: Bool
    var max_samples: Int
    var samples_ns: List[Float64]
    var samples_seen: Int
    var _rng: UInt64

    def __init__(
        out self,
        num_iters: Int,
        *,
        sample_each: Bool = False,
        max_samples: Int = 0,
        seed: UInt64 = 0x9E3779B97F4A7C15,
    ):
        """Create the handle the suite passes to one benchmark call.

        Args:
            num_iters: How many times `iter` runs the timed closure.
            sample_each: Time every iteration separately and keep the samples.
            max_samples: Ceiling on retained samples; beyond it, reservoir
                sampling replaces rather than appends.
            seed: Reservoir RNG seed, fixed so a rerun keeps the same
                positions.
        """
        self.num_iters = num_iters
        self.elapsed = 0
        self.metric = None
        self.count = 0
        self.sample_each = sample_each
        self.max_samples = max_samples
        self.samples_ns = List[Float64]()
        self.samples_seen = 0
        self._rng = seed if seed != 0 else UInt64(0x9E3779B97F4A7C15)

    def throughput(mut self, var metric: Metric, count: Int):
        """Declare how much work one iteration does, for the rate column.

        Args:
            metric: What `count` measures.
            count: Bytes/elements/flops processed per iteration.
        """
        self.metric = metric^
        self.count = count

    def _offer(mut self, sample_ns: Float64):
        """Algorithm R: every iteration keeps an equal chance of being kept.

        The alternative -- retaining the first `max_samples` and dropping the
        rest -- would take its percentiles from the start of the run, which is
        the part most contaminated by cache warming.
        """
        self.samples_seen += 1
        if len(self.samples_ns) < self.max_samples:
            self.samples_ns.append(sample_ns)
            return
        var j = Int(_next_rand(self._rng) % UInt64(self.samples_seen))
        if j < self.max_samples:
            self.samples_ns[j] = sample_ns

    def iter[f: def() capturing raises -> None](mut self) raises:
        """Run `f` `num_iters` times and record the elapsed nanoseconds.

        Batched: one timer pair wraps the loop, so the clock's cost is divided
        by `num_iters` and what comes out is a mean.

        Per-iteration: each call is timed on its own, and every sample carries
        the cost of one clock read. The suite only turns this on when an
        iteration costs at least a hundred such reads, which puts that bias
        under 1% -- far below the run-to-run spread the percentiles exist to
        show. `elapsed` is then the sum of the samples, i.e. time inside `f`,
        with the clock reads between iterations excluded.
        """
        if not self.sample_each:
            var t0 = perf_counter_ns()
            for _ in range(self.num_iters):
                f()
            self.elapsed = perf_counter_ns() - t0
            return

        var total = 0
        for _ in range(self.num_iters):
            var t0 = perf_counter_ns()
            f()
            var dt = perf_counter_ns() - t0
            total += dt
            self._offer(Float64(dt))
        self.elapsed = total


# ── Results ─────────────────────────────────────────────────────────────────


struct BenchResult(Copyable, Movable):
    """One benchmark's timings, plus statistics over them.

    Two vectors, and which one the statistics came from is the whole point:

    * `runs_ns` -- one entry per repetition, each the mean nanoseconds per
      iteration over that repetition. Always populated, always
      `num_repetitions` long, which is three or five.
    * `samples_ns` -- one entry per timed iteration, populated only when the
      suite ran this benchmark in per-iteration mode. `per_iteration` says so;
      `samples_seen` counts the iterations that were timed, which exceeds
      `len(samples_ns)` when the reservoir cap bit.

    Every statistic is computed over `samples_ns` in per-iteration mode and
    over `runs_ns` otherwise. Percentiles are the exception that proves the
    rule: they are `None` in batched mode rather than being computed over a
    handful of repetition means, because a p90 over five averages would look
    rigorous and describe nothing.
    """

    var name: String
    var iters: Int
    var runs_ns: List[Float64]
    var metric: Optional[Metric]
    var count: Int
    var samples_ns: List[Float64]
    var per_iteration: Bool
    var samples_seen: Int
    var reference_ns: Float64
    var reference_ratio: Float64

    def __init__(
        out self,
        var name: String,
        iters: Int,
        var runs_ns: List[Float64],
        var metric: Optional[Metric],
        count: Int,
    ):
        """A batched result: statistics over the per-repetition means."""
        self.name = name^
        self.iters = iters
        self.runs_ns = runs_ns^
        self.metric = metric^
        self.count = count
        self.samples_ns = List[Float64]()
        self.per_iteration = False
        self.samples_seen = 0
        # Filled in by the suite once every probe of the run is in hand; a
        # result built by hand carries no reference and reports none.
        self.reference_ns = 0.0
        self.reference_ratio = 1.0

    @staticmethod
    def sampled(
        var name: String,
        iters: Int,
        var runs_ns: List[Float64],
        var metric: Optional[Metric],
        count: Int,
        var samples_ns: List[Float64],
        samples_seen: Int,
    ) -> Self:
        """A per-iteration result: statistics over individual iterations.

        Args:
            name: Benchmark name.
            iters: Calibrated iterations per repetition.
            runs_ns: Per-repetition means, kept so the JSON shape is unchanged.
            metric: Throughput declaration, if the body made one.
            count: Work units per iteration.
            samples_ns: Retained per-iteration timings.
            samples_seen: Iterations timed, which may exceed the retained
                count.
        """
        var r = Self(name^, iters, runs_ns^, metric^, count)
        r.samples_ns = samples_ns^
        r.per_iteration = True
        r.samples_seen = samples_seen
        return r^

    # -- what the statistics were computed over -----------------------------

    def sampling(self) -> String:
        """`per-iteration` or `batched` -- printed everywhere a number is."""
        if self.per_iteration:
            return String("per-iteration")
        return String("batched")

    def num_samples(self) -> Int:
        """Retained per-iteration samples; zero in batched mode."""
        return len(self.samples_ns)

    def subsampled(self) -> Bool:
        """True when the reservoir cap dropped some of what was timed."""
        return self.per_iteration and self.samples_seen > len(self.samples_ns)

    def has_percentiles(self) -> Bool:
        return self.per_iteration and len(self.samples_ns) > 0

    def headline_ns(self) -> Float64:
        """The number a reader takes away: the p50 where there is one.

        Everything that compares one measurement of a benchmark with another
        one -- the end-of-run recheck, above all -- goes through here, so the
        comparison is always against the number that was printed.
        """
        var p50 = self.p50_ns()
        return p50.value() if p50 else self.mean_ns()

    def contended(self, tolerance: Float64) -> Bool:
        """True when the machine measurably slowed while this row was timed.

        Args:
            tolerance: Fractional slowdown of the reference kernel that counts
                as interference; 0.05 is five percent.
        """
        return (
            self.reference_ns > 0.0 and self.reference_ratio > 1.0 + tolerance
        )

    # -- statistics ---------------------------------------------------------

    def mean_ns(self) -> Float64:
        if self.per_iteration:
            return _mean(self.samples_ns)
        return _mean(self.runs_ns)

    def min_ns(self) -> Float64:
        if self.per_iteration:
            return _min(self.samples_ns)
        return _min(self.runs_ns)

    def max_ns(self) -> Float64:
        if self.per_iteration:
            return _max(self.samples_ns)
        return _max(self.runs_ns)

    def median_ns(self) -> Float64:
        if self.per_iteration:
            return _percentile(self.samples_ns, 0.5)
        return _percentile(self.runs_ns, 0.5)

    def stddev_ns(self) -> Float64:
        """Sample standard deviation (n-1), matching pytest-benchmark."""
        if self.per_iteration:
            return _stddev(self.samples_ns)
        return _stddev(self.runs_ns)

    def percentile_ns(self, q: Float64) -> Optional[Float64]:
        """The `q` quantile over measured iterations, or `None`.

        `None` is the answer in batched mode, and it is the honest one: there
        is no per-iteration distribution to take a quantile of.

        Args:
            q: Fraction in [0, 1]. 0.5 for the median.
        """
        if not self.has_percentiles():
            return None
        return _percentile(self.samples_ns, q)

    def p50_ns(self) -> Optional[Float64]:
        return self.percentile_ns(0.5)

    def p90_ns(self) -> Optional[Float64]:
        return self.percentile_ns(0.9)

    def p99_ns(self) -> Optional[Float64]:
        return self.percentile_ns(0.99)

    def rate(self) -> Float64:
        """Throughput in units of a billion per second, or 0 if undeclared.

        Still against the mean, and deliberately: it is the one central value
        both modes have, so a rate stays comparable across a run that switched
        modes and across history recorded before the modes existed.
        """
        var mean = self.mean_ns()
        if self.count == 0 or mean <= 0.0:
            return 0.0
        return Float64(self.count) / mean


# ── Statistics over a sample vector ─────────────────────────────────────────


def _mean(values: List[Float64]) -> Float64:
    if len(values) == 0:
        return 0.0
    var total = Float64(0)
    for i in range(len(values)):
        total += values[i]
    return total / Float64(len(values))


def _min(values: List[Float64]) -> Float64:
    if len(values) == 0:
        return 0.0
    var lo = values[0]
    for i in range(1, len(values)):
        if values[i] < lo:
            lo = values[i]
    return lo


def _max(values: List[Float64]) -> Float64:
    if len(values) == 0:
        return 0.0
    var hi = values[0]
    for i in range(1, len(values)):
        if values[i] > hi:
            hi = values[i]
    return hi


def _stddev(values: List[Float64]) -> Float64:
    var n = len(values)
    if n < 2:
        return 0.0
    var m = _mean(values)
    var acc = Float64(0)
    for i in range(n):
        var d = values[i] - m
        acc += d * d
    return sqrt(acc / Float64(n - 1))


def _percentile(values: List[Float64], q: Float64) -> Float64:
    """Linear interpolation between closest ranks, over a sorted copy.

    This is numpy's default and pytest-benchmark's, so a p90 printed here means
    what a p90 means elsewhere. `q` is a fraction: 0.5 is the median, and for
    an even count it reduces to the average of the middle pair, which is what
    `median_ns` reported before percentiles existed.

    Zero for an empty vector, the single value for a vector of one.
    """
    var n = len(values)
    if n == 0:
        return 0.0
    var qq = q
    if qq < 0.0:
        qq = 0.0
    if qq > 1.0:
        qq = 1.0
    var s = _sorted(values)
    if n == 1:
        return s[0]
    var h = qq * Float64(n - 1)
    var lo = Int(h)
    if lo >= n - 1:
        return s[n - 1]
    return s[lo] + (s[lo + 1] - s[lo]) * (h - Float64(lo))


def _sift_down(mut a: List[Float64], var root: Int, size: Int):
    while True:
        var child = 2 * root + 1
        if child >= size:
            return
        if child + 1 < size and a[child + 1] > a[child]:
            child += 1
        if a[root] >= a[child]:
            return
        var swap = a[root]
        a[root] = a[child]
        a[child] = swap
        root = child


comptime _INSERTION_SORT_MAX = 48
"""Above this, `_sorted` switches to heapsort."""


def _sorted(values: List[Float64]) -> List[Float64]:
    """Ascending copy, leaving the input alone.

    Insertion sort for the handful of numbers a batched run produces. Above
    `_INSERTION_SORT_MAX` -- which is to say once per-iteration sampling makes
    the vector thousands long -- heapsort, because O(n^2) over twenty thousand
    samples would cost more than the benchmark it is summarising and there is
    still no `List.sort` here that stays put across toolchains.
    """
    var out = values.copy()
    var n = len(out)

    if n <= _INSERTION_SORT_MAX:
        for i in range(1, n):
            var v = out[i]
            var j = i - 1
            while j >= 0 and out[j] > v:
                out[j + 1] = out[j]
                j -= 1
            out[j + 1] = v
        return out^

    for start in range(n // 2 - 1, -1, -1):
        _sift_down(out, start, n)
    for end in range(n - 1, 0, -1):
        var swap = out[0]
        out[0] = out[end]
        out[end] = swap
        _sift_down(out, 0, end)
    return out^


# ── Was this run worth believing? ───────────────────────────────────────────


def _load_average() -> Float64:
    """Runnable threads on this machine, averaged over the last minute.

    The one question the harness cannot answer by timing itself: a run where
    every measurement is uniformly inflated by a neighbouring process looks,
    from the inside, exactly like a run of slower code. The operating system
    knows, so ask it.

    Negative when it cannot be determined, which is never treated as a
    problem -- an unknown load says nothing, and inventing a number to warn
    about would be worse than staying quiet.
    """

    comptime if CompilationTarget.is_macos():
        # struct loadavg { fixpt_t ldavg[3]; long fscale; } -- three scaled
        # averages and the scale they are in, 2048 in practice.
        var key = _cstr("vm.loadavg")
        var size = List[UInt64](length=1, fill=UInt64(32))
        var buf = List[UInt32](length=8, fill=UInt32(0))
        var rc = external_call["sysctlbyname", Int32](
            key.unsafe_ptr(),
            buf.unsafe_ptr(),
            size.unsafe_ptr(),
            Int(0),
            Int(0),
        )
        if rc != 0 or Int(size[0]) < 20 or buf[4] == 0:
            return -1.0
        return Float64(Int(buf[0])) / Float64(Int(buf[4]))
    elif CompilationTarget.is_linux():
        return _proc_load_average()
    return -1.0


def _proc_load_average() -> Float64:
    """The first field of `/proc/loadavg`, or -1 if it cannot be read.

    A plain function rather than a branch of `_load_average`, so the parsing
    is compiled -- and unit tested -- on every platform rather than only on
    the one that can run it. `_proc_field` is a normal function for the same
    reason.
    """
    try:
        with open("/proc/loadavg", "r") as f:
            var text = f.read()
            var space = text.find(" ")
            if space <= 0:
                return -1.0
            return Float64(String(text[byte=0:space]))
    except:
        return -1.0


struct Stability(Copyable, Movable):
    """Evidence about the run itself, gathered alongside the measurements.

    A harness that reports only what it measured cannot tell the difference
    between code that is slow and a machine that was busy, and neither can the
    person reading the table. Three questions get answered here, all of them
    about the run rather than the code, and each catching something the others
    cannot.

    * **Did the machine hold its speed?** `_reference_kernel` is timed either
      side of every benchmark; it touches no memory and allocates nothing, so
      the only things that move it belong to the machine. The baseline is the
      median probe, not the fastest: a core coming out of idle can spend one
      probe at a boost clock it will not hold, and treating that as the
      standard would condemn every honest run that followed it. A result
      whose slower bracketing probe is more than `tolerance` above the median
      was measured while the machine was slow, and is named.
    * **Does the first benchmark still measure the same?** It is re-timed at
      the end of the run, with the iteration count it calibrated. That is the
      direct test of what a harness owes its readers -- a number must not
      depend on what else was selected, or on when in the run it was taken --
      and it costs one repetition.
    * **Was anything else running?** The two checks above compare the run
      against itself, so a machine that was *uniformly* busy from start to
      finish passes both while every number is inflated, so the load average
      is read from the OS as well, and a machine with more runnable threads
      than it has cores to spare is reported as what it is.

    None of this rescues a suite run under `mojo run`. JIT-executing several
    benchmarks in one process inflated an allocation-heavy one by 1.9x --
    62 ms against 33 ms for the same benchmark alone -- while the same program
    built ahead of time measured 32.8 ms in the suite against 32.1 ms alone.
    That spread was first blamed on benchmark ordering and then on a busy
    machine; it was neither, and both diagnoses survived a while because the
    isolated and suite readings were taken through different execution paths.
    Build the benchmark and run the binary. `Stability` reports the machine,
    not the toolchain.

    `ok` is all three, and it is what `--strict` exits on. Nothing here changes
    a measurement; it changes whether one can be read as if it were
    comparable.
    """

    var probes_ns: List[Float64]
    var results_ns: List[Float64]
    var rechecked: Bool
    var recheck_name: String
    var recheck_before_ns: Float64
    var recheck_after_ns: Float64
    var load: Float64
    var load_budget: Float64
    var tolerance: Float64

    def __init__(
        out self, tolerance: Float64 = 0.05, load_budget: Float64 = 0.0
    ):
        """A verdict that has learned nothing yet.

        Args:
            tolerance: Fractional disagreement treated as noise rather than
                interference.
            load_budget: Load average above which the machine counts as busy.
                Zero disables the check.
        """
        self.probes_ns = List[Float64]()
        self.results_ns = List[Float64]()
        self.rechecked = False
        self.recheck_name = String("")
        self.recheck_before_ns = 0.0
        self.recheck_after_ns = 0.0
        self.load = -1.0
        self.load_budget = load_budget
        self.tolerance = tolerance

    # -- what was observed --------------------------------------------------

    def observe(mut self, reference_ns: Float64):
        """Record one reference probe. Zero and below are dropped."""
        if reference_ns > 0.0:
            self.probes_ns.append(reference_ns)

    def observe_result(mut self, reference_ns: Float64):
        """Record the reference a single result ended up being judged by."""
        self.results_ns.append(reference_ns)

    def observe_load(mut self, load: Float64):
        """Record the busiest load average seen around the run."""
        if load > self.load:
            self.load = load

    def record_recheck(
        mut self, var name: String, before_ns: Float64, after_ns: Float64
    ):
        """Record the first benchmark\'s headline, then and now."""
        self.rechecked = True
        self.recheck_name = name^
        self.recheck_before_ns = before_ns
        self.recheck_after_ns = after_ns

    # -- what it means ------------------------------------------------------

    def measured(self) -> Bool:
        return len(self.probes_ns) > 0

    def baseline_ns(self) -> Float64:
        """The speed the machine held for most of the run."""
        return _percentile(self.probes_ns, 0.5)

    def best_ns(self) -> Float64:
        return _min(self.probes_ns)

    def worst_ns(self) -> Float64:
        return _max(self.probes_ns)

    def ratio(self, reference_ns: Float64) -> Float64:
        """How slow the machine was at one probe, against the baseline."""
        var base = self.baseline_ns()
        if base <= 0.0 or reference_ns <= 0.0:
            return 1.0
        return reference_ns / base

    def spread(self) -> Float64:
        """Fraction the slowest probe of the run lost against the fastest."""
        if not self.measured() or self.best_ns() <= 0.0:
            return 0.0
        return self.worst_ns() / self.best_ns() - 1.0

    def slowest_ratio(self) -> Float64:
        """The worst slowdown any result was measured through."""
        var worst = 1.0
        for i in range(len(self.results_ns)):
            var r = self.ratio(self.results_ns[i])
            if r > worst:
                worst = r
        return worst

    def steady(self) -> Bool:
        """True when no result was measured while the machine was slow."""
        if not self.measured():
            return True
        return self.slowest_ratio() <= 1.0 + self.tolerance

    def recheck_delta(self) -> Float64:
        """Fraction the re-timed benchmark moved, in either direction."""
        if not self.rechecked or self.recheck_before_ns <= 0.0:
            return 0.0
        var d = self.recheck_after_ns / self.recheck_before_ns - 1.0
        return d if d >= 0.0 else -d

    def reproducible(self) -> Bool:
        """True when the re-timed benchmark agreed with its first timing."""
        if not self.rechecked:
            return True
        return self.recheck_delta() <= self.tolerance

    def quiet(self) -> Bool:
        """True when the OS did not report other work competing for the CPU."""
        if self.load < 0.0 or self.load_budget <= 0.0:
            return True
        return self.load <= self.load_budget

    def ok(self) -> Bool:
        """True when nothing observed makes these numbers untrustworthy."""
        return self.steady() and self.reproducible() and self.quiet()

    def as_json(self) -> String:
        return String(
            '{"ok": ', "true" if self.ok() else "false",
            ', "tolerance": ', self.tolerance,
            ', "reference_probes": ', len(self.probes_ns),
            ', "reference_baseline_ns": ', self.baseline_ns(),
            ', "reference_best_ns": ', self.best_ns(),
            ', "reference_worst_ns": ', self.worst_ns(),
            ', "reference_spread": ', self.spread(),
            ', "slowest_ratio": ', self.slowest_ratio(),
            ', "steady": ', "true" if self.steady() else "false",
            ', "rechecked": ', "true" if self.rechecked else "false",
            ', "recheck_name": "', self.recheck_name,
            '", "recheck_before_ns": ', self.recheck_before_ns,
            ', "recheck_after_ns": ', self.recheck_after_ns,
            ', "recheck_delta": ', self.recheck_delta(),
            ', "reproducible": ', "true" if self.reproducible() else "false",
            ', "load_average": ', self.load,
            ', "load_budget": ', self.load_budget,
            ', "quiet": ', "true" if self.quiet() else "false",
            "}",
        )


# ── Registration ────────────────────────────────────────────────────────────


@fieldwise_init
struct _Bench(Copyable):
    comptime fn_type = def(mut Benchmark) thin raises

    var bench_fn: Self.fn_type
    var name: String


# ── The suite ───────────────────────────────────────────────────────────────


comptime _RESERVOIR_SEED = UInt64(0x9E3779B97F4A7C15)
"""Fixed, and offset per repetition, so a rerun keeps the same positions."""


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
    var max_samples: Int
    var resolution_factor: Int
    var force_batched: Bool
    var timer_resolution_ns: Float64
    var recheck: Bool
    var strict: Bool
    var stability_tolerance: Float64
    var reference_reps: Int

    def __init__(
        out self,
        *,
        min_runtime_secs: Float64 = 1.0,
        num_warmup_iters: Int = 2,
        num_repetitions: Int = 5,
        max_iters: Int = 100_000_000,
        max_samples: Int = 20_000,
        resolution_factor: Int = 100,
        recheck: Bool = True,
        stability_tolerance: Float64 = 0.05,
    ):
        """Configure a suite.

        Args:
            min_runtime_secs: Calibration target for one repetition.
            num_warmup_iters: Untimed passes before calibrating.
            num_repetitions: Timed repetitions; each becomes a `runs_ns` entry.
            max_iters: Ceiling on the calibrated iteration count.
            max_samples: Ceiling on retained per-iteration samples, split
                evenly across repetitions. Zero disables per-iteration
                sampling entirely.
            resolution_factor: How many clock reads one iteration must cost
                before it is worth timing on its own.
            recheck: Re-time the first benchmark once at the end of the run
                and report whether it still measures the same. Costs one
                repetition.
            stability_tolerance: Fractional disagreement -- between the
                reference probes, or between a benchmark and its recheck --
                treated as noise rather than interference.
        """
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
        self.max_samples = max_samples
        self.resolution_factor = resolution_factor
        self.force_batched = False
        # Measured once, here, so every benchmark is judged against the same
        # number and the report can say what it was.
        self.timer_resolution_ns = _timer_resolution_ns()
        self.recheck = recheck
        self.strict = False
        self.stability_tolerance = stability_tolerance
        # Calibrated in `execute`, not here: constructing a suite is something
        # the unit tests do dozens of times, and none of them wants to pay for
        # a reference kernel it will never run.
        self.reference_reps = 0

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
        max_samples: Int = 20_000,
        resolution_factor: Int = 100,
        recheck: Bool = True,
        stability_tolerance: Float64 = 0.05,
    ) raises:
        """Discover every `bench_*` in the module, then parse argv and run.

        Parameters:
            funcs: Pass `__functions_in_module()`.

        Args:
            min_runtime_secs: Calibration target for one repetition.
            num_warmup_iters: Untimed passes before calibrating.
            num_repetitions: Timed repetitions; each becomes a reported run.
            max_iters: Ceiling on the calibrated iteration count.
            max_samples: Ceiling on retained per-iteration samples.
            resolution_factor: How many clock reads one iteration must cost
                before it is timed on its own.
            recheck: Re-time the first benchmark at the end of the run and
                report whether it still measures the same.
            stability_tolerance: Fractional disagreement treated as noise
                rather than interference.
        """
        var suite = Self(
            min_runtime_secs=min_runtime_secs,
            num_warmup_iters=num_warmup_iters,
            num_repetitions=num_repetitions,
            max_iters=max_iters,
            max_samples=max_samples,
            resolution_factor=resolution_factor,
            recheck=recheck,
            stability_tolerance=stability_tolerance,
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
            elif arg == "--batched":
                # An escape hatch, mostly for reproducing a number recorded
                # before per-iteration sampling existed.
                self.force_batched = True
                mode = String("")
            elif arg == "--strict":
                self.strict = True
                mode = String("")
            elif arg == "--no-recheck":
                self.recheck = False
                mode = String("")
            elif arg == "--recheck":
                self.recheck = True
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

    def _load_budget(self, host: Host) -> Float64:
        """Load average this machine can carry before a timing is suspect.

        Half the cores, and never less than two. A benchmark occupies one of
        them and a developer machine idles with a thread or two of background
        noise; past that the run is sharing the machine with something, and
        which core the scheduler hands it -- performance or efficiency --
        stops being predictable. Deliberately generous: the failure this
        catches inflates a number by half or more, and a warning that fires on
        a quiet machine would be ignored within a week.
        """
        var cores = Float64(host.physical_cores)
        if cores <= 0.0:
            return 0.0
        var budget = cores * 0.5
        return budget if budget > 2.0 else 2.0

    def _index_of(self, name: String) -> Int:
        """Where a benchmark sits in `benches`, or -1. Used by the recheck to
        find the first benchmark again after the run."""
        for i in range(len(self.benches)):
            if self.benches[i].name == name:
                return i
        return -1

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

    def _config_json(self) -> String:
        return String(
            '{"min_runtime_secs": ', self.min_runtime_secs,
            ', "num_warmup_iters": ', self.num_warmup_iters,
            ', "num_repetitions": ', self.num_repetitions,
            ', "max_iters": ', self.max_iters,
            ', "max_samples": ', self.max_samples,
            ', "resolution_factor": ', self.resolution_factor,
            ', "timer_resolution_ns": ', self.timer_resolution_ns,
            ', "force_batched": ', "true" if self.force_batched else "false",
            ', "recheck": ', "true" if self.recheck else "false",
            ', "stability_tolerance": ', self.stability_tolerance,
            ', "reference_reps": ', self.reference_reps,
            "}",
        )

    def _samples_per_iteration(self, per_iter_ns: Float64) -> Bool:
        """Whether one iteration is expensive enough to time on its own.

        `timer_resolution_ns` was measured on this machine at construction, so
        this is a comparison against a real number rather than a guess.
        Demanding `resolution_factor` clock ticks' worth of work per iteration
        (100 by default) holds the clock's own contribution -- its cost, and
        more importantly its quantisation -- under 1% of a sample, an order of
        magnitude below the run-to-run spread the percentiles exist to expose.
        Cheaper than that and batching is the only way to measure the thing at
        all, so the harness batches and reports no percentile rather than a
        fabricated one.

        Args:
            per_iter_ns: Calibrated cost of one iteration.
        """
        if self.force_batched or self.max_samples <= 0:
            return False
        return (
            per_iter_ns
            >= Float64(self.resolution_factor) * self.timer_resolution_ns
        )

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
        # Set on every pass of the loop below, which always runs at least once.
        var per_iter_ns: Float64
        while True:
            var cal = Benchmark(num_iters)
            b.bench_fn(cal)
            per_iter_ns = Float64(cal.elapsed) / Float64(num_iters)
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

        # The calibrated cost of one iteration decides how the repetitions are
        # timed. Splitting the sample budget evenly across repetitions rather
        # than pooling one reservoir over all of them keeps each repetition
        # equally represented, which matters because a repetition is the unit a
        # scheduler hiccup lands in.
        var per_iteration = self._samples_per_iteration(per_iter_ns)
        var budget = 0
        if per_iteration and self.num_repetitions > 0:
            budget = self.max_samples // self.num_repetitions
            if budget < 1:
                budget = 1

        return self._repetitions(
            b,
            num_iters,
            per_iteration,
            budget,
            self.num_repetitions,
            metric^,
            count,
        )

    def _repetitions(
        self,
        b: _Bench,
        num_iters: Int,
        per_iteration: Bool,
        budget: Int,
        num_repetitions: Int,
        var metric: Optional[Metric],
        count: Int,
    ) raises -> BenchResult:
        """Time an already-calibrated benchmark `num_repetitions` times.

        Split out of `_run_one` so the end-of-run recheck can repeat exactly
        the protocol that produced the printed number -- same iteration count,
        same sampling mode, same reservoir budget -- and differ only in how
        many repetitions it can afford. A recheck measured any other way would
        be comparing two things and blaming the machine for the difference.

        Args:
            b: The registered benchmark.
            num_iters: Calibrated iterations per repetition.
            per_iteration: Whether to time each iteration on its own.
            budget: Retained samples per repetition.
            num_repetitions: How many timed repetitions to run.
            metric: The throughput declaration read during calibration.
            count: Work units per iteration.
        """
        var runs_ns = List[Float64](capacity=num_repetitions)
        var samples = List[Float64]()
        var seen = 0
        for rep_index in range(num_repetitions):
            var rep = Benchmark(
                num_iters,
                sample_each=per_iteration,
                max_samples=budget,
                seed=_RESERVOIR_SEED + UInt64(rep_index),
            )
            b.bench_fn(rep)
            runs_ns.append(Float64(rep.elapsed) / Float64(num_iters))
            for i in range(len(rep.samples_ns)):
                samples.append(rep.samples_ns[i])
            seen += rep.samples_seen

        if per_iteration and len(samples) > 0:
            return BenchResult.sampled(
                String(b.name),
                num_iters,
                runs_ns^,
                metric^,
                count,
                samples^,
                seen,
            )
        return BenchResult(
            String(b.name), num_iters, runs_ns^, metric^, count
        )

    def _recheck_one(self, b: _Bench, first: BenchResult) raises -> Float64:
        """Re-time an already-measured benchmark, once, and return its
        headline.

        Args:
            b: The registered benchmark, which must be the one `first`
                describes.
            first: What it measured the first time, for its iteration count
                and sampling mode.
        """
        var budget = 0
        if first.per_iteration and self.num_repetitions > 0:
            budget = self.max_samples // self.num_repetitions
            if budget < 1:
                budget = 1
        var again = self._repetitions(
            b,
            first.iters,
            first.per_iteration,
            budget,
            1,
            first.metric.copy(),
            first.count,
        )
        return again.headline_ns()

    def execute(mut self) raises:
        """Parse argv, run the selected benchmarks, and report."""
        self._parse_args()

        if self.list_only:
            var names = List[String]()
            for b in self.benches:
                names.append(String(b.name))
            print(_json_string_array(names))
            return

        var host = Host.detect()

        # One reference measurement per boundary: before the first benchmark,
        # between each pair, and after the last. A benchmark is bracketed by
        # two of them, and the slower of the two is what its row is judged by,
        # so interference has to miss both windows to go unreported.
        self.reference_reps = _calibrate_reference()
        var stability = Stability(
            tolerance=self.stability_tolerance,
            load_budget=self._load_budget(host),
        )
        stability.observe_load(_load_average())
        var references = List[Float64]()

        var results = List[BenchResult]()
        for b in self.benches:
            if not self._selected(b.name):
                continue
            if not self.json:
                print("running", b.name, "...")
            var before = _reference_ns(self.reference_reps)
            references.append(before)
            stability.observe(before)
            results.append(self._run_one(b))
        var after_last = _reference_ns(self.reference_reps)
        references.append(after_last)
        stability.observe(after_last)

        for i in range(len(results)):
            var bracketing = references[i]
            if references[i + 1] > bracketing:
                bracketing = references[i + 1]
            results[i].reference_ns = bracketing
            results[i].reference_ratio = stability.ratio(bracketing)
            stability.observe_result(bracketing)

        # The direct test of the property this harness owes its readers: the
        # first benchmark, measured again after everything else has run, on the
        # same iteration count. If the number moved, it depended on something
        # other than the code -- what else was selected, or when it ran -- and
        # saying so is worth one repetition.
        if self.recheck and len(results) > 0:
            var idx = self._index_of(results[0].name)
            if idx >= 0:
                if not self.json:
                    print("re-checking", results[0].name, "...")
                var again = self._recheck_one(self.benches[idx], results[0])
                stability.record_recheck(
                    String(results[0].name), results[0].headline_ns(), again
                )

        stability.observe_load(_load_average())
        var payload = _json_report(
            host, self._config_json(), results, stability
        )
        if self.json:
            print(payload)
        else:
            print(host.summary())
            print(
                "clock:",
                _format_ns(self.timer_resolution_ns),
                "resolution; per-iteration sampling above",
                _format_ns(
                    Float64(self.resolution_factor) * self.timer_resolution_ns
                ),
                "per iteration",
            )
            print(_table(results))
            print(_stability_report(stability, results))
        if self.out_path.byte_length() > 0:
            with open(self.out_path, "w") as f:
                f.write(payload)
                f.write("\n")
            if not self.json:
                print("wrote", self.out_path)

        # `--strict` is for whatever publishes these numbers: a run the harness
        # could not vouch for should fail the job rather than reach a README.
        if self.strict and not stability.ok():
            exit(1)


# ── Output ──────────────────────────────────────────────────────────────────


def _json_string_array(names: List[String]) -> String:
    var out = String("[")
    for i in range(len(names)):
        if i > 0:
            out += ", "
        out += '"' + names[i] + '"'
    out += "]"
    return out^


def _stability_report(
    stability: Stability, results: List[BenchResult]
) -> String:
    """One line when the run holds up, and a paragraph when it does not.

    The line is not decoration. A table of timings with nothing said about the
    machine invites the reader to assume the machine was quiet, and on a shared
    machine that assumption is wrong often enough to have produced a 1.9x
    spread on identical code. Stating the evidence -- the reference kernel
    varied by this much, the first benchmark still measures the same -- is what
    makes the numbers quotable; stating its absence is what stops them being
    quoted by mistake.
    """
    if not stability.measured() and not stability.rechecked:
        return String("machine: not checked")

    if stability.ok():
        var out = String("machine: steady")
        if stability.measured():
            out += String(
                " — reference work within ",
                _round(stability.spread() * 100.0, 1),
                "% across the run",
            )
        if stability.load >= 0.0:
            out += String("; load average ", _round(stability.load, 2))
        if stability.rechecked:
            out += String(
                "; ",
                stability.recheck_name,
                " re-timed within ",
                _round(stability.recheck_delta() * 100.0, 1),
                "%",
            )
        return out^

    var out = String("machine: NOT STEADY — these numbers are not comparable")
    if not stability.quiet():
        out += String(
            "\n  WARNING: the load average was ",
            _round(stability.load, 2),
            " on a machine budgeted for ",
            _round(stability.load_budget, 2),
            (
                ". Something else was running, and everything below may be"
                " inflated by it -- uniformly, which is why the timings"
                " themselves look consistent."
            ),
        )
    if not stability.steady():
        out += String(
            "\n  WARNING: reference work ran ",
            _round((stability.slowest_ratio() - 1.0) * 100.0, 1),
            "% slower for part of this run than for the rest of it (",
            _format_ns(stability.baseline_ns()),
            " typical, ",
            _format_ns(stability.worst_ns()),
            (
                " at worst). The rows measured through that are inflated"
                " against the others."
            ),
        )
        var named = String("")
        for i in range(len(results)):
            ref r = results[i]
            if not r.contended(stability.tolerance):
                continue
            if named.byte_length() > 0:
                named += ", "
            named += String(
                r.name,
                " (+",
                _round((r.reference_ratio - 1.0) * 100.0, 1),
                "%)",
            )
        if named.byte_length() > 0:
            out += String("\n  measured while the machine was slow: ", named)
    if not stability.reproducible():
        out += String(
            "\n  WARNING: ",
            stability.recheck_name,
            " re-timed at the end of the run came out ",
            _round(stability.recheck_delta() * 100.0, 1),
            "% away from its first timing (",
            _format_ns(stability.recheck_before_ns),
            " then, ",
            _format_ns(stability.recheck_after_ns),
            (
                " now). This run is not reproducible against itself; do not"
                " publish it."
            ),
        )
    out += String(
        "\n  Re-run on an idle machine. `--strict` turns this into a"
        " non-zero exit."
    )
    return out^


def _json_report(
    host: Host,
    config: String,
    results: List[BenchResult],
    stability: Stability,
) -> String:
    """What it ran on, how it was run, whether the run holds up, and what it
    measured.

    `stability` is the fourth part and the newest: a report that cannot say
    whether the machine was steady while it measured is a report whose numbers
    cannot be compared with anyone else's, or with their own history.

    Still no commit or timestamp -- the binary has no business shelling out to
    git, and whatever saves these adds them. The machine is different: a
    timing is not comparable to anything without it, and the binary is the
    only thing that knows for certain.
    """
    return String(
        '{"host": ', host.as_json(),
        ', "config": ', config,
        ', "stability": ', stability.as_json(),
        ', "results": ', _json_results(results),
        "}",
    )


def _json_results(results: List[BenchResult]) -> String:
    """The measurements, in the order the benchmarks ran.

    `sampling` says what every other number in the object was computed over,
    and it is always present. `p50_ns`, `p90_ns`, `p99_ns`, `samples` and
    `samples_seen` appear only on a per-iteration result -- omitted rather than
    null, the same way the `throughput_*` fields are omitted when no metric was
    declared, so a consumer that finds a percentile key knows it was measured.

    The per-iteration samples themselves are not dumped: twenty thousand floats
    per benchmark would dwarf the rest of the report, and the statistics are
    what a document or a trend line reads. `runs_ns` keeps its old meaning and
    its old length -- one entry per repetition -- in both modes.
    """
    var out = String("[\n")
    for i in range(len(results)):
        ref r = results[i]
        out += '  {"name": "' + r.name + '"'
        out += ', "unit": "ns"'
        out += ", " + '"iters": ' + String(r.iters)
        out += ", " + '"reps": ' + String(len(r.runs_ns))
        out += ", " + '"sampling": "' + r.sampling() + '"'
        out += ", " + '"mean_ns": ' + String(r.mean_ns())
        out += ", " + '"min_ns": ' + String(r.min_ns())
        out += ", " + '"max_ns": ' + String(r.max_ns())
        out += ", " + '"median_ns": ' + String(r.median_ns())
        out += ", " + '"stddev_ns": ' + String(r.stddev_ns())
        if r.has_percentiles():
            out += ", " + '"p50_ns": ' + String(r.p50_ns().value())
            out += ", " + '"p90_ns": ' + String(r.p90_ns().value())
            out += ", " + '"p99_ns": ' + String(r.p99_ns().value())
            out += ", " + '"samples": ' + String(r.num_samples())
            out += ", " + '"samples_seen": ' + String(r.samples_seen)
        if r.reference_ns > 0.0:
            out += ", " + '"reference_ns": ' + String(r.reference_ns)
            out += ", " + '"reference_ratio": ' + String(r.reference_ratio)
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


def _scaled_rate(giga_per_s: Float64, base_unit: String) -> String:
    """Pick the SI prefix that puts the number in a readable range.

    The JSON always reports against the fixed giga unit; this is display only.
    """
    var per_s = giga_per_s * 1.0e9
    if per_s >= 1.0e9:
        return _rate_str(per_s / 1.0e9) + " G" + base_unit
    if per_s >= 1.0e6:
        return _rate_str(per_s / 1.0e6) + " M" + base_unit
    if per_s >= 1.0e3:
        return _rate_str(per_s / 1.0e3) + " K" + base_unit
    return _rate_str(per_s) + " " + base_unit


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


def _sampling_cell(r: BenchResult) -> String:
    """`iters x reps`, plus what those iterations were actually timed with.

    A row has to be self-describing: pasted into a README it is separated from
    every other clue about how it was produced. `batched` means the numbers
    beside it are over repetition means and there are no percentiles;
    `per-iter` means they are over individual iterations. When the reservoir
    cap bit, the kept-of-seen count goes here too, because a percentile over a
    subsample is honest only when it says it is one.
    """
    var cell = String(r.iters, " x ", len(r.runs_ns))
    if not r.per_iteration:
        return cell + " batched"
    cell += " per-iter"
    if r.subsampled():
        cell += String(
            " (", r.num_samples(), " of ", r.samples_seen, " kept)"
        )
    return cell^


def _table(results: List[BenchResult]) -> String:
    """A markdown table: the p50 headline, p90 beside it, the floor and the
    ceiling, how it was sampled, and the rate.

    `p50` and `p90` read `n/a` on a batched row and the sampling column says
    why -- see `_sampling_cell`. `mean` stays because it is the one central
    value both modes have and the rate is derived from it, and because the gap
    between it and the p50 is itself the finding: one 125 ms first call among
    two hundred 2 ms ones moves a mean by 30% and a median not at all.
    """
    comptime NCOL = 7
    var headers = List[String]()
    headers.append(String("benchmark"))
    headers.append(String("p50"))
    headers.append(String("p90"))
    headers.append(String("mean"))
    headers.append(String("min - max"))
    headers.append(String("iters x reps"))
    headers.append(String("rate"))

    var rows = List[List[String]]()
    for i in range(len(results)):
        ref r = results[i]
        var row = List[String]()
        row.append(String(r.name))
        var p50 = r.p50_ns()
        var p90 = r.p90_ns()
        row.append(_format_ns(p50.value()) if p50 else String("n/a"))
        row.append(_format_ns(p90.value()) if p90 else String("n/a"))
        row.append(_format_ns(r.mean_ns()))
        row.append(_format_ns(r.min_ns()) + " - " + _format_ns(r.max_ns()))
        row.append(_sampling_cell(r))
        if r.metric:
            row.append(_scaled_rate(r.rate(), r.metric.value().base_unit))
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
