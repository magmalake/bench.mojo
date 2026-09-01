#!/usr/bin/env bash
# Drives every flag of a built bench binary against tests/example_bench.mojo.
# This is the only coverage of the `__functions_in_module()` discovery path —
# see the note at the top of that file for why the unit tests cannot do it.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

bin="$work/example_bench"
mojo build -I "$root/src" "$root/tests/example_bench.mojo" -o "$bin"

fail() { echo "check_cli: $1" >&2; exit 1; }

# --list names both benchmarks and skips the underscore-prefixed helper.
list="$("$bin" --list)"
[[ "$list" == *'"bench_sum"'* ]] || fail "--list missing bench_sum: $list"
[[ "$list" == *'"bench_max"'* ]] || fail "--list missing bench_max: $list"
[[ "$list" != *"_helper_ignored_by_discovery"* ]] || fail "--list leaked a helper"

# --only restricts the run; --json replaces the table with the JSON report.
json="$("$bin" --only bench_sum --json)"
[[ "$json" == \{* ]] || fail "--json did not start with '{': $json"
[[ "$json" == *'"name": "bench_sum"'* ]] || fail "--json missing bench_sum"
[[ "$json" != *"bench_max"* ]] || fail "--only did not exclude bench_max"
[[ "$json" == *'"throughput_unit": "GElems/s"'* ]] || fail "--json missing throughput"
[[ "$json" == *'"runs_ns": ['* ]] || fail "--json missing per-repetition runs"

# A script file rather than `python3 -c`, because stdin is carrying the JSON.
cat > "$work/check_report.py" <<'PYEOF'
import json, sys

d = json.load(sys.stdin)
assert len(d["results"]) == 1, d
assert len(d["results"][0]["runs_ns"]) == 3, d
assert d["config"]["num_repetitions"] == 3, d

h = d["host"]
# cpu and memory can legitimately be unknown on an unfamiliar platform; the
# rest comes from std.sys and always resolves.
assert h["os"] in ("macos", "linux"), h
assert h["arch"] in ("arm64", "x86_64"), h
assert h["physical_cores"] > 0 and h["logical_cores"] > 0, h
print("  host:", h["cpu"] or "(unknown cpu)", h["os"] + "/" + h["arch"],
      h["physical_cores"], "cores,", h["memory_bytes"], "bytes")
PYEOF
python3 "$work/check_report.py" <<<"$json" \
  || fail "--json report is not the expected shape"

# --skip is the complement.
skipped="$("$bin" --skip bench_sum --json)"
[[ "$skipped" == *'"name": "bench_max"'* ]] || fail "--skip dropped the wrong one"
[[ "$skipped" != *"bench_sum"* ]] || fail "--skip did not exclude bench_sum"

# --out writes the same payload to a file.
"$bin" --only bench_sum --out "$work/r.json" >/dev/null
[[ -s "$work/r.json" ]] || fail "--out wrote nothing"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['results'][0]['name']=='bench_sum', d" "$work/r.json" \
  || fail "--out did not write valid JSON"

# An unknown name is an error, not a silent empty run.
if "$bin" --only no_such_bench >/dev/null 2>&1; then
  fail "--only with an unknown name should have failed"
fi

# The default output is the human table.
table="$("$bin" --only bench_sum)"
[[ "$table" == *"benchmark"* ]] || fail "table missing header: $table"
[[ "$table" == *"GElems/s"* ]] || fail "table missing rate column"

echo "check_cli: ok"
