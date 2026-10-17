"""Independently reproduce all structural summaries from the compact catalog."""
import bisect
from collections import defaultdict
import math
from pathlib import Path
import sys
import tomllib
from verify_catalog_classifications import rows, verify_manifest, digest


def verify(catalog, packet):
    verify_manifest(catalog)
    verify_manifest(packet)
    meta = tomllib.loads((packet / "study.toml").read_text())
    assert meta["completed"] and meta["schema_version"] == 1
    assert meta["catalog_manifest_sha256"] == digest(catalog / "checksums.toml")
    index = rows(catalog / "chunks.csv")
    contexts = {r["context_id"]: r for chunk in index if chunk["kind"] == "contexts"
                for r in rows(catalog / chunk["file"])}
    groups = defaultdict(list)
    for chunk in index:
        if chunk["kind"] != "roots":
            continue
        for r in rows(catalog / chunk["file"]):
            scope = contexts[r["context_id"]]["scope"]
            if scope not in ("screen", "detail_geometry"):
                continue
            a, b = int(r["E_arm"]), int(r["I_branch"])
            for key in ((scope, a, b), (scope, a, -1), (scope, -1, b), (scope, -1, -1)):
                groups[key].append(r)
    quantiles = rows(packet / "quantiles.csv")
    histograms = rows(packet / "histograms.csv.gz")
    stability = rows(packet / "stability.csv")
    assert len(stability) == len(groups)
    assert len(quantiles) == 4 * len(groups)
    assert len(histograms) == 50 * len(quantiles)
    expected = {}
    for key, rs in groups.items():
        for subset in ("all", "attracting_nonsingular"):
            selected = rs if subset == "all" else [r for r in rs if r["stability"] == "1" and r["near_singular"] == "false"]
            for field in ("E", "I"):
                expected[key + (subset, field)] = sorted(float(r[field]) for r in selected)
    seen = set()
    for r in quantiles:
        key = (r["scope"], int(r["E_arm"]), int(r["I_branch"]), r["subset"], r["variable"])
        assert key not in seen
        seen.add(key)
        xs = expected[key]
        assert int(r["n"]) == len(xs)
        for name, q in zip(("minimum", "q05", "q25", "median", "q75", "q95", "maximum"), (0, .05, .25, .5, .75, .95, 1)):
            assert (float(r[name]) == xs[max(0, math.ceil(q*len(xs))-1)]) if xs else r[name] == ""
        assert int(r["exact_zero"]) == xs.count(0)
        assert int(r["exact_ceiling"]) == xs.count(.5)
    assert seen == set(expected)
    seen_bins = set()
    for r in histograms:
        key = (r["scope"], int(r["E_arm"]), int(r["I_branch"]), r["subset"], r["variable"])
        b = int(r["bin"])
        assert (key, b) not in seen_bins and 1 <= b <= 50
        seen_bins.add((key, b))
        xs = expected[key]
        lo, hi = float(r["lo"]), float(r["hi"])
        assert lo == (b-1)/100 and hi == b/100
        end = bisect.bisect_right(xs, hi) if b == 50 else bisect.bisect_left(xs, hi)
        count = end - bisect.bisect_left(xs, lo)
        assert int(r["count"]) == count and int(r["n"]) == len(xs)
    seen = set()
    for r in stability:
        key = r["scope"], int(r["E_arm"]), int(r["I_branch"])
        assert key not in seen
        seen.add(key)
        rs = groups[key]
        assert int(r["n"]) == len(rs)
        for field, code in (("attracting", "1"), ("saddle", "2"), ("repelling", "3"), ("unresolved", "0")):
            assert int(r[field]) == sum(x["stability"] == code for x in rs)
        assert int(r["near_singular"]) == sum(x["near_singular"] == "true" for x in rs)
        assert int(r["eligible"]) == len(expected[key + ("attracting_nonsingular", "E")])
    assert seen == set(groups)
    joint_expected = defaultdict(int)
    edges = [i/200 for i in range(101)]
    for key, rs in groups.items():
        for r in rs:
            e = min(100, bisect.bisect_right(edges, float(r["E"])))
            i = min(100, bisect.bisect_right(edges, float(r["I"])))
            joint_expected[key + ("all", e, i)] += 1
            if r["stability"] == "1" and r["near_singular"] == "false":
                joint_expected[key + ("attracting_nonsingular", e, i)] += 1
    observed = {}
    for r in rows(packet / "joint_histograms.csv.gz"):
        key = (r["scope"], int(r["E_arm"]), int(r["I_branch"]), r["subset"], int(r["E_bin"]), int(r["I_bin"]))
        assert key not in observed
        observed[key] = int(r["count"])
        assert int(r["n"]) == len(expected[key[:4] + ("E",)])
    assert observed == joint_expected
    print(f"Verified {len(observed)} nonempty joint E-I bins (including implicit empty bins).")
    print(f"Verified {len(quantiles)} distributions, {len(histograms)} histogram bins, and {len(stability)} stability summaries.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_structural_distributions.py CATALOG SUMMARIES")
    verify(*map(Path, sys.argv[1:]))
