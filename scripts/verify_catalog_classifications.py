"""Independent standard-library verification of catalog-derived classification tables.

Recomputes proposed memberships, activity bands, and every context summary directly
from catalog rows; does not import the Julia classifier or use its summary counts.
"""
import csv
import gzip
import hashlib
from pathlib import Path
import sys
import tomllib
from collections import Counter, defaultdict

if not __debug__:
    raise RuntimeError("Verification requires assertions; run Python without -O.")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(path):
    manifest = tomllib.loads((path / "checksums.toml").read_text())
    assert manifest["algorithm"] == "SHA-256"
    files = {}
    for p in path.rglob("*"):
        assert not p.is_symlink(), p
        if p.is_file() and p != path / "checksums.toml":
            files[p.relative_to(path).as_posix()] = digest(p)
    assert files == manifest["files"], "manifest mismatch"


def rows(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        return list(csv.DictReader(f))


def verify(catalog, packet):
    verify_manifest(catalog)
    verify_manifest(packet)
    meta = tomllib.loads((packet / "study.toml").read_text())
    assert meta["completed"] and meta["schema_version"] == 1
    assert meta["catalog_manifest_sha256"] == digest(catalog / "checksums.toml")
    index = rows(catalog / "chunks.csv")
    contexts = {int(r["context_id"]): r for chunk in index if chunk["kind"] == "contexts"
                for r in rows(catalog / chunk["file"])}
    annotations = rows(packet / "annotations.csv.gz")
    lookup = {(int(r["context_id"]), int(r["root_id"])): r for r in annotations}
    assert len(lookup) == len(annotations) == meta["roots"]
    grouped = defaultdict(list)
    expected_counts = Counter()
    agreement = Counter()
    matched = set()
    policies = meta["policies"]
    assert [p["id"] for p in policies] == ["narrow_endpoints", "reference", "broad_endpoints"]
    for chunk in index:
        if chunk["kind"] != "roots":
            continue
        for r in rows(catalog / chunk["file"]):
            key = int(r["context_id"]), int(r["root_id"])
            assert key not in matched
            matched.add(key)
            context = contexts[key[0]]
            e, i = float(r["E"]), float(r["I"])
            codes = []
            for p in policies:
                memberships = [e <= p["low_E"] and i <= p["low_I"],
                    e > p["low_E"] and i <= p["low_I"] and r["I_branch"] == "2",
                    e >= p["high_E"] and i >= p["high_I"],
                    e >= p["high_E"] and p["low_I"] < i < p["high_I"] and r["I_branch"] == "2"]
                assert sum(memberships) <= 1
                code = memberships.index(True) + 1 if any(memberships) else 0
                codes.append(code)
                expected_counts[(context["scope"], p["id"], str(code), r["stability"], r["near_singular"])] += 1
            a = lookup[key]
            assert [int(a[n]) for n in ("narrow", "reference", "broad")] == codes
            p = policies[1]
            band_e = 1 if e <= p["low_E"] else 3 if e >= p["high_E"] else 2
            band_i = 1 if i <= p["low_I"] else 3 if i >= p["high_I"] else 2
            assert int(a["activity"]) == 3 * (band_e - 1) + band_i
            assert (a["agreement"] == "true") == (len(set(codes)) == 1)
            grouped[key[0]].append((r, codes[1]))
            if r["stability"] == "1" and r["near_singular"] == "false":
                agreement[(context["scope"], a["agreement"])] += 1
    assert matched == set(lookup)
    observed_counts = {}
    for r in rows(packet / "counts.csv"):
        key = tuple(r[n] for n in ("scope", "policy", "candidate", "stability", "near_singular"))
        assert key not in observed_counts
        observed_counts[key] = int(r["count"])
    assert expected_counts == observed_counts
    seen = set()
    for s in rows(packet / "contexts.csv.gz"):
        key = int(s["context_id"])
        assert key not in seen
        seen.add(key)
        c = contexts[key]
        for n in ("scope", "case_id"):
            assert s[n] == c[n]
        for n in ("B_E", "B_I"):
            assert float(s[n]) == float(c[n])
        group = grouped[key]
        sinks = [(r, code) for r, code in group if r["stability"] == "1" and r["near_singular"] == "false"]
        assert int(s["roots"]) == len(group) == int(c["roots"])
        assert int(s["sinks"]) == len(sinks)
        for code, name in enumerate(("descriptive", "rest", "seizure", "herald", "partial")):
            assert int(s[name]) == sum(label == code for _, label in sinks)
        for name, predicate in (("middle", lambda r: r["E_arm"] == "2"),
                                ("descending", lambda r: r["I_branch"] == "2"),
                                ("complex", lambda r: float(r["omega"]) > 0)):
            assert int(s[name]) == sum(predicate(r) for r, _ in sinks)
        for prefix, name in (("E", "E"), ("I", "I"), ("alpha", "lambda_max_real")):
            for suffix, op in (("min", min), ("max", max)):
                observed = s[prefix + "_" + suffix]
                assert (float(observed) == op(float(r[name]) for r, _ in sinks)) if sinks else observed == ""
    assert seen == set(contexts) and len(seen) == meta["contexts"]
    print(f"Verified all {len(matched):,} annotations and {len(seen):,} context summaries; all category counts agree.")
    for scope in ("screen", "detail_geometry"):
        same, different = agreement[(scope, "true")], agreement[(scope, "false")]
        print(f"{scope}: {same:,} same codes, {different:,} boundary-sensitive records ({100*different/(same+different):.2f}%).")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_catalog_classifications.py CATALOG CLASSIFICATIONS")
    verify(*map(Path, sys.argv[1:]))
