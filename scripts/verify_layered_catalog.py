"""Independently check layered gallery counts and bind them to source manifests."""
from collections import Counter
from pathlib import Path
import sys
import tomllib
import xml.etree.ElementTree as ET
from verify_catalog_classifications import rows, verify_manifest, digest
from verify_structural_distributions import verify


def verify_layers(catalog, summaries, gallery):
    verify(catalog, summaries)
    verify_manifest(gallery)
    meta = tomllib.loads((gallery / 'provenance.toml').read_text())
    study = tomllib.loads((summaries / 'study.toml').read_text())
    scripts = Path(__file__).resolve().parent
    for key, filename in [('renderer_sha256', 'render_structural_distributions.jl'),
                          ('layered_renderer_sha256', 'render_layered_catalog.jl'),
                          ('catalog_reader_sha256', 'equilibrium_catalog.jl')]:
        assert meta[key] == digest(scripts / filename), key
    for key, filename in [('analysis_sha256', 'catalog_structural_distributions.jl'),
                          ('quantile_helper_sha256', 'equilibrium_nomenclature.jl'),
                          ('catalog_reader_sha256', 'equilibrium_catalog.jl')]:
        assert study[key] == digest(scripts / filename), key
    assert meta['catalog_manifest_sha256'] == digest(catalog / 'checksums.toml')
    assert meta['summary_manifest_sha256'] == digest(summaries / 'checksums.toml')
    assert meta['interpretation_status'] == 'deferred'
    chunks = rows(catalog / 'chunks.csv')
    contexts = {r['context_id']: r for c in chunks if c['kind'] == 'contexts'
                for r in rows(catalog / c['file'])}
    scopes = ('screen', 'detail_geometry')
    expected = Counter({(s, str(st), complex_, near): 0 for s in scopes for st in range(4)
                        for complex_ in ('false', 'true') for near in ('false', 'true')})
    for c in chunks:
        if c['kind'] != 'roots':
            continue
        for r in rows(catalog / c['file']):
            scope = contexts[r['context_id']]['scope']
            if scope in scopes:
                key = (scope, r['stability'], str(float(r['omega']) > 0).lower(), r['near_singular'])
                expected[key] += 1
    observed = {}
    for r in rows(gallery / 'dynamics_counts.csv'):
        key = (r['scope'], r['stability'], r['complex'], r['near_singular'])
        assert key not in observed
        observed[key] = int(r['n'])
    assert observed == dict(expected)
    svgs = list(gallery.glob('*.svg'))
    assert len(svgs) == 8
    for p in svgs:
        ET.parse(p)
    print(f'Verified {sum(expected.values())} cohort root records in {len(expected)} dynamics groups; eight SVGs and exact provenance.')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit('usage: verify_layered_catalog.py CATALOG SUMMARIES GALLERY')
    verify_layers(*map(Path, sys.argv[1:]))
