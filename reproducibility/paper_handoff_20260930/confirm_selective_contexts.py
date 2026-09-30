"""Independently replay the 13 archived selective-withdrawal screen contexts."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STUDY = ROOT / 'output/input_response_20260929_v3'
EQUATIONS = HERE / 'independent_equations.py'
nodes = []
for node in ast.parse(EQUATIONS.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
        t.id == 'results' for t in node.targets):
        break
    nodes.append(node)
namespace = {'__file__': str(EQUATIONS)}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EQUATIONS), 'exec'), namespace)
np, roots, evolve, jac = [namespace[k] for k in ('np', 'roots', 'evolve', 'jac')]

contexts = json.loads((HERE / 'selective_screen_contexts.json').read_text())
source_contexts = {c['case']: c for c in json.loads((HERE / 'source_contexts.json').read_text())}
checks = []
for item in contexts:
    source_context = source_contexts[item['case']]
    saved = source_context['equilibria']
    raw = item['parameters']
    p = dict(zip(('a', 'b', 'c', 'd', 'h', 'r'),
                 (raw[k] for k in ('e_to_e', 'i_to_e', 'e_to_i',
                                  'i_to_i', 'theta_off', 'tau_ratio'))))
    baseline = item['baseline']
    target_input = [0., baseline[1]]
    independent = {}
    references = {}
    for label, B in [('source', baseline), ('target', target_input)]:
        stored = [r for r in saved if [float(r['B_E']), float(r['B_I'])] == B]
        found = roots(p, B)
        assert len(found) == len(stored), (item['case'], label, len(found), len(stored))
        assert all(min(np.max(np.abs(x - np.array([float(r['E']), float(r['I'])])))
                       for r in stored) < 1e-7 for x in found)
        independent[label] = found
        references[label] = stored
    trials = []
    for role in ('herald', 'seizure'):
        src = [r for r in references['source'] if r['role'] == role and r['stability'] == 'Attracting']
        dst = [r for r in references['target'] if r['role'] == item[role] and r['stability'] == 'Attracting']
        assert len(src) == 1, (item['case'], role, 'source')
        initial = np.array([float(src[0]['E']), float(src[0]['I'])])
        # The study can retain a baseline role by unique nearby correspondence
        # when the destination has no same-context role partner. Verify that
        # geometry rather than requiring an unavailable destination label.
        correspondence_used = False
        if not dst and item[role] == role:
            candidates = sorted((float(np.max(np.abs(initial - np.array(
                [float(r['E']), float(r['I'])])))), j)
                for j, r in enumerate(references['target']))
            assert candidates[0][0] < .05
            assert len(candidates) == 1 or candidates[1][0] - candidates[0][0] > 1e-6
            dst = [references['target'][candidates[0][1]]]
            assert dst[0]['stability'] == 'Attracting'
            correspondence_used = True
        assert len(dst) == 1, (item['case'], role, 'target')
        expected = np.array([float(dst[0]['E']), float(dst[0]['I'])])
        for name, B, expected_state in [('withdrawal', target_input, expected),
                                        ('held_input', baseline, initial)]:
            final = evolve(initial, p, B)
            error = float(np.max(np.abs(final - expected_state)))
            eig = np.linalg.eigvals(jac(expected_state, p, B) /
                                   np.array([[7.8], [7.8 * p['r']]]))
            assert error < 1e-7 and max(eig.real) < 0, (item['case'], role, name, error)
            trials.append(dict(source=role, protocol=name, final=final.tolist(),
                               expected=item[role] if name == 'withdrawal' else role,
                               reference_correspondence_used=correspondence_used and name == 'withdrawal',
                               max_coordinate_error=error,
                               maximum_eigenvalue_real=float(max(eig.real))))
    checks.append(dict(case=item['case'], baseline=baseline, parameters=raw,
                       source_roots=len(independent['source']),
                       target_roots=len(independent['target']), trials=trials,
                       equilibrium_table_sha256=source_context['equilibrium_table_sha256'],
                       behavior_sha256=source_context['behavior_sha256']))
    print(item['case'], 'confirmed', flush=True)

report = dict(scipy=namespace['scipy'].__version__, checks=checks,
              horizon_ms=10000., abstol=1e-12, reltol=1e-12,
              independent_equations_sha256=hashlib.sha256(EQUATIONS.read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              interpretation='Finite-window withdrawal and held-input checks at archived parameters; '
                             'does not establish induction, full neighborhoods, or biological roles.')
(HERE / 'selective_confirmation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(contexts=len(checks), trajectories=sum(len(c['trials']) for c in checks))))
