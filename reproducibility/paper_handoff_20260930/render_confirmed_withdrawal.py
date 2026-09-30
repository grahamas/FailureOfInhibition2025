"""Render independent trajectories for three already screened parameter contexts."""
from pathlib import Path
import ast
import csv
import hashlib
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STUDY = ROOT / 'output/input_response_20260929_v3'
source = HERE / 'independent_equations.py'
nodes = []
for node in ast.parse(source.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'results' for t in node.targets):
        break
    nodes.append(node)
ns = {'__file__': str(source)}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
checks = json.loads((HERE / 'selective_confirmation.json').read_text())
ids = ['selective_withdrawal_e_to_e_17.25', 'figure4_joint_29', 'input_dependent_e_to_e_4.25']
by_id = {c['case']: c for c in checks['checks']}
source_contexts = {c['case']: c for c in json.loads((HERE/'source_contexts.json').read_text())}
colors = {'herald': '#007f86', 'seizure': '#8e367e'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True, sharey=True)
records = []
for col, case_id in enumerate(ids):
    check = by_id[case_id]; raw = check['parameters']; B = check['baseline']
    p = dict(zip(('a','b','c','d','h','r'), (raw[k] for k in
        ('e_to_e','i_to_e','e_to_i','i_to_i','theta_off','tau_ratio'))))
    rows = source_contexts[case_id]['equilibria']
    for role in ('herald', 'seizure'):
        row = [r for r in rows if [float(r['B_E']), float(r['B_I'])] == B
               and r['role'] == role and r['stability'] == 'Attracting']
        assert len(row) == 1
        initial = np.array([float(row[0]['E']), float(row[0]['I'])])
        for protocol, target in [('withdrawal', [0., B[1]]), ('held_input', B)]:
            times = np.r_[np.linspace(0., 500., 1001), np.linspace(9800., 10000., 201)]
            sol = ns['solve_ivp'](lambda t,x: ns['balance'](x,p,target)/[7.8,7.8*p['r']],
                                 (0.,10000.),initial,method='DOP853',
                                 atol=1e-12,rtol=1e-12,t_eval=times)
            assert sol.success
            expected = next(t for t in check['trials'] if t['source'] == role and t['protocol'] == protocol)
            assert np.max(np.abs(sol.y[:,-1] - expected['final'])) < 1e-7
            name = f'{case_id}_{role}_{protocol}.csv'
            with (HERE / name).open('w') as stream:
                writer = csv.writer(stream); writer.writerow(['time','E','I'])
                writer.writerows(zip(sol.t, sol.y[0], sol.y[1]))
            records.append(dict(case=case_id, source=role, protocol=protocol,
                                input=target, file=name, destination=expected['expected']))
            for axis, trace in zip(axes[:,col], sol.y):
                axis.plot(sol.t, trace, color=colors[role],
                          linestyle='-' if protocol == 'withdrawal' else '--',
                          linewidth=2 if protocol == 'withdrawal' else 1.1,
                          alpha=1 if protocol == 'withdrawal' else .6)
    dest = next(t['expected'] for t in check['trials'] if t['source']=='herald' and t['protocol']=='withdrawal')
    axes[0,col].set_title(f"Herald → {dest}; seizure persists\nτI/τE = {p['r']:.3g}, baseline B = ({B[0]:g}, {B[1]:g})",
                         fontsize=11, pad=12)
    axes[1,col].set_xlabel('Time after E-input withdrawal (ms)')
    for ax in axes[:,col]:
        ax.set_xlim(0,500);ax.set_ylim(-.015,.53);ax.grid(alpha=.14)
axes[0,0].set_ylabel('Excitatory activity E')
axes[1,0].set_ylabel('Inhibitory activity I')
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],color=colors['herald'],lw=2,label='Herald source'),
                    Line2D([],[],color=colors['seizure'],lw=2,label='Seizure source'),
                    Line2D([],[],color='#343434',lw=2,label='E input withdrawn'),
                    Line2D([],[],color='#777777',lw=1.1,ls='--',label='Input held')],
           loc='upper center',bbox_to_anchor=(.5,.91),ncol=4,frameon=False)
fig.suptitle('Input withdrawal distinguishes coexisting high-activity states',y=.985,fontsize=15)
fig.text(.5,.018,'Three confirmed parameter contexts • 500 ms shown; outcomes checked through 10,000 ms • coordinate roles',
         ha='center',fontsize=9,color='#414141')
fig.subplots_adjust(top=.78,bottom=.12,left=.07,right=.98,wspace=.13,hspace=.24)
fig.savefig(HERE/'confirmed_withdrawal.png',dpi=180)
fig.savefig(HERE/'confirmed_withdrawal.svg')
plt.close(fig)
table=''.join('<tr><td>'+html.escape(name)+'</td><td>'+', '.join(str(by_id[name]['parameters'][k]) for k in
    ('e_to_e','i_to_e','e_to_i','i_to_i','theta_off','tau_ratio'))+'</td></tr>' for name in ids)
(HERE/'confirmed_withdrawal.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Confirmed selective withdrawal</title><style>body{font:17px/1.55 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#24303a}img{width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}code{font-size:.85em}</style>
<h1>Selective withdrawal beyond the first example</h1>
<p>Independent equations reproduce herald recovery alongside seizure persistence in these three contexts. The full check covers 13 contexts and 52 withdrawal/control trajectories.</p>
<img src="confirmed_withdrawal.png" alt="E and I time courses for herald and seizure sources in three parameter contexts, comparing complete E-input withdrawal with held input.">
<p><a href="confirmed_withdrawal.svg">Vector figure</a> · <a href="selective_confirmation.json">Independent checks</a> · <a href="source_contexts.json">Source equilibrium records</a></p>
<table><tr><th>Context</th><th>E→E, I→E, E→I, I→I, failure threshold, time-scale ratio</th></tr>'''+table+'''</table>
<p>Both response slopes are 5, onset thresholds are E=1.5 and I=4, and τE=7.8 ms. Each withdrawal sets BE to zero while preserving BI. All trials begin at the recorded source equilibrium; input induction is a separate test. Dashed controls retain the original input. These contexts do not establish a connected region or biological calibration.</p>
<p>Replay: <code>python3 confirm_selective_contexts.py</code>, then <code>python3 render_confirmed_withdrawal.py</code>. Both scripts use the compact context records and independent equations retained beside them; NumPy, SciPy, and Matplotlib are required. The source scripts, selected context records, trajectories, and hashes are retained.</p></html>''')
metadata = dict(contexts=ids, trajectories=records, matplotlib=matplotlib.__version__,
                independent_equations_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                confirmation_sha256=hashlib.sha256((HERE/'selective_confirmation.json').read_bytes()).hexdigest())
(HERE/'figure_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Rendered three confirmed contexts and twelve replayed trajectories.')
