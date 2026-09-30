"""Render retained two-input observations; does not assign biological validity."""
from pathlib import Path
import argparse, csv, hashlib, html, json, math, platform, shutil, tomllib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'rest':'#64748b','active':'#16856b','herald':'#9261ba','seizure':'#c44150',
        'oscillatory':'#cf861a','unresolved':'#444444','integration_failed':'#000000'}

def read_csv(path):
    return list(csv.DictReader(path.open())) if path.is_file() else []

def read_toml(path):
    return tomllib.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(source):
    manifest=read_toml(source/'checksums.toml')['files']
    for relative, expected in manifest.items():
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts or digest(source/p)!=expected:
            raise ValueError(f'invalid or changed artifact: {relative}')
    return len(manifest)

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def plot_geometry(case,out):
    rows=read_csv(case/'geometry/inputs.csv');curves=read_csv(case/'geometry/critical.csv')
    bounds=read_toml(case/'geometry/bounds.toml')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,zoom in zip(axes,(False,True)):
        sc=ax.scatter([float(x['B_E']) for x in rows],[float(x['B_I']) for x in rows],
                      c=[int(x['sinks']) for x in rows],s=12,cmap='viridis',vmin=0,vmax=4)
        for kind,color in [('fold','#d34343'),('hopf','#2276b5')]:
            points=[x for x in curves if x['kind']==kind]
            if points: ax.scatter([float(x['B_E']) for x in points],[float(x['B_I']) for x in points],
                                  s=2,color=color,label=kind+' candidates',rasterized=True)
        ax.set(xlabel='Tonic E input, B_E',ylabel='Tonic I input, B_I',
               xlim=(0,1 if zoom else bounds['B_E']),ylim=(0,1 if zoom else bounds['B_I']),
               title='Low-input detail' if zoom else 'Declared input domain')
        if curves: ax.legend(fontsize=8,loc='upper right')
    fig.colorbar(sc,ax=axes,label='Discovered attracting equilibria',ticks=range(5),shrink=.8)
    fig.suptitle(case.name+'\nEquilibrium discovery and candidate critical sets',fontsize=13)
    save(fig,out,'geometry')

def response_baselines(case):
    root=case/'responses'
    return sorted(root.glob('baseline_*'),key=lambda p:int(p.name.split('_')[-1])) if root.exists() else []

def plot_responses(case,out):
    baselines=response_baselines(case);chosen=None
    for baseline in baselines:
        roles=read_toml(baseline/'roles.toml')
        if all(x in roles for x in ['herald','seizure']): chosen=baseline;break
    if chosen is None: return None
    roles=read_toml(chosen/'roles.toml');inputs=read_toml(chosen/'input.toml')
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    bounds=read_toml(case/'geometry/bounds.toml')
    for ax,source,zoom in zip(axes.flat,['herald','seizure']*2,[False,False,True,True]):
        rows=read_csv(chosen/f'root_{roles[source]}/sustained/destinations.csv')
        for destination in sorted({r['destination'] for r in rows}):
            group=[r for r in rows if r['destination']==destination]
            ax.scatter([float(r['B_E']) for r in group],[float(r['B_I']) for r in group],
                       color=COLORS.get(destination,'#aaaaaa'),s=13,label=destination)
        ax.scatter([inputs['B_E']],[inputs['B_I']],marker='*',s=160,color='black',label='starting input')
        ax.set(xlabel='Held E input, B_E',ylabel='Held I input, B_I',title='Starting from '+source+(' · detail' if zoom else ''),
               xlim=(0,max(1.,inputs['B_E']*1.5) if zoom else bounds['B_E']),
               ylim=(0,max(2.,inputs['B_I']*1.5) if zoom else bounds['B_I']))
        ax.legend(fontsize=7,loc='upper right')
    fig.suptitle(f"Sustained changes from B_E={inputs['B_E']:g}, B_I={inputs['B_I']:g}\nSampled destinations; gaps and unresolved cells are not exclusions",fontsize=12)
    save(fig,out,'destinations')
    return chosen

def plot_displacement(case,out):
    entries=[]
    for b in response_baselines(case):
        p=b/'displacement/summary.toml'
        if not p.is_file(): continue
        r=read_toml(p);v=r.get('seizure_over_herald')
        if isinstance(v,(int,float)) and math.isfinite(v): entries.append((read_toml(b/'input.toml'),r))
    if not entries: return
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    positions=range(len(entries))
    ax.scatter(positions,[r['herald'] for _,r in entries],label='Herald',color=COLORS['herald'])
    ax.scatter(positions,[r['seizure'] for _,r in entries],label='Seizure',color=COLORS['seizure'])
    ax.set_xticks(list(positions),[f"({b['B_E']:.3g}, {b['B_I']:.3g})\nR={r['seizure_over_herald']:.3g}" for b,r in entries],rotation=35,ha='right',fontsize=8)
    ax.set(xlabel='Starting (B_E, B_I); R = seizure / herald',ylabel='Smallest observed successful E displacement',title='Direct state displacement at fixed input')
    ax.legend();save(fig,out,'displacement')

def plot_hysteresis(case,out,chosen):
    if chosen is None:return
    roles=read_toml(chosen/'roles.toml')
    rows=read_csv(chosen/f"root_{roles['herald']}/hysteresis/sweeps.csv")
    if not rows:return
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,axis in zip(axes,['B_E','B_I']):
        for first_direction in ['-1','1']:
            group=[r for r in rows if r['axis']==axis and r['first_direction']==first_direction]
            ax.plot([float(r['input']) for r in group],[float(r['E']) for r in group],lw=.8,
                    label='First decrease' if first_direction=='-1' else 'First increase')
        for status,color,label in [('periodic_compatible',COLORS['oscillatory'],'Periodic-compatible'),('unresolved','#222222','Unresolved')]:
            group=[r for r in rows if r['axis']==axis and r['status']==status]
            if group:ax.scatter([float(r['input']) for r in group],[float(r['E']) for r in group],s=12,color=color,label=label,zorder=3)
        ax.set(xlabel=axis,ylabel='Terminal E',title='Held-input step protocol');ax.legend(fontsize=8)
    fig.suptitle('Herald-source sweeps carry actual endpoints; periodic endpoints are phase samples',fontsize=11)
    save(fig,out,'hysteresis')

def plot_pareto(case,out,chosen):
    if chosen is None:return
    roles=read_toml(chosen/'roles.toml')
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    found=False;missing=[]
    for role in ('herald','seizure'):
        rows=read_csv(chosen/f"root_{roles[role]}/sustained/pareto.csv")
        if rows:
            found=True
            ax.scatter([float(r['E_withdrawal']) for r in rows],[float(r['I_stimulation']) for r in rows],
                       label=role,color=COLORS[role],s=35)
        else:missing.append(role)
    if found:ax.legend()
    else:ax.text(.5,.5,'No successful combinations in the sampled observations',ha='center',va='center',transform=ax.transAxes,wrap=True)
    ax.set(xlabel='E input withdrawn',ylabel='I input added',title='Nondominated sampled recovery combinations\nSustained changes; no scalar cost assigned')
    ax.set_xlim(left=0);ax.set_ylim(bottom=0)
    if found and missing:ax.text(.98,.95,'No sampled recovery: '+', '.join(missing),ha='right',va='top',transform=ax.transAxes,fontsize=9)
    save(fig,out,'pareto')

def plot_pulses(case,out,chosen):
    if chosen is None:return
    roles=read_toml(chosen/'roles.toml')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for column,role in enumerate(('herald','seizure')):
        rows=read_csv(chosen/f"root_{roles[role]}/pulses/pulses.csv")
        for row,(direction,title) in enumerate((('E_withdrawal','E input withdrawn'),('I_increase','I input added'))):
            ax=axes[row,column];group=[r for r in rows if r['axis']==direction]
            for destination in sorted({r['destination'] for r in group}):
                selected=[r for r in group if r['destination']==destination]
                ax.scatter([float(r['amplitude']) for r in selected],[float(r['duration']) for r in selected],
                           s=8,color=COLORS.get(destination,'#aaaaaa'),label=destination)
            ax.set(xlabel=title,ylabel='Pulse duration (ms)',yscale='log',title='Starting from '+role)
            if group:
                ax.legend(fontsize=7,loc='upper right')
                if max(float(r['amplitude']) for r in group)==0:
                    ax.set_xticks([0.])
                    ax.text(.02,.94,'No tonic input available to withdraw',transform=ax.transAxes,fontsize=9,va='top')
                else:ax.set_xlim(left=0)
    fig.suptitle('Destinations after pulse withdrawal and return to the starting tonic input\nSampled amplitudes and durations; unresolved outcomes retained',fontsize=12)
    save(fig,out,'pulses')

def plot_cycle_phases(case,out):
    chosen=None
    for baseline in response_baselines(case):
        summary=baseline/'summary.toml'
        if summary.is_file() and read_toml(summary).get('validated_cycles',0)>0:
            chosen=baseline;break
    if chosen is None:return
    summary=read_toml(chosen/'summary.toml');inputs=read_toml(chosen/'input.toml')
    rows=[]
    for source in summary['sources']:
        if source['source'].startswith('cycle_1_phase_'):
            rows.extend(read_csv(chosen/source['source']/'pulses/pulses.csv'))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,duration in enumerate((10.,100.)):
        for column,(direction,label) in enumerate((('E_withdrawal','E input withdrawn'),('I_increase','I input added'))):
            ax=axes[row,column]
            selected=[r for r in rows if r['axis']==direction and float(r['duration'])==duration]
            for destination in sorted({r['destination'] for r in selected}):
                group=[r for r in selected if r['destination']==destination]
                ax.scatter([float(r['amplitude']) for r in group],[float(r['phase']) for r in group],
                           s=9,color=COLORS.get(destination,'#aaaaaa'),label=destination)
            ax.set(xlabel=label,ylabel='Starting phase (fraction of cycle)',ylim=(-.03,1.03),title=f'{duration:g} ms pulse')
            ax.set_xlim(left=0)
            if selected:ax.legend(fontsize=7,loc='upper right')
    fig.suptitle(f"Cycle-source outcomes at B_E={inputs['B_E']:g}, B_I={inputs['B_I']:g}\nSampled phases and amplitudes; first discovered cycle at the first cycle-bearing representative",fontsize=11)
    save(fig,out,'cycle_phases')


STYLE="body{font:16px system-ui;color:#263444;max-width:1250px;margin:32px auto;padding:0 20px}p{line-height:1.5}img{max-width:100%}table{border-collapse:collapse}td,th{padding:7px 12px;border-bottom:1px solid #ddd;text-align:left}details{overflow-x:auto}nav a{margin-right:16px}canvas{width:100%;max-width:900px;border:1px solid #ddd}select{font:inherit;margin:10px}small{color:#596779}"

def render(source,out,partial=False):
    source=source.resolve();out=out.resolve()
    verified=None if partial else verify(source)
    if out.exists() and any(out.iterdir()):raise ValueError('output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    cases=[]
    for category in ['anchors','representatives']:
        root=source/category
        if root.exists(): cases.extend(p for p in sorted(root.iterdir()) if (p/'geometry/summary.toml').is_file())
    overview=[];interactive=[]
    for case in cases:
        name=case.parent.name+'_'+case.name;folder=out/name;folder.mkdir()
        g=read_toml(case/'geometry/summary.toml');bounds=read_toml(case/'geometry/bounds.toml')
        plot_geometry(case,folder);chosen=plot_responses(case,folder)
        plot_displacement(case,folder);plot_hysteresis(case,folder,chosen);plot_pareto(case,folder,chosen);plot_pulses(case,folder,chosen)
        plot_cycle_phases(case,folder)
        inputs=read_csv(case/'geometry/inputs.csv')
        interactive.append({'name':name,'bounds':[bounds['B_E'],bounds['B_I']],
                            'points':[[float(r['B_E']),float(r['B_I']),int(r['sinks']),int(r['roots']),r['roles']] for r in inputs]})
        links=''.join(f"<h2>{html.escape(p.stem.replace('_',' '))}</h2><a href='{p.with_suffix('.pdf').name}'>PDF</a><img src='{p.name}' alt='{html.escape(p.stem)}'>" for p in sorted(folder.glob('*.png')))
        baselines=response_baselines(case)
        completed=sum((p/'done.toml').is_file() for p in baselines)
        expected=len(g['representatives'])
        table=''.join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k,v in read_toml(case/'geometry/parameters.toml').items() if k!='baselines')
        (folder/'index.html').write_text(f"<!doctype html><meta charset='utf-8'><title>{html.escape(name)}</title><style>{STYLE}</style><nav><a href='../report.html'>All cases</a></nav><h1>{html.escape(name)}</h1><p>Numerical discovery; provisional coordinate roles. {g['inputs']} input contexts, {g['budget_unresolved_cells']} geometry cells left unresolved by budget. Response baselines completed: {completed}/{expected}.</p><table>{table}</table>{links}")
        overview.append(f"<tr><td><a href='{name}/index.html'>{html.escape(name)}</a></td><td>{bounds['B_E']:g} × {bounds['B_I']:g}</td><td>{g['inputs']}</td><td>{g['budget_unresolved_cells']}</td><td>{completed}/{expected}</td></tr>")
    comparison=[]
    expansion=source/'expansion'
    selection_file=source/'expansion_selection.toml'
    selection=read_toml(selection_file).get('representatives',{}) if selection_file.is_file() else {}
    if expansion.exists():
        for case in sorted(expansion.iterdir()):
            if not (case/'geometry/summary.toml').is_file():continue
            p=read_toml(case/'geometry/parameters.toml');g=read_toml(case/'geometry/summary.toml')
            b=read_toml(case/'behavior.toml') if (case/'behavior.toml').is_file() else {}
            values=[p['id'],p['e_to_e'],p['i_to_e'],p['e_to_i'],p['i_to_i'],p['theta_off'],p['tau_ratio'],g['inputs'],len(g['signatures']),b.get('observations','pending')]
            representatives=sorted({selection[r] for r in b.get('regimes',[]) if r in selection})
            links=[]
            for representative in representatives:
                target=f'representatives_{representative}/index.html'
                label=html.escape(representative)
                links.append(f"<a href='{html.escape(target,quote=True)}'>{label}</a>" if (out/target).is_file() else label+' (pending)')
            comparison.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'<td>'+'<br>'.join(links)+'</td></tr>')
    comparison_html=f"<details><summary>Parameter comparison screen ({len(comparison)} cases)</summary><p>Screening density is lower than the detailed maps. Signatures summarize observations, not equivalence or prevalence. Representative links identify detailed cases selected for the same observed baseline coexistence and fixed-protocol outcomes.</p><table><tr><th>Case</th><th>E→E</th><th>I→E</th><th>E→I</th><th>I→I</th><th>Failure</th><th>τI/τE</th><th>Inputs</th><th>Signatures</th><th>Behavior probes</th><th>Selected detailed comparisons</th></tr>{''.join(comparison)}</table></details>"
    data=json.dumps(interactive).replace('</','<\\/')
    script="""
const cases=JSON.parse(document.getElementById('data').textContent), select=document.getElementById('case'), zoom=document.getElementById('zoom'), canvas=document.getElementById('atlas'), ctx=canvas.getContext('2d');
const colors=['#dddddd','#39558c','#21918c','#9dc943','#fde725'];
cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=c.name;select.appendChild(o)});
function draw(){const c=cases[+select.value];if(!c)return;const xmax=zoom.value==='low'?1:c.bounds[0], ymax=zoom.value==='low'?1:c.bounds[1];ctx.clearRect(0,0,900,650);ctx.font='15px sans-serif';ctx.fillStyle='#263444';ctx.fillText('B_I',15,25);ctx.fillText('B_E',845,625);for(let t=0;t<=4;t++){let x=70+780*t/4,y=570-520*t/4;ctx.fillText((xmax*t/4).toPrecision(3),x-15,600);ctx.fillText((ymax*t/4).toPrecision(3),5,y+5);ctx.strokeStyle='#e4e7eb';ctx.beginPath();ctx.moveTo(x,50);ctx.lineTo(x,570);ctx.moveTo(70,y);ctx.lineTo(850,y);ctx.stroke()}for(const p of c.points){if(p[0]>xmax||p[1]>ymax)continue;ctx.fillStyle=colors[Math.min(p[2],4)];ctx.beginPath();ctx.arc(70+780*p[0]/xmax,570-520*p[1]/ymax,3,0,7);ctx.fill()}ctx.fillStyle='#263444';ctx.fillText('Color: 0 gray · 1 blue · 2 teal · 3 green · 4 yellow attracting equilibria discovered',70,640)}
select.onchange=zoom.onchange=draw;draw();
canvas.onmousemove=e=>{const r=canvas.getBoundingClientRect(),c=cases[+select.value];if(!c)return;let x=(e.clientX-r.left)*900/r.width,y=(e.clientY-r.top)*650/r.height,xmax=zoom.value==='low'?1:c.bounds[0],ymax=zoom.value==='low'?1:c.bounds[1],best=null,d=100;for(const p of c.points){let q=Math.hypot(70+780*p[0]/xmax-x,570-520*p[1]/ymax-y);if(q<d){d=q;best=p}}document.getElementById('detail').textContent=best&&d<20?`B_E=${best[0]}, B_I=${best[1]} · ${best[2]} attracting / ${best[3]} total equilibria discovered · ${best[4]||'roles unassigned'}`:'Hover over a sampled input point.'};
"""
    status='PARTIAL ARTIFACT PREVIEW' if partial else f'{verified:,} input-artifact file checksums verified'
    if not read_toml(source/'metadata.toml').get('completed',False):status+='; the full all-case study is not completed'
    (out/'report.html').write_text(f"<!doctype html><meta charset='utf-8'><title>Two-input response study</title><style>{STYLE}</style><h1>Responses to excitatory and inhibitory input</h1><p>{status}. These maps retain input-dependent states, numerical critical-set candidates, and finite-window transitions. A failed sampled intervention does not establish impossibility. Herald-to-seizure and seizure-to-herald transitions are distinct from recovery.</p><h2>Explore equilibrium observations</h2><label>Case <select id='case'></select></label><label>View <select id='zoom'><option value='full'>Full input domain</option><option value='low'>0–1 detail</option></select></label><canvas id='atlas' width='900' height='650'></canvas><p id='detail'>Hover over a sampled input point.</p><h2>Case reports</h2><table><thead><tr><th>Case</th><th>Input upper bounds</th><th>Contexts</th><th>Unresolved geometry cells</th><th>Response baselines complete</th></tr></thead><tbody>{''.join(overview)}</tbody></table>{comparison_html}<p>Input-withdrawal costs and direct state displacement have different units. Oscillatory destinations remain separate from provisional biological roles. Tail bounds concern response saturation, not global attractor completeness.</p><script id='data' type='application/json'>{data}</script><script>{script}</script>")
    shutil.copy2(__file__,out/Path(__file__).name)
    metadata={'input':str(source),'partial':partial,'verified_input_files':verified,'python':platform.python_version(),'matplotlib':matplotlib.__version__,'input_manifest_sha256':digest(source/'checksums.toml') if (source/'checksums.toml').is_file() else None}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    hashes={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'checksums.json').write_text(json.dumps({'files':hashes},indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'files':len(hashes),'report':str(out/'report.html')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();render(args.input,args.output,args.allow_partial)
