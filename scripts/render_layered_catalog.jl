"""Continuous activity, structure/dynamics, and deferred biological interpretation."""
module RenderLayeredCatalog
include("render_structural_distributions.jl")
using CairoMakie
const R=RenderStructuralDistributions
const M=R.M
const C=M.C

function dynamics(catalog,out)
    groups=M.load_groups(catalog)
    cohorts=[sort(groups[(scope,-1,-1)];by=d->(d.context_id,d.root_id)) for scope in M.COHORTS]
    counts=NamedTuple[]
    for (scope,ds) in zip(M.COHORTS,cohorts), status in 0:3, complex in (false,true), near in (false,true)
        selected=filter(d->d.stability==status && (d.omega>0)==complex && d.near_singular==near,ds)
        push!(counts,(;scope,stability=status,complex,near_singular=near,n=length(selected)))
    end
    C.CSV.write(joinpath(out,"dynamics_counts.csv"),counts)
    colors=("#777777","#0072B2","#D55E00","#CC79A7")
    markers=(:rect,:circle,:utriangle,:diamond)
    names=("Unresolved","Attracting","Saddle","Repelling")
    f=Figure(size=(1250,770));R.heading(f,"7   Stability across continuous E–I activity")
    for (col,ds) in enumerate(cohorts)
        ax=Axis(f[1,col],title="$(R.NAMES[col]) · all roots n=$(length(ds))",xlabel="E occupancy",ylabel="I occupancy",aspect=1)
        scatter!(ax,[d.E for d in ds],[d.I for d in ds];color=[colors[Int(d.stability)+1] for d in ds],
            marker=[markers[Int(d.stability)+1] for d in ds],markersize=3,rasterize=true)
        xlims!(ax,-.005,.505);ylims!(ax,-.005,.505)
    end
    Legend(f[2,:],[MarkerElement(color=colors[i],marker=markers[i]) for i in 1:4],collect(names);orientation=:horizontal)
    R.foot(f,3,"All recorded roots, including near-singular and unresolved records. Colors and symbols encode original numerical stability.\nExact E/I coordinates; no bins, smoothing, or activity thresholds. Draw order is catalog context ID then root ID; later points can obscure earlier points.\nThese overlapping samples do not measure prevalence or basin size. Biological interpretation is deferred.")
    R.savefig(f,out,"07_stability_in_activity_space")
    sinks=[filter(M.eligible,ds) for ds in cohorts]
    values=[d.lambda_max_real for ds in sinks for d in ds]
    isempty(values) && error("no attracting nonsingular roots to display")
    lo,hi=extrema(values)
    if lo==hi
        lo,hi=lo-1.,hi+1.
    end
    f=Figure(size=(1250,790));R.heading(f,"8   Local decay rate and spectrum across E–I activity")
    for (col,ds) in enumerate(sinks)
        ax=Axis(f[1,col],title="$(R.NAMES[col]) · eligible n=$(length(ds))",xlabel="E occupancy",ylabel="I occupancy",aspect=1)
        scatter!(ax,[d.E for d in ds],[d.I for d in ds];color=[d.lambda_max_real for d in ds],
            colorrange=(lo,hi),colormap=:viridis,marker=[d.omega==0 ? :circle : :utriangle for d in ds],markersize=4,rasterize=true)
        xlims!(ax,-.005,.505);ylims!(ax,-.005,.505)
    end
    Colorbar(f[1,3];limits=(lo,hi),colormap=:viridis,label="Largest eigenvalue real part (ms⁻¹); linear scale")
    Legend(f[2,:],[MarkerElement(color=:black,marker=:circle),MarkerElement(color=:black,marker=:utriangle)],
        ["Real spectrum","Complex conjugate spectrum"];orientation=:horizontal)
    R.foot(f,3,"Attracting, nonsingular roots. Shared color limits include every displayed value; less negative means slower asymptotic local decay.\nCircles: real eigenvalues; triangles: complex pair. Source angular frequency is retained in the interface as ω (rad/ms).\nExact coordinates, deterministic context/root draw order; overlapping marks can hide spectrum types. No basin or finite-pulse response is inferred.\nActivity remains continuous. Biological interpretation is deferred.")
    R.savefig(f,out,"08_local_dynamics_in_activity_space")
end

function render(directory,packet,out)
    R.render(directory,packet,out)
    catalog=C.read_catalog(directory)
    with_theme(Theme(fontsize=16)) do
        dynamics(catalog,out)
    end
    index=joinpath(out,"index.html")
    page=read(index,String)
    page=replace(page,"<h1>Activity distributions within structural categories</h1>"=>
        "<h1>Three-layer equilibrium catalog</h1><p><b>Activity:</b> continuous E and I. <b>Structure and dynamics:</b> arms, response branches, stability, and spectrum. <b>Biological interpretation:</b> deferred; no biological labels or activity bands assigned.</p>")
    open(index,"w") do io
        write(io,page)
        for (stem,title) in (("07_stability_in_activity_space","Source stability across activity space"),("08_local_dynamics_in_activity_space","Local decay rate and spectrum across activity space"))
            write(io,"<section><h2>$title</h2><a href='$stem.pdf'>PDF</a> · <a href='$stem.svg'>SVG</a><img src='$stem.png' alt='$title'></section>")
        end
    end
    meta=C.TOML.parsefile(joinpath(out,"provenance.toml"))
    meta["layered_renderer_sha256"]=C.IOChecks.digest(@__FILE__)
    meta["catalog_reader_sha256"]=C.IOChecks.digest(joinpath(@__DIR__,"equilibrium_catalog.jl"))
    meta["interpretation_status"]="deferred"
    meta["activity_representation"]="continuous original occupancies; no bands or thresholds"
    meta["spectrum_units"]="real parts ms^-1; omega rad/ms"
    meta["draw_order"]="ascending catalog context_id, root_id; no random subsampling"
    C.write_toml(joinpath(out,"provenance.toml"),meta)
    C.IOChecks.write_manifest(out)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==3 || error("usage: render_layered_catalog.jl CATALOG SUMMARIES NEW_OUTPUT")
    RenderLayeredCatalog.render(abspath.(ARGS)...)
end
