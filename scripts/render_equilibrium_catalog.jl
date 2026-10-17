"""Render a small descriptor table, without introducing biological classifications."""
module RenderEquilibriumCatalog
using CairoMakie
include("equilibrium_catalog.jl")
const C=EquilibriumCatalog

function examples(catalog)
    data=C.Descriptor[];C.foreach_descriptor(d->push!(data,d),catalog)
    result=C.Descriptor[]
    function pick(predicate,key)
        candidates=filter(predicate,data)
        isempty(candidates) || push!(result,first(sort(candidates;by=d->(key(d),d.context_id,d.root_id))))
    end
    pick(d->d.E_arm==1 && d.I_branch==1 && d.stability==1,d->max(d.E,d.I))
    pick(d->d.E_arm==2 && d.I_branch==1 && d.stability==1,d->max(d.E,d.I))
    pick(d->d.E_arm==3 && d.I_branch==2 && d.stability==1,d->d.I)
    pick(d->d.E_arm==3 && d.I_branch==2 && d.stability==1,d->-d.I)
    pick(d->d.E_arm==3 && d.I_branch==1 && d.stability==1,d->-d.E)
    pick(d->d.E_arm==0 && d.stability==1,d->max(d.E,d.I))
    pick(d->d.stability==2,d->d.context_id)
    pick(d->d.stability==3 && d.omega>0,d->d.context_id)
    pick(d->d.stability==1 && d.omega>0,d->d.context_id)
    result
end

function render(directory,out)
    catalog=C.read_catalog(directory)
    ispath(out) && error("output exists");mkpath(out)
    ds=examples(catalog)
    C.CSV.write(joinpath(out,"examples.csv"),C.record.(ds))
    f=Figure(size=(1320,610))
    Label(f[0,:],"Compact equilibrium descriptors preserve geometry, activity, and the spectrum";
        fontsize=22,font=:bold,halign=:left,tellwidth=false)
    headers=("Context / root","E arm","I branch","E","I","max Re(λ)","min Re(λ)","|Im(λ)|","Stability","Near singular")
    for (j,h) in enumerate(headers)
        Label(f[1,j],h;fontsize=15,font=:bold)
    end
    small(x)=string(round(x;sigdigits=4))
    for (i,d) in enumerate(ds)
        ename=d.E_arm==0 ? "nonfolded" : C.name(C.E_ARMS,d.E_arm)
        iname=d.I_branch==1 ? "rising" : d.I_branch==2 ? "falling" : C.name(C.I_BRANCHES,d.I_branch)
        fields=("$(d.context_id) / $(d.root_id)","$(d.E_arm) $ename","$(d.I_branch) $iname",small(d.E),small(d.I),
            small(d.lambda_max_real),small(d.lambda_min_real),small(d.omega),C.name(C.STABILITY,d.stability),string(d.near_singular))
        for (j,value) in enumerate(fields)
            Label(f[i+1,j],value;fontsize=14)
        end
    end
    Label(f[length(ds)+2,:],"E and I: occupancy. Spectral values: ms⁻¹. I branch describes the response function, not a numbering of nullcline folds.\n"*
        "Illustrative records selected by geometry/spectrum; no biological labels or population frequencies are inferred.\n"*
        "Display rounded to four significant digits; the catalog and examples.csv retain original Float64 precision.\n"*
        "For a real pair: λ = min/max Re(λ). For a complex pair: λ = Re(λ) ± i|Im(λ)|. Flags retain the source numerical assessment.";
        fontsize=13,halign=:left,justification=:left,tellwidth=false)
    colgap!(f.layout,10);rowgap!(f.layout,12)
    for (ext,kw) in (("png",(;px_per_unit=1.5)),("pdf",(;pt_per_unit=.6)),("svg",(;pt_per_unit=.6)))
        save(joinpath(out,"descriptors.$ext"),f;kw...)
    end
    open(joinpath(out,"index.html"),"w") do io
        write(io,"<!doctype html><meta charset='utf-8'><title>Compact equilibrium descriptors</title><style>body{font:18px sans-serif;margin:2em;max-width:1400px}img{width:100%}</style><h1>Compact equilibrium catalog</h1><p>Geometry, activity, full two-dimensional eigenvalue spectrum, and numerical status. No biological labels are assigned.</p><a href='descriptors.pdf'>PDF</a> · <a href='descriptors.svg'>SVG</a> · <a href='examples.csv'>Exact example records</a><img src='descriptors.png' alt='Example equilibrium descriptors'>")
    end
    C.write_toml(joinpath(out,"provenance.toml"),Dict("catalog_manifest_sha256"=>C.IOChecks.digest(joinpath(directory,"checksums.toml")),
        "renderer_sha256"=>C.IOChecks.digest(@__FILE__)))
    C.IOChecks.write_manifest(out)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("usage: render_equilibrium_catalog.jl CATALOG NEW_OUTPUT")
    RenderEquilibriumCatalog.render(abspath.(ARGS)...)
end
