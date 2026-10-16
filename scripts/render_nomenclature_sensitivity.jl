"""Render candidate-boundary diagnostics without changing manuscript figures."""
module NomenclatureSensitivityRenderer
using CairoMakie
import CSV, TOML
include("figure_packet_io.jl")
const IOChecks=FigurePacketIO
const COHORTS=("screen","detail_geometry")
const LABELS=("232-case screen","Eight detailed circuits")
const ROLES=("rest","active","herald","seizure","unassigned")
const COLORS=("#0072B2","#009E73","#CC79A7","#D55E00","#666666")
const FAMILIES=("rest","seizure","herald","partial_failure")
rows(p,n)=collect(CSV.File(joinpath(p,n);stringtype=String))
function chunks(p,prefix)
    files=sort(filter(n->startswith(n,prefix*"_") && endswith(n,".csv.gz"),readdir(p)))
    reduce(vcat,[open(io->collect(CSV.File(io;stringtype=String)),`gzip -dc $(joinpath(p,n))`) for n in files])
end
title!(f,s)=Label(f[0,:],s;fontsize=22,font=:bold,halign=:left,tellwidth=false)
note!(f,row,s)=Label(f[row,:],s;fontsize=13,halign=:left,justification=:left,tellwidth=false)
function savefig(f,out,name)
    save(joinpath(out,name*".png"),f;px_per_unit=1.5)
    save(joinpath(out,name*".pdf"),f;pt_per_unit=.6)
    save(joinpath(out,name*".svg"),f;pt_per_unit=.6)
end

function distributions(p,out)
    f=Figure(size=(1180,850));title!(f,"Activity spans a continuum beyond the existing state names")
    zeros=String[]
    for (j,cohort) in enumerate(COHORTS)
        rs=chunks(p,"membership_support_$cohort")
        ax=Axis(f[1,j],title="$(Char('A'+j-1))   $(LABELS[j]): attracting roots",xlabel="E occupancy",ylabel="I occupancy")
        for (role,c,marker) in zip(ROLES,COLORS,(:circle,:rect,:utriangle,:diamond,:cross))
            group=filter(r->r.role==role,rs)
            isempty(group) || scatter!(ax,getproperty.(group,:E),getproperty.(group,:I);color=(c,.35),marker,markersize=3,rasterize=true,label=role)
        end
        xlims!(ax,-.01,.51);ylims!(ax,-.01,.51)
        axislegend(ax;position=:lc,labelsize=11,nbanks=2)
        a=Axis(f[2,j],title="$(Char('C'+j-1))   Coordinate distributions (same records)",
            xlabel="log10 occupancy (positive values)",ylabel="Cumulative fraction of records",limits=((-18,log10(.5)),(0,1)))
        for (field,c,style) in (("E","#0072B2",:solid),("I","#D55E00",:dash))
            cdf=sort(chunks(p,"cdf_$(cohort)_$field");by=r->r.value)
            positive=filter(r->r.value>0,cdf)
            lines!(a,log10.(getproperty.(positive,:value)),[r.cumulative/r.total for r in positive];color=c,linestyle=style,label=field)
            zero=sum(r.cumulative for r in cdf if r.value==0;init=0)
            below=count(r->0<getproperty(r,Symbol(field))<1e-18,rs)
            push!(zeros,"$(LABELS[j]), $field: $zero exact zeros; $below positive values below displayed range")
        end
        axislegend(a;position=:rb)
    end
    note!(f,3,"Colors/shapes show original local names; all attracting nonsingular records are plotted (overlap can hide multiplicity).\nCDFs include all records in the denominator; logarithmic displays omit zero and crop values below 10^-18.\n"*join(zeros,";\n")*". Counts are sampled records, not prevalence.")
    savefig(f,out,"01_activity_distributions")
end

function surfaces(p,out)
    f=Figure(size=(1240,810));title!(f,"Candidate boundaries change membership; no cutoff is adopted")
    for (i,cohort) in enumerate(COHORTS)
        data=chunks(p,"surfaces_$cohort")
        for (j,family) in enumerate(("rest","seizure","herald"))
            selected=filter(r->r.family==family,data)
            values=zeros(101,101)
            for r in selected;values[r.E_percentile+1,r.I_percentile+1]=100r.count/r.total;end
            ax=Axis(f[i,j],title="$(Char('A'+(i-1)*3+j-1))   $(LABELS[i]): $family",titlesize=14,
                xlabel="E boundary: empirical percentile",ylabel="I boundary: empirical percentile")
            heatmap!(ax,0:100,0:100,values;colorrange=(0,100),colormap=:viridis)
        end
    end
    Colorbar(f[1:2,4];limits=(0,100),colormap=:viridis,label="Members / cohort records (%)")
    note!(f,3,"Rest: E ≤ E0, I ≤ I0. Seizure: E > E0, I ≤ I0, descending inhibitory response. Herald: E ≥ E0, I ≥ I0.\nEach panel varies its own two boundaries independently; every plotted cutoff is an observed coordinate.\nPercentiles are within each cohort, not biological activity levels. Repeated cutoffs produce plateaus; exact values are in the quantile tables.")
    savefig(f,out,"02_boundary_surfaces")
end

function comparisons(p,out)
    f=Figure(size=(1180,860));title!(f,"Independent candidate definitions expose overlaps and gaps")
    names=["rest","seizure","herald","partial\nfailure","overlap","gap"]
    for (j,cohort) in enumerate(COHORTS)
        rs=rows(p,"scenario_counts_$cohort.csv")
        ax=Axis(f[1,j],title="$(Char('A'+j-1))   $(LABELS[j]): 729 boundary combinations",
            ylabel="Fraction of cohort records (%)",xticks=(1:6,names),xticklabelsize=11)
        for (k,field) in enumerate((:rest,:seizure,:herald,:partial_failure,:overlap,:gap))
            values=sort([100getproperty(r,field)/r.total for r in rs])
            lines!(ax,[k,k],[first(values),last(values)];color=:black,linewidth=3)
            scatter!(ax,[k],[values[cld(length(values),2)]];color=:black,markersize=9)
        end
        ylims!(ax,0,100)
        ir=chunks(p,"interventions_$cohort")
        keys=sort(unique((r.id,r.axis,r.value,r.root) for r in ir))
        mat=zeros(length(keys),4);lookup=Dict(k=>i for (i,k) in enumerate(keys))
        for r in ir
            labels=ismissing(r.memberships) ? String[] : split(r.memberships,'|')
            for (k,family) in enumerate(FAMILIES)
                family in labels && (mat[lookup[(r.id,r.axis,r.value,r.root)],k]+=1/729)
            end
        end
        a=Axis(f[2,j],title="$(Char('C'+j-1))   84 intervention destinations; $(LABELS[j]) cutoffs",titlesize=14,
            xlabel="Destination index (sorted by circuit, axis, value)",ylabel="Candidate membership",
            yticks=(1:4,["rest","seizure","herald","partial failure"]))
        heatmap!(a,1:length(keys),1:4,mat;colorrange=(0,1),colormap=:viridis)
    end
    Colorbar(f[2,3];limits=(0,1),colormap=:viridis,label="Fraction of 729 boundary combinations")
    note!(f,3,"Bars: minimum–maximum across the declared sensitivity grid; dots: median. These are not confidence intervals.\nMemberships are independent: overlaps are retained; gaps can include ordinary active states and other unnamed states.\nIntervention rows retain original local and inherited names in the packet. Fractions describe cutoff sensitivity, not classification probability.\nThe grid deliberately includes broad stress probes that need not be plausible definitions of low activity.")
    savefig(f,out,"03_membership_sensitivity")
end

function low_activity(p,out)
    f=Figure(size=(1260,850));title!(f,"Low activity on the middle E arm can require inhibitory stabilization")
    self_axes=Axis[];decay_axes=Axis[]
    for (i,cohort) in enumerate(COHORTS)
        rs=filter(r->r.joint_activity<=.05,chunks(p,"dynamics_$cohort"))
        a=Axis(f[i,1],title="$(Char('A'+(i-1)*3))   $(LABELS[i]): activity zoom",titlesize=14,xlabel="E occupancy",ylabel="I occupancy")
        b=Axis(f[i,2],title="$(Char('B'+(i-1)*3))   E self-response",titlesize=14,xlabel="max(E, I)",ylabel="J_EE (ms⁻¹)")
        c=Axis(f[i,3],title="$(Char('C'+(i-1)*3))   Slowest linear decay",titlesize=14,xlabel="max(E, I)",ylabel="−1 / max Re(λ) (ms)",yscale=log10)
        push!(self_axes,b);push!(decay_axes,c)
        for (arm,color,marker) in (("lower","#0072B2",:circle),("middle","#D55E00",:utriangle))
            group=filter(r->r.arm==arm,rs)
            scatter!(a,getproperty.(group,:E),getproperty.(group,:I);color=(color,.7),marker,markersize=4,rasterize=true,label="$arm E arm")
            scatter!(b,getproperty.(group,:joint_activity),getproperty.(group,:J_EE);color=(color,.7),marker,markersize=4,rasterize=true)
            scatter!(c,getproperty.(group,:joint_activity),getproperty.(group,:linear_decay_ms);color=(color,.7),marker,markersize=4,rasterize=true)
        end
        hlines!(b,[0.];color=:black,linestyle=:dash)
        xlims!(a,0,.05);ylims!(a,0,.05);xlims!(b,0,.05);xlims!(c,0,.05)
        axislegend(a;position=:lt,labelsize=11)
    end
    linkyaxes!(self_axes...);linkyaxes!(decay_axes...)
    note!(f,3,"Only attracting, nonsingular lower/middle-arm roots are shown. Display zoom max(E,I) ≤ 0.05 is not a rest cutoff.\nPositive J_EE means an E perturbation grows with I held fixed; the full coupled system is locally attracting.\nDecay times describe eigenmodes, not finite perturbation recovery, switching thresholds, or basin size. Full-coordinate data are retained.")
    savefig(f,out,"04_low_activity_dynamics")
end

function render(packet,out)
    IOChecks.verify_manifest(packet)
    meta=TOML.parsefile(joinpath(packet,"study.toml"));meta["completed"] || error("incomplete packet")
    ispath(out) && error("output exists");mkpath(out)
    distributions(packet,out);surfaces(packet,out);comparisons(packet,out);low_activity(packet,out)
    names=("01_activity_distributions","02_boundary_surfaces","03_membership_sensitivity","04_low_activity_dynamics")
    open(joinpath(out,"index.html"),"w") do io
        write(io,"<!doctype html><meta charset='utf-8'><title>Equilibrium nomenclature sensitivity</title><style>body{font:18px sans-serif;max-width:1250px;margin:2em auto}img{width:100%}</style><h1>Candidate definitions: author review</h1><p>No cutoff or new state name has been adopted. Original classifications remain unchanged.</p>")
        for name in names
            write(io,"<h2>$(replace(name,'_'=>' '))</h2><a href='$name.pdf'>PDF</a> · <a href='$name.svg'>SVG</a><img src='$name.png' alt='$name'>")
        end
    end
    open(joinpath(out,"provenance.toml"),"w") do io
        TOML.print(io,Dict("packet_manifest_sha256"=>IOChecks.digest(joinpath(packet,"checksums.toml")),
            "renderer_sha256"=>IOChecks.digest(@__FILE__));sorted=true)
    end
    IOChecks.write_manifest(out)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("usage: render_nomenclature_sensitivity.jl PACKET NEW_OUTPUT")
    NomenclatureSensitivityRenderer.render(abspath.(ARGS)...)
end
