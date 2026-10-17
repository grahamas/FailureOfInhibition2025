"""Render conditional occupancy distributions and explicit structural denominators."""
module RenderStructuralDistributions
using CairoMakie
include("catalog_structural_distributions.jl")
const M=CatalogStructuralDistributions
const C=M.C
const NAMES=("Screen","Detailed geometry")
const ARM_NAMES=Dict(0=>"No E folds",1=>"Lower E arm",2=>"Middle E arm",3=>"Upper E arm",4=>"E turning point",255=>"Unresolved E arm")
const SIDE_NAMES=Dict(0=>"unresolved",1=>"ascending",2=>"descending",3=>"peak")
const ARM_COLORS=Dict(0=>"#777777",1=>"#0072B2",2=>"#D55E00",3=>"#009E73",4=>"#CC79A7",255=>"#000000")
const STYLES=Dict(0=>:dashdot,1=>:solid,2=>:dash,3=>:dot,4=>:dashdot,255=>:solid)
const SIDE_COLORS=Dict(0=>"#777777",1=>"#0072B2",2=>"#D55E00",3=>"#CC79A7")
heading(f,t)=Label(f[0,:],t;fontsize=22,font=:bold,halign=:left,tellwidth=false)
foot(f,row,t)=Label(f[row,:],t;fontsize=13,halign=:left,justification=:left,tellwidth=false)
function savefig(f,out,name)
    save(joinpath(out,name*".png"),f;px_per_unit=1.4)
    save(joinpath(out,name*".pdf"),f;pt_per_unit=.65)
    save(joinpath(out,name*".svg"),f;pt_per_unit=.65)
    println("Rendered $name");flush(stdout)
end

function intervals(summary,keys,out)
    f=Figure(size=(1380,1020));heading(f,"1   How much E and I activity occurs in each structural category?")
    for (row,scope) in enumerate(M.COHORTS)
        rs=filter(r->r.scope==scope && r.subset=="attracting_nonsingular",summary)
        labels=String[]
        for (a,b) in keys
            group=filter(r->r.E_arm==a && r.I_branch==b && r.variable=="E",rs)
            n=isempty(group) ? 0 : only(group).n
            push!(labels,"$(ARM_NAMES[a]) / $(SIDE_NAMES[b])\nn = $n")
        end
        for (col,variable) in enumerate(("E","I"))
            ax=Axis(f[row,col],title="$(NAMES[row]): $variable occupancy",xlabel="$variable occupancy",
                yticks=(1:length(keys),labels),yticklabelsize=12,yreversed=true)
            col==2 && hideydecorations!(ax;grid=false)
            for (i,(a,b)) in enumerate(keys)
                group=filter(r->r.E_arm==a && r.I_branch==b && r.variable==variable,rs)
                if isempty(group) || only(group).n==0
                    text!(ax,.25,i;text="no eligible records",align=(:center,:center),fontsize=11,color=:gray)
                    continue
                end
                r=only(group);c=SIDE_COLORS[b]
                lines!(ax,[r.q05,r.q95],[i,i];color=c,linewidth=2)
                lines!(ax,[r.q25,r.q75],[i,i];color=c,linewidth=7)
                scatter!(ax,[r.median],[i];color=:black,marker=b==1 ? :circle : b==2 ? :utriangle : :rect,markersize=8)
            end
            xlims!(ax,-.008,.508);ylims!(ax,.4,length(keys)+.6)
        end
    end
    Legend(f[3,:],[LineElement(color=SIDE_COLORS[1],linewidth=3),LineElement(color=SIDE_COLORS[2],linewidth=3),
        MarkerElement(color=:black,marker=:circle),MarkerElement(color=:black,marker=:utriangle)],
        ["Ascending I response","Descending I response","Median (ascending)","Median (descending)"];
        orientation=:horizontal,framevisible=false,labelsize=13)
    foot(f,4,"Attracting, nonsingular equilibria. Dot/triangle = median; thick segment = middle 50%; thin segment = middle 90% of records.\nIntervals describe the observed distribution, not uncertainty in a mean. Quantiles are observed values; no smoothing or interpolation.\nE and I summaries are separate marginals, not paired coordinates of a typical root. Each record has equal weight within its structural category.\nCounts are adaptive-sample records, not prevalence. Cohorts overlap and remain separate. Missing categories mean none found in this catalog.")
    savefig(f,out,"01_typical_activity_by_structure")
end

function ecdfs(groups,out)
    f=Figure(size=(1200,900));heading(f,"2   What fraction of each E arm lies below a given activity level?")
    for (col,scope) in enumerate(M.COHORTS), (row,variable) in enumerate((:E,:I))
        ax=Axis(f[row,col],title="$(NAMES[col]): $variable by E arm",xlabel="$variable occupancy",ylabel="Records at or below x (%)")
        arms=sort([a for (s,a,b) in keys(groups) if s==scope && a>=0 && b==-1])
        for arm in arms
            ds=filter(M.eligible,groups[(scope,arm,-1)])
            isempty(ds) && continue
            x,y=M.ecdf_steps(getproperty.(ds,variable))
            lines!(ax,x,100 .* y;color=ARM_COLORS[arm],linestyle=STYLES[arm],linewidth=2,
                label="$(ARM_NAMES[arm]) (n=$(length(ds)))",rasterize=true)
        end
        xlims!(ax,-.005,.505);ylims!(ax,-1,101)
        axislegend(ax;position=:rb,labelsize=12,framevisible=true)
    end
    foot(f,3,"Each curve is normalized within its own E-arm category among attracting, nonsingular roots. A steep rise means activity concentrates there.\nExample reading: y = 80 at x = 0.1 means 80% of that arm's records have occupancy ≤ 0.1. Exact zeros and the 0.5 ceiling are included.\nThese are exact empirical cumulative distributions. No bins or smoothing. Counts are shown so sparse groups are not mistaken for equally supported groups.\nArm boundaries vary with circuit parameters; these pooled distributions describe the catalog, not a universal occupancy threshold for an arm.")
    savefig(f,out,"02_occupancy_cdfs_by_E_arm")
end

function branch_histograms(histograms,out)
    f=Figure(size=(1400,880));heading(f,"3   The inhibitory response side separates very different I distributions")
    for (row,scope) in enumerate(M.COHORTS), (col,arm) in enumerate((1,2,3))
        ax=Axis(f[row,col],title="$(NAMES[row]): $(ARM_NAMES[arm])",titlesize=16,
            xlabel="I occupancy",ylabel="Within-branch records per bin (%)")
        for branch in (1,2)
            rs=sort(filter(r->r.scope==scope && r.E_arm==arm && r.I_branch==branch && r.variable=="I" && r.subset=="attracting_nonsingular",histograms);by=r->r.bin)
            n=isempty(rs) ? 0 : first(rs).n
            if n==0
                text!(ax,.03,branch==2 ? .76 : .88;text="$(SIDE_NAMES[branch]): no eligible records",space=:relative,fontsize=11)
                continue
            end
            x=Float64[];y=Float64[]
            for r in rs
                append!(x,(r.lo,r.hi));append!(y,(100r.count/n,100r.count/n))
            end
            lines!(ax,x,y;color=SIDE_COLORS[branch],linestyle=branch==1 ? :solid : :dash,
                linewidth=2,label="$(SIDE_NAMES[branch]) (n=$n)")
        end
        xlims!(ax,-.005,.505);ylims!(ax,-1,101)
        axislegend(ax;position=:rt,labelsize=11)
    end
    foot(f,3,"Attracting, nonsingular roots; each response-side distribution sums to 100% separately. Fifty bins of width 0.01; final bin includes 0.5.\nThe traces show piecewise-constant bin percentages, not interpolated densities. Their sum is not an estimate of category frequency.\nThe no-E-fold category is omitted here (2,310 eligible screen records; no detailed records) and retained in Figures 1–2 and the tables.\nAscending/descending describe the I response function. Descending does not by itself mean near-zero inhibitory activity.")
    savefig(f,out,"03_I_distributions_by_arm_and_response")
end

function stability_view(rows,keys,out)
    f=Figure(size=(1320,750));heading(f,"4   Which structural categories contain attracting or unstable equilibria?")
    fields=(:attracting,:saddle,:repelling,:unresolved)
    colors=("#0072B2","#D55E00","#CC79A7","#777777")
    labels=["$(ARM_NAMES[a]) / $(SIDE_NAMES[b])" for (a,b) in keys]
    for (col,scope) in enumerate(M.COHORTS)
        ax=Axis(f[1,col],title=NAMES[col],xlabel="Within-category records (%)",yticks=(1:length(keys),labels),
            yticklabelsize=13,yreversed=true)
        col==2 && hideydecorations!(ax;grid=false)
        for (i,(a,b)) in enumerate(keys)
            rs=filter(r->r.scope==scope && r.E_arm==a && r.I_branch==b,rows)
            if isempty(rs)
                text!(ax,50,i;text="no records",align=(:center,:center),color=:gray,fontsize=12)
                continue
            end
            r=only(rs);left=0.
            for (field,color) in zip(fields,colors)
                width=100getproperty(r,field)/r.n
                width>0 && poly!(ax,Rect2f(left,i-.33,width,.66);color,strokewidth=0)
                left+=width
            end
            text!(ax,101,i;text="n=$(r.n)",align=(:left,:center),fontsize=11)
        end
        xlims!(ax,0,125);ylims!(ax,.4,length(keys)+.6);ax.xticks=0:25:100
    end
    Legend(f[2,:],[PolyElement(color=c) for c in colors],["Attracting","Saddle","Repelling","Unresolved"];
        orientation=:horizontal,framevisible=false)
    foot(f,3,"All discovered equilibrium records in each structural category are included here; each bar sums to 100%. Numerical stability is inherited.\nThe distribution plots select attracting, nonsingular roots. This panel shows the selection context; near-singular counts are retained in stability.csv.\nNo root in these two cohorts is flagged near singular in this catalog. A 100% bar describes these samples and does not prove a global stability rule.\nEmpty means no catalog records, not mathematical impossibility. These are not basin sizes, occurrence probabilities, or independent branches.")
    savefig(f,out,"04_stability_within_structural_categories")
end

function joint_views(joint,summary,out)
    selected=filter(r->r.subset=="attracting_nonsingular",joint)
    lower=floor(minimum(log10(100r.count/r.n) for r in selected))
    limits=(lower,2.)
    ticks=collect(lower:2.);ticklabels=[string(round(10.0^v;sigdigits=3)) for v in ticks]
    for (which,scope) in enumerate(M.COHORTS)
        f=Figure(size=(1480,1160));heading(f,"$(which+4)   $(NAMES[which]): joint E–I occupancy within structural categories")
        for (row,branch) in enumerate((-1,1,2)), (col,arm) in enumerate((-1,1,2,3))
            keyrows=filter(r->r.scope==scope && r.E_arm==arm && r.I_branch==branch,selected)
            q=filter(r->r.scope==scope && r.E_arm==arm && r.I_branch==branch && r.subset=="attracting_nonsingular" && r.variable=="E",summary)
            n=isempty(q) ? 0 : only(q).n
            aname=arm==-1 ? "All E geometry" : ARM_NAMES[arm]
            bname=branch==-1 ? "All I response sides" : "I response: $(SIDE_NAMES[branch])"
            ax=Axis(f[row,col],title="$aname\n$bname · n=$n",titlesize=14,
                xlabel="E occupancy",ylabel="I occupancy",xticks=[0.,.25,.5],yticks=[0.,.25,.5],aspect=1,
                xgridvisible=false,ygridvisible=false)
            h=fill(NaN,100,100)
            for r in keyrows;h[r.E_bin,r.I_bin]=log10(100r.count/n);end
            heatmap!(ax,range(0.,.5;length=101),range(0.,.5;length=101),h;
                colorrange=limits,colormap=:viridis,nan_color=:white,rasterize=true)
            n==0 && text!(ax,.25,.25;text="No eligible\nrecords",align=(:center,:center),fontsize=14,color=:gray)
            xlims!(ax,-.005,.505);ylims!(ax,-.005,.505)
        end
        Colorbar(f[1:3,5];limits,colormap=:viridis,ticks=(ticks,ticklabels),label="Panel records per E–I bin (%) · logarithmic color scale")
        foot(f,4,"Attracting, nonsingular roots. Each nonempty panel sums to 100% independently; identical 100 × 100 bins and color limits in both cohort figures.\nE is horizontal; I is vertical. Bin width = 0.005 in each population. Exact 0 and 0.5 are retained; white bins contain no sampled records.\nTop row pools I response sides; lower rows condition on response side. Columns/rows overlap and must not be added as independent data.\nAll E geometry includes no-fold records; separate lower/middle/upper columns do not. These are catalog distributions, not basin weights or prevalence.")
        savefig(f,out,which==1 ? "05_joint_EI_screen" : "06_joint_EI_detailed")
    end
end

function render(directory,packet,out)
    catalog=C.read_catalog(directory);C.IOChecks.verify_manifest(packet)
    meta=C.TOML.parsefile(joinpath(packet,"study.toml"))
    meta["completed"] && meta["schema_version"]==1 || error("invalid study")
    meta["catalog_manifest_sha256"]==C.IOChecks.digest(joinpath(directory,"checksums.toml")) || error("catalog mismatch")
    meta["analysis_sha256"]==C.IOChecks.digest(joinpath(@__DIR__,"catalog_structural_distributions.jl")) || error("analysis changed")
    groups=M.load_groups(catalog)
    joint=sort(unique((a,b) for (s,a,b) in keys(groups) if a>=0 && b>=0))
    summary=C.read_csv(joinpath(packet,"quantiles.csv"));histograms=C.read_gzip(joinpath(packet,"histograms.csv.gz"));stability=C.read_csv(joinpath(packet,"stability.csv"))
    joint_hist=C.read_gzip(joinpath(packet,"joint_histograms.csv.gz"))
    ispath(out) && error("output exists");mkpath(out)
    with_theme(Theme(fontsize=16)) do
        intervals(summary,joint,out);ecdfs(groups,out);branch_histograms(histograms,out);stability_view(stability,joint,out);joint_views(joint_hist,summary,out)
    end
    names=("01_typical_activity_by_structure","02_occupancy_cdfs_by_E_arm","03_I_distributions_by_arm_and_response","04_stability_within_structural_categories","05_joint_EI_screen","06_joint_EI_detailed")
    captions=("Typical E and I activity: medians, middle 50%, and middle 90% within each E-arm / I-response category.",
        "Exact cumulative distributions answer how much of an E-arm category lies below any chosen activity level.",
        "I histograms distinguish low-I and high-I concentrations within each E arm and inhibitory response side.",
        "Stability proportions explain which roots enter the attracting-equilibrium distributions.",
        "Joint E–I histograms: screen cohort, conditioned on E arm and inhibitory response side.",
        "Joint E–I histograms: detailed cohort, using the same bins and color scale.")
    open(joinpath(out,"index.html"),"w") do io
        write(io,"<!doctype html><meta charset='utf-8'><title>Occupancy by structural category</title><style>body{font:18px sans-serif;max-width:1400px;margin:2em auto;padding:0 1em}img{width:100%}section{margin:3em 0}</style><h1>Activity distributions within structural categories</h1><p>E arm, I response side, and stability are separate descriptors. No proposed biological thresholds are used. Screen and detailed cohorts remain separate; distributions describe sampled records.</p>")
        for (n,c) in zip(names,captions)
            write(io,"<section><h2>$c</h2><a href='$n.pdf'>PDF</a> · <a href='$n.svg'>SVG</a><img src='$n.png' alt='$c'></section>")
        end
    end
    C.write_toml(joinpath(out,"provenance.toml"),Dict("catalog_manifest_sha256"=>meta["catalog_manifest_sha256"],
        "summary_manifest_sha256"=>C.IOChecks.digest(joinpath(packet,"checksums.toml")),"renderer_sha256"=>C.IOChecks.digest(@__FILE__)))
    C.IOChecks.write_manifest(out)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==3 || error("usage: render_structural_distributions.jl CATALOG SUMMARIES NEW_OUTPUT")
    RenderStructuralDistributions.render(abspath.(ARGS)...)
end
