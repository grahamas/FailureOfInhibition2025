"""Render catalog comparisons without relabeling manuscript data or interpolating input regions."""
module RenderCatalogClassifications
using CairoMakie
include("catalog_classifications.jl")
const M=CatalogClassifications
const C=M.C
const COHORTS=("screen","detail_geometry")
const COHORT_NAMES=("Screen","Detailed geometry")
const CASES=("selective_withdrawal","input_dependent")
const CASE_NAMES=("Reference circuit","Weak recurrent excitation")
const COLORS=("#777777","#0072B2","#D55E00","#CC79A7","#E69F00")
const MARKERS=(:circle,:rect,:utriangle,:diamond,:cross)
const SHORT=("Descriptive","Rest?","Seizure?","Herald?","High-E / mid-I failure")
const ARMS=(0,1,2,3,4,255)
const ARM_COLORS=("#777777","#0072B2","#E69F00","#009E73","#CC79A7","#000000")
const ARM_NAMES=("No E folds","Lower E arm","Middle E arm","Upper E arm","E turning point","Unresolved E arm")
heading(f,t)=Label(f[0,:],t;fontsize=23,font=:bold,halign=:left,tellwidth=false)
foot(f,row,t)=Label(f[row,:],t;fontsize=14,halign=:left,justification=:left,tellwidth=false)
function savefig(f,out,name)
    save(joinpath(out,name*".png"),f;px_per_unit=1.25)
    save(joinpath(out,name*".pdf"),f;pt_per_unit=.65)
    save(joinpath(out,name*".svg"),f;pt_per_unit=.65)
    println("Rendered $name");flush(stdout)
end
function points!(ax,rs,x,y;kwargs...)
    isempty(rs) || scatter!(ax,getproperty.(rs,x),getproperty.(rs,y);rasterize=true,kwargs...)
end
function portraits(groups,out)
    f=Figure(size=(1320,960));heading(f,"1   Continuous activity and proposed endpoint names")
    histograms=Matrix{Float64}[]
    for ds in groups
        h=zeros(120,120)
        for d in ds
            h[clamp(floor(Int,240*d.E)+1,1,120),clamp(floor(Int,240*d.I)+1,1,120)]+=1
        end
        push!(histograms,[v==0 ? NaN : log10(v) for v in h])
    end
    top=maximum(maximum(filter(isfinite,h)) for h in histograms)
    for (row,ds) in enumerate(groups)
        a=Axis(f[row,1],title="$(COHORT_NAMES[row]): $(length(ds)) attracting roots",xlabel="E occupancy",ylabel="I occupancy")
        heatmap!(a,range(0,.5;length=121),range(0,.5;length=121),histograms[row];colormap=:viridis,colorrange=(0,top),nan_color=:white,rasterize=true)
        b=Axis(f[row,3],title="Same roots: reference candidate policy",xlabel="E occupancy",ylabel="I occupancy")
        for code in 0:4
            selected=filter(d->M.candidate(d,M.POLICIES[2])==code,ds)
            points!(b,selected,:E,:I;color=(COLORS[code+1],.7),marker=MARKERS[code+1],markersize=3)
        end
        for ax in (a,b);xlims!(ax,-.006,.506);ylims!(ax,-.006,.506);end
    end
    Colorbar(f[1:2,2];limits=(0,top),colormap=:viridis,label="log10 records per bin (120 × 120)")
    Legend(f[3,:],[MarkerElement(color=c,marker=m) for (c,m) in zip(COLORS,MARKERS)],collect(SHORT);orientation=:horizontal,nbanks=2,framevisible=false)
    foot(f,4,"Attracting, nonsingular equilibria only; cohorts shown separately and not independent. Counts describe adaptive samples, not prevalence.\nDensity bins contain all records, including E or I = 0 and 0.5. Scatter overlap can hide roots. No interpolation or fitted boundaries.\nTrial cutoffs: low E/I ≤ 0.01; high E ≥ 0.45; retained I ≥ 0.30. Seizure/partial require descending I response; herald does not.\n? marks a proposed name. Gray roots retain activity and geometry descriptions; they are not numerically unresolved.")
    savefig(f,out,"01_activity_and_candidates")
end

function input_axis(f,pos,title)
    ax=Axis(f[pos...],title=title,titlesize=16,xlabel="Tonic E input, B_E",ylabel="Tonic I input, B_I")
    xlims!(ax,-.2,16.2);ylims!(ax,-.2,16.2);ax
end
function names_map(summaries,out)
    f=Figure(size=(1500,900));heading(f,"2   Coexistence maps preserve every discovered attracting equilibrium")
    fields=(:sinks,:rest,:seizure,:herald)
    titles=("Number of attracting roots","Rest candidate present","Seizure candidate present","Herald candidate present")
    allrows=filter(s->s.case_id in CASES && s.scope=="detail_geometry",summaries)
    cap=maximum(s.sinks for s in allrows)
    for (i,id) in enumerate(CASES)
        rows=filter(s->s.case_id==id,allrows)
        for (j,field) in enumerate(fields)
            ax=input_axis(f,(i,j),"$(CASE_NAMES[i])\n$(titles[j])")
            if j==1
                points!(ax,rows,:B_E,:B_I;color=getproperty.(rows,field),colorrange=(-.5,cap+.5),colormap=cgrad(:viridis,cap+1;categorical=true),markersize=4)
            else
                absent=filter(s->getproperty(s,field)==0,rows)
                present=filter(s->getproperty(s,field)>0,rows)
                points!(ax,absent,:B_E,:B_I;color="#cccccc",markersize=3)
                points!(ax,present,:B_E,:B_I;color=COLORS[j],markersize=4)
            end
        end
    end
    Colorbar(f[3,1];limits=(-.5,cap+.5),colormap=cgrad(:viridis,cap+1;categorical=true),ticks=0:cap,vertical=false,label="Discovered attracting roots")
    Legend(f[3,2:4],[MarkerElement(color=c,marker=:circle) for c in ("#cccccc",COLORS[2],COLORS[3],COLORS[4])],
        ["No member found","Rest? present","Seizure? present","Herald? present"];orientation=:horizontal,framevisible=false)
    foot(f,4,"Every dot is a sampled input context; white space is unmeasured. No cell fills or interpolation. All maps use the same input limits.\nMembership panels are separate: rest, seizure, and herald may coexist at the same input. Multiple roots with one name remain counted.\nReference policy; attracting, nonsingular roots only. No sink found does not imply no attractor: cycles and basin accessibility are outside this catalog.\nDiscovery remains CompletenessNotCertified. These are the two circuits shown in the current Figure 2.")
    savefig(f,out,"02_input_candidate_maps")
end

function raw_maps(summaries,out)
    allrows=filter(s->s.case_id in CASES && s.scope=="detail_geometry",summaries)
    finite=filter(s->s.sinks>0,allrows)
    alpha_limits=(minimum(s.alpha_min for s in finite),0.)
    for (idx,id) in enumerate(CASES)
        rows=filter(s->s.case_id==id && s.sinks>0,allrows)
        absent=filter(s->s.case_id==id && s.sinks==0,allrows)
        f=Figure(size=(1320,900));heading(f,"$(idx+2)   $(CASE_NAMES[idx]): activity and stability without biological cutoffs")
        for (i,extreme) in enumerate(("min","max")), (j,(prefix,name)) in enumerate((("E","E occupancy"),("I","I occupancy"),("alpha","max Re(λ), ms⁻¹")))
            field=Symbol(prefix*"_"*extreme)
            title=prefix=="alpha" ? (i==1 ? "Most negative spectral abscissa" : "Least negative spectral abscissa") : "$(i==1 ? "Minimum" : "Maximum") $name"
            ax=input_axis(f,(i,2j-1),title)
            limits=j==3 ? alpha_limits : (0.,.5)
            points!(ax,rows,:B_E,:B_I;color=Float64.(getproperty.(rows,field)),colorrange=limits,colormap=:viridis,markersize=4)
            points!(ax,absent,:B_E,:B_I;color=:black,marker=:xcross,markersize=5)
            Colorbar(f[i,2j];limits,colormap=:viridis,label=name)
        end
        foot(f,3,"Extrema are over all discovered attracting, nonsingular roots at each sampled input. Min and max may belong to different roots.\nE and I extrema are independent envelopes, not coordinates of one chosen state. No branch matching, interpolation, or biological names.\nBoth circuits use identical input limits and color ranges. Black × = no eligible sink found ($(length(absent)) contexts); white = unmeasured.\nSpectral abscissa is the largest eigenvalue real part for one root. More negative values indicate faster asymptotic linear decay only.")
        savefig(f,out,idx==1 ? "03_reference_raw_maps" : "04_weak_recurrence_raw_maps")
    end
end

function partial_maps(summaries,out)
    f=Figure(size=(1400,920));heading(f,"5   Incomplete classifications retain distinct mechanistic information")
    fields=(:middle,:descending,:complex)
    titles=("Attracting middle-E-arm roots","Attracting descending-I-response roots","Attracting roots with complex eigenvalues")
    allrows=filter(s->s.case_id in CASES && s.scope=="detail_geometry",summaries)
    cap=maximum(getproperty(s,k) for s in allrows for k in fields)
    for (i,id) in enumerate(CASES), (j,k) in enumerate(fields)
        rows=filter(s->s.case_id==id,allrows)
        ax=input_axis(f,(i,j),"$(CASE_NAMES[i])\n$(titles[j])")
        points!(ax,rows,:B_E,:B_I;color=getproperty.(rows,k),colorrange=(-.5,cap+.5),colormap=cgrad(:viridis,cap+1;categorical=true),markersize=4)
    end
    Colorbar(f[1:2,4];limits=(-.5,cap+.5),colormap=cgrad(:viridis,cap+1;categorical=true),ticks=0:cap,label="Number of matching attracting roots")
    foot(f,3,"These overlapping descriptors require no biological activity cutoff. A context may satisfy every column or none. White = unmeasured.\nMiddle E arm describes self-excitatory geometry; descending I response describes the failure side, not low I occupancy by itself.\nComplex eigenvalues describe local damped oscillatory modes. They do not establish a limit cycle or a finite perturbation response.\nDots preserve the measured input contexts. Zero means no matching eligible root found; it does not certify absence.")
    savefig(f,out,"05_partial_geometry_and_dynamics")
end

function local_geometry(allgroups,groups,out)
    f=Figure(size=(1320,960));heading(f,"6   Geometry and stability distinguish roots that activity labels combine")
    allalpha=vcat([[d.lambda_max_real for d in ds] for ds in allgroups]...)
    allbeta=vcat([[d.lambda_min_real for d in ds] for ds in allgroups]...)
    limits=(minimum(allbeta),maximum(allbeta),minimum(allalpha),maximum(allalpha))
    for (i,ds) in enumerate(groups)
        ax=Axis(f[1,i],title="$(COHORT_NAMES[i]): low-activity zoom",xlabel="E occupancy",ylabel="I occupancy")
        for (arm,color) in zip(ARMS,ARM_COLORS)
            selected=filter(d->d.E_arm==arm && max(d.E,d.I)<=.03,ds)
            for (branch,marker) in ((1,:circle),(2,:utriangle),(3,:rect),(0,:xcross))
                points!(ax,filter(d->d.I_branch==branch,selected),:E,:I;color=(color,.8),marker,markersize=5)
            end
        end
        lines!(ax,[0.,.01,.01],[.01,.01,0.];color=:black,linestyle=:dash,linewidth=1.5)
        xlims!(ax,-.0005,.0305);ylims!(ax,-.0005,.0305)
        ax=Axis(f[2,i],title="$(COHORT_NAMES[i]): full real-part spectrum",xlabel="min Re(λ) (ms⁻¹)",ylabel="max Re(λ) (ms⁻¹)")
        for (status,color,marker) in ((1,"#0072B2",:circle),(2,"#D55E00",:utriangle),(3,"#CC79A7",:rect),(0,"#000000",:xcross))
            points!(ax,filter(d->d.stability==status,allgroups[i]),:lambda_min_real,:lambda_max_real;color=(color,.4),marker,markersize=3)
        end
        hlines!(ax,[0.];color=:black,linestyle=:dash);vlines!(ax,[0.];color=:black,linestyle=:dash)
        xlims!(ax,limits[1]-.02,limits[2]+.02);ylims!(ax,limits[3]-.02,limits[4]+.02)
    end
    Legend(f[3,:],[MarkerElement(color=c,marker=:circle) for c in ARM_COLORS[1:4]],collect(ARM_NAMES[1:4]);orientation=:horizontal,framevisible=false)
    foot(f,4,"Top: all eligible roots with max(E,I) ≤ 0.03; dashed corner marks the reference rest proposal (≤ 0.01 for both populations).\nTop symbols: circle = ascending I response; triangle = descending. Colors are E arms. Low activity does not determine E arm.\nBottom: all roots in each cohort, including unstable roots. Blue circles = attracting; orange triangles = saddle; purple squares = repelling.\nReal parts are raw catalog values, with identical limits across cohorts. Complex pairs share real parts; the diagonal alone does not identify them.")
    savefig(f,out,"06_low_activity_and_spectra")
end

function sensitivity(groups,out)
    f=Figure(size=(1400,1000));heading(f,"7   Endpoint proposals and descriptive activity bands expose naming choices")
    policy_colors=("#0072B2","#777777","#D55E00")
    for (i,ds) in enumerate(groups)
        ax=Axis(f[1,i],title="$(COHORT_NAMES[i]): candidate counts",ylabel="Records (% of cohort)",xticks=(0:4,collect(SHORT)),xticklabelrotation=pi/10,xticklabelsize=12)
        for (k,p) in enumerate(M.POLICIES)
            values=[100count(d->M.candidate(d,p)==code,ds)/length(ds) for code in 0:4]
            barplot!(ax,collect(0:4).+.24*(k-2),values;width=.23,color=policy_colors[k],label=replace(p.id,'_'=>' '))
        end
        ylims!(ax,0,65)
        same=count(d->M.annotation(d).agreement,ds)
        Label(f[2,i],"Same candidate code in all 3 proposals: $same / $(length(ds)) ($(round(100same/length(ds);digits=1))%)";fontsize=15)
        ax=Axis(f[3,i],title="Reference activity bands: no biological name required",xlabel="I activity band",ylabel="E activity band",
            xticks=(1:3,collect(M.BANDS)),yticks=(1:3,collect(M.BANDS)))
        values=zeros(3,3)
        for d in ds
            code=M.activity_code(d,M.POLICIES[2]);values[mod(code-1,3)+1,div(code-1,3)+1]+=1
        end
        heatmap!(ax,.5:1:3.5,.5:1:3.5,log10.(values.+1);colorrange=(0,5),colormap=:viridis)
        for x in 1:3,y in 1:3
            text!(ax,x,y;text=string(Int(values[x,y])),align=(:center,:center),color=log10(values[x,y]+1)>2.5 ? :black : :white,fontsize=20)
        end
    end
    Legend(f[4,:],[PolyElement(color=c) for c in policy_colors],[replace(p.id,'_'=>' ') for p in M.POLICIES];orientation=:horizontal,framevisible=false)
    foot(f,5,"Trial boundaries (low E, low I; high E, retained I): narrow (0.005, 0.001; 0.475, 0.45), reference (0.01, 0.01; 0.45, 0.30), broad (0.02, 0.02; 0.40, 0.20).\nThe policies change several boundaries together; bar differences cannot isolate a single cutoff. Agreement is not confidence or probability.\nThe 3 × 3 activity partition is exhaustive: low ≤ low cutoff, high ≥ high cutoff, intermediate between. Integers are counts; fill = log10(count + 1), range 0–5.\nAttracting, nonsingular roots only. No default active category is imposed on the remainder; use its activity bands plus geometry and spectrum.")
    savefig(f,out,"07_policy_comparison_and_activity_bands")
end

function render(directory,packet,out)
    catalog=C.read_catalog(directory);C.IOChecks.verify_manifest(packet)
    meta=C.TOML.parsefile(joinpath(packet,"study.toml"))
    meta["completed"] && meta["schema_version"]==1 || error("incomplete or unknown study")
    meta["catalog_manifest_sha256"]==C.IOChecks.digest(joinpath(directory,"checksums.toml")) || error("catalog identity mismatch")
    meta["implementation_sha256"]==C.IOChecks.digest(joinpath(@__DIR__,"catalog_classifications.jl")) || error("classification source mismatch")
    annotations=C.read_gzip(joinpath(packet,"annotations.csv.gz"))
    lookup=Dict((r.context_id,r.root_id)=>r for r in annotations)
    length(lookup)==length(annotations)==meta["roots"] || error("duplicate or missing annotations")
    allgroups=[C.Descriptor[] for _ in COHORTS];seen=0
    C.foreach_descriptor(catalog) do d
        r=lookup[(d.context_id,d.root_id)]
        NamedTuple(r)==M.annotation(d) || error("annotation mismatch")
        seen+=1
        i=findfirst(==(C.context(catalog,d).scope),COHORTS)
        i===nothing || push!(allgroups[i],d)
    end
    seen==length(lookup) || error("annotation coverage mismatch")
    groups=[filter(M.eligible,ds) for ds in allgroups]
    summaries=C.read_gzip(joinpath(packet,"contexts.csv.gz"))
    ispath(out) && error("output exists");mkpath(out)
    with_theme(Theme(fontsize=16)) do
        portraits(groups,out);names_map(summaries,out);raw_maps(summaries,out)
        partial_maps(summaries,out);local_geometry(allgroups,groups,out);sensitivity(groups,out)
    end
    names=("01_activity_and_candidates","02_input_candidate_maps","03_reference_raw_maps","04_weak_recurrence_raw_maps",
        "05_partial_geometry_and_dynamics","06_low_activity_and_spectra","07_policy_comparison_and_activity_bands")
    captions=("Compare raw occupancy density with proposed endpoint names; gray retains descriptive information.",
        "Figure 2 style input maps: presence panels retain coexistence without combinatorial labels.",
        "Reference circuit: coordinate and spectral envelopes over all discovered sinks.",
        "Weak recurrent circuit: identical scales show the contrast without biological cutoffs.",
        "Three overlapping, cutoff-free descriptors: middle E arm, descending inhibitory response, complex spectrum.",
        "Low-activity geometry and full real-part spectra separate activity, mechanism, and stability.",
        "Three declared trial policies, plus an exhaustive 3 × 3 activity partition.")
    open(joinpath(out,"index.html"),"w") do io
        write(io,"<!doctype html><meta charset='utf-8'><title>Catalog classification proposals</title><style>body{font:18px sans-serif;max-width:1450px;margin:2em auto;padding:0 1em}img{width:100%}nav{line-height:1.8}p{max-width:1000px}section{margin:3em 0}</style><h1>Equilibrium catalog: candidate classifications</h1><p>Exploratory figures for author review. The reference proposal uses occupancy boundaries 0.01 (low E/I), 0.45 (high E), and 0.30 (retained I). No manuscript or production labels have changed. All data retain CompletenessNotCertified.</p><nav>")
        for (n,c) in zip(names,captions);write(io,"<a href='#$n'>$(first(n,2)) — $c</a><br>");end
        write(io,"</nav>")
        for (n,c) in zip(names,captions)
            write(io,"<section id='$n'><h2>$(replace(n,'_'=>' '))</h2><p>$c</p><a href='$n.pdf'>PDF</a> · <a href='$n.svg'>SVG</a> · <a href='$n.png'>PNG</a><img src='$n.png' alt='$c'></section>")
        end
    end
    C.write_toml(joinpath(out,"provenance.toml"),Dict("catalog_manifest_sha256"=>meta["catalog_manifest_sha256"],
        "classification_manifest_sha256"=>C.IOChecks.digest(joinpath(packet,"checksums.toml")),"renderer_sha256"=>C.IOChecks.digest(@__FILE__),
        "cohorts"=>collect(COHORTS),"input_cases"=>collect(CASES),"interpolation"=>"none","density_bins_per_axis"=>120))
    C.IOChecks.write_manifest(out)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==3 || error("usage: render_catalog_classifications.jl CATALOG CLASSIFICATIONS NEW_OUTPUT")
    RenderCatalogClassifications.render(abspath.(ARGS)...)
end
