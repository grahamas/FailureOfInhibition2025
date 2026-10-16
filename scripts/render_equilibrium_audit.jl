"""Diagnostic figures for author decisions about unassigned equilibria."""
module EquilibriumAuditRenderer
using CairoMakie
import CSV,TOML
include("figure_packet_io.jl")
const COLORS=Dict("lower/ascending"=>"#0072B2","lower/descending"=>"#56B4E9","middle/ascending"=>"#009E73",
    "middle/descending"=>"#999999","upper/ascending"=>"#CC79A7","upper/descending"=>"#D55E00",
    "no_turning_points/ascending"=>"#7A68A6","no_turning_points/descending"=>"#A891CA")
rows(p,n)=collect(CSV.File(joinpath(p,n);stringtype=String))
gzrows(p,n)=open(`gzip -dc $(joinpath(p,n))`) do io
    collect(CSV.File(io;stringtype=String))
end
color(x)=get(COLORS,x,"#777777")
function title!(f,text)
    Label(f[0,:],text;fontsize=23,font=:bold,halign=:left,tellwidth=false)
end
note!(p,text)=Label(p,text;fontsize=14,halign=:left,justification=:left,tellwidth=false)
function savefig(f,out,name)
    save(joinpath(out,name*".png"),f;px_per_unit=2)
    save(joinpath(out,name*".pdf"),f;pt_per_unit=.5)
    save(joinpath(out,name*".svg"),f;pt_per_unit=.5)
end

function overview(p,out)
    f=Figure(size=(1150,830));title!(f,"Unassigned is a naming outcome: several geometries share the label")
    summary=rows(p,"summary.csv");s=filter(r->r.scope=="screen" && r.sinks>0,summary)
    sort!(s;by=r->-r.sinks)
    labels=[replace(split(r.key,"::")[1],"missing_higher_I_sink|no_unique_seizure_for_herald"=>"No higher-I comparison sink",
        "no_unique_seizure_for_herald"=>"Herald requires a named seizure","multiple_rest_candidates"=>"Multiple rest candidates",
        "multiple_seizure_candidates"=>"Multiple seizure candidates","multiple_active_candidates"=>"Multiple active candidates",
        "no_E_turning_points"=>"No E turning points")*"\n"*
        replace(split(r.key,"::")[2],"_"=>" ") for r in s]
    ax=Axis(f[1,1:2],title="A   Causes among attracting unassigned roots in the 232-case screen",
        xlabel="Root records in sampled contexts (not prevalence)",yticks=(1:length(s),labels),yreversed=true,yticklabelsize=12)
    barplot!(ax,1:length(s),[r.sinks for r in s];direction=:x,color=[color(split(r.key,"::")[2]) for r in s])
    for (k,r) in enumerate(s)
        text!(ax,r.sinks+170,k;text=string(r.sinks),align=(:left,:center),fontsize=12)
    end
    xlims!(ax,0,maximum(r.sinks for r in s)*1.1)
    for (j,id) in enumerate(("selective_withdrawal","input_dependent"))
        r=gzrows(p,"map_"*id*".csv.gz")
        a=Axis(f[2,j],title=j==1 ? "B   Reference: unassigned attracting roots" : "C   Weak circuit: unassigned attracting roots",
            xlabel="Tonic E input, B_E",ylabel="Tonic I input, B_I")
        for d in sort(unique(x.descriptor for x in r if x.role=="unassigned"))
            group=filter(x->x.role=="unassigned" && x.descriptor==d,r)
            scatter!(a,getproperty.(group,:B_E),getproperty.(group,:B_I);color=color(d),markersize=3,label=replace(d,"/"=>" / "))
        end
        xlims!(a,0,16);ylims!(a,0,16)
        axislegend(a;position=:rt,labelsize=11,framevisible=true)
    end
    note!(f[3,1:2],"Colors describe the E-nullcline arm and the side of the inhibitory response peak.\nDots are sampled roots; several roots can share an input. Blank areas are not an absence claim. No biological names are reassigned.")
    savefig(f,out,"01_inventory")
end

function portraits(p,out)
    f=Figure(size=(1200,850));title!(f,"What the unnamed states look like")
    seeds=rows(p,"seeds.csv");roots=rows(p,"seed_roots.csv")
    for (index,k) in enumerate((1,2,6,8,9,11))
        seed=only(filter(r->r.seed==k,seeds));row=cld(index,3);column=mod1(index,3)
        names=Dict(1=>"Cycle plus upper descending sink",2=>"Recruitment: upper ascending sink",6=>"Two lower-arm sinks",
            8=>"Multiple middle-arm candidates",9=>"No E turning points",11=>"Multiple upper descending candidates")
        context="seed $k; B=($(round(seed.B_E;digits=3)), $(round(seed.B_I;digits=3)))"*(k==8 ? "; zoom" : "")
        ax=Axis(f[row,column],title="$(Char('A'+index-1))   $(names[k])\n"*context,titlesize=14,
            xlabel="E occupancy",ylabel="I occupancy",aspect=DataAspect())
        nulls=rows(p,"seed_$(k)_nullclines.csv")
        for (pop,c,style) in (("E","#0072B2",:solid),("I","#D55E00",:dash))
            r=filter(x->x.population==pop,nulls);lines!(ax,getproperty.(r,:E),getproperty.(r,:I);color=c,linestyle=style,linewidth=2)
        end
        for r in filter(r->r.seed==k,roots)
            scatter!(ax,[r.E],[r.I];marker=r.stability=="Attracting" ? :circle : :xcross,color=r.role=="unassigned" ? :black : :gray,
                markersize=10)
        end
        scatter!(ax,[seed.E],[seed.I];color=:transparent,strokecolor="#CC79A7",strokewidth=2,markersize=18)
        if k==1
            orbit=rows(p,"cycle.csv");lines!(ax,getproperty.(orbit,:E),getproperty.(orbit,:I);color="#B88700",linewidth=2)
        end
        xlims!(ax,k==8 ? (-.002,.09) : (-.02,.53));ylims!(ax,k==8 ? (-.002,.065) : (-.02,.53))
    end
    note!(f[3,1:3],"Blue solid / orange dashed: E / I nullclines. Circles: attracting; crosses: other stability. Black: unassigned; gray: named.\nMagenta outline: selected root. Gold: validated periodic orbit. Close roots can overlap at this scale; exact coordinates are in seed_roots.csv.")
    savefig(f,out,"02_geometry")
end

function branches(p,out)
    f=Figure(size=(1150,820));title!(f,"A continuous equilibrium branch can gain or lose a contextual name")
    seeds=rows(p,"seeds.csv")
    choices=((1,"B_I"),(2,"e_to_i"),(9,"B_E"),(11,"B_E"))
    labels=Dict("B_E"=>"Tonic E input, B_E","B_I"=>"Tonic I input, B_I","e_to_i"=>"Recruitment, E→I")
    for (j,(seed,axis)) in enumerate(choices)
        ax=Axis(f[cld(j,2),mod1(j,2)],title="$(Char('A'+j-1))   Seed $seed: "*only(filter(r->r.seed==seed,seeds)).id,
            titlesize=15,xlabel=labels[axis],ylabel="E / I occupancy")
        r=filter(r->r.variant=="default",rows(p,"seed_$(seed)_$(axis)_points.csv"))
        for dir in (-1,1)
            group=filter(r->r.direction==dir,r)
            for (a,b) in zip(group,group[2:end]), (field,c) in ((:E,"#0072B2"),(:I,"#D55E00"))
                lines!(ax,[a.parameter,b.parameter],[getproperty(a,field),getproperty(b,field)];color=c,
                    linestyle=a.stability=="Attracting" && b.stability=="Attracting" ? :solid : :dash,linewidth=2)
            end
        end
        connections=filter(r->r.validated,rows(p,"seed_$(seed)_$(axis)_connections.csv"))
        for role in sort(unique(r.role for r in connections))
            group=filter(r->r.role==role,connections)
            scatter!(ax,getproperty.(group,:parameter),getproperty.(group,:E);marker=role=="unassigned" ? :circle : :rect,
                color=role=="unassigned" ? :white : :black,strokecolor=:black,strokewidth=1,markersize=8)
        end
        ax.title[]=ax.title[]*"\nLocal labels: "*join(sort(unique(r.role for r in connections)),", ")
    end
    note!(f[3,1:2],"Blue: E; orange: I. Solid: attracting segments; dashed: other stability. Open circles: locally unassigned; filled squares: named.\nMarkers denote independently searched contexts. Endpoints are computational or domain stops; branch continuity is not a control result.")
    savefig(f,out,"03_connections")
end

function interventions(p,out)
    f=Figure(size=(1150,780));title!(f,"Intervention destinations: coordinates and label provenance")
    data=rows(p,"intervention_labels.csv")
    for (row,id) in enumerate(("selective_withdrawal","threshold_to_active")), (col,axis) in enumerate(("theta_off","e_to_i"))
        r=sort(filter(x->x.id==id && x.axis==axis,data);by=x->x.value)
        ax=Axis(f[row,col],title=(row==1 ? "Reference" : "Contrast")*": "*(axis=="theta_off" ? "failure threshold" : "recruitment"),
            xlabel=axis=="theta_off" ? "Failure threshold, θoff" : "Recruitment, E→I",ylabel="Destination E / I occupancy")
        scatterlines!(ax,getproperty.(r,:value),getproperty.(r,:E);color="#0072B2",markersize=6,label="E")
        scatterlines!(ax,getproperty.(r,:value),getproperty.(r,:I);color="#D55E00",linestyle=:dash,markersize=6,label="I")
        inherited=filter(x->x.origin=="reference_coordinate_match",r)
        isempty(inherited) || scatter!(ax,getproperty.(inherited,:value),getproperty.(inherited,:I);
            color=:transparent,strokecolor=:black,strokewidth=1.8,markersize=13,label="Name inherited from original context")
        axislegend(ax;position=:lb,labelsize=12,framevisible=true);ylims!(ax,-.03,.53)
    end
    note!(f[3,1:2],"Each setting starts from its circuit's original seizure coordinates. Lines connect sampled outcomes; they are not equilibrium continuation.\nOutlined points have a reported name supplied by conservative coordinate matching, despite having no local role. Original labels are retained.")
    savefig(f,out,"04_intervention_identity")
end

function timescales(p,out)
    f=Figure(size=(1100,620));title!(f,"Changing another equilibrium's stability can remove the seizure label")
    data=filter(r->r.id=="input_dependent",rows(p,"timescale_labels.csv"))
    a=Axis(f[1,1],title="A   Weak circuit at fixed input B=(2,1)",xlabel="Time-scale ratio, τI / τE",
        ylabel="Largest real eigenvalue (ms⁻¹)")
    b=Axis(f[1,2],title="B   The upper equilibrium's coordinates stay fixed",xlabel="Time-scale ratio, τI / τE",ylabel="E / I occupancy")
    for (root,c,label) in ((1,"#009E73","Middle equilibrium"),(3,"#D55E00","Upper equilibrium"))
        r=sort(filter(r->r.root==root,data);by=r->r.ratio)
        lines!(a,getproperty.(r,:ratio),getproperty.(r,:alpha);color=c,linewidth=2,label)
    end
    hlines!(a,[0.];color=:gray,linestyle=:dash);axislegend(a;position=:rb,labelsize=13)
    r=sort(filter(r->r.root==3,data);by=r->r.ratio)
    lines!(b,getproperty.(r,:ratio),getproperty.(r,:E);color="#0072B2",linewidth=2,label="E")
    lines!(b,getproperty.(r,:ratio),getproperty.(r,:I);color="#D55E00",linestyle=:dash,linewidth=2,label="I")
    for role in ("seizure","unassigned")
        group=filter(r->r.local_role==role,r)
        scatter!(b,getproperty.(group,:ratio),getproperty.(group,:E);color=role=="seizure" ? :black : :white,
            strokecolor=:black,strokewidth=1,markersize=6,label="Local label: "*role)
    end
    axislegend(b;position=:rc,labelsize=12)
    note!(f[2,1:2],"The role rule requires a higher-I attracting equilibrium. When the middle equilibrium loses stability, that comparison disappears.\nThe upper equilibrium stays attracting and keeps exactly the same coordinates throughout this recorded scan.\nThis is a contextual naming change; the scan does not establish the birth of the periodic orbit.")
    savefig(f,out,"05_contextual_naming")
end

function render(packet,output)
    FigurePacketIO.verify_manifest(packet);ispath(output) && error("render output exists");mkpath(output)
    with_theme(Theme(fontsize=17,Axis=(titlesize=17,titlealign=:left,xlabelsize=16,ylabelsize=16,xticklabelsize=13,yticklabelsize=13))) do
        overview(packet,output);portraits(packet,output);branches(packet,output);interventions(packet,output);timescales(packet,output)
    end
    open(joinpath(output,"index.html"),"w") do io
        write(io,"<!doctype html><meta charset='utf-8'><title>Unassigned equilibrium audit</title><style>body{max-width:1150px;margin:2em auto;font-family:system-ui}img{width:100%}</style><h1>Unassigned equilibrium audit</h1><p>Diagnostic evidence for author review. Existing biological roles remain unchanged.</p>")
        for name in ("01_inventory","02_geometry","03_connections","04_intervention_identity","05_contextual_naming")
            write(io,"<section><h2>",replace(name,'_'=>' '),"</h2><a href='",name,".pdf'>PDF</a> <a href='",name,".svg'>SVG</a><img src='",name,".png'></section>")
        end
    end
    open(joinpath(output,"provenance.toml"),"w") do io
        TOML.print(io,Dict("renderer_sha256"=>FigurePacketIO.digest(@__FILE__),"packet_manifest_sha256"=>FigurePacketIO.digest(joinpath(packet,"checksums.toml")),
            "julia_version"=>string(VERSION),"cairomakie_version"=>string(pkgversion(CairoMakie)));sorted=true)
    end
    FigurePacketIO.write_manifest(output)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("usage: render_equilibrium_audit.jl PACKET NEW_OUTPUT")
    EquilibriumAuditRenderer.render(ARGS...)
end
