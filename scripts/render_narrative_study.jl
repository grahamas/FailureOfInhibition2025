"""Render the narrative study's retained observations, without assigning biological validity."""
module NarrativeRenderer
using CairoMakie, FailureOfInhibition2025
import CSV, TOML, SHA
include("narrative_models.jl")
const E_COLOR="#177eab"
const I_COLOR="#d3782e"
const ROLES=Dict("rest"=>"#64748b","active"=>"#23866d","herald"=>"#9863c4","seizure"=>"#c44851")
hashfile(path)=bytes2hex(SHA.sha256(read(path)))

function verify(input)
    manifest=TOML.parsefile(joinpath(input,"checksums.toml"))["files"]
    for (relative,hash) in manifest
        isabspath(relative) || ".." in splitpath(relative) ? throw(ArgumentError("invalid artifact path")) : nothing
        hashfile(joinpath(input,relative))==hash || throw(ArgumentError("checksum mismatch: $relative"))
    end
    metadata=TOML.parsefile(joinpath(input,"metadata.toml"))
    for (relative,hash) in metadata["source_sha256"]
        startswith(relative,"src/") || relative=="scripts/narrative_models.jl" || continue
        hashfile(joinpath(@__DIR__,"..",relative))==hash || throw(ArgumentError("loaded model differs from archive"))
    end
    return manifest
end

function save_figure(fig,output,name)
    save(joinpath(output,name*".png"),fig;px_per_unit=1.5)
    save(joinpath(output,name*".pdf"),fig)
end

function draw_context!(ax,model,record,roles)
    axis=collect(range(0.,0.505;length=301))
    dE=zeros(length(axis),length(axis));dI=similar(dE)
    balance=zeros(2)
    for (j,i) in enumerate(axis),(k,e) in enumerate(axis)
        point_rhs!(balance,[e,i],model,0.)
        dE[k,j]=balance[1];dI[k,j]=balance[2]
    end
    contour!(ax,axis,axis,dE;levels=[0.],color=E_COLOR,linewidth=2)
    contour!(ax,axis,axis,dI;levels=[0.],color=I_COLOR,linewidth=2,linestyle=:dash)
    for (index,r) in enumerate(record["equilibria"])
        stable=r["stability"]["classification"]=="Attracting"
        role=something(findfirst(==(index),roles),"unassigned")
        scatter!(ax,[r["state"][1]],[r["state"][2]];marker=stable ? :circle : :xcross,
            color=get(ROLES,role,"#495566"),markersize=stable ? 15 : 10)
        role=="unassigned" || text!(ax,r["state"][1],r["state"][2];text=role,
            offset=role=="seizure" ? (-65,12) : role=="herald" ? (-65,12) : (7,10),fontsize=13)
    end
    xlims!(ax,-0.015,0.535);ylims!(ax,-0.015,0.535)
end

function draw_trace!(ax,path;offset=0.,label="")
    data=CSV.File(path)
    lines!(ax,data.time.+offset,data.E;color=E_COLOR,label=isempty(label) ? "E" : label*" E")
    lines!(ax,data.time.+offset,data.I;color=I_COLOR,label=isempty(label) ? "I" : label*" I")
end

function confirmed_context_path(input,selected)
    grids=TOML.parsefile(joinpath(input,"config.toml"))["search"]["confirmation_grids"]
    grid=get(selected,"final_grid",last(grids))
    grid isa Integer && !(grid isa Bool) && grid>=2 && grid in grids ||
        throw(ArgumentError("selected confirmation grid does not match archived configuration"))
    joinpath(input,selected["confirmation_directory"],"grid$grid")
end

function render(input_dir,output_dir)
    input,output=abspath(input_dir),abspath(output_dir)
    verify(input)
    isdir(output) && !isempty(readdir(output)) && throw(ArgumentError("output must be new"))
    mkpath(output)
    selected=TOML.parsefile(joinpath(input,"selected.toml"));p=selected["parameters"]
    pair=NarrativeModels.models(p)
    confirmed=confirmed_context_path(input,selected)
    roots=TOML.parsefile(joinpath(confirmed,"context.toml"))
    roles=TOML.parsefile(joinpath(confirmed,"roles.toml"))
    control=TOML.parsefile(joinpath(input,"matched_control","context.toml"))
    fig=Figure(size=(1300,760),fontsize=17)
    Label(fig[1,1:3],"Coexistence at the same background input",fontsize=26)
    ax=Axis(fig[2,1],title="Inhibitory recruitment",xlabel="Input to I",ylabel="Response")
    u=range(0.,14.;length=600)
    lines!(ax,u,response.(Ref(pair.control.inhibitory.response),u);label="Monotone",color="#64748b")
    lines!(ax,u,response.(Ref(pair.failure_of_inhibition.inhibitory.response),u);label="Fire then fail",color=I_COLOR)
    axislegend(ax;position=:rt)
    ax=Axis(fig[2,2],title="Matched monotone model",xlabel="E",ylabel="I")
    draw_context!(ax,pair.control,control,Dict())
    ax=Axis(fig[2,3],title="Failure of inhibition",xlabel="E",ylabel="I")
    draw_context!(ax,pair.failure_of_inhibition,roots,roles)
    Label(fig[3,1:3],"B_E = $(p["B_E"]) · tau_I/tau_E = $(p["tau_ratio"]) · labels denote provisional coordinate roles",fontsize=15)
    Label(fig[4,1:3],"The ordering result concerns two equilibria under common parameters and input. It does not exclude all seizure-like dynamics.",fontsize=14)
    save_figure(fig,output,"01_mechanism")

    fig=Figure(size=(1300,780),fontsize=17)
    Label(fig[1,1:2],"Rest–active switching and access to high-activity states",fontsize=25)
    for (j,key) in enumerate(("rest_to_active","active_to_rest"))
        ax=Axis(fig[2,j],title=replace(key,"_"=>" "),xlabel="Time (ms)",ylabel="Activity")
        path=joinpath(confirmed,"cycle1_"*key*"_pulse.csv")
        pulse=CSV.File(path);draw_trace!(ax,path)
        draw_trace!(ax,joinpath(confirmed,"cycle1_"*key*".csv");offset=last(pulse.time))
        vlines!(ax,[last(pulse.time)];color="#64748b",linestyle=:dash)
        xlims!(ax,0,max(100.,last(pulse.time)+150));ylims!(ax,-0.01,0.52)
    end
    for (j,source) in enumerate(("herald","seizure"))
        ax=Axis(fig[3,j],title="Induction of $source candidate",xlabel="Time (ms)",ylabel="Activity")
        dir=joinpath(input,"induced_recovery",source)
        if isfile(joinpath(dir,"induction_pulse.csv"))
            pulse=CSV.File(joinpath(dir,"induction_pulse.csv"))
            draw_trace!(ax,joinpath(dir,"induction_pulse.csv"))
            draw_trace!(ax,joinpath(dir,"induction.csv");offset=last(pulse.time))
            vlines!(ax,[last(pulse.time)];color="#64748b",linestyle=:dash)
            xlims!(ax,0,max(100.,last(pulse.time)+150))
        else
            text!(ax,0.1,0.2;text="No confirmed induction witness")
        end
        ylims!(ax,-0.01,0.52)
    end
    Label(fig[4,1:2],"Blue: E · orange: I · dashed line: stimulus ends. Plots show early dynamics; classification uses the retained full follow-up.",fontsize=14)
    save_figure(fig,output,"02_transitions")

    fig=Figure(size=(1400,850),fontsize=16,figure_padding=30)
    Label(fig[1,1:2],"Herald and seizure: two distinct reduction experiments",fontsize=24,tellwidth=false)
    data=collect(CSV.File(joinpath(input,"recovery","recovery.csv")))
    for (j,protocol) in enumerate(("permanent_withdrawal","direct_E_displacement"))
        for (i,source) in enumerate(("herald","seizure"))
            ax=Axis(fig[i+1,j],title="$source\n"*(j==1 ? "Sustained input withdrawal" : "Instantaneous E displacement"),
                xlabel=j==1 ? "Input removed, ΔB_E" : "Activity removed, ΔE",ylabel="Final E",yticks=[0.,0.2,0.4])
            rows=sort(filter(r->r.protocol==protocol && r.source==source,data);by=r->r.reduction)
            for destination in unique(r.destination for r in rows)
                group=filter(r->r.destination==destination,rows)
                scatter!(ax,[r.reduction for r in group],[r.final_E for r in group];
                    color=get(ROLES,destination,"#495566"),markersize=8,label=destination)
            end
            ylims!(ax,-0.01,0.52);axislegend(ax;position=:lb,labelsize=12)
        end
    end
    Label(fig[4,1:2],"Finite-follow-up destinations. Withdrawal changes the system; displacement leaves tonic input fixed. Herald ↔ seizure is not recovery.",fontsize=14,tellwidth=false)
    save_figure(fig,output,"03_reduction")

    fig=Figure(size=(1300,800),fontsize=17)
    Label(fig[1,1:2],"Intervention effects on ordinary switching and seizure persistence",fontsize=24)
    rows=collect(CSV.File(joinpath(input,"interventions","summary.csv")))
    for (index,axis) in enumerate(("theta_off","e_to_i","e_to_e","i_to_e"))
        ax=Axis(fig[2+(index-1)÷2,1+(index-1)%2],title=axis,xlabel="Parameter value",ylabel="Observed property",yticks=([0,1,2],["No","Yes"," "]))
        group=sort(filter(r->r.axis==axis,rows);by=r->r.value)
        x=[r.value for r in group]
        scatterlines!(ax,x,[Float64(r.switching_preserved) for r in group];label="Rest–active switching",color=ROLES["active"])
        scatterlines!(ax,x,[Float64(r.seizure_after_parameter_change=="seizure")+0.08 for r in group];label="Seizure persists",color=ROLES["seizure"],linestyle=:dash)
        ylims!(ax,-0.15,1.55);axislegend(ax;position=:rt,labelsize=12)
    end
    Label(fig[4,1:2],"A seizure-to-herald transition is not recovery. Switching is remeasured; shifted thresholds and coordinates remain in the tables.",fontsize=14)
    save_figure(fig,output,"04_interventions")
    fig=Figure(size=(1500,700),fontsize=16)
    Label(fig[1,1:3],"Local input map and numerical equilibrium branches",fontsize=25,tellwidth=false)
    maprows=collect(CSV.File(joinpath(input,"map","map.csv")))
    ax=Axis(fig[2,1],title="Fixed switching protocols",xlabel="Recurrent excitation",ylabel="Tonic E input")
    for retained in (false,true)
        group=filter(r->(r.rest_to_active=="active" && r.active_to_rest=="rest")==retained,maprows)
        scatter!(ax,[r.e_to_e for r in group],[r.B_E for r in group];
            color=retained ? ROLES["active"] : "#b5bcc5",markersize=14,label=retained ? "Both transitions retained" : "Not both retained")
    end
    axislegend(ax;position=:rt,labelsize=12)
    for coordinate in 1:2
        ax=Axis(fig[2,coordinate+1],title="Continued branches · local view",xlabel="Tonic E input",ylabel="Equilibrium "*(coordinate==1 ? "E" : "I"))
        for role in sort(collect(keys(roles)))
            result=TOML.parsefile(joinpath(input,"map","continuation_"*role*".toml"))
            for direction in ("negative","positive")
                points=result[direction]["points"]
                isempty(points) && continue
                lines!(ax,[x["parameter"] for x in points],[x["state"][coordinate] for x in points];color=ROLES[role],label=direction=="positive" ? role : nothing)
            end
        end
        xlims!(ax,0.,0.6)
        axislegend(ax;position=:rc,labelsize=12)
    end
    Label(fig[3,1:3],"Colors identify starting branches, including unstable portions. Continued segments are bounded evidence; no fold location is certified.",fontsize=14,tellwidth=false)
    save_figure(fig,output,"05_local_map")
    # Standalone report, with relative images for easy offline viewing.
    open(joinpath(output,"report.html"),"w") do io
        write(io,"<!doctype html><meta charset='utf-8'><title>Narrative study</title><style>body{max-width:1200px;margin:40px auto;font:18px system-ui;color:#243348}img{width:100%}p{line-height:1.5}</style><h1>Activity, persistence, and selective control</h1><p>Exploratory point-model evidence. Coordinate roles are provisional; finite trajectories do not establish biological identity or global attractor completeness.</p>")
        for name in ("01_mechanism","02_transitions","03_reduction","04_interventions","05_local_map")
            write(io,"<img src='$name.png' alt='$(replace(name,"_"=>" "))'>")
        end
    end
    cp(joinpath(input,"source"),joinpath(output,"source"))
    for file in ("render_narrative_study.jl","narrative_models.jl")
        cp(joinpath(@__DIR__,file),joinpath(output,"source","scripts",file);force=true)
    end
    for file in ("Project.toml","Manifest.toml")
        mkpath(joinpath(output,"source","plotting"))
        cp(joinpath(@__DIR__,"..","plotting",file),joinpath(output,"source","plotting",file);force=true)
    end
    metadata=Dict("input"=>input,"input_checksums_sha256"=>hashfile(joinpath(input,"checksums.toml")),
        "renderer_sha256"=>hashfile(@__FILE__),
        "replay"=>"julia --project=source/plotting source/scripts/render_narrative_study.jl $input replay")
    open(joinpath(output,"metadata.toml"),"w") do io; TOML.print(io,metadata;sorted=true);end
    hashes=Dict(relpath(joinpath(root,file),output)=>hashfile(joinpath(root,file))
        for (root,_,files) in walkdir(output) for file in files)
    open(joinpath(output,"checksums.toml"),"w") do io;TOML.print(io,Dict("files"=>hashes);sorted=true);end
    return output
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("usage: render_narrative_study.jl INPUT OUTPUT")
    NarrativeRenderer.render(ARGS...)
end
