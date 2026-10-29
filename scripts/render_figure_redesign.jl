"""Five scientific figure drafts rendered only from a checked portable packet."""
module FigureRedesignRenderer
using CairoMakie
import CSV,TOML,SHA
include("figure_packet_io.jl")
include("nullcline_overlays.jl")
const E_COLOR="#0072B2"
const I_COLOR="#D55E00"
const ROLES=("rest","active","herald","seizure","oscillatory","unassigned","unresolved")
const COLORS=Dict("rest"=>"#6B7280","active"=>"#009E73","herald"=>"#8B5FB0",
    "seizure"=>"#C83D5B","oscillatory"=>"#B88700","unassigned"=>"#333333","unresolved"=>"#B5B5B5")
const MARKERS=Dict("rest"=>:circle,"active"=>:rect,"herald"=>:diamond,
    "seizure"=>:utriangle,"oscillatory"=>:star5,"unassigned"=>:cross,"unresolved"=>:xcross)
rows(packet,relative)=collect(CSV.File(joinpath(packet,relative);stringtype=String))
const PERIODIC_DIR="timescales/input_dependent_2.0_1.0"
function decreasing_endpoint(cycles,stops)
    branch=filter(r->r.direction==-1,cycles)
    isempty(branch) && return nothing
    stop=only(filter(r->r.direction==-1,stops))
    point=branch[argmax(getproperty.(branch,:index))]
    (;point,termination=stop.termination)
end
termination_label(status)=replace(status,'_'=>' ')
function continuation_caption(packet)
    cyclefile=joinpath(packet,PERIODIC_DIR,"cycles.csv")
    isfile(cyclefile) || return "No validated periodic continuation was recorded in this run."
    endpoint=decreasing_endpoint(rows(packet,joinpath(PERIODIC_DIR,"cycles.csv")),
        rows(packet,joinpath(PERIODIC_DIR,"stops.csv")))
    endpoint===nothing && return "No accepted points were recorded on the initially decreasing periodic continuation."
    "Initially decreasing periodic continuation termination: $(termination_label(endpoint.termination)). Its birth is not identified."
end
color(role)=get(COLORS,role,"#333333")
marker(role)=get(MARKERS,role,:cross)
function savefig(fig,out,name)
    for ext in ("pdf","svg","png")
        kwargs=ext=="png" ? (;px_per_unit=2) : (;pt_per_unit=.5)
        save(joinpath(out,name*"."*ext),fig;kwargs...)
    end
end
function role_legend(position;roles=collect(ROLES))
    Legend(position,[MarkerElement(color=color(r),marker=marker(r),markersize=12) for r in roles],
        [r=="unassigned" ? "unassigned equilibrium" : r for r in roles];
        orientation=:horizontal,framevisible=false,nbanks=1,labelsize=15)
end
function heading(fig,title)
    Label(fig[0,:],title;fontsize=24,font=:bold,halign=:left,tellwidth=false)
end
function note(position,text)
    Label(position,text;fontsize=15,halign=:left,tellwidth=false,justification=:left)
end

function mechanism_overlays(packet,condition,roots;show_basins=true,show_separatrices=true)
    folder=joinpath(packet,"mechanism","overlays");path=joinpath(folder,"metadata.toml")
    isfile(path) || return (;basins=nothing,separatrices=nothing)
    metadata=TOML.parsefile(path);entry=metadata["conditions"][condition]
    basins=show_basins && haskey(entry,"basin_options") ?
        NullclineOverlays.read_basins(folder,condition,roots,entry) : nothing
    separatrices=show_separatrices && haskey(entry,"separatrix_options") ?
        NullclineOverlays.read_separatrices(folder,condition,roots,entry) : nothing
    (;basins,separatrices)
end
function pale_role_color(role)
    c=to_color(color(role));weight=.16
    RGBAf(1-weight+weight*c.r,1-weight+weight*c.g,1-weight+weight*c.b,1)
end
function draw_mechanism_overlays!(ax,overlays,roots)
    if overlays.basins!==nothing
        grid=overlays.basins
        palette=vcat([to_color("#F4F4F4")],[pale_role_color(r.role) for r in roots])
        heatmap!(ax,grid.E,grid.I,grid.labels;colormap=cgrad(palette;categorical=true),
            colorrange=(-.5,length(roots)+.5),interpolate=false,rasterize=true)
    end
    if overlays.separatrices!==nothing
        for branch in overlays.separatrices
            branch.converged || continue
            lines!(ax,branch.E,branch.I;color=("#374151",.8),linestyle=:dash,linewidth=1.4)
        end
    end
end

function mechanism(packet,out;show_basins=true,show_separatrices=true)
    fig=Figure(size=(1120,590));heading(fig,"1   Failure of inhibition changes the available equilibrium ordering")
    response=rows(packet,"mechanism/responses.csv");nulls=rows(packet,"mechanism/nullclines.csv")
    roots=rows(packet,"mechanism/roots.csv")
    a=Axis(fig[1,1],title="A   Population response",xlabel="Effective input to I",ylabel="Response")
    for condition in ("monotone","failure")
        r=filter(x->x.condition==condition,response)
        lines!(a,getproperty.(r,:input),getproperty.(r,:response);
            color=condition=="monotone" ? "#64748B" : I_COLOR,
            linestyle=condition=="monotone" ? :dash : :solid,linewidth=3,
            label=condition=="monotone" ? "Monotone" : "Fire then fail")
    end
    vlines!(a,[4.,8.];color=(:gray,.45),linestyle=:dot)
    text!(a,[Point2f(4.,.08),Point2f(8.,.08)];text=["onset","failure"],rotation=pi/2,fontsize=13)
    axislegend(a;position=:rt,framevisible=false,labelsize=14)
    for (column,condition,title) in ((2,"monotone","B   Matched monotone circuit"),(3,"failure","C   Failure circuit"))
        ax=Axis(fig[1,column],title=title,xlabel="E occupancy",ylabel="I occupancy",aspect=DataAspect())
        group=filter(x->x.condition==condition,roots)
        overlays=mechanism_overlays(packet,condition,group;show_basins,show_separatrices)
        draw_mechanism_overlays!(ax,overlays,group)
        for (population,c,style) in (("E",E_COLOR,:solid),("I",I_COLOR,:dash))
            r=filter(x->x.condition==condition && x.population==population,nulls)
            lines!(ax,getproperty.(r,:E),getproperty.(r,:I);color=c,linestyle=style,linewidth=2)
        end
        for r in filter(x->x.condition==condition,roots)
            scatter!(ax,[r.E],[r.I];color=r.stability=="Attracting" ? color(r.role) : "#96A0AE",
                marker=r.stability=="Attracting" ? marker(r.role) : :xcross,markersize=12)
            if condition=="failure" && r.role!="unassigned"
                positions=Dict("rest"=>(.035,.04),"active"=>(.27,.35),"herald"=>(.37,.46),"seizure"=>(.36,.035))
                text!(ax,positions[r.role]...;text=r.role,color=color(r.role),fontsize=14)
            end
        end
        xlims!(ax,0.,.5);ylims!(ax,0.,.5)
    end
    # Full physical state domain is context, rather than another main panel.
    inset=Axis(fig[1,1],width=75,height=75,halign=.04,valign=.94,backgroundcolor=:white,
        xticks=[0.,1.],yticks=[0.,1.],xticklabelsize=10,yticklabelsize=10,title="state domain",titlesize=11,
        xlabel="E",ylabel="I",xlabelsize=10,ylabelsize=10,
        tellwidth=false,tellheight=false)
    poly!(inset,Rect2f(0,0,.5,.5);color=(E_COLOR,.12),strokecolor=E_COLOR,strokewidth=1)
    xlims!(inset,0,1);ylims!(inset,0,1);hidespines!(inset,:t,:r)
    note(fig[2,1:3],"Same couplings, tonic input, and time scales. Blue/orange: E/I nullclines; crosses: other discovered roots.\nPale colour (when present): compatible attracting root; neutral: unresolved. Dashed grey (when shown): converged stable manifold.")
    note(fig[3,1:3],"Ordering result: with nondecreasing inhibition, E₂ ≥ E₁ and I₂ < I₁ cannot hold for two equilibria\nof the same fixed-input system. The failure response permits this ordering; it does not guarantee it.")
    rowsize!(fig.layout,1,Relative(.75));savefig(fig,out,"01_mechanism")
end

function repertoire_label(signature)
    short=Dict("rest"=>"R","active"=>"A","herald"=>"H","seizure"=>"S","unassigned"=>"U")
    parts=split(signature,'|');labels=String[]
    for part in parts
        startswith(part,"Attracting:") || continue
        push!(labels,get(short,last(split(part,':')),"U"))
    end
    isempty(labels) ? "no sink found" : join(sort(labels),"+")
end

function repertoire(packet,out)
    fig=Figure(size=(1120,880));heading(fig,"2   Inputs organize the repertoire; time scales change its stability")
    ids=("selective_withdrawal","input_dependent")
    data=[rows(packet,"cases/$id/geometry/inputs.csv") for id in ids]
    names=sort(unique(repertoire_label(r.signature) for group in data for r in group))
    palette=cgrad(:tab20,length(names);categorical=true)
    colors=Dict(n=>palette[k] for (k,n) in enumerate(names))
    for (column,id) in enumerate(ids)
        ax=Axis(fig[1,column],title=column==1 ? "A   Reference circuit" : "B   Weak recurrent excitation",
            xlabel="Tonic E input, B_E",ylabel="Tonic I input, B_I")
        lookup=Dict((r.B_E,r.B_I)=>r for r in data[column])
        rectangles=Rect2f[];cell_colors=RGBAf[]
        label_cells=Dict{String,Tuple{Float64,Float64,Float64}}()
        for cell in rows(packet,"cases/$id/geometry/cells.csv")
            center=((cell.x0+cell.x1)/2,(cell.y0+cell.y1)/2)
            observed=get(lookup,center,nothing)
            resolved=cell.status=="sampled_homogeneous" && observed!==nothing
            c=resolved ? colors[repertoire_label(observed.signature)] : "#E5E7EB"
            push!(rectangles,Rect2f(cell.x0,cell.y0,cell.x1-cell.x0,cell.y1-cell.y0))
            push!(cell_colors,to_color(c))
            if resolved && cell.x1-cell.x0>=.7 && cell.y1-cell.y0>=.5
                name=repertoire_label(observed.signature)
                area=(cell.x1-cell.x0)*(cell.y1-cell.y0)
                area>first(get(label_cells,name,(0.,0.,0.))) && (label_cells[name]=(area,center...))
            end
        end
        poly!(ax,rectangles;color=cell_colors,strokewidth=0)
        # All discovered samples remain visible, including probes inside unresolved cells.
        for n in names
            selected=filter(r->repertoire_label(r.signature)==n,data[column])
            isempty(selected) && continue
            scatter!(ax,getproperty.(selected,:B_E),getproperty.(selected,:B_I);color=colors[n],markersize=2.5)
        end
        for (name,(_,x,y)) in label_cells
            text!(ax,x,y;text=name,align=(:center,:center),fontsize=13,color=:black)
        end
        bound=TOML.parsefile(joinpath(packet,"cases/$id/geometry/bounds.toml"))
        xlims!(ax,0,bound["B_E"]);ylims!(ax,0,bound["B_I"])
        for b in filter(r->r.id==id,rows(packet,"baselines.csv"))
            scatter!(ax,[b.B_E],[b.B_I];color=:black,marker=b.validated_cycles>0 ? :star5 : :circle,markersize=12)
        end
    end
    Legend(fig[2,1:2],[PolyElement(color=colors[n]) for n in names],names;orientation=:horizontal,
        nbanks=2,framevisible=false,labelsize=13,patchsize=(13,13),colgap=10)
    spectrum=rows(packet,"timescales/selective_withdrawal_0.35_0.0/equilibria.csv")
    central=filter(r->.01<r.E<.45 && .05<r.I<.49,spectrum)
    ax=Axis(fig[3,1],title="C   Stability at fixed equilibrium coordinates",
        xlabel="Time-scale ratio, τI / τE",ylabel="Largest real eigenvalue (ms⁻¹)")
    for root in sort(unique(r.root for r in central))
        r=sort(filter(x->x.root==root,central);by=x->x.ratio)
        lines!(ax,getproperty.(r,:ratio),getproperty.(r,:alpha);linewidth=2,label="E = $(round(first(r).E;digits=3))")
    end
    hlines!(ax,[0.];color=:black,linestyle=:dash,linewidth=1)
    axislegend(ax;position=:rb,labelsize=13,framevisible=false)
    cyclepath=joinpath(PERIODIC_DIR,"cycles.csv")
    grid=GridLayout(fig[3,2]);amp=Axis(grid[1,1],title="D   Continued periodic dynamics",ylabel="E peak-to-peak")
    per=Axis(grid[2,1],xlabel="Time-scale ratio, τI / τE",ylabel="Period (ms)")
    if isfile(joinpath(packet,cyclepath))
        cycles=rows(packet,cyclepath)
        for direction in unique(r.direction for r in cycles)
            r=filter(x->x.direction==direction,cycles)
            lines!(amp,getproperty.(r,:ratio),[x.E_max-x.E_min for x in r];color=color("oscillatory"),linewidth=2)
            lines!(per,getproperty.(r,:ratio),getproperty.(r,:period);color=color("oscillatory"),linewidth=2)
            unstable=filter(x->x.stability!="PeriodicOrbitAttracting",r)
            isempty(unstable) || scatter!(amp,getproperty.(unstable,:ratio),[x.E_max-x.E_min for x in unstable];marker=:xcross,color=:black,markersize=6)
        end
        endpoint=decreasing_endpoint(cycles,rows(packet,joinpath(PERIODIC_DIR,"stops.csv")))
        if endpoint!==nothing
            point=endpoint.point
            scatter!(amp,[point.ratio],[point.E_max-point.E_min];color=:white,strokecolor=:black,
                strokewidth=1,marker=:rect,markersize=8)
            text!(amp,.02,.98;text="□ initially decreasing: $(termination_label(endpoint.termination))",
                space=:relative,align=(:left,:top),fontsize=12)
        end
    else
        text!(amp,.5,.5;text="No validated continuation in this run",space=:relative,align=(:center,:center),fontsize=15)
    end
    linkxaxes!(amp,per);hidexdecorations!(amp;grid=false)
    note(fig[4,1:2],"R rest · A active · H herald · S seizure · U unassigned equilibrium. Cell fills summarize agreeing equilibrium samples.\nGray cells remain unresolved; dots are measured inputs. Stars mark tested cycles. Root and branch discovery remain incomplete.")
    rowsize!(fig.layout,2,Auto(.2));savefig(fig,out,"02_repertoire")
end

function outcome_scatter!(ax,data;x=:amplitude,y=:duration)
    for destination in sort(unique(r.destination for r in data))
        selected=filter(r->r.destination==destination,data)
        scatter!(ax,getproperty.(selected,x),getproperty.(selected,y);color=color(destination),marker=marker(destination),markersize=7)
    end
end

function accessibility(packet,out)
    fig=Figure(size=(1120,880));heading(fig,"3   The same circuit supports repeated switching and access to high states")
    events=rows(packet,"sequence/events.csv");trace=rows(packet,"sequence/trajectory.csv")
    ax=Axis(fig[1,1:3],title="A   One connected simulation",ylabel="E / I occupancy",xticklabelsvisible=false)
    stimulus=Axis(fig[2,1:3],ylabel="Applied input",xlabel="Time (s; settled intervals omitted)")
    ticks=Float64[];labels=String[]
    for (k,event) in enumerate(events)
        left=event.onset-50.;right=event.offset+250.;shift=(k-1)*420.
        selected=filter(r->left<=r.time<=right,trace)
        x=[r.time-left+shift for r in selected]
        lines!(ax,x,getproperty.(selected,:E);color=E_COLOR,linewidth=2,label=k==1 ? "E" : nothing)
        lines!(ax,x,getproperty.(selected,:I);color=I_COLOR,linestyle=:dash,linewidth=2,label=k==1 ? "I" : nothing)
        be=.35+(event.axis=="E_increase" ? event.amplitude : event.axis=="E_withdrawal" ? -event.amplitude : 0.)
        bi=event.axis=="I_increase" ? event.amplitude : 0.
        xs=shift .+ [0.,50.,50.,50. + event.duration,50. + event.duration,400.]
        lines!(stimulus,xs,[.35,.35,be,be,.35,.35];color=E_COLOR,linewidth=1.8)
        lines!(stimulus,xs,[0.,0.,bi,bi,0.,0.];color=I_COLOR,linewidth=1.8,linestyle=:dash)
        text!(ax,shift+220.,.535;text=event.destination,color=color(event.destination),align=(:center,:bottom),fontsize=14)
        push!(ticks,shift+50.);push!(labels,string(round(event.onset/1000;digits=2)))
        k==length(events) || text!(ax,shift+410.,.02;text="//",align=(:center,:center),fontsize=17)
    end
    stimulus.xticks=(ticks,labels);linkxaxes!(ax,stimulus);ylims!(ax,-.02,.58)
    text!(ax,0.,.535;text="rest",color=color("rest"),align=(:left,:bottom),fontsize=14)
    axislegend(ax;position=:lt,orientation=:horizontal,framevisible=false,labelsize=13)
    roots=rows(packet,"cases/selective_withdrawal/baseline_1/roots.csv")
    for (column,source,direction,title) in ((1,"rest","E_increase","B   E stimulation from rest"),
        (2,"active","I_increase","C   I stimulation from active"))
        k=only(filter(r->r.role==source,roots)).root
        pulse=filter(r->r.axis==direction,rows(packet,"cases/selective_withdrawal/baseline_1/root_$k/pulses/pulses.csv"))
        a=Axis(fig[3,column],title=title,xlabel="Pulse increment",ylabel="Duration (ms)",yscale=log10,
            yticks=[1,2,5,10,20,50,100,200])
        outcome_scatter!(a,pulse)
        for event in filter(e->e.source==source && e.axis==direction,events)
            scatter!(a,[event.amplitude],[event.duration];color=:transparent,strokecolor=:black,strokewidth=2,markersize=17)
        end
    end
    k=only(filter(r->r.role=="rest",roots)).root
    sweep=filter(r->r.axis=="B_E",rows(packet,"cases/selective_withdrawal/baseline_1/root_$k/hysteresis/sweeps.csv"))
    h=Axis(fig[3,3],title="D   Input history selects the state",xlabel="Tonic E input, B_E",ylabel="Settled E occupancy")
    previous_E=only(filter(r->r.role=="rest",roots)).E
    labeled=Set{Int}()
    for r in filter(x->x.first_direction==-1,sweep)
        direction=sign(r.input-r.previous_input)
        direction==0 && continue
        lines!(h,[r.previous_input,r.input],[previous_E,r.E];color=direction==1 ? E_COLOR : I_COLOR,
            linestyle=direction==1 ? :solid : :dash,linewidth=2,
            label=direction in labeled ? nothing : direction==1 ? "increasing input" : "decreasing input")
        push!(labeled,Int(direction));previous_E=r.E
    end
    xlims!(h,0,2);axislegend(h;position=:rb,labelsize=12,framevisible=false)
    role_legend(fig[4,1:3])
    note(fig[5,1:3],"Each event starts at the preceding simulated endpoint. Breaks omit settled portions; destinations use full recorded follow-up.\nCircles outline chosen pulses. D connects settled samples along one sweep: B_E = 0.35 → 0 → 16 → 0.35; the axis shows its transition detail.")
    rowsize!(fig.layout,1,Relative(.29));rowsize!(fig.layout,2,Relative(.12));savefig(fig,out,"03_accessibility")
end

function phase_data(packet)
    root=joinpath(packet,"cases","input_dependent")
    baselines=filter(name->startswith(name,"baseline_"),readdir(root))
    selected=filter(name->begin
        summary=TOML.parsefile(joinpath(root,name,"summary.toml"))
        summary["B_E"]==2. && summary["B_I"]==1.
    end,baselines)
    length(selected)==1 || error("phase panel requires one B=(2,1) baseline")
    baseline=joinpath(root,only(selected))
    count=TOML.parsefile(joinpath(baseline,"summary.toml"))["validated_cycles"]
    count==0 && return NamedTuple[],nothing
    count==1 || error("phase panel requires a declared orbit when multiple cycles are validated")
    isfile(joinpath(baseline,"orbit_1.toml")) || error("phase panel lacks its declared orbit")
    all=NamedTuple[]
    for name in readdir(baseline)
        startswith(name,"cycle_1_phase_") || continue
        file=joinpath(baseline,name,"pulses","pulses.csv")
        isfile(file) || continue
        append!(all,NamedTuple.(collect(CSV.File(file;stringtype=String))))
    end
    isempty(all) && return all,nothing
    choices=sort(unique((r.axis,r.duration) for r in all if r.axis!="joint"))
    isempty(choices) && return NamedTuple[],nothing
    score(choice)=begin
        r=filter(x->(x.axis,x.duration)==choice,all)
        sum(length(unique(x.destination for x in r if x.amplitude==a))>1 for a in unique(x.amplitude for x in r))
    end
    chosen=first(sort(choices;by=c->(-score(c),c)))
    filter(r->(r.axis,r.duration)==chosen,all),chosen
end

function control(packet,out)
    fig=Figure(size=(1120,900));heading(fig,"4   Control depends on the source state, available input, and phase")
    restriction=TOML.parsefile(joinpath(packet,"mechanism","restriction.toml"))
    restriction["descending_branch_condition"] || error("reference source does not meet the invariant-region condition")
    pulse=rows(packet,"paired_controls/pulses.csv")
    a=Axis(fig[1,1],title="A   Herald response to matched withdrawal",
        xlabel="E-input decrement",ylabel="Duration (ms)",yscale=log10,yticks=[1,2,5,10,20,50,100,200])
    for r in pulse
        scatter!(a,[r.amplitude],[r.duration];color=color(r.herald),marker=marker(r.herald),markersize=8)
    end
    seizure_outcomes=unique(r.seizure for r in pulse)
    text!(a,.97,.95;text="Seizure source: "*join(seizure_outcomes,", ")*"\nat matched sampled pulses",
        space=:relative,align=(:right,:top),color=color("seizure"),fontsize=13)
    paired=rows(packet,"paired_controls/sustained.csv")
    # Each row is a measured context, not an independent replicate or prevalence estimate.
    context_names=Dict("selective_withdrawal"=>"Reference","narrative"=>"Low drive circuit",
        "input_dependent"=>"Weak recurrence","input_dependent_e_to_e_4.25"=>"Nearby recurrence (E→E=4.25)",
        "figure4_joint_29"=>"Joint parameter contrast")
    labels=["$(get(context_names,r.id,r.id))\nB=($(r.B_E), $(r.B_I))" for r in paired]
    b=Axis(fig[1,2],title="B   Complete withdrawal across contexts",xticks=([1,2],["herald source","seizure source"]),
        yticks=(collect(1:length(paired)),labels),yreversed=true,yticklabelsize=14,xgridvisible=false,ygridvisible=false)
    for (k,r) in enumerate(paired), (x,dest) in ((1,r.herald),(2,r.seizure))
        scatter!(b,[x],[k];color=color(dest),marker=marker(dest),markersize=11)
        text!(b,x+.08,k;text=dest,color=color(dest),fontsize=14,align=(:left,:center))
    end
    xlims!(b,.8,2.65);ylims!(b,length(paired)+.5,.5)
    proof=Axis(fig[2,1],title="C   Why added drive can fail categorically",limits=(0,1,0,1))
    hidedecorations!(proof);hidespines!(proof)
    poly!(proof,Rect2f(.35,.12,.58,.55);color=(color("seizure"),.14),strokecolor=color("seizure"),strokewidth=2)
    scatter!(proof,[.35],[.67];color=color("seizure"),marker=:utriangle,markersize=14)
    text!(proof,.34,.73;text="Starting equilibrium (E*, I*)",fontsize=15,align=(:left,:bottom))
    text!(proof,.63,.4;text="E ≥ E*\nI ≤ I*",fontsize=22,align=(:center,:center))
    text!(proof,.05,.88;text="ΔB_E ≥ 0 and ΔB_I ≥ 0",fontsize=18)
    text!(proof,.05,.015;text="Inward or tangent boundary flow (schematic).\nDescending I response; drives stay at or above baseline.",fontsize=13)
    for (x,y,txt) in ((.36,.38,"→"),(.62,.65,"↓"),(.91,.38,"←"),(.63,.13,"↑"))
        text!(proof,x,y;text=txt,fontsize=24,align=(:center,:center))
    end
    phase,choice=phase_data(packet)
    axis_names=Dict("E_increase"=>"E pulse increment","I_increase"=>"I pulse increment",
        "E_withdrawal"=>"E-input decrement","I_withdrawal"=>"I-input decrement")
    d=Axis(fig[2,2],title="D   Timing within a cycle changes the response",xlabel=choice===nothing ? "Pulse increment" : axis_names[choice[1]],
        ylabel="Starting cycle phase",yticks=0:.25:1)
    if !isempty(phase)
        outcome_scatter!(d,phase;y=:phase)
        text!(d,.98,.98;text="Weak circuit, B=(2,1), orbit 1; $(choice[2]) ms",space=:relative,align=(:right,:top),fontsize=12)
    else
        text!(d,.5,.5;text="No phase-resolved pulse data",space=:relative,align=(:center,:center))
    end
    ylims!(d,-.03,1.06)
    role_legend(fig[3,1:2])
    note(fig[4,1:2],"A restores baseline after a temporary decrement; B permanently sets E input to zero with I input unchanged.\nPaired sources share parameters and inputs. D displays the measured protocol with greatest sampled phase variation; no phase-completeness claim.")
    savefig(fig,out,"04_control")
end

function interventions(packet,out)
    fig=Figure(size=(1180,700));heading(fig,"5   Intervention outcomes and retained switching must be measured together")
    axes=("theta_off","e_to_e","e_to_i")
    titles=("Failure threshold, θoff","Recurrent excitation, E→E","Recruitment, E→I")
    for (row,id) in enumerate(("selective_withdrawal","threshold_to_active"))
        data=rows(packet,"interventions/$id/summary.csv")
        for (column,axis) in enumerate(axes)
            r=sort(filter(x->x.axis==axis,data);by=x->x.value)
            title=(row==1 ? string(Char('A'+column-1))*"   " : string(Char('D'+column-1))*"   ")*titles[column]
            a=Axis(fig[row,column],title=title,xlabel=titles[column],
                yticks=([1,2],["Original high-state\ndestination","Ordinary switching"]),
                yticklabelsize=14,ygridvisible=false)
            for x in r
                scatter!(a,[x.value],[1.];color=color(x.destination),marker=marker(x.destination),markersize=12)
                original=x.fixed_switching=="retained"
                remeasured=x.remeasured_switching=="retained"
                unavailable=x.fixed_switching=="sources_unavailable"
                scatter!(a,[x.value],[2.];color=original ? "#111111" : remeasured ? :white : "#888888",
                    strokecolor=remeasured ? :black : :gray,strokewidth=remeasured ? 1.5 : 0,
                    marker=remeasured ? :circle : unavailable ? :xcross : :utriangle,markersize=11)
            end
            ylims!(a,.5,2.5)
            column==1 || hideydecorations!(a;grid=false)
        end
        Label(fig[row,0],row==1 ? "Reference\nB_E = 0.35" : "Contrast\nB_E = 0.5";
            fontsize=15,rotation=pi/2,tellheight=false)
    end
    role_legend(fig[3,1:3];roles=["rest","active","herald","seizure","oscillatory","unassigned"])
    Legend(fig[4,1:3],[MarkerElement(marker=:circle,color=:black),
        MarkerElement(marker=:circle,color=:white,strokecolor=:black,strokewidth=1.5),MarkerElement(marker=:utriangle,color=:gray),
        MarkerElement(marker=:xcross,color=:gray)],
        ["original pulses retained","remeasured pulses only","not demonstrated","sources unavailable"];
        orientation=:horizontal,framevisible=false,labelsize=14)
    note(fig[5,1:3],"Each intervention starts from its circuit’s original seizure source. Markers are sampled settings, not certified parameter intervals.\nA preserved original protocol is also a successful remeasured protocol; failed original pulses trigger a new bounded stimulus search.")
    savefig(fig,out,"05_interventions")
end

function render(packet,output;show_basins=true,show_separatrices=true)
    FigurePacketIO.verify_manifest(packet)
    ispath(output) && error("output already exists")
    mkpath(output)
    with_theme(Theme(fontsize=18,Axis=(titlesize=18,titlealign=:left,xlabelsize=17,ylabelsize=17,
        xticklabelsize=14,yticklabelsize=14,spinewidth=1,xgridcolor=(:gray,.12),ygridcolor=(:gray,.12)))) do
        mechanism(packet,output;show_basins,show_separatrices);repertoire(packet,output);accessibility(packet,output)
        control(packet,output);interventions(packet,output)
    end
    open(joinpath(output,"provenance.toml"),"w") do io
        TOML.print(io,Dict("packet_manifest_sha256"=>FigurePacketIO.digest(joinpath(packet,"checksums.toml")),
            "renderer_sha256"=>FigurePacketIO.digest(@__FILE__),
            "overlay_helper_sha256"=>FigurePacketIO.digest(joinpath(@__DIR__,"nullcline_overlays.jl")),
            "show_basins"=>show_basins,"show_separatrices"=>show_separatrices,"julia_version"=>string(VERSION),
            "cairomakie_version"=>string(pkgversion(CairoMakie)),
            "plotting_project_sha256"=>FigurePacketIO.digest(joinpath(@__DIR__,"..","plotting","Project.toml")),
            "plotting_manifest_sha256"=>FigurePacketIO.digest(joinpath(@__DIR__,"..","plotting","Manifest.toml")),
            "purpose"=>"Five figure drafts for author review; numerical verification is recorded in the data packet");sorted=true)
    end
    titles=["Mechanism and equilibrium ordering","Input repertoire and time-scale dynamics",
        "Connected switching and accessibility","Source-dependent and phase-dependent control","Intervention tradeoffs"]
    names=["01_mechanism","02_repertoire","03_accessibility","04_control","05_interventions"]
    captions=[
        "The response mechanism is compared at identical couplings, drives, and time scales. The ordering restriction concerns equilibria of one fixed-input system.",
        "Two-input maps show sampled equilibrium repertoires. Time scales change stability while equilibrium coordinates stay fixed. $(continuation_caption(packet))",
        "One connected simulation visits rest, active, herald, and seizure without resetting state. Aligned inputs show all seven pulses. Additional panels measure amplitude-duration dependence and input history.",
        "Matched withdrawal can preserve both high states, recover both, or recover only herald across the measured circuits. A specified class of drive increments has an analytical obstruction. The periodic-source panel compares pulse outcomes across starting phases.",
        "Intervention destination and ordinary switching are separate measurements. Filled circles preserve the original switching protocol; open circles require remeasured pulses. Missing sources and unassigned equilibria retain distinct labels."]
    open(joinpath(output,"index.html"),"w") do io
        write(io,"<!doctype html><meta charset='utf-8'><title>Point-model figure redesign</title><style>body{max-width:1200px;margin:2rem auto;font-family:system-ui;line-height:1.5}img{width:100%}section{margin:3rem 0}a{margin-right:1rem}</style><h1>Point-model figure redesign</h1><p>Five figure drafts for author review. Numerical searches are bounded; state names are provisional coordinate roles.</p>")
        for (title,name,caption) in zip(titles,names,captions)
            write(io,"<section><h2>",title,"</h2><p>",caption,"</p><a href='",name,".pdf'>PDF</a><a href='",name,".svg'>SVG</a><img src='",name,".png' alt='",title,"'></section>")
        end
        write(io,"<p><a href='provenance.toml'>Rendering provenance</a><a href='checksums.toml'>Artifact checksums</a></p>")
    end
    FigurePacketIO.write_manifest(output)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)>=2 || error("usage: render_figure_redesign.jl PACKET NEW_OUTPUT [--no-basins] [--no-separatrices]")
    flags=ARGS[3:end]
    all(in(("--no-basins","--separatrices","--no-separatrices")),flags) || error("unknown rendering flag")
    FigureRedesignRenderer.render(ARGS[1],ARGS[2];show_basins=!("--no-basins" in flags),show_separatrices=!("--no-separatrices" in flags))
end
