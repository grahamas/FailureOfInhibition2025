import Attractors

"""Finite-resolution equilibrium basin map. Bounds select initial conditions only;
trajectories and the persistent recurrence cache cover the full physical square.
`sample_dt=nothing` chooses a deterministic interval from a 17×17 speed probe.
The recurrence adapter is intentionally pinned to Attractors 2.0.5.
"""
struct BasinGridOptions{P<:PulseExperimentOptions}
    grid_points::Int
    bounds::NTuple{2,Tuple{Float64,Float64}}
    sample_dt::Union{Nothing,Float64}
    consecutive_basin_steps::Int
    consecutive_recurrences::Int
    attractor_locate_steps::Int
    maximum_iterations::Int
    pulse_options::P
end
function BasinGridOptions(; grid_points=100, bounds=((0.,.5),(0.,.5)),
    sample_dt=nothing, consecutive_basin_steps=10, consecutive_recurrences=100,
    attractor_locate_steps=100, maximum_iterations=100000,
    pulse_options=PulseExperimentOptions())
    for (key,value,minimum) in ((:grid_points,grid_points,2),
        (:consecutive_basin_steps,consecutive_basin_steps,1),
        (:consecutive_recurrences,consecutive_recurrences,2),
        (:attractor_locate_steps,attractor_locate_steps,2),
        (:maximum_iterations,maximum_iterations,1))
        value isa Integer && !(value isa Bool) && minimum<=value<=typemax(Int) ||
            throw(ArgumentError("$key must be an integer ≥ $minimum"))
    end
    length(bounds)==2 && all(b->length(b)==2 && all(x->x isa Real && !(x isa Bool) && isfinite(x),b) &&
        0<=b[1]<b[2]<=1,bounds) || throw(ArgumentError("bounds must be two intervals within [0,1]"))
    sample_dt===nothing || (sample_dt isa Real && !(sample_dt isa Bool) && isfinite(sample_dt) && sample_dt>0) ||
        throw(ArgumentError("sample_dt must be positive or nothing"))
    pulse_options isa PulseExperimentOptions || throw(ArgumentError("invalid pulse_options"))
    BasinGridOptions(Int(grid_points),Tuple((Float64(b[1]),Float64(b[2])) for b in bounds),
        sample_dt===nothing ? nothing : Float64(sample_dt),Int(consecutive_basin_steps),
        Int(consecutive_recurrences),Int(attractor_locate_steps),Int(maximum_iterations),pulse_options)
end

function _basin_sample_dt(model, spacing)
    du=zeros(2); speeds=Float64[]
    for e in range(0,1;length=17), i in range(0,1;length=17)
        point_rhs!(du,[e,i],model,0.)
        speed=maximum(abs,du)
        isfinite(speed) && speed>0 && push!(speeds,speed)
    end
    isempty(speeds) && return 1.
    sort!(speeds)
    # A typical step crosses about two cells. Unlike automatic_Δt_recurrences,
    # this uses no random sampling or global random state.
    clamp(2minimum(spacing)/speeds[cld(length(speeds),2)],1e-4,10.)
end

# A single sampled path does not establish a whole cell's basin. Require an
# interior 3×3 patch of one identity before reusing a transient cell. Attractor
# cells are anchored by the independently checked known sink seeds.
function _basin_cache_interior(info,cell,protected)
    protected[cell] && return false
    code=info.BoA.basins[cell]
    code<=0 && return false
    iseven(code) && return code<info.visited_cell_label
    id=(code-1)÷2
    for de in -1:1, di in -1:1
        other=cell+CartesianIndex(de,di)
        checkbounds(Bool,protected,other) || continue
        protected[other] && return false
        value=info.BoA.basins[other]
        value>0 && value<info.visited_cell_label || return false
        (iseven(value) ? value÷2 : (value-1)÷2)==id || return false
    end
    true
end

# Narrow compatibility adapter: Attractors 2.0.5 mapper(u0) erases visited
# transient cells (cleanup_visited_cells!). Drive its same FSM here, retaining
# accepted sample-cell indices before cleanup and committing them on success.
# Internal labels are 2k for attractors and 2k+1 for their basins.
function _cached_recurrence!(mapper, state, protected, options; discover=false)
    info=mapper.bsn_nfo; ds=mapper.ds
    cell=Attractors.basin_cell_index(state,info.BoA.grid)
    code=info.BoA.basins[cell]
    if code>0 && code<info.visited_cell_label && _basin_cache_interior(info,cell,protected)
        return (label=iseven(code) ? code÷2 : (code-1)÷2,time=0.,steps=0,kind=:cached_start)
    end
    Attractors.reinit!(ds,state)
    provisional_label=info.current_att_label
    Attractors.reset_basins_counters!(info)
    info.safety_counter=0
    path=CartesianIndex{2}[cell]
    label=0; elapsed=0.; kind=:unresolved
    for iteration in 1:options.maximum_iterations
        Attractors.step!(ds,info.Δt,true)
        info.safety_counter=iteration
        elapsed=Float64(Attractors.current_time(ds))
        if !Attractors.successful_step(ds)
            kind=:integration_failed;break
        end
        u=Attractors.current_state(ds)
        if _state_outside_domain(u,options.pulse_options.domain_atol)
            kind=:integration_failed;break
        end
        cell=Attractors.basin_cell_index(u,info.BoA.grid)
        cell[1]==-1 && (kind=:integration_failed;break)
        push!(path,cell)
        old=info.BoA.basins[cell]
        unsafe=old>0 && old<info.visited_cell_label && !_basin_cache_interior(info,cell,protected)
        unsafe && (info.BoA.basins[cell]=0)
        label=Attractors.finite_state_machine!(info,cell,u;
            consecutive_basin_steps=options.consecutive_basin_steps,
            consecutive_recurrences=discover ? options.consecutive_recurrences : typemax(Int),
            attractor_locate_steps=options.attractor_locate_steps,
            show_progress=false)
        unsafe && (info.BoA.basins[cell]=old)
        if label!=0
            kind=info.return_code
            break
        end
    end
    Attractors.cleanup_visited_cells!(info)
    if label>0
        id=iseven(label) ? label÷2 : (label-1)÷2
        for index in path
            old=info.BoA.basins[index]
            # Never overwrite attractor cells, another basin, or cells close
            # to known nonattracting equilibria with a transient path label.
            !protected[index] && old==0 && (info.BoA.basins[index]=2id+1)
        end
        return (label=id,time=elapsed,steps=info.safety_counter,kind)
    end
    # An interrupted :att_found phase must not leave provisional attractor
    # cells/points available for subsequent classifications.
    for index in eachindex(info.BoA.basins)
        info.BoA.basins[index]==provisional_label && (info.BoA.basins[index]=0)
    end
    delete!(info.BoA.attractors,provisional_label÷2)
    return (label=0,time=elapsed,steps=info.safety_counter,kind)
end

"""
    basin_grid(model, equilibria; options=BasinGridOptions())

Map initial `[E,I]` states to discovered attracting equilibrium indices using a
shared dense recurrence cache. Zero labels are unresolved, including unmatched
recurrent sets and trajectories at nonattracting equilibria. Cell membership is
a finite-resolution approximation, not a certified basin enclosure. The result
includes per-point integration times and cache-hit statistics (not convergence
times to the equilibrium). Periodic attractors are not assigned equilibrium IDs.
"""
function basin_grid(model::PointModelParameters,equilibria;options=BasinGridOptions())
    options isa BasinGridOptions || throw(ArgumentError("invalid basin options"))
    _pulse_context(model,equilibria)
    pkgversion(Attractors)==v"2.0.5" || error("recurrence adapter requires Attractors 2.0.5; revalidate before upgrading")
    E=collect(range(options.bounds[1]...;length=options.grid_points))
    I=collect(range(options.bounds[2]...;length=options.grid_points))
    spacing=(step(range(options.bounds[1]...;length=options.grid_points)),
        step(range(options.bounds[2]...;length=options.grid_points)))
    # Separate known roots on the internal lattice, even when the requested
    # image grid is coarse. Do not permit a sink and saddle to share a cell.
    roots=equilibria.equilibria
    separation=minimum((maximum(abs.(roots[a].state.-roots[b].state))
        for a in eachindex(roots) for b in eachindex(roots) if a<b);init=Inf)
    support_spacing=min.(spacing,separation/8)
    support_lengths=Tuple(ceil(Int,1/h)+1 for h in support_spacing)
    maximum(support_lengths)<=4097 || throw(ArgumentError("known roots require an internal basin grid finer than 4097 per axis"))
    support=Tuple(range(0.,1.;length=n) for n in support_lengths)
    h=maximum(step.(support)); match_distance=2sqrt(2)*h
    dt=something(options.sample_dt,_basin_sample_dt(model,spacing))
    po=options.pulse_options
    ds=Attractors.CoupledODEs(point_rhs!,[E[1],I[1]],model;
        diffeq=(alg=Tsit5(),abstol=po.abstol,reltol=po.reltol,maxiters=po.maxiters,
            isoutofdomain=(u,p,t)->_state_outside_domain(u,po.domain_atol)))
    mapper=Attractors.BasinMapRecurrences(ds,support;sparse=false,Δt=dt,stop_at_Δt=true)
    protected=falses(length.(support)...)
    for r in equilibria.equilibria
        r.stability.classification==Attracting && continue
        axes=Tuple(searchsortedfirst(support[k],r.state[k]-2h):
            searchsortedlast(support[k],r.state[k]+2h) for k in 1:2)
        for e in axes[1], i in axes[2]
            protected[e,i]=true
        end
    end
    # Warm known sinks first, so recurrence discovery cannot invent a stable
    # label for a nearby slow transient before the actual sink is visited.
    for r in equilibria.equilibria
        r.stability.classification==Attracting && _cached_recurrence!(mapper,r.state,protected,options;discover=true)
    end
    raw=zeros(Int,length(E),length(I)); statuses=fill(:unresolved,size(raw))
    times=zeros(size(raw)); iterations=zeros(Int,size(raw))
    cached_starts=0;basin_hits=0;attractor_hits=0
    for index in CartesianIndices(raw)
        state=[E[index[1]],I[index[2]]]
        if any(r->r.stability.classification!=Attracting &&
            maximum(abs.(state.-r.state))<=po.diagnostic_options.coordinate_atol,equilibria.equilibria)
            statuses[index]=:nonattracting;continue
        end
        outcome=_cached_recurrence!(mapper,state,protected,options)
        raw[index]=outcome.label;times[index]=outcome.time;iterations[index]=outcome.steps
        statuses[index]=outcome.kind==:integration_failed ? :integration_failed : :unresolved
        cached_starts+=outcome.kind==:cached_start
        basin_hits+=outcome.kind==:bas_hit
        attractor_hits+=outcome.kind==:att_hit
    end
    # Discovery is restricted to known sink seeds; arbitrary recurrent sets
    # (including cycles and slow saddles) must never seed the shared cache.
    # Recurrence IDs are not root IDs. Confirm each entire recurrent set is
    # local to one known sink, then independently diagnose a representative.
    matches=Dict{Int,Int}()
    for (id,attractor) in Attractors.extract_attractors(mapper)
        isempty(attractor) && continue
        candidates=findall(r->r.stability.classification==Attracting &&
            all(u->maximum(abs.(u.-r.state))<=match_distance,attractor),equilibria.equilibria)
        length(candidates)==1 || continue
        diagnostic=basin_destination(model,equilibria,collect(last(attractor));options=po)
        diagnostic.status==:compatible && diagnostic.destination==only(candidates) &&
            (matches[id]=only(candidates))
    end
    labels=zeros(Int,size(raw))
    for index in CartesianIndices(raw)
        labels[index]=get(matches,raw[index],0)
        labels[index]>0 && (statuses[index]=:compatible)
    end
    # A categorical image cell touching a different sampled destination is
    # not resolved to sub-cell precision. Keep both sides neutral instead of
    # interpolating identities. This does not bound earlier cache errors.
    ambiguous=falses(size(labels))
    for index in CartesianIndices(labels), delta in (CartesianIndex(1,0),CartesianIndex(0,1))
        neighbor=index+delta
        if checkbounds(Bool,labels,neighbor) && labels[index]!=labels[neighbor]
            ambiguous[index]=true;ambiguous[neighbor]=true
        end
    end
    for index in CartesianIndices(labels)
        if ambiguous[index] && labels[index]>0
            labels[index]=0;statuses[index]=:boundary_unresolved
        end
    end
    (;E,I,labels,statuses,options,statistics=(;sample_dt=dt,
        support_shape=size(protected),cached_starts,basin_hits,attractor_hits,
        integrated_times=times,iterations,total_steps=sum(iterations),
        unresolved=count(==(0),labels),recurrence_to_equilibrium=matches))
end
