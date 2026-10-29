# Warmed basin reuse/scaling benchmark and independent classification checks.
# Usage: julia --project=. scripts/benchmark_nullcline_basins.jl NEW_OUTPUT
using FailureOfInhibition2025
import CSV, TOML
include("input_response_models.jl")
function benchmark_nullcline_basins(output)
    ispath(output) && error("output must be new")
    mkpath(output)
    registry=TOML.parsefile(joinpath(@__DIR__,"..","reproducibility","figure_redesign_20261006","cases.toml"))
    p=only(filter(p->p["id"]=="selective_withdrawal",registry["cases"]))
    timings=NamedTuple[]; validations=NamedTuple[]; decisions=Dict{String,Bool}()
    for (condition,control) in (("monotone",true),("failure",false))
        model=InputResponseModels.model(p,.35,0.;control)
        search=find_equilibria(model;seeds=[[e,i] for e in range(0,.5;length=41) for i in range(0,.5;length=41)])
        basin_grid(model,search;options=BasinGridOptions(grid_points=8))
        trace_separatrices(model,search)
        grid100=Float64[]; traces=Float64[]
        for n in (50,100,200), repeat in 1:3
            grid=nothing
            elapsed=@elapsed grid=basin_grid(model,search;options=BasinGridOptions(grid_points=n))
            times=vec(grid.statistics.integrated_times); width=max(1,length(times)÷10)
            early=sum(times[1:width])/width;late=sum(times[end-width+1:end])/width
            push!(timings,(;condition,method="basin_grid",n,repeat,seconds=elapsed,
                cached_starts=grid.statistics.cached_starts,basin_hits=grid.statistics.basin_hits,
                total_steps=grid.statistics.total_steps,early_mean_ms=early,late_mean_ms=late,
                unresolved=grid.statistics.unresolved))
            n==100 && push!(grid100,elapsed)
            println(condition," ",n," run ",repeat," seconds=",elapsed," cached=",grid.statistics.cached_starts," basin_hits=",grid.statistics.basin_hits);flush(stdout)
        end
        for repeat in 1:3
            branches=nothing
            elapsed=@elapsed branches=trace_separatrices(model,search)
            push!(traces,elapsed)
            push!(timings,(;condition,method="separatrices",n=0,repeat,seconds=elapsed,
                cached_starts=0,basin_hits=0,total_steps=0,early_mean_ms=0.,late_mean_ms=0.,
                unresolved=count(b->!b.converged,branches)))
        end
        sort!(traces);sort!(grid100);decisions[condition]=traces[2]<grid100[2]
        # Validate a complete coarse grid, then finer-grid label transitions.
        for n in (17,100)
            grid=basin_grid(model,search;options=BasinGridOptions(grid_points=n))
            indices=n==17 ? collect(CartesianIndices(grid.labels)) :
                [index for index in CartesianIndices(grid.labels) if any(
                    checkbounds(Bool,grid.labels,index+delta) && grid.labels[index]!=grid.labels[index+delta]
                    for delta in (CartesianIndex(1,0),CartesianIndex(0,1),CartesianIndex(-1,0),CartesianIndex(0,-1)))]
            # Bound fine checks deterministically, retaining points across the full boundary.
            n==100 && length(indices)>100 && (indices=indices[round.(Int,range(1,length(indices);length=100))])
            for index in indices
                state=[grid.E[index[1]],grid.I[index[2]]]
                independent=basin_destination(model,search,state)
                destination=something(independent.destination,0)
                comparable=grid.labels[index]>0 && independent.status==:compatible
                comparison=!comparable ? "unassessed" : grid.labels[index]==destination ? "agreed" : "disagreed"
                matched=comparable && grid.labels[index]==destination
                push!(validations,(;condition,n,E=state[1],I=state[2],label=grid.labels[index],
                    independent=destination,independent_status=string(independent.status),comparison,matched))
            end
        end
    end
    CSV.write(joinpath(output,"timings.csv"),timings)
    CSV.write(joinpath(output,"validation.csv"),validations)
    open(joinpath(output,"decision.toml"),"w") do io
        TOML.print(io,Dict("separatrices_default"=>all(values(decisions)),"cheaper_by_condition"=>decisions,
            "validation_disagreements"=>count(r->r.comparison=="disagreed",validations),
            "validation_comparable"=>count(r->r.comparison!="unassessed",validations),
            "validation_unassessed"=>count(r->r.comparison=="unassessed",validations),
            "julia_version"=>string(VERSION),"attractors_version"=>string(pkgversion(FailureOfInhibition2025.Attractors))))
    end
    println("decision: ",decisions," validation disagreements=",count(r->r.comparison=="disagreed",validations))
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==1 || error("usage: benchmark_nullcline_basins.jl NEW_OUTPUT")
    benchmark_nullcline_basins(ARGS[1])
end
