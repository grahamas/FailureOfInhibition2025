"""Recheck unmatched continuation samples using both sides of every refined fold.

An interpolated seed can converge to the neighboring root at a fold. Retain
every correction, and accept a match only if a validated candidate reproduces
the original coordinate within 1e-7. This supplements, never overwrites, the
first continuation comparison.
"""
module EquilibriumFoldMatches
using FailureOfInhibition2025
import CSV,TOML
include("continue_equilibrium_audit.jl")
const C=EquilibriumAuditContinuation

function candidates(points,parameter,state)
    seeds=Vector{Float64}[]
    for direction in (-1,1)
        group=filter(r->r.variant=="half_step" && r.direction==direction,points)
        for (a,b) in zip(group,group[2:end])
            min(a.parameter,b.parameter)<=parameter<=max(a.parameter,b.parameter) || continue
            a.parameter==b.parameter && continue
            x=[a.E,a.I];y=[b.E,b.I];t=(parameter-a.parameter)/(b.parameter-a.parameter)
            for s in (x,y,(1-t).*x.+t.*y)
                maximum(abs.(s.-state))<=.05 && push!(seeds,s)
            end
        end
    end
    unique(seeds)
end

function check(audit,continuation,output)
    C.IOChecks.verify_manifest(audit);C.IOChecks.verify_manifest(continuation)
    ispath(output) && error("fold-check output exists");mkpath(output)
    checks=collect(CSV.File(joinpath(continuation,"checks.csv");stringtype=String))
    seeds=collect(CSV.File(joinpath(audit,"seeds.csv");stringtype=String))
    entries=TOML.parsefile(joinpath(audit,"seed_parameters.toml"))["seeds"]
    results=NamedTuple[];attempts=NamedTuple[]
    for c in filter(c->c.refinement!="agrees",checks)
        points=collect(CSV.File(joinpath(continuation,"seed_$(c.seed)",c.axis*"_points.csv");stringtype=String))
        target=only(filter(r->r.variant=="default" && r.direction==c.direction && r.index==c.index,points))
        state=[target.E,target.I]
        seed=only(filter(r->r.seed==c.seed,seeds));entry=only(filter(e->e["seed"]==c.seed,entries))
        change,_,_=C.setup(seed,entry,c.axis);p,be,bi=change(c.parameter);model=C.A.M.model(p,be,bi)
        starts=candidates(points,c.parameter,state);best=Inf
        for (k,s) in enumerate(starts)
            a=solve_equilibrium(model,s;options=EquilibriumOptions(residual_atol=1e-12,solver_reltol=1e-12)).attempt
            error=maximum(abs.(a.candidate.-state));valid=a.validation==AdmissibleCandidate
            valid && (best=min(best,error))
            push!(attempts,(;seed=c.seed,axis=c.axis,direction=c.direction,index=c.index,start=k,
                E_start=s[1],I_start=s[2],E=a.candidate[1],I=a.candidate[2],valid,residual=a.residual_norm,error))
        end
        status=best<1e-7 ? "agrees_after_fold_seed_check" : isempty(starts) ? "refined_segment_not_available" : "unresolved_branch_match"
        push!(results,(;seed=c.seed,axis=c.axis,direction=c.direction,index=c.index,parameter=c.parameter,
            original=c.refinement,role=c.role,status,error=best,attempts=length(starts)))
    end
    CSV.write(joinpath(output,"checks.csv"),results);CSV.write(joinpath(output,"attempts.csv"),attempts)
    C.write_record(joinpath(output,"summary.toml"),Dict("completed"=>true,"checks"=>length(results),
        "resolved"=>count(r->r.status=="agrees_after_fold_seed_check",results),
        "remaining"=>count(r->r.status!="agrees_after_fold_seed_check",results),
        "source_sha256"=>C.IOChecks.digest(@__FILE__),"continuation_manifest_sha256"=>C.IOChecks.digest(joinpath(continuation,"checksums.toml"))))
    C.IOChecks.write_manifest(output)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==3 || error("usage: check_equilibrium_fold_matches.jl AUDIT CONTINUATION NEW_OUTPUT")
    EquilibriumFoldMatches.check(abspath.(ARGS)...)
end
