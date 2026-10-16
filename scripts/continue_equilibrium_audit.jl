"""Bounded branch checks for the unassigned-equilibrium decision package."""
module EquilibriumAuditContinuation
using FailureOfInhibition2025, LinearAlgebra
import CSV,TOML,ForwardDiff
include("run_equilibrium_audit.jl")
include("run_input_response_study.jl")
include("verify_figure_redesign.jl")
const A=EquilibriumAuditStudy.A
const S=InputResponseStudy
const IOChecks=EquilibriumAuditStudy.IOChecks
const write_record=EquilibriumAuditStudy.write_record

function records(search)
    [Dict("state"=>r.state,"near_singular"=>r.near_singular,"balance_residual"=>r.balance_residual,
        "stability"=>Dict("classification"=>string(r.stability.classification))) for r in search.equilibria]
end

function independent_checks(p,be,bi,state)
    model=A.M.model(p,be,bi);jac=zeros(2,2);point_jacobian!(jac,state,model,0.)
    rhs(x)=begin
        dx=similar(x);VerifyFigureRedesign.independent_rhs!(dx,x,(p,be,bi),0.);dx
    end
    ij=ForwardDiff.jacobian(rhs,state)
    residual=maximum(abs.(rhs(state)));error=maximum(abs.(ij.-jac))
    (;independent_residual=residual,jacobian_error=error,passed=residual<1e-8 && error<1e-8)
end

function search_check(p,be,bi,state,cfg)
    ctx=S.context(p,be,bi,cfg;tight=true,grid=41)
    other=find_equilibria(ctx.model;seeds=[[e,i] for e in range(0,.5;length=41)
        for i in range(0,A.M.state_bounds(ctx.model)[2];length=41)],
        options=EquilibriumOptions(residual_atol=1e-11,solver_reltol=1e-11))
    agreement=length(other.equilibria)==length(ctx.search.equilibria) &&
        all(r->any(q->maximum(abs.(q.state.-r.state))<1e-7,other.equilibria),ctx.search.equilibria)
    distances=[maximum(abs.(r.state.-state)) for r in ctx.search.equilibria]
    k=isempty(distances) ? 0 : argmin(distances)
    found=k>0 && distances[k]<1e-7
    role=found ? S.label(ctx,k) : "root_not_rediscovered"
    (;ctx,agreement,found,role,error=isempty(distances) ? Inf : minimum(distances))
end

function setup(seed,entry,axis)
    p=entry["parameters"];be=entry["B_E"];bi=entry["B_I"]
    if axis in ("B_E","B_I")
        bounds=A.M.input_bounds(A.M.model(p,be,bi))
        upper=axis=="B_E" ? bounds.B_E : bounds.B_I
        change(x)=(p,axis=="B_E" ? x : be,axis=="B_I" ? x : bi)
        initial=axis=="B_E" ? be : bi
        return change,initial,(0.,upper)
    end
    base=TOML.parsefile(joinpath(EquilibriumAuditStudy.ROOT,"output/figure_details_20261006/selection.toml"))["cases"]
    q=only(filter(q->q["id"]==seed.id,base))
    bounds=axis=="theta_off" ? (6.,12.) : axis=="e_to_e" ? (.6*q[axis],q[axis]) : (max(12.,.6*q[axis]),q[axis])
    x->(merge(p,Dict(axis=>x)),be,bi),p[axis],bounds
end

function refinement_error(point,refined,factory)
    seeds=Vector{Float64}[]
    for branch in (refined.negative,refined.positive)
        for (a,b) in zip(branch.points,branch.points[2:end])
            min(a.parameter,b.parameter)<=point.parameter<=max(a.parameter,b.parameter) || continue
            a.parameter==b.parameter && continue
            t=(point.parameter-a.parameter)/(b.parameter-a.parameter)
            push!(seeds,(1-t).*a.state.+t.*b.state)
        end
        for p in branch.points
            abs(p.parameter-point.parameter)<1e-8 && push!(seeds,p.state)
        end
    end
    isempty(seeds) && return (;refinement="not_covered",refinement_error=NaN)
    distances=[maximum(abs.(s.-point.state)) for s in seeds];k=argmin(distances)
    distances[k]>.05 && return (;refinement="ambiguous",refinement_error=distances[k])
    attempt=solve_equilibrium(factory(point.parameter),seeds[k];options=EquilibriumOptions(residual_atol=1e-11,solver_reltol=1e-11)).attempt
    err=maximum(abs.(attempt.candidate.-point.state))
    (;refinement=attempt.validation==AdmissibleCandidate && err<1e-7 ? "agrees" : "disagrees",refinement_error=err)
end

function run(audit,output)
    IOChecks.verify_manifest(audit);ispath(output) && error("continuation output exists");mkpath(output)
    seeds=collect(CSV.File(joinpath(audit,"seeds.csv");stringtype=String))
    entries=TOML.parsefile(joinpath(audit,"seed_parameters.toml"))["seeds"]
    cfg=S.load_config(joinpath(EquilibriumAuditStudy.ROOT,"experiments/figure_redesign.toml"))
    checks=NamedTuple[];stops=NamedTuple[];allconnections=NamedTuple[]
    for seed in seeds
        entry=only(filter(e->e["seed"]==seed.seed,entries));p=entry["parameters"]
        initial=[seed.E,seed.I];checked=search_check(p,seed.B_E,seed.B_I,initial,cfg)
        checked.found || error("seed root not rediscovered: $(seed.seed)")
        folder=joinpath(output,"seed_$(seed.seed)");mkpath(folder)
        S.save_context(folder,checked.ctx)
        axes=seed.scope=="intervention" ? [seed.axis] : ["B_E","B_I"]
        for axis in axes
            change,parameter,bounds=setup(seed,entry,axis)
            factory(x)=A.M.model(change(x)...)
            result=continue_equilibria(factory,initial,parameter;parameter_bounds=bounds)
            refined=continue_equilibria(factory,initial,parameter;parameter_bounds=bounds,
                options=ContinuationOptions(initial_step=.015,maximum_step=.05,max_steps=500))
            points=NamedTuple[];attempts=NamedTuple[];candidates=NamedTuple[];connections=NamedTuple[]
            for (variant,run) in (("default",result),("half_step",refined)), branch in (run.negative,run.positive)
                push!(stops,(;seed=seed.seed,axis,variant,direction=branch.direction,termination=string(branch.termination),points=length(branch.points)))
                for (k,point) in enumerate(branch.points)
                    q,be,bi=change(point.parameter);m=point.model
                    row=only(first(A.explain(m,records((;equilibria=[(;state=point.state,near_singular=point.equilibrium_attempt.near_singular,
                        balance_residual=point.equilibrium_attempt.balance_residual,stability=point.stability)])))))
                    independent=independent_checks(q,be,bi,point.state)
                    independent.passed || error("independent equations disagree")
                    push!(points,merge((;variant,direction=branch.direction,index=k,parameter=point.parameter),
                        (;E=row.E,I=row.I,stability=row.stability,arm=row.arm,I_side=row.I_side,alpha=row.alpha,residual=row.residual),independent))
                end
                for (k,a) in enumerate(branch.attempts)
                    push!(attempts,(;variant,direction=branch.direction,index=k,parameter=a.candidate[3],E=a.candidate[1],I=a.candidate[2],
                        accepted=a.accepted,status=string(a.status),step=a.step,iterations=a.iterations,arclength_residual=a.arclength_residual))
                end
                for c in branch.candidates
                    push!(candidates,(;variant,direction=branch.direction,kind=string(c.kind),first_point=c.first_point,second_point=c.second_point))
                end
                variant=="default" || continue
                indices=sort(unique(vcat([1,length(branch.points)],collect(1:25:length(branch.points)))))
                # Also inspect every intrinsic geometry/stability transition, not just a coarse sample.
                localrows=filter(r->r.variant=="default" && r.direction==branch.direction,points)
                for k in 2:length(localrows)
                    a,b=localrows[k-1],localrows[k]
                    (a.arm,a.I_side,a.stability)!=(b.arm,b.I_side,b.stability) && append!(indices,[k-1,k])
                end
                previous_role="unassigned"
                for k in sort(unique(indices))
                    isempty(branch.points) && continue
                    point=branch.points[k];q,be,bi=change(point.parameter)
                    ctx=S.context(q,be,bi,cfg;tight=true)
                    dist=[maximum(abs.(r.state.-point.state)) for r in ctx.search.equilibria]
                    root=isempty(dist) ? 0 : argmin(dist)
                    role=root>0 && dist[root]<1e-7 ? S.label(ctx,root) : "root_not_rediscovered"
                    validate=k in (1,length(branch.points)) || role!=previous_role || !(k in collect(1:25:length(branch.points)))
                    if validate
                        chk=search_check(q,be,bi,point.state,cfg);role=chk.role
                        ref=refinement_error(point,refined,factory)
                        push!(checks,merge((;seed=seed.seed,axis,direction=branch.direction,index=k,parameter=point.parameter,
                            root_found=chk.found,search_agreement=chk.agreement,coordinate_error=chk.error,role),ref))
                        S.save_context(joinpath(folder,"$(axis)_$(branch.direction)_$k"),chk.ctx)
                    end
                    push!(connections,(;seed=seed.seed,axis,direction=branch.direction,index=k,parameter=point.parameter,E=point.state[1],I=point.state[2],
                        role,validated=validate,stability=string(point.stability.classification)))
                    previous_role=role
                end
            end
            CSV.write(joinpath(folder,axis*"_points.csv"),points);CSV.write(joinpath(folder,axis*"_attempts.csv"),attempts)
            isempty(candidates) || CSV.write(joinpath(folder,axis*"_candidates.csv"),candidates)
            CSV.write(joinpath(folder,axis*"_connections.csv"),connections);append!(allconnections,connections)
            println("seed ",seed.seed," ",seed.id," ",axis," roles=",join(sort(unique(r.role for r in connections)),","));flush(stdout)
        end
    end
    CSV.write(joinpath(output,"checks.csv"),checks);CSV.write(joinpath(output,"stops.csv"),stops)
    CSV.write(joinpath(output,"connections.csv"),allconnections)
    write_record(joinpath(output,"summary.toml"),Dict("completed"=>true,"seeds"=>length(seeds),"checks"=>length(checks),
        "search_disagreements"=>count(r->!r.search_agreement,checks),"missing_roots"=>count(r->!r.root_found,checks),
        "refinement_disagreements"=>count(r->r.refinement=="disagrees",checks),
        "refinement_uncovered"=>count(r->r.refinement in ("not_covered","ambiguous"),checks),
        "source_sha256"=>IOChecks.digest(@__FILE__),"audit_manifest_sha256"=>IOChecks.digest(joinpath(audit,"checksums.toml")),
        "completeness"=>"CompletenessNotCertified"))
    IOChecks.write_manifest(output)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("usage: continue_equilibrium_audit.jl AUDIT NEW_OUTPUT")
    EquilibriumAuditContinuation.run(abspath.(ARGS)...)
end
