module AtlasNormalFormRejectionTests
using FailureOfInhibition2025, LinearAlgebra, Test, TOML
include(joinpath(@__DIR__,"..","scripts","atlas","core.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","events.jl"))

rejection(kind="cusp",reason="fixture_expected_rejection")=AtlasNormalFormUnresolved(kind,reason,Dict{String,Any}("state"=>[0.,0.]))
roundtrip(value)=TOML.parse(sprint(io->TOML.print(io,value)))
function cusp_job()
    merge(equilibrium_chart(8.,0.,0.,0.,-log(2.),0.,log(9.)),Dict("event_type"=>"cusp"))
end
function unresolved_candidate(value)
    @test value["status"]=="unresolved"
    @test !value["validated"]
    @test isempty(value["followups"])
    @test value["completeness"]=="CompletenessNotCertified"
    @test roundtrip(value)["status"]=="unresolved"
end

@testset "Expected numerical normal-form rejections retain their type" begin
    nilpotent=x->[0. 1.;0. 0.]
    for (jac,kind,reason) in ((nilpotent,"cusp","zero_eigenvalue_not_simple"),
            (x->zeros(2,2),"bt","double_zero_jordan_chain_unresolved"),
            (x->zeros(2,2),"cusp","singular_cusp_bordered_system"),
            (x->[1. 0.;0. 2.],"bt","singular_bt_dual_basis"))
        err=try atlas_nexus_coefficients(jac,zeros(2),kind);nothing catch error;error end
        @test err isa AtlasNormalFormUnresolved
        @test err.kind==kind
        @test err.reason==reason
        @test roundtrip(atlas_normal_form_evidence(err))["reason"]==reason
    end
    err=try atlas_hopf_coefficient(x->[-1. 0.;0. -2.],zeros(2));nothing catch error;error end
    @test err isa AtlasNormalFormUnresolved
    @test err.kind=="hopf" && err.reason=="hopf_spectrum_unresolved"
    @test_throws ArgumentError atlas_nexus_coefficients(x->error("must not evaluate"),zeros(2),"unknown")
    @test_throws ArgumentError atlas_nexus_coefficients(x->throw(ArgumentError("jacobian defect")),zeros(2),"cusp")
    # Only the bordered solve itself converts SingularException. A tensor
    # callback exception while constructing its RHS must propagate unchanged.
    calls=Ref(0)
    B=(u,v)->begin calls[]+=1;calls[]==2 && throw(SingularException(7));zeros(2) end
    C=(u,v,w)->zeros(2)
    @test_throws SingularException atlas_nexus_coefficients(x->[0. 0.;0. -1.],zeros(2),"cusp";tensors=(B,C))
end

@testset "Actual degenerate candidates preserve unresolved evidence" begin
    # The exact PR163 regression: a BT point cannot use the simple-zero cusp
    # normal form, although it is a valid reduced model and equilibrium.
    wrong_kind=merge(analytic_seed(),Dict("event_type"=>"cusp"))
    result=atlas_nexus_job(wrong_kind,Dict{String,Any}())
    unresolved_candidate(result)
    @test result["reason"]=="no_admissible_unfolding_pair"
    @test result["parameters"]==wrong_kind["params"]
    @test result["state"]==wrong_kind["state"]
    selection=first(result["attempts"])["pair_selection_attempts"]
    @test length(selection)==28
    @test all(row["status"]=="unresolved_normal_form" for row in selection)
    @test all(row["normal_form_rejection"]["reason"]=="zero_eigenvalue_not_simple" for row in selection)
    # A regular uncoupled equilibrium requested as BT has a singular dual
    # basis; numerical nonconvergence must not discard its final tuple.
    regular=equilibrium_chart(0.,0.,0.,0.,0.,0.,log(9.))
    result=atlas_nexus_job(merge(regular,Dict("event_type"=>"bt")),Dict("nexus_max_iterations"=>0))
    unresolved_candidate(result)
    @test result["rejection_phase"]=="final_coefficients"
    @test result["normal_form_rejection"]["kind"]=="bt"
    @test result["state"]==regular["state"]
    # A trace-zero BT tuple is not a resolved oscillatory Hopf candidate.
    result=atlas_periodic_job(analytic_seed(),Dict{String,Any}())
    unresolved_candidate(result)
    @test result["rejection_phase"]=="hopf_coefficients"
    @test result["normal_form_rejection"]["reason"]=="hopf_spectrum_unresolved"
end

@testset "Newton records expected center, probe and trial rejections" begin
    x,ok,attempts=atlas_newton(z->throw(rejection()),[1.],z->true)
    @test !ok && x==[1.]
    @test first(attempts)["rejection_phase"]=="corrector_center"
    @test roundtrip(Dict("attempts"=>attempts))["attempts"][1]["status"]=="unresolved_normal_form"
    calls=Ref(0)
    f=z->begin calls[]+=1;calls[]==2 && throw(rejection());[z[1]] end
    x,ok,attempts=atlas_newton(f,[1.],z->true)
    @test !ok && x==[1.]
    @test last(attempts)["rejection_phase"]=="corrector_derivative_center"
    # One unusable probe leaves an admissible one-sided stencil.
    stencils=Dict{String,Any}[]
    f=z->isapprox(z[1],1e-5;atol=1e-15,rtol=0) ? throw(rejection()) : [z[1]-1]
    @test only(atlas_difference(f,[0.];diagnostics=stencils))≈1 atol=1e-10
    @test only(stencils)["scheme"]=="backward_second_order"
    @test length(only(stencils)["normal_form_probe_rejections"])==1
    # No available probe remains a bounded unresolved derivative.
    f=z->z[1]==0 ? [-1.] : throw(rejection())
    x,ok,attempts=atlas_newton(f,[0.],z->true)
    @test !ok && x==[0.]
    @test last(attempts)["derivative_status"]=="unresolved"
    @test !isempty(last(attempts)["derivative_stencils"][1]["normal_form_probe_rejections"])
    @test !isempty(roundtrip(Dict("attempts"=>attempts))["attempts"])
    # Full Newton trial is rejected, while smaller existing reductions remain
    # available. No rejected point is accepted merely to force convergence.
    f=z->abs(z[1])<1e-10 ? throw(rejection()) : [z[1]]
    x,ok,attempts=atlas_newton(f,[1.],z->true;iterations=3)
    @test !ok && 0<x[1]<1
    @test !isempty(first(attempts)["normal_form_rejected_trials"])
    @test first(first(attempts)["normal_form_rejected_trials"])["reduction"]==0
    # The same bounded loop records all eleven unavailable trial reductions.
    f=z->z[1]<.9999 ? throw(rejection()) : [z[1]]
    x,ok,attempts=atlas_newton(f,[1.],z->true)
    @test !ok && x==[1.]
    @test length(first(attempts)["normal_form_rejected_trials"])==11
    @test_throws ArgumentError atlas_newton(z->throw(ArgumentError("callback defect")),[1.],z->true)
    @test_throws SingularException atlas_newton(z->throw(SingularException(3)),[1.],z->true)
    f=z->abs(z[1])<1e-10 ? throw(ArgumentError("trial callback defect")) : [z[1]]
    @test_throws ArgumentError atlas_newton(f,[1.],z->true)
    f=z->z[1]>0 ? throw(ArgumentError("probe callback defect")) : [-1.]
    @test_throws ArgumentError atlas_difference(f,[0.])
end

@testset "Final normal-form boundaries reject only expected numerical failures" begin
    job=cusp_job()
    calls=Ref(0)
    differentiate=(f,z;kwargs...)->begin
        calls[]+=1
        calls[]==29 && throw(rejection())
        atlas_difference(f,z;kwargs...)
    end
    result=atlas_nexus_job(job,Dict{String,Any}();differentiate=differentiate)
    unresolved_candidate(result)
    @test result["rejection_phase"]=="final_conditioning"
    @test result["state"]==job["state"]
    for rejected_call in 1:3
        calls=Ref(0)
        solver=(jac,x,kind;kwargs...)->begin
            calls[]+=1
            calls[]==rejected_call && throw(rejection())
            atlas_nexus_coefficients(jac,x,kind;kwargs...)
        end
        result=atlas_nexus_job(job,Dict{String,Any}();coefficient_solver=solver)
        unresolved_candidate(result)
        @test result["rejection_phase"]=="final_coefficients"
        @test result["state"]==job["state"]
        pair=result["parameter_pair"]
        @test vcat(result["state"],result["parameters"][pair[1]],result["parameters"][pair[2]])==result["converged_coordinates"]
    end
    complete=Ref(false);calls=Ref(0)
    solver=(jac,x,kind;kwargs...)->begin
        calls[]+=1;value=atlas_nexus_coefficients(jac,x,kind;kwargs...)
        calls[]==3 && (complete[]=true)
        value
    end
    evaluator=(p,pair,z,kind)->complete[] ? throw(rejection()) : atlas_nexus_vector_at(p,pair,z,kind)
    result=atlas_nexus_job(job,Dict{String,Any}();coefficient_solver=solver,event_evaluator=evaluator)
    unresolved_candidate(result)
    @test result["rejection_phase"]=="saved_event_conditions"
    for error in (ArgumentError("coefficient callback defect"),SingularException(4),LinearAlgebra.LAPACKException(2))
        solver=(args...;kwargs...)->throw(error)
        @test_throws typeof(error) atlas_nexus_job(job,Dict{String,Any}();coefficient_solver=solver)
        @test_throws typeof(error) atlas_nexus_job(job,Dict{String,Any}();differentiate=solver)
    end
    hp=equilibrium_chart(12.,8.,8.,0.,-log(2.),0.,log(9.);r=2.1)
    for rejected_call in 1:3
        calls=Ref(0)
        solver=(jac,x;kwargs...)->begin
            calls[]+=1
            calls[]==rejected_call && throw(rejection("hopf"))
            atlas_hopf_coefficient(jac,x;kwargs...)
        end
        result=atlas_periodic_job(hp,Dict{String,Any}();coefficient_solver=solver)
        unresolved_candidate(result)
        @test result["rejection_phase"]=="hopf_coefficients"
    end
    @test_throws ArgumentError atlas_periodic_job(hp,Dict{String,Any}();coefficient_solver=(args...;kwargs...)->throw(ArgumentError("Hopf callback defect")))
end
end
