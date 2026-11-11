module AtlasEventsTests
using FailureOfInhibition2025, LinearAlgebra, Test, TOML
include(joinpath(@__DIR__,"..","scripts","atlas","core.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","events.jl"))

@testset "Nexus coefficient conventions and degeneracy rejection" begin
    # Canonical BT: x'=y, y'=mu1+mu2*y+x^2+x*y.
    Jbt=x->[0.0 1.0;2x[1]+x[2] x[1]]
    c=atlas_nexus_coefficients(Jbt,zeros(2),"bt")
    @test c["quadratic"] ≈ 1 atol=1e-10
    @test c["mixed"] ≈ 1 atol=1e-10
    @test c["jordan_residual"]<1e-12
    @test_throws ArgumentError atlas_nexus_coefficients(x->zeros(2,2),zeros(2),"bt")
    # Scalar cusp suspended by y'=-y.
    cusp=x->[3x[1]^2 0.0;0.0 -1.0]
    c=atlas_nexus_coefficients(cusp,zeros(2),"cusp")
    @test abs(c["quadratic"])<1e-12
    @test c["cubic"] ≈ 1 atol=1e-8
    # F1=20*x^5 has zero exact cubic term. Finite differences must not turn
    # the discretization error into a numerically nondegenerate cusp.
    quintic=x->[100x[1]^4 0.0;0.0 -1.0]
    coarse=atlas_nexus_coefficients(quintic,zeros(2),"cusp")["cubic"]
    fine=atlas_nexus_coefficients(quintic,zeros(2),"cusp";step=5e-4)["cubic"]
    @test !atlas_coefficient_resolved(coarse,fine)
    @test !atlas_coefficient_resolved(1e-4,-1e-4)
    @test !atlas_coefficient_resolved(1e-6,1e-6)
    @test_throws ArgumentError atlas_nexus_coefficients(Jbt,zeros(2),"cusp")
    # Generic four-equation cusp and BT unfolding must be transverse.
    fbt=z->[z[2],z[3]+z[4]*z[2]+z[1]^2+z[1]*z[2],z[4]+z[1],-2z[1]-z[2]]
    fcusp=z->[z[1]^3+z[3]+z[4]*z[1],-z[2],-(3z[1]^2+z[4]),3z[1]]
    @test minimum(svdvals(atlas_difference(fbt,zeros(4))))>0.1
    @test minimum(svdvals(atlas_difference(fcusp,zeros(4))))>0.1
    @test minimum(svdvals(atlas_difference(z->[z[1],z[2],z[3],z[3]],zeros(4))))<1e-10
    root,ok,attempts=atlas_newton(z->[z[1]^2-2,z[2]-3],[1.0,1.0],z->all(isfinite,z))
    @test ok
    @test root ≈ [sqrt(2),3] atol=1e-8
    @test !isempty(attempts)
    _,ok,_=atlas_newton(z->ones(2),ones(2),z->true)
    @test !ok
end

@testset "History independent BT and local section evidence" begin
    result=atlas_nexus_job(analytic_seed(),Dict{String,Any}())
    @test result["status"]=="numerically_validated"
    @test result["classification"]=="bt"
    @test result["event_residual"]<1e-8
    @test result["unfolding_smallest_singular_value"]>1e-7
    @test result["completeness"]=="CompletenessNotCertified"
    @test length(result["followups"])==1
    cuspseed=merge(equilibrium_chart(8.0,0.0,0.0,0.0,-log(2.0),0.0,log(9.0)),
        Dict("kind"=>"nexus","event_type"=>"cusp"))
    cuspevidence=atlas_nexus_job(cuspseed,Dict{String,Any}())
    @test cuspevidence["status"]=="numerically_validated"
    @test cuspevidence["classification"]=="cusp"
    @test cuspevidence["event_residual"]<1e-8
    cusptraces,_=atlas_section_traces(cuspevidence["parameters"],cuspevidence["state"],
        cuspevidence["parameter_pair"],Dict("section_curve_max_steps"=>1))
    @test all(t->t["termination"]=="initial_point_not_on_event_locus",
        filter(t->t["event"]=="hopf",cusptraces))
    @test all(t->isempty(t["points"]),filter(t->t["event"]=="hopf",cusptraces))
    section=atlas_section_job(first(result["followups"]),Dict("section_angles"=>4,"section_curve_max_steps"=>2))
    @test length(section["section_samples"])==12
    @test length(section["transition_curves"])==4
    @test all(row->haskey(row,"status"),section["section_samples"])
    @test all(curve->haskey(curve,"termination"),section["transition_curves"])
    io=IOBuffer();TOML.print(io,section)
    @test !isempty(take!(io))
end

@testset "Hopf coefficients and stable/unstable periodic branches" begin
    # x'=-y+c*x*(x^2+y^2), y'=x+c*y*(x^2+y^2).
    for c in (-1.0,1.0)
        J=x->[c*(3x[1]^2+x[2]^2) -1+2c*x[1]*x[2];
            1+2c*x[1]*x[2] c*(x[1]^2+3x[2]^2)]
        l1,omega,q=atlas_hopf_coefficient(J,zeros(2))
        @test sign(l1)==sign(c)
        @test omega ≈ 1
        @test norm(q) ≈ 1
    end
    @test_throws ArgumentError atlas_hopf_coefficient(x->[-1.0 0.0;0.0 -2.0],zeros(2))
    fixture=atlas_periodic_fixture(Dict("periodic_max_steps"=>1))
    @test fixture["origin"]=="synthetic_normal_form"
    @test length(fixture["fixtures"])==2
    @test Set(row["stability"] for row in fixture["fixtures"])==Set(["PeriodicOrbitAttracting","PeriodicOrbitRepelling"])
    @test all(row->row["validation"]=="NumericallyValidatedPeriodicOrbit",fixture["fixtures"])
    @test all(row->length(row["branches"])==2,fixture["fixtures"])
end
end
