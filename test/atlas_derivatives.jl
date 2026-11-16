module AtlasDerivativesTests
using FailureOfInhibition2025, LinearAlgebra, Test, TOML
include(joinpath(@__DIR__,"..","scripts","atlas","core.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","events.jl"))

@testset "Analytic original-time mixed state derivatives" begin
    # Mixed couplings, narrow inhibitory width, both sides of its maximum,
    # and a saturated excitatory response exercise the product jets.
    for (A,B,C,D,a,b,h,r) in ((12.,8.,8.,2.,2.,1.,log(9.),2.1),
            (128.,64.,32.,16.,30.,4.,1e-5,0.01),
            (64.,128.,128.,64.,-25.,-16.,32.,100.)), x in ([.17,.31],[.49,.01])
        p=Dict(zip(ATLAS_COORDINATES,(A,B,C,D,a,b,h,r)))
        H,C3=atlas_reduced_tensor_arrays(p,x)
        Hr,Cr=atlas_reference_tensor_arrays(p,x)
        Hf,Cf=atlas_reference_tensor_arrays(p,x;step=ATLAS_DERIVATIVE_REFERENCE_STEP/2)
        for (actual,reference,refined) in ((H,Hr,Hf),(C3,Cr,Cf))
            @test maximum(abs.(BigFloat.(actual).-refined)) <= 2e-12*max(1,maximum(abs,refined))
            @test maximum(abs.(reference.-refined)) <= big"1e-23"*max(1,maximum(abs,refined))
        end
        @test H≈permutedims(H,(1,3,2)) rtol=1e-14
        @test C3≈permutedims(C3,(1,3,2,4)) rtol=1e-14
        @test C3≈permutedims(C3,(1,4,3,2)) rtol=1e-14
        p2=copy(p);p2["r"]=2r
        H2,C2=atlas_reduced_tensor_arrays(p2,x)
        @test H2[1,:,:]==H[1,:,:]
        @test H2[2,:,:]==H[2,:,:]/2
        @test C2[2,:,:,:]==C3[2,:,:,:]/2
        # Package kernel equality is an arithmetic-contract check, distinct
        # from the independent derivative stencil comparison.
        @test atlas_rhs(p,x)≈Float64.(atlas_reference_rhs(
            merge(Dict(k=>BigFloat(p[k]) for k in ATLAS_COORDINATES),
                Dict("_failure_threshold"=>BigFloat(p["b"]+p["h"]))),BigFloat.(x))) atol=2e-14 rtol=2e-13
    end
    p=analytic_seed()["params"]
    bad=merge(p,Dict("b"=>1e20,"h"=>1.0))
    @test_throws ArgumentError reduced_model(bad)
    @test_throws ArgumentError atlas_reduced_tensors(bad,[.2,.3])
    @test_throws ArgumentError atlas_reference_tensors(bad,[.2,.3])
    @test_throws ArgumentError atlas_reduced_tensors(p,[.2])
end

@testset "Analytic cusp correction and original-time BT/Hopf conventions" begin
    config=Dict{String,Any}("balance_atol"=>1e-11,"event_atol"=>1e-9)
    cusp=merge(equilibrium_chart(8.,0.,0.,0.,-log(2.),0.,log(9.)),
        Dict("kind"=>"nexus","event_type"=>"cusp"))
    cusp["state"] += [.003,-.002]
    corrected=atlas_nexus_job(cusp,config)
    @test corrected["validated"]
    @test corrected["state"][1]≈.25 atol=1e-10
    @test corrected["analytic_degeneracy_residual"]<=1e-9
    @test corrected["refined_degeneracy_residual"]<=1e-9
    @test corrected["numerical_policy_version"]==ATLAS_NUMERICAL_POLICY_VERSION
    @test haskey(corrected["conditioning"],"linearized_scaled_coordinate_residual_estimate")
    for job in (analytic_seed(),)
        result=atlas_nexus_job(job,config)
        @test result["validated"]
        @test all(values(result["coefficient_refinement_checks"]))
        @test result["reference_coefficient_refinement_difference"]<1e-10
    end
    hp=equilibrium_chart(12.,8.,8.,0.,-log(2.),0.,log(9.);r=2.1)
    p,x=hp["params"],hp["state"]
    l1,omega,_=atlas_hopf_coefficient(y->atlas_jac(p,y),x;tensors=atlas_reduced_tensors(p,x))
    lr,wr,_=atlas_hopf_coefficient(y->atlas_jac(p,y),x;tensors=atlas_reference_tensors(p,x))
    @test l1≈lr rtol=2e-12
    @test omega==wr
    @test l1<0
    # Polynomial callback fixtures retain their original finite-difference path.
    J=x->[3x[1]^2 0.;0. -1.]
    @test atlas_nexus_coefficients(J,zeros(2),"cusp")["cubic"]≈1 atol=1e-8
    almost_double=atlas_nexus_coefficients(x->[0. 0.;0. -1e-10],zeros(2),"cusp")
    @test almost_double["simple_pairing"]==1
    @test almost_double["transverse_eigenvalue_magnitude"]<1e-7
    @test !atlas_cusp_simple_zero_resolved(almost_double,1e-9)
    @test atlas_cusp_simple_zero_resolved(atlas_nexus_coefficients(J,zeros(2),"cusp"),1e-9)
    @test !atlas_coefficient_resolved(1e-6,1e-6)
    @test !atlas_coefficient_resolved(1.,-1.)
    @test_throws ArgumentError atlas_nexus_coefficients(x->zeros(2,2),zeros(2),"bt")
end
end
