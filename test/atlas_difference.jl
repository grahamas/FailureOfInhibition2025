module AtlasDifferenceTests
using FailureOfInhibition2025, LinearAlgebra, Test, TOML, SHA
include(joinpath(@__DIR__,"..","scripts","atlas","core.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","events.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","science.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","pilot.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","targeted.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","targeted_events.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","campaign_science.jl"))

@testset "Admissible centered and one-sided corrector derivatives" begin
    domain=x->all(0 .<= x .<= 1)
    calls=Vector{Float64}[]
    f=x->begin
        @test domain(x)
        push!(calls,copy(x))
        [x[1]^2+3x[2]^2,x[1]*x[2]+2x[2]]
    end
    for x in ([0.,0.],[1.,1.],[0.,1.],[1.,0.],[.4,.6])
        stencils=Dict{String,Any}[]
        result=atlas_difference(f,x;admissible=domain,diagnostics=stencils)
        @test result≈[2x[1] 6x[2];x[2] x[1]+2] atol=2e-10 rtol=2e-10
        @test length(stencils)==2
        @test all(row->row["halvings"]==0,stencils)
        @test all(row->row["scheme"]==(x[row["coordinate"]]==0 ? "forward_second_order" :
            x[row["coordinate"]]==1 ? "backward_second_order" : "centered_second_order"),stencils)
    end
    @test all(domain,calls)
    # Interior arithmetic remains identical to the original centered stencil.
    x=[.4,.6];unconstrained=z->[exp(z[1])+z[2]^3,z[1]^2-z[2]]
    expected=hcat(((unconstrained(x+1e-5*[j==1,j==2])-unconstrained(x-1e-5*[j==1,j==2]))/2e-5 for j in 1:2)...)
    @test atlas_difference(unconstrained,x;admissible=domain)==expected
    # Cubic forward/backward truncation error scales as h^2.
    for x in ([0.],[1.])
        true_derivative=3x[1]^2
        e1=abs(only(atlas_difference(z->[z[1]^3],x;step=.01,admissible=domain))-true_derivative)
        e2=abs(only(atlas_difference(z->[z[1]^3],x;step=.005,admissible=domain))-true_derivative)
        @test e1/e2≈4 rtol=1e-6
    end
    narrow=x->0<=x[1]<=1e-8
    stencils=Dict{String,Any}[]
    @test only(atlas_difference(x->[2x[1]+x[1]^2],[0.];admissible=narrow,diagnostics=stencils))≈2 atol=1e-12
    @test only(stencils)["halvings"]>0
    # Represented offsets around 1.0 may have unequal one/two-step spacing.
    tiny=x->1<=x[1]<=nextfloat(1.,4)
    @test only(atlas_difference(x->[x[1]^2],[1.];step=1.3eps(),admissible=tiny))≈2 atol=1e-14
    singleton=x->x==[0.]
    @test_throws AtlasDifferenceUnresolved atlas_difference(x->[x[1]+1],[0.];admissible=singleton)
    coordinates,ok,attempts=atlas_newton(x->[x[1]+1],[0.],singleton)
    @test !ok && coordinates==[0.]
    @test last(attempts)["derivative_status"]=="unresolved"
    @test last(attempts)["derivative_reason"]=="no_admissible_finite_representable_derivative_stencil"
    @test last(attempts)["derivative_coordinate"]==1
    @test_throws ErrorException atlas_difference(x->error("implementation defect"),[0.];admissible=domain)
end

const BOUNDARY_CASES=TOML.parsefile(joinpath(@__DIR__,"fixtures","atlas_boundary_paths.toml"))
fixture_point(record)=Dict{String,Any}("id"=>record["id"],"signature"=>record["signature"],
    "parameters"=>record["parameters"],"roots"=>[Dict("state"=>x) for x in record["states"]],"attempts"=>Any[])
@testset "All nine frozen failed path endpoint correctors" begin
    cases=BOUNDARY_CASES["cases"]
    @test length(cases)==9
    @test Set(row["job_id"] for row in cases)==Set("job-"*lpad(string(i),6,'0') for i in (773,774,777,778,781,782,785,786,789))
    for case in cases
        # In local audits verify the immutable original result binding too;
        # standalone checkouts retain the exact compact numerical inputs.
        source=joinpath(BOUNDARY_CASES["source_packet"],case["source_result"])
        if isfile(source)
            @test bytes2hex(sha256(read(source)))==case["source_result_sha256"]
        end
        p0,p1=case["from"]["parameters"],case["to"]["parameters"]
        for (endpoint,name) in ((0.,"from"),(1.,"to")),kind in ("fold","trace_zero")
            seed=vcat(case[name]["states"][1],endpoint)
            bounds=endpoint==0 ? (0.,1/128) : (127/128,1.)
            # This reproduces the former unguarded derivative failure exactly.
            @test_throws ArgumentError atlas_difference(z->targeted_event_residual(p0,p1,z,kind),seed)
            result=targeted_corrector(p0,p1,seed,kind,bounds)
            @test result["converged"] isa Bool
            @test all(isfinite,result["coordinates"])
            @test bounds[1]<=result["coordinates"][3]<=bounds[2]
            @test result["balance_atol"]==1e-11 && result["event_atol"]==1e-9
            if result["converged"]
                @test result["balance_residual"]<=1e-11
                @test result["event_condition_residual"]<=1e-9
            end
            @test any(haskey(row,"derivative_stencils") for row in result["attempts"])
        end
    end
end

@testset "Actual failed job 773 bounded path replay" begin
    case=first(BOUNDARY_CASES["cases"])
    job=Dict{String,Any}("id"=>case["job_id"],"kind"=>"campaign_path",
        "from"=>fixture_point(case["from"]),"to"=>fixture_point(case["to"]))
    result=execute_campaign_path(job,Dict{String,Any}())
    @test result["status"]=="completed"
    evidence=only(result["evidence"]["campaign_paths"])
    @test evidence["corrector_count"]<=64
    @test evidence["inventory_location_count"]<=256
    @test !isempty(result["evidence"]["paths"])
    @test !any(event->event["validated"],result["evidence"]["events"])
    @test result["evidence"]["paths"][1]["continuous_coverage_certified"]===false
end

@testset "Boundary domains reach every nexus and section derivative caller" begin
    cusp=merge(equilibrium_chart(8.,0.,0.,0.,-log(2.),0.,log(9.)),
        Dict("kind"=>"nexus","event_type"=>"cusp"))
    calls=Ref(0)
    function guarded(f,z;kwargs...)
        calls[]+=1
        @test haskey(kwargs,:admissible)
        @test kwargs[:admissible](z)
        atlas_difference(f,z;kwargs...)
    end
    result=atlas_nexus_job(cusp,Dict{String,Any}();differentiate=guarded)
    @test result["validated"]
    @test calls[]==29 # 28 candidate pairs, then the final conditioning system.
    selection=first(result["attempts"])["pair_selection_attempts"]
    @test length(selection)==28
    # Zero coupling parameters are now scored with admissible one-sided probes.
    @test any(row->row["status"]=="scored"&&any(stencil->stencil["scheme"]=="forward_second_order",row["derivative_stencils"]),selection)
    @test all(haskey(row,"derivative_stencils") for row in selection)
    @test length(result["conditioning"]["derivative_stencils"])==4
    @test !haskey(result,"pair_scoring") # unsuccessful selection alternatives are solver attempts.
    no_stencil(f,z;kwargs...)=throw(AtlasDifferenceUnresolved(3,"injected_no_admissible_stencil"))
    skipped=atlas_nexus_job(cusp,Dict{String,Any}();differentiate=no_stencil)
    @test skipped["status"]=="unresolved" && !skipped["validated"]
    @test skipped["reason"]=="no_admissible_unfolding_pair"
    @test length(first(skipped["attempts"])["pair_selection_attempts"])==28
    @test isempty(skipped["followups"])
    calls[]=0
    function fail_conditioning(f,z;kwargs...)
        calls[]+=1
        calls[]==29 && throw(AtlasDifferenceUnresolved(3,"injected_conditioning_stencil_failure"))
        atlas_difference(f,z;kwargs...)
    end
    candidate=atlas_nexus_job(cusp,Dict{String,Any}();differentiate=fail_conditioning)
    @test candidate["status"]=="unresolved" && !candidate["validated"]
    @test candidate["reason"]=="unresolved_conditioning_derivative_stencil"
    @test candidate["conditioning"]["reason"]=="injected_conditioning_stencil_failure"
    @test candidate["state"]==cusp["state"]
    @test candidate["parameters"]==cusp["params"]
    pair=candidate["parameter_pair"]
    @test vcat(candidate["state"],candidate["parameters"][pair[1]],candidate["parameters"][pair[2]])==candidate["converged_coordinates"]
    @test isempty(candidate["followups"])
    programmer_error(f,z;kwargs...)=throw(ArgumentError("unexpected callback defect"))
    @test_throws ArgumentError atlas_nexus_job(cusp,Dict{String,Any}();differentiate=programmer_error)

    # A genuine fold with B=0 lies on the section parameter-domain boundary.
    e=(1-sqrt(1-8/12))/4
    p=Dict{String,Any}("A"=>12.,"B"=>0.,"C"=>0.,"D"=>0.,
        "a"=>12*e-log(e/(1-2*e)),"b"=>0.,"h"=>log(9.),"r"=>1.)
    center=[e,2/7];section_pair=["B","a"]
    calls[]=0
    traces,_=atlas_section_traces(p,center,section_pair,Dict("section_curve_max_steps"=>1);differentiate=guarded)
    @test calls[]==2 # Both fold directions; trace-zero starts off the locus.
    folds=filter(row->row["event"]=="fold",traces)
    @test all(!isempty(row["attempts"]) for row in folds)
    @test all(any(stencil->stencil["scheme"]=="forward_second_order",
        first(row["attempts"])["tangent_derivative_stencils"]) for row in folds)
    @test !any(row->row["termination"]=="unresolved_derivative_stencil",folds)
    rejected,_=atlas_section_traces(p,center,section_pair,Dict("section_curve_max_steps"=>1);differentiate=no_stencil)
    @test all(row["termination"]=="unresolved_derivative_stencil" for row in filter(row->row["event"]=="fold",rejected))
    @test all(first(row["attempts"])["reason"]=="injected_no_admissible_stencil" for row in filter(row->row["event"]=="fold",rejected))
    @test_throws ArgumentError atlas_section_traces(p,center,section_pair,Dict("section_curve_max_steps"=>1);differentiate=programmer_error)
    outside,_=atlas_section_traces(p,[-.01,center[2]],section_pair,Dict("section_curve_max_steps"=>1);differentiate=programmer_error)
    @test all(row["termination"]=="unresolved_initial_point_outside_admissible_domain" for row in outside)
end

end
