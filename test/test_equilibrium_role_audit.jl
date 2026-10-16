using Test
include(joinpath(@__DIR__,"..","scripts","run_equilibrium_audit.jl"))
include(joinpath(@__DIR__,"..","scripts","continue_equilibrium_audit.jl"))
@testset "Equilibrium role audit preserves contextual classification" begin
    A=EquilibriumAuditStudy.A
    p=Dict("e_to_e"=>4.,"i_to_e"=>9.,"e_to_i"=>19.,"i_to_i"=>4.,"tau_ratio"=>4.4,"theta_off"=>8.)
    record(e,i;stability="Attracting",singular=false)=Dict("state"=>[e,i],"near_singular"=>singular,
        "balance_residual"=>[0.,0.],"stability"=>Dict("classification"=>stability))
    m=A.M.model(p,2.,1.)
    # Actual cycle-coexisting sink: its geometry does not supply a comparison sink.
    rec=[record(.49999906816480066,3.727233130807358e-6)]
    rows,roles=A.explain(m,rec)
    @test isempty(roles)
    @test only(rows).descriptor=="upper/descending"
    @test occursin("missing_higher_I_sink",only(rows).reasons)
    @test EquilibriumAuditContinuation.independent_checks(p,2.,1.,rec[1]["state"]).passed
    factory(be)=A.M.model(p,be,1.)
    branch=FailureOfInhibition2025.continue_equilibria(factory,rec[1]["state"],2.;parameter_bounds=(1.9,2.1))
    check=EquilibriumAuditContinuation.refinement_error(first(branch.positive.points),branch,factory)
    @test check.refinement=="agrees"
    @test check.refinement_error<1e-7
    # Adding a higher-I sink changes contextual naming, not the first root's geometry.
    push!(rec,record(.49,.4))
    rows,roles=A.explain(m,rec)
    @test roles["seizure"]==1
    @test roles["herald"]==2
    @test rows[1].descriptor=="upper/descending"
    @test rows[1].reasons==""
    rows,_=A.explain(m,[record(.18,.16;stability="Repelling"),record(.42,.22;stability="Saddle"),record(.5,0.;singular=true)])
    @test rows[1].reasons=="non_attracting"
    @test rows[2].reasons=="non_attracting"
    @test rows[3].reasons=="near_singular"
    flat=A.M.model(merge(p,Dict("e_to_e"=>0.)),2.,1.)
    rows,roles=A.explain(flat,[record(.001,.001)])
    @test isempty(roles)
    @test only(rows).reasons=="no_E_turning_points"
    rows,_=A.explain(m,[record(.001,.001),record(.002,.002)])
    @test all(r->r.reasons=="multiple_rest_candidates",rows)
    rows,_=A.explain(m,[record(.35,.001)])
    @test occursin("middle_without_positive_I_slope",only(rows).reasons)
    @test isempty(A.retained_context("status = \"completed\"\n")["equilibria"])
    @test A.retained_context("[[attempts]]\nlarge = 4\n[[equilibria]]\nstate = [0.0,0.0]\n")["equilibria"][1]["state"]==[0.,0.]
    search(xs)=(;equilibria=[(;state=x) for x in xs])
    old=search([[.1,.2]]);new=search([[.11,.2]])
    @test A.destination_label(new,Dict("active"=>1),1;reference_search=old,reference_roles=Dict("rest"=>1)).origin=="local"
    matched=A.destination_label(new,Dict{String,Int}(),1;reference_search=old,reference_roles=Dict("active"=>1))
    @test matched.label=="active" && matched.origin=="reference_coordinate_match"
    tied=search([[.09,.2],[.11,.2]])
    @test A.destination_label(tied,Dict{String,Int}(),1;reference_search=old,reference_roles=Dict("active"=>1)).label=="equilibrium_1"
    collision=search([[.1,.2],[.12,.2]])
    @test isempty(A.M.correspondence(collision,new))
    @test A.destination_label(search([[.4,.4]]),Dict{String,Int}(),1;reference_search=old,reference_roles=Dict("active"=>1)).origin=="numbered_local_root"
    row(k;scope="screen",cycles=0,reason="missing_higher_I_sink")=(;context="context_$k",root=1,id="case",B_E=Float64(k),B_I=0.,E=.5,I=0.,
        role="unassigned",stability="Attracting",near_singular=false,scope,cycles,axis="",value=NaN,reasons=reason,descriptor="upper/descending")
    selected,uncovered=EquilibriumAuditStudy.select_seeds([row(1),row(2;scope="baseline",cycles=1),row(3;reason="no_E_turning_points")];limit=1)
    @test only(selected).context=="context_2"
    @test length(uncovered)==1
end
