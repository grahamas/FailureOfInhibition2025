using LinearAlgebra
include(joinpath(@__DIR__,"..","scripts","run_input_response_study.jl"))
@testset "Two-input chart and response contracts" begin
    A=InputResponseStudy;M=A.M
    cfg=A.load_config(joinpath(@__DIR__,"..","experiments","input_response.toml");smoke=true)
    p=A.anchors()[6]
    m=M.model(p)
    @test M.model(p,.3,.7).drive.baseline==(.3,.7)
    for bad in (-1.,true,NaN,Inf)
        @test_throws ArgumentError M.model(p,0.,bad)
    end
    for (u,v) in ((2.,6.),(3.,8.),(8.,9.),(12.,15.))
        x=M.chart(m,u,v)
        if minimum(x.inputs)>=0
            b=zeros(2);point_balance!(b,x.state,M.model(p,x.inputs...),0.)
            @test maximum(abs.(b))<1e-12
        end
        h=1e-5
        numeric=hcat((M.chart(m,u+h,v).inputs-M.chart(m,u-h,v).inputs)/(2h),
            (M.chart(m,u,v+h).inputs-M.chart(m,u,v-h).inputs)/(2h))
        @test x.derivative≈numeric atol=1e-7
        f=response(m.excitatory.response,u);g=response(m.inhibitory.response,v)
        @test LinearAlgebra.det(x.balance_jacobian)≈(1+f)*(1+g)*LinearAlgebra.det(x.derivative) atol=1e-10
        y=M.chart(M.model(merge(p,Dict("tau_ratio"=>.4))),u,v)
        @test y.state==x.state && y.inputs==x.inputs
        @test y.jacobian[1,:]==x.jacobian[1,:]
        @test y.jacobian[2,:]≈x.jacobian[2,:]/2
    end
    wide=M.input_bounds(M.model(merge(p,Dict("theta_off"=>20.))))
    @test wide.B_I>16 && wide.I_tail_resolved
    @test wide.I_tail_value<=1e-6
    blocked=M.input_bounds(M.model(merge(p,Dict("theta_off"=>20.)));max_expansions=0)
    @test !blocked.I_tail_resolved && blocked.B_I==16
    @test_throws ArgumentError M.input_bounds(m;epsilon=1.)
    for kind in ("fold","hopf")
        rows=vcat([M.critical_states(m,v,kind) for v in 2.:.1:10.]...)
        @test !isempty(rows)
        if kind=="fold"
            @test maximum(abs(r.determinant) for r in rows)<1e-8
        else
            @test maximum(abs(r.trace) for r in rows)<1e-8
            @test all(r->r.determinant>0,rows)
        end
    end
    f(x,y)=.2<x<.4 && .2<y<.4 ? "island" : "outside"
    sample=M.sample_rectangle(f,[0.,.25,.5,1.],[0.,.25,.5,1.];width=.01,max_evaluations=1000)
    @test "island" in values(sample.cache)
    limited=M.sample_rectangle(f,[0.,.5,1.],[0.,.5,1.];width=.01,max_evaluations=5)
    @test any(x->x.status=="budget_unresolved",limited.leaves)
    # Equal corner signatures on opposite sides of a mixed cell do not
    # establish a connected observed region.
    fake(role)=(;search=(equilibria=[(;stability=(classification=:Attracting,))],),
        roles=Dict(role=>1),discovery_status="index_consistent_not_complete")
    contexts=Dict{Tuple{Float64,Float64},Any}()
    for x in (0.,.5,1.,2.,2.5,3.), y in (0.,.5,1.)
        contexts[(x,y)]=fake("A")
    end
    contexts[(1.5,.5)]=fake("B")
    leaves=[(;x0=0.,x1=1.,y0=0.,y1=1.,status="sampled_homogeneous"),
        (;x0=1.,x1=2.,y0=0.,y1=1.,status="boundary_bracket"),
        (;x0=2.,x1=3.,y0=0.,y1=1.,status="sampled_homogeneous")]
    representatives=A.observed_representatives((;leaves),contexts,NamedTuple[],
        (;B_E=3.,B_I=1.))
    @test length(representatives)==2
    @test any(x->x[1]<=1.,representatives)
    @test any(x->x[1]>=2.,representatives)
    costs=[(E_withdrawal=1.,I_stimulation=0.),(E_withdrawal=0.,I_stimulation=1.),
        (E_withdrawal=1.,I_stimulation=1.)]
    @test length(M.nondominated(costs))==2

    ctx=A.context(p,.35,0.,cfg;grid=21)
    @test length(ctx.search.equilibria)==7
    @test all(k->haskey(ctx.roles,k),("rest","active","herald","seizure"))
    low=A.context(p,.1,0.,cfg;grid=21)
    for source in ("herald","seizure")
        initial=ctx.search.equilibria[ctx.roles[source]].state
        result=A.observe(low,initial,cfg;reference=ctx)
        @test result.status=="compatible"
        @test result.destination==(source=="herald" ? "active" : "seizure")
    end
    old=A.context(merge(p,Dict("e_to_e"=>16.)),.5,0.,cfg;grid=21)
    new=A.context(merge(p,Dict("e_to_e"=>16.,"theta_off"=>12.)),.5,0.,cfg;grid=21)
    result=A.observe(new,old.search.equilibria[old.roles["seizure"]].state,cfg;reference=old)
    @test result.destination=="active"
    high=A.context(p,.35,16.,cfg)
    @test all(r->r.state[2]<1e-6,high.search.equilibria)
    @test all(isfinite, M.susceptibility(ctx.model,ctx.search.equilibria[ctx.roles["active"]].state).gain)
    @test isempty(M.correspondence(ctx.search,(equilibria=[ctx.search.equilibria[1],ctx.search.equilibria[1]],)))
    unchanged=A.pulse(ctx,ctx.search.equilibria[ctx.roles["active"]].state,.35,0.,10.,cfg)
    @test unchanged.destination=="active"
    # A saturated inhibitory tail can produce a tiny negative endpoint under
    # a permissive domain tolerance; a subsequent phase requires an exact
    # in-domain initial state. Domain rejection must preserve the actual state.
    tail=A.context(A.anchors()[3],.25,9.,cfg;grid=21)
    tailpulse=A.pulse(tail,tail.search.equilibria[tail.roles["seizure"]].state,.25,6.,10.,cfg;retain=true)
    @test minimum(last(tailpulse.pulse_solution.u))>=0
    @test tailpulse.status=="compatible"
    cyclep=merge(A.anchors()[5],Dict("tau_ratio"=>.5))
    cyclectx=A.context(cyclep,0.,0.,cfg;grid=21)
    candidate=A.periodic_candidate(cyclectx,[.25,.25],cfg)
    @test candidate.orbit!==nothing
    @test candidate.orbit.validation==NumericallyValidatedPeriodicOrbit
    @test candidate.orbit.stability==PeriodicOrbitAttracting
    @test candidate.orbit.period≈5.978735418864944 atol=1e-5
    @test length(periodic_orbit_phases(candidate.orbit,collect(0:7)./8))==8
    early=A.observe(cyclectx,[.25,.25],cfg;cycles=true)
    late=A.observe(cyclectx,[.25,.25],merge(cfg,(;horizons=[20000.]));cycles=true)
    @test early.status==late.status=="periodic_compatible"
    @test early.phase.horizon==first(cfg.horizons)
    @test early.orbit.period≈late.orbit.period atol=1e-5
    # Resume identity protects both source and configuration, including smoke mode.
    mktempdir() do dir
        path=joinpath(@__DIR__,"..","experiments","input_response.toml")
        metadata=A.initialize(path,cfg,dir)
        @test occursin("two-input response",metadata["purpose"])
        @test occursin("run_input_response_study.jl",metadata["replay_from_artifact_directory"])
        @test endswith(metadata["replay_from_artifact_directory"]," --smoke")
        @test A.initialize(path,cfg,dir)["smoke"]
        @test_throws ArgumentError A.initialize(path,A.load_config(path),dir)
    end
    mktempdir() do dir
        path=joinpath(@__DIR__,"..","experiments","input_response.toml")
        metadata=A.initialize(path,A.load_config(path),dir)
        @test !occursin("--smoke",metadata["replay_from_artifact_directory"])
    end
end
