using Test
include(joinpath(@__DIR__,"..","scripts","run_figure_details.jl"))
include(joinpath(@__DIR__,"..","scripts","verify_figure_redesign.jl"))
@testset "Figure redesign study contract" begin
    cfg=FigureDetails.S.load_config(joinpath(@__DIR__,"..","experiments","figure_redesign.toml"))
    cases=FigureDetails.F.registry(cfg)
    @test length(cases)==232
    @test count(p->p["category"]=="anchors",cases)==8
    @test count(p->p["category"]=="expansion",cases)==224
    @test cfg.raw["redesign"]["detailed_limit"]==8
    @test all(0<=p["e_to_e"]<=24 && 12<=p["e_to_i"]<=28 &&
        6<=p["theta_off"]<=12 && .2<=p["tau_ratio"]<=4.4 for p in cases)
    row(a,d;status="compatible")=(;axis="E_increase",amplitude=a,duration=100.,status,destination=d)
    rows=[row(0.,"rest"),row(.25,"active"),row(.5,"active"),row(.75,"herald"),row(1.,"active")]
    witness=FigureDetails.interior_pulse(rows,"active")
    @test witness.amplitude==.375
    @test witness.width==.25
    @test FigureDetails.interior_pulse(rows,"seizure")===nothing
    @test FigureDetails.interior_pulse([row(0.,"active"),row(.25,"active";status="unresolved"),row(.5,"active")],"active")===nothing
    reference=only(filter(p->p["id"]=="selective_withdrawal",cases))
    weak=only(filter(p->p["id"]=="input_dependent",cases))
    @test [2.,1.] in FigureDetails.target_baselines(weak)
    @test FigureDetails.target_baselines(reference)==reference["baselines"]
    for state in ([0.,0.],[.3,.32],[.5,0.],[1.,1.]), input in ((.35,0.),(0.,4.),(8.,1.))
        model=FigureDetails.S.M.model(reference,input...)
        package=zeros(2);independent=zeros(2)
        FigureDetails.point_rhs!(package,state,model,0.)
        VerifyFigureRedesign.independent_rhs!(independent,state,(reference,input...),0.)
        @test package≈independent atol=1e-12 rtol=1e-12
    end
    mktempdir() do dir
        ctx=FigureDetails.confirmed_context(reference,[.35,0.],cfg,dir)
        @test all(haskey(ctx.roles,r) for r in ("rest","active","herald","seizure"))
        @test FigureDetails.input_for(ctx,(;axis="E_withdrawal",amplitude=.35))==(0.,0.)
        @test FigureDetails.input_for(ctx,(;axis="I_increase",amplitude=3.))==(.35,3.)
        rest=ctx.search.equilibria[ctx.roles["rest"]].state
        held=VerifyFigureRedesign.integrate(rest,reference,.35,0.,[0.,1.])
        @test length(held.u)==2
        @test maximum(abs.(last(held.u).-rest))<1e-8
    end
end
