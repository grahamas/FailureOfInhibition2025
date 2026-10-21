using Test
include(joinpath(@__DIR__,"..","scripts","run_figure_details.jl"))
include(joinpath(@__DIR__,"..","scripts","verify_figure_redesign.jl"))
include(joinpath(@__DIR__,"..","scripts","build_figure_redesign.jl"))
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
        path=joinpath(dir,"trace.csv")
        FigureDetails.CSV.write(path,[(;time=t,E=x[1],I=x[2]) for (t,x) in zip(held.t,held.u)])
        @test VerifyFigureRedesign.compare(path,reference,.35,0.,dir).file=="trace.csv"
    end
end

@testset "Packet builder rejects changed verified inputs" begin
    C=FigureRedesignData.IOChecks
    mktempdir() do root
        details=joinpath(root,"details");screen=joinpath(root,"screen")
        stage=joinpath(details,"sequence");mkpath(stage)
        mkpath(screen)
        config=joinpath(details,"config.toml")
        cp(joinpath(@__DIR__,"..","experiments","figure_redesign.toml"),config)
        screen_metadata=joinpath(screen,"metadata.toml")
        FigureRedesignData.S.write_record(screen_metadata,Dict(
            "completed"=>true,"config_sha256"=>C.digest(config)))
        screen_data=joinpath(screen,"screen.csv")
        write(screen_data,"id,regime\nexample,rest\n")
        FigureRedesignData.D.F.A.refresh_study_manifest(screen)
        data=joinpath(stage,"trajectory.csv");marker=joinpath(stage,"done.toml")
        function finish_stage(value)
            write(data,"time,E,I\n0,$value,0\n")
            FigureRedesignData.S.write_record(marker,Dict("files"=>Dict("trajectory.csv"=>C.digest(data))))
        end
        finish_stage(0)
        FigureRedesignData.S.write_record(joinpath(details,"metadata.toml"),Dict(
            "completed"=>true,"completed_stages"=>["sequence"],
            "config_sha256"=>C.digest(config),"source_sha256"=>Dict{String,String}()))
        FigureRedesignData.S.write_record(joinpath(details,"selection.toml"),Dict(
            "cases"=>Any[],"screen_metadata_sha256"=>C.digest(joinpath(screen,"metadata.toml")),
            "screen_manifest_sha256"=>C.digest(joinpath(screen,"checksums.toml"))))
        report=joinpath(details,"verification","trajectories.csv")
        mkpath(dirname(report))
        write(report,"file,maximum_coordinate_error,passed\nsequence/event_1,0.0,true\n")
        verification=Dict("passed"=>true,
            "verifier_sha256"=>C.digest(joinpath(@__DIR__,"..","scripts","verify_figure_redesign.jl")),
            "packet_io_sha256"=>C.digest(joinpath(@__DIR__,"..","scripts","figure_packet_io.jl")),
            "details_metadata_sha256"=>C.digest(joinpath(details,"metadata.toml")),
            "selection_sha256"=>C.digest(joinpath(details,"selection.toml")),
            "trajectories_sha256"=>C.digest(report),
            "completed_stage_manifests_sha256"=>C.completed_stage_manifests(details,["sequence"]))
        FigureRedesignData.S.write_record(joinpath(details,"verification","summary.toml"),verification)
        function build_error()
            try
                FigureRedesignData.build(screen,details,joinpath(root,"packet"))
                nothing
            catch e
                e
            end
        end
        finish_stage(1)
        err=build_error()
        @test err isa ErrorException
        @test occursin("verified stage contents changed",sprint(showerror,err))
        finish_stage(0)
        write(screen_data,"id,regime\nexample,active\n")
        err=build_error()
        @test err isa ArgumentError
        @test occursin("study manifest does not match output",sprint(showerror,err))
        FigureRedesignData.D.F.A.refresh_study_manifest(screen)
        err=build_error()
        @test err isa ErrorException
        @test occursin("screen contents changed since selection",sprint(showerror,err))
        write(screen_data,"id,regime\nexample,rest\n")
        FigureRedesignData.D.F.A.refresh_study_manifest(screen)
        write(report,"file,maximum_coordinate_error,passed\nsequence/event_1,1.0,false\n")
        err=build_error()
        @test err isa ErrorException
        @test occursin("verification trajectory report changed",sprint(showerror,err))
        FigureRedesignData.S.write_record(screen_metadata,Dict(
            "completed"=>true,"config_sha256"=>"different"))
        cfg=FigureRedesignData.S.load_config(config)
        err=try
            FigureRedesignData.D.select_cases(cfg,screen,details,C.digest(config))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("screen configuration differs from detail study",sprint(showerror,err))
        err=build_error()
        @test err isa ErrorException
        @test occursin("screen configuration differs from detail study",sprint(showerror,err))
    end
end
