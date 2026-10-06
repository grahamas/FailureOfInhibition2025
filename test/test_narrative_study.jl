include(joinpath(@__DIR__, "..", "scripts", "run_narrative_study.jl"))
@testset "Narrative study contracts" begin
    N=NarrativeStudy; M=N.NarrativeModels
    config=N.load_config(joinpath(@__DIR__,"..","experiments","narrative_study.toml");smoke=true)
    @test N.confirmation_grid(config)==41
    mktempdir() do root
        source=read(joinpath(@__DIR__,"..","experiments","narrative_study.toml"),String)
        invalid=joinpath(root,"invalid.toml")
        write(invalid,replace(source,"confirmation_grids = [21, 41]"=>"confirmation_grids = [21.5]"))
        @test_throws ArgumentError N.load_config(invalid)
        write(invalid,replace(source,"confirmation_grids = [21, 41]"=>"confirmation_grids = [21]"))
        @test N.confirmation_grid(N.load_config(invalid))==21
        for (section,key,value) in (("map","baseline_step",0.),
            ("map","coupling_halfwidth",-1.),("interventions","fractions",Float64[]),
            ("interventions","output_factors",[-1.]),("interventions","threshold_step",0.))
            raw=N.TOML.parse(source)
            raw[section][key]=value
            open(invalid,"w") do stream;N.TOML.print(stream,raw);end
            @test_throws ArgumentError N.load_config(invalid)
        end
        raw=N.TOML.parse(source)
        delete!(raw["map"],"coupling_step")
        open(invalid,"w") do stream;N.TOML.print(stream,raw);end
        @test_throws ArgumentError N.load_config(invalid)
    end
    command=N.replay_command("run_narrative_study.jl";stage="screen",case_filter="figure3",smoke=true)
    @test occursin("--stage screen --case 'figure3' --smoke",command)
    @test N.shell_quote("a'b")=="'a'\\''b'"
    metadata=Dict{String,Any}()
    N.complete_replay!(metadata,"run_narrative_study.jl";stage="screen",case_filter="figure3",smoke=true)
    N.complete_replay!(metadata,"run_narrative_study.jl";stage="confirm",case_filter=nothing,smoke=true)
    @test metadata["replay_invocations"]==[command,
        N.replay_command("run_narrative_study.jl";stage="confirm",smoke=true)]
    @test metadata["replay_from_artifact_directory"]==join(metadata["replay_invocations"]," && ")
    p=Dict{String,Any}("family"=>"figure3","e_to_e"=>17.,"i_to_e"=>9.,"e_to_i"=>19.,
        "i_to_i"=>4.,"theta_off"=>8.,"tau_ratio"=>0.2,"B_E"=>0.125)
    for b in (0.,0.125)
        pair=M.models(merge(p,Dict("B_E"=>b)))
        @test pair.control.coupling == pair.failure_of_inhibition.coupling
        @test pair.control.drive.baseline == pair.failure_of_inhibition.drive.baseline == (b,0.)
    end
    for bad in (true,NaN,Inf,-0.1)
        @test_throws ArgumentError M.models(merge(p,Dict("B_E"=>bad)))
    end
    @test_throws ArgumentError M.models(merge(p,Dict("tau_ratio"=>0.)))
    @test [M.radical_inverse(n,2) for n in 1:4] == [0.5,0.25,0.75,0.125]
    island(x)=(status="compatible",destination=0.2<x<0.4 ? "active" : "source")
    cache=M.sample_line(island,1.;step=0.25,width=0.005)
    @test any(x->0.2<x<0.4,keys(cache))
    @test any(x->0.19<x<0.2,keys(cache))
    @test any(x->0.4<x<0.41,keys(cache))
    uncertain(x)=(status=0.3<x<0.7 ? "unresolved" : "compatible",destination="source")
    cache=M.sample_line(uncertain,1.;step=0.5,width=0.05)
    @test count(v->v.status=="unresolved",values(cache))>3
    @test length(M.sample_line(island,0.;step=0.25,width=0.01))==1
    @test_throws ArgumentError M.sample_line(island,1.;step=0.,width=0.01)

    ctx=N.context(p,21)
    zero_ctx=N.context(merge(p,Dict("B_E"=>0.)),21)
    @test all(k->haskey(zero_ctx.roles,k),("rest","active","herald","seizure"))
    @test !haskey(ctx.roles,"active") # positive input can remove the intermediate branch
    rising=N.context(merge(p,Dict("e_to_e"=>19.,"i_to_e"=>13.,"e_to_i"=>14.,
        "i_to_i"=>6.,"theta_off"=>6.,"tau_ratio"=>4.4,"B_E"=>0.)),21)
    @test haskey(rising.roles,"herald") && !haskey(rising.roles,"active")
    pulse=N.pulse(ctx,ctx.search.equilibria[ctx.roles["rest"]].state,"E",0.,20.,config)
    @test pulse.status=="compatible" && pulse.destination=="rest"
    unchanged=N.recovery_trial(ctx,"herald",0.,"permanent_withdrawal",config)
    @test unchanged.destination=="herald"
    @test !unchanged.recovery
    displaced=N.recovery_trial(ctx,"herald",0.1,"direct_E_displacement",config)
    original=ctx.search.equilibria[ctx.roles["herald"]].state
    @test displaced.initial ≈ [original[1]-0.1,original[2]]
    @test displaced.context.model.drive.baseline == ctx.model.drive.baseline
    @test_throws ArgumentError N.recovery_trial(ctx,"herald",p["B_E"]+1.,"permanent_withdrawal",config)
    @test_throws ArgumentError N.recovery_trial(ctx,"herald",original[1]+1.,"direct_E_displacement",config)

    # Driven source discovery does not depend on a zero-input counterpart.
    driven=N.context(merge(p,Dict("e_to_e"=>4.,"B_E"=>8.)),21)
    off=N.context(merge(p,Dict("e_to_e"=>4.,"B_E"=>0.)),21)
    @test length(N.IR.sink_indices(off.search))==1
    @test haskey(driven.roles,"seizure")
    released=N.recovery_trial(driven,"seizure",8.,"permanent_withdrawal",config)
    @test released.context.model.drive.baseline==(0.,0.)
    @test released.E<0.001
    @test released.destination=="unassigned" # no fabricated correspondence to an absent driven rest role
    @test !released.recovery
    @test isempty(M.match_roles(ctx.search,Dict("a"=>ctx.roles["rest"],"b"=>ctx.roles["rest"]),ctx.search))

    mktempdir() do dir
        N.write_record(joinpath(dir,"data.toml"),Dict("value"=>1))
        @test !N.checked_unit(dir)
        N.complete_unit(dir)
        @test N.checked_unit(dir)
        N.write_record(joinpath(dir,"data.toml"),Dict("value"=>2))
        @test_throws ArgumentError N.checked_unit(dir)
    end
    mktempdir() do dir
        path=joinpath(@__DIR__,"..","experiments","narrative_study.toml")
        metadata=N.initialize(path,dir,config;stage="screen",case_filter="figure3")
        @test metadata["replay_from_artifact_directory"]==command
        @test N.initialize(path,dir,config)["source_sha256"]==metadata["source_sha256"]
        archived=joinpath(dir,"source","scripts","narrative_models.jl")
        open(archived,"a") do io
            write(io,"\n# corruption\n")
        end
        @test_throws ArgumentError N.initialize(path,dir,config)
    end
    mktempdir() do dir
        path=joinpath(@__DIR__,"..","experiments","narrative_study.toml")
        output=N.run_study(path,joinpath(dir,"study");stage="screen",smoke=true,
            case_filter="no_matching_case")
        metadata=N.TOML.parsefile(joinpath(output,"metadata.toml"))
        @test metadata["last_completed_stage"]=="screen"
        @test !metadata["completed"]
        @test metadata["replay_invocations"]==[N.replay_command("run_narrative_study.jl";
            stage="screen",case_filter="no_matching_case",smoke=true)]
    end
end
