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
    row(a,d;status="compatible",duration=100.)=(;axis="E_increase",amplitude=a,duration,status,destination=d)
    rows=[row(0.,"rest"),row(.25,"active"),row(.5,"active"),row(.75,"herald"),row(1.,"active")]
    witness=FigureDetails.interior_pulse(rows,"active")
    @test witness.amplitude==.375
    @test witness.width==.25
    @test FigureDetails.interior_pulse(rows,"seizure")===nothing
    @test FigureDetails.interior_pulse([row(0.,"active"),row(.25,"active";status="unresolved"),row(.5,"active")],"active")===nothing
    longer=[row(0.,"rest";duration=200.),row(.25,"active";duration=200.),row(.5,"active";duration=200.)]
    @test FigureDetails.interior_pulse(longer,"active";duration=200.).amplitude==.375
    @test FigureDetails.interior_pulse(longer,"active")===nothing
    reference=only(filter(p->p["id"]=="selective_withdrawal",cases))
    weak=only(filter(p->p["id"]=="input_dependent",cases))
    durations=cfg.raw["durations"]
    paired_be=reference["baselines"][1][1]
    paired_coarse=unique(vcat(collect(0.:paired_be/8:paired_be),[paired_be]))
    paired_amplitudes=union(Set(paired_coarse),
        Set((a+b)/2 for (a,b) in zip(paired_coarse,paired_coarse[2:end])))
    paired_grid=[(;id="selective_withdrawal",baseline=1,duration,amplitude,
        herald_status="compatible",herald="herald",
        seizure_status="compatible",seizure="seizure")
        for duration in durations for amplitude in paired_amplitudes]
    mktempdir() do paired_dir
        for duration in durations
            FigureDetails.S.write_record(joinpath(paired_dir,"pulse_intervals_$(duration).toml"),
                Dict("unresolved"=>Any[]))
        end
        @test VerifyFigureRedesign.verify_paired_pulse_grid(paired_grid,durations,paired_be,
            cfg.raw,paired_dir)
        @test_throws ErrorException VerifyFigureRedesign.verify_paired_pulse_grid(
            paired_grid[2:end],durations,paired_be,cfg.raw,paired_dir)
        @test_throws ErrorException VerifyFigureRedesign.verify_paired_pulse_grid(
            filter(r->r.duration!=first(durations),paired_grid),durations,paired_be,cfg.raw,paired_dir)
        @test_throws ErrorException VerifyFigureRedesign.verify_paired_pulse_grid(
            vcat(paired_grid,paired_grid[1]),durations,paired_be,cfg.raw,paired_dir)
        @test_throws ErrorException VerifyFigureRedesign.verify_paired_pulse_grid(
            vcat(paired_grid,merge(first(paired_grid),(;duration=3.))),durations,
            paired_be,cfg.raw,paired_dir)
        @test_throws ErrorException VerifyFigureRedesign.verify_paired_pulse_grid(
            vcat(paired_grid[1:end-1],merge(last(paired_grid),(;id="other"))),
            durations,paired_be,cfg.raw,paired_dir)
        sample=FigureDetails.S.M.sample_line(a->a<paired_be/3 ?
            ("compatible","herald","compatible","seizure") :
            ("compatible","active","compatible","seizure"),0.,paired_be;
            step=paired_be/8,width=cfg.raw["input_width"],key=identity,
            max_evaluations=cfg.raw["line_budget"])
        adaptive=[(;id="selective_withdrawal",baseline=1,duration=first(durations),
            amplitude,herald_status=value[1],herald=value[2],
            seizure_status=value[3],seizure=value[4]) for (amplitude,value) in sample.cache]
        adaptive_grid=vcat(filter(r->r.duration!=first(durations),paired_grid),adaptive)
        @test VerifyFigureRedesign.verify_paired_pulse_grid(adaptive_grid,durations,paired_be,
            cfg.raw,paired_dir)
        refined=first(filter(a->a ∉ paired_amplitudes,keys(sample.cache)))
        @test_throws ErrorException VerifyFigureRedesign.verify_paired_pulse_grid(
            filter(r->r.duration!=first(durations) || r.amplitude!=refined,adaptive_grid),
            durations,paired_be,cfg.raw,paired_dir)
    end
    sustained_grid=[(;id="selective_withdrawal",baseline=1),(;id="narrative",baseline=1)]
    expected_sustained=Set((row.id,row.baseline) for row in sustained_grid)
    @test VerifyFigureRedesign.verify_paired_sustained_grid(sustained_grid,expected_sustained)
    @test_throws ErrorException VerifyFigureRedesign.verify_paired_sustained_grid(
        sustained_grid[1:1],expected_sustained)
    @test_throws ErrorException VerifyFigureRedesign.verify_paired_sustained_grid(
        vcat(sustained_grid,sustained_grid[1]),expected_sustained)
    unit=Dict("B_E"=>1.,"B_I"=>1.)
    cells=[(;x0=0.,x1=.5,y0=0.,y1=1.),(;x0=.5,x1=1.,y0=0.,y1=1.)]
    @test VerifyFigureRedesign.verify_cell_partition(cells,unit)
    @test_throws ErrorException VerifyFigureRedesign.verify_cell_partition(cells[1:1],unit)
    @test_throws ErrorException VerifyFigureRedesign.verify_cell_partition(
        vcat(cells,cells[1]),unit)
    initial_phases=["cycle_1_phase_$(fraction)" for fraction in
        collect(0:cfg.raw["phase_count"]-1)./cfg.raw["phase_count"]]
    phase_summary=Dict("validated_cycles"=>1,"sources"=>[
        Dict("source"=>source,"phase_signature"=>"same") for source in initial_phases])
    @test VerifyFigureRedesign.verify_cycle_sources(phase_summary,initial_phases,cfg.raw)
    @test_throws ErrorException VerifyFigureRedesign.verify_cycle_sources(
        phase_summary,initial_phases[2:end],cfg.raw)
    phase_summary["sources"][1]["phase_signature"]="different"
    @test_throws ErrorException VerifyFigureRedesign.verify_cycle_sources(
        phase_summary,initial_phases,cfg.raw)
    refined_phases=["cycle_1_phase_$(fraction)" for fraction in
        collect(1:2:cfg.raw["phase_confirmation_count"]-1)./
        cfg.raw["phase_confirmation_count"]]
    append!(phase_summary["sources"],[Dict("source"=>source,"phase_signature"=>"refined")
        for source in refined_phases])
    @test VerifyFigureRedesign.verify_cycle_sources(phase_summary,
        vcat(initial_phases,refined_phases),cfg.raw)
    intervention_axes=(("theta_off",vcat(reference["theta_off"],collect(6.:.25:12.))),
        ("e_to_e",filter(x->0<=x<=24,reference["e_to_e"].*[1.,.95,.9,.85,.8,.75,.7,.65,.6])),
        ("e_to_i",filter(x->12<=x<=28,reference["e_to_i"].*[1.,.95,.9,.85,.8,.75,.7,.65,.6])))
    intervention_grid=[(;axis,value) for (axis,values) in intervention_axes for value in unique(values)]
    @test length(intervention_grid)==42
    @test VerifyFigureRedesign.verify_intervention_grid(intervention_grid,reference)
    @test_throws ErrorException VerifyFigureRedesign.verify_intervention_grid(intervention_grid[2:end],reference)
    @test_throws ErrorException VerifyFigureRedesign.verify_intervention_grid(vcat(intervention_grid,intervention_grid[1]),reference)
    bounds=VerifyFigureRedesign.independent_input_bounds(reference,cfg.raw)
    @test VerifyFigureRedesign.verify_geometry_bounds(bounds,reference,cfg.raw)==bounds
    @test_throws ErrorException VerifyFigureRedesign.verify_geometry_bounds(
        merge(bounds,Dict("B_E"=>bounds["B_E"]/2)),reference,cfg.raw)
    @test_throws ErrorException VerifyFigureRedesign.verify_geometry_bounds(
        merge(bounds,Dict("I_tail_resolved"=>false)),reference,cfg.raw)
    branch=[(;ratio=1.)]
    failed=[Dict("direction"=>-1,"accepted"=>false,
        "status"=>"corrector_line_search_failed","step"=>1.5e-5)]
    @test VerifyFigureRedesign.verify_continuation_stop((;termination="minimum_step"),
        branch,failed,"cycles.csv",-1)===nothing
    @test_throws ErrorException VerifyFigureRedesign.verify_continuation_stop(
        (;termination="minimum_step"),branch,Any[],"cycles.csv",-1)
    @test_throws ErrorException VerifyFigureRedesign.verify_continuation_stop(
        (;termination="minimum_step"),branch,
        [merge(only(failed),Dict("step"=>3e-5))],"cycles.csv",-1)
    @test_throws ErrorException VerifyFigureRedesign.verify_continuation_stop(
        (;termination="minimum_step"),branch,
        [merge(only(failed),Dict("direction"=>1))],"cycles.csv",-1)
    @test VerifyFigureRedesign.verify_continuation_stop((;termination="minimum_step"),
        [(;ratio=.2+1.02e-5)],Any[],"cycles.csv",1)===nothing
    @test VerifyFigureRedesign.verify_continuation_stop((;termination="parameter_boundary"),
        [(;ratio=4.4)],Any[],"cycles.csv",-1)===nothing
    protocol_config=merge(cfg.raw,Dict("durations"=>[1.]))
    baseline=(.35,0.)
    protocol_rows=NamedTuple[]
    for axis in ("E_increase","E_withdrawal","I_increase","I_withdrawal")
        cap=axis=="E_increase" ? bounds["B_E"]-baseline[1] :
            axis=="E_withdrawal" ? baseline[1] :
            axis=="I_increase" ? bounds["B_I"]-baseline[2] : baseline[2]
        step=max(min(1.,cap/8),eps())
        coarse=unique(vcat(collect(0.:step:cap),[cap]))
        for amplitude in union(Set(coarse),Set((a+b)/2 for (a,b) in zip(coarse,coarse[2:end])))
            be=baseline[1]+(axis=="E_increase" ? amplitude : axis=="E_withdrawal" ? -amplitude : 0.)
            bi=baseline[2]+(axis=="I_increase" ? amplitude : axis=="I_withdrawal" ? -amplitude : 0.)
            push!(protocol_rows,(;axis,duration=1.,amplitude,B_E=be,B_I=bi,
                status="compatible",destination="rest"))
        end
    end
    for amplitude in (0.,baseline[1]/2,baseline[1]),
        stimulation in (0.,bounds["B_I"]/2,bounds["B_I"])
        push!(protocol_rows,(;axis="joint",duration=1.,amplitude,
            B_E=baseline[1]-amplitude,B_I=stimulation))
    end
    mktempdir() do pulse_dir
        @test VerifyFigureRedesign.verify_pulse_protocol_grid(protocol_rows,reference,
            baseline,protocol_config,pulse_dir)
        @test_throws ErrorException VerifyFigureRedesign.verify_pulse_protocol_grid(
            protocol_rows[2:end],reference,baseline,protocol_config,pulse_dir)
        @test_throws ErrorException VerifyFigureRedesign.verify_pulse_protocol_grid(
            vcat(protocol_rows,protocol_rows[1]),reference,baseline,protocol_config,pulse_dir)
        key(x)=(x.status,x.destination)
        line=FigureDetails.S.M.sample_line(a->(;status=a<.5 ? "compatible" : "unresolved",
            destination=a<.5 ? "rest" : "active"),0.,1.;step=.5,width=.05,key,
            max_evaluations=30)
        adaptive=[merge((;amplitude),value) for (amplitude,value) in line.cache]
        found,stopped=VerifyFigureRedesign.verify_line_samples(adaptive,1.,.5,.05,30,
            "E_increase",1.)
        @test !isempty(found)
        @test length(stopped)==length(line.unresolved)
        @test !isempty(stopped)
        refined=first(filter(a->a ∉ Set([0.,.25,.5,.75,1.]),keys(line.cache)))
        @test_throws ErrorException VerifyFigureRedesign.verify_line_samples(
            filter(r->r.amplitude!=refined,adaptive),1.,.5,.05,30,"E_increase",1.)
        FigureDetails.CSV.write(joinpath(pulse_dir,"boundaries.csv"),found)
        @test_throws ErrorException VerifyFigureRedesign.verify_recorded_rows(
            joinpath(pulse_dir,"unresolved_intervals.csv"),stopped)
        isempty(stopped) || FigureDetails.CSV.write(
            joinpath(pulse_dir,"unresolved_intervals.csv"),stopped)
        @test VerifyFigureRedesign.verify_recorded_rows(joinpath(pulse_dir,"boundaries.csv"),found)===nothing
        @test_throws ErrorException VerifyFigureRedesign.verify_recorded_rows(
            joinpath(pulse_dir,"boundaries.csv"),found[2:end])
    end
    @test [2.,1.] in FigureDetails.target_baselines(weak)
    @test FigureDetails.target_baselines(reference)==reference["baselines"]
    for state in ([0.,0.],[.3,.32],[.5,0.],[1.,1.]), input in ((.35,0.),(0.,4.),(8.,1.))
        model=FigureDetails.S.M.model(reference,input...)
        package=zeros(2);independent=zeros(2)
        FigureDetails.point_rhs!(package,state,model,0.)
        VerifyFigureRedesign.independent_rhs!(independent,state,(reference,input...),0.)
        @test package≈independent atol=1e-12 rtol=1e-12
        j=VerifyFigureRedesign.independent_jacobian(state,reference,input...)
        for coordinate in 1:2
            shifted=copy(state);shifted[coordinate]+=1e-6
            plus=zeros(2);VerifyFigureRedesign.independent_rhs!(plus,shifted,(reference,input...),0.)
            shifted[coordinate]-=2e-6
            minus=zeros(2);VerifyFigureRedesign.independent_rhs!(minus,shifted,(reference,input...),0.)
            column=(plus.-minus)./2e-6
            expected=coordinate==1 ? [j.j11,j.j21] : [j.j12,j.j22]
            @test column≈expected atol=1e-7 rtol=1e-7
        end
    end
    mktempdir() do dir
        ctx=FigureDetails.confirmed_context(reference,[.35,0.],cfg,dir)
        @test all(haskey(ctx.roles,r) for r in ("rest","active","herald","seizure"))
        mechanism_details=joinpath(dir,"mechanism_details")
        mechanism_reference=joinpath(mechanism_details,"cases","selective_withdrawal","baseline_1")
        mkpath(mechanism_reference)
        for name in ("context.toml","roles.toml")
            cp(joinpath(dir,name),joinpath(mechanism_reference,name))
        end
        FigureDetails.mechanism(reference,cfg,mechanism_details)
        @test all(r->r.passed,VerifyFigureRedesign.verify_mechanism(mechanism_details,reference))
        response_file=joinpath(mechanism_details,"mechanism","responses.csv")
        original_response=read(response_file,String)
        response_lines=readlines(response_file)
        response_fields=split(response_lines[2],',')
        response_fields[3]="1.0"
        response_lines[2]=join(response_fields,",")
        write(response_file,join(response_lines,"\n")*"\n")
        @test !all(r->r.passed,VerifyFigureRedesign.verify_mechanism(mechanism_details,reference))
        write(response_file,original_response)
        roots_file=joinpath(mechanism_details,"mechanism","roots.csv")
        original_roots=read(roots_file,String)
        root_rows=[NamedTuple(r) for r in FigureDetails.CSV.File(roots_file)]
        monotone_rows=findall(r->r.condition=="monotone",root_rows)
        @test length(monotone_rows)>1
        truncated=copy(root_rows);deleteat!(truncated,last(monotone_rows))
        FigureDetails.CSV.write(roots_file,truncated)
        @test_throws ErrorException VerifyFigureRedesign.verify_mechanism(mechanism_details,reference)
        write(roots_file,original_roots)
        monotone_root=findfirst(r->r.condition=="monotone" && r.stability=="Attracting",root_rows)
        @test monotone_root!==nothing
        root_rows[monotone_root]=merge(root_rows[monotone_root],(;role="seizure"))
        FigureDetails.CSV.write(roots_file,root_rows)
        @test_throws ErrorException VerifyFigureRedesign.verify_mechanism(mechanism_details,reference)
        write(roots_file,original_roots)
        @test FigureDetails.input_for(ctx,(;axis="E_withdrawal",amplitude=.35))==(0.,0.)
        @test FigureDetails.input_for(ctx,(;axis="I_increase",amplitude=3.))==(.35,3.)
        rest=ctx.search.equilibria[ctx.roles["rest"]].state
        group=[(;E=x.state[1],I=x.state[2],B_E=.35,B_I=0.,
            stability=string(x.stability.classification)) for x in ctx.search.equilibria]
        @test VerifyFigureRedesign.independent_geometry_roles(group,reference,0.)==ctx.roles
        roots=FigureDetails.TOML.parsefile(joinpath(dir,"context.toml"))["equilibria"]
        @test VerifyFigureRedesign.independent_target(reference,(.35,0.),rest,"E_increase",.5,
            100.,roots,ctx.roles["active"],cfg.raw["followup_times"])
        @test !VerifyFigureRedesign.independent_target(reference,(.35,0.),rest,"E_increase",0.,
            100.,roots,ctx.roles["active"],cfg.raw["followup_times"])
        for (winning_axis,winning_amplitudes) in
            (("E_increase",(.25,.5)),("I_increase",(.5,.75)))
            search=[(;axis,amplitude,duration=1.,
                success=axis==winning_axis && amplitude in winning_amplitudes)
                for axis in ("E_increase","I_increase") for amplitude in 0.:.25:8.]
            witness=VerifyFigureRedesign.remeasured_witness(search,[1.])
            @test (witness.axis,witness.amplitude,witness.duration)==
                (winning_axis,(first(winning_amplitudes)+last(winning_amplitudes))/2,1.)
        end
        sequence_search=[(;axis="E_withdrawal",amplitude,duration=100.,
            success=amplitude in (.25,.5)) for amplitude in 0.:.25:1.]
        sequence_witness=VerifyFigureRedesign.remeasured_witness(sequence_search,[100.];
            axes=("E_increase","I_increase","E_withdrawal","I_withdrawal"))
        @test (sequence_witness.axis,sequence_witness.amplitude,sequence_witness.duration)==
            ("E_withdrawal",.375,100.)
        successful=(;axis="E_increase",amplitude=.5,duration=100.,
            status="compatible",destination="active")
        @test VerifyFigureRedesign.verified_replacement_success(successful,reference,(.35,0.),
            rest,roots,ctx.roles,"active",cfg.raw["followup_times"],"test")
        @test_throws ErrorException VerifyFigureRedesign.verified_replacement_success(
            merge(successful,(;destination="rest")),reference,(.35,0.),rest,roots,
            ctx.roles,"active",cfg.raw["followup_times"],"test")
        fake_times=[0.,.5,1.]
        fake_orbit=VerifyFigureRedesign.integrate([.2,.2],weak,2.,1.,fake_times)
        orbit_file=joinpath(dir,"orbit_1.csv")
        FigureDetails.CSV.write(orbit_file,[(;time=t,E=x[1],I=x[2]) for (t,x) in
            zip(fake_times,fake_orbit.u)])
        FigureDetails.S.write_record(joinpath(dir,"orbit_1.toml"),Dict(
            "period"=>1.,"validation"=>"NumericallyValidatedPeriodicOrbit",
            "stability"=>"PeriodicOrbitAttracting","transverse_real"=>.1,
            "transverse_imag"=>0.,"amplitudes"=>[.1,.1]))
        @test_throws ErrorException VerifyFigureRedesign.verify_source_orbit(dir,orbit_file,weak,(2.,1.))
        @test_throws ErrorException VerifyFigureRedesign.verify_orbit_values(
            (;closure=0.,E_min=.1,E_max=.2,multiplier=1.2),
            Dict("validation"=>"NumericallyValidatedPeriodicOrbit",
                 "stability"=>"PeriodicOrbitAttracting",
                 "transverse_multiplier"=>Dict("real"=>1.2,"imaginary"=>0.)),"repelling outcome";
            attracting_only=true)
        pulse=FigureDetails.S.pulse(ctx,rest,.85,0.,100.,cfg)
        switch=last(pulse.pulse_solution.u)
        sampled=(;axis="E_increase",amplitude=.5,duration=100.,B_E=.85,B_I=0.,
            status=pulse.status,destination=pulse.destination,switch_E=switch[1],switch_I=switch[2])
        pulse_file=joinpath(dir,"sampled_pulses.csv")
        @test VerifyFigureRedesign.verify_axial_pulse_row(dir,pulse_file,1,sampled,rest,reference,
            (.35,0.),roots,ctx.roles,nothing,cfg.raw["followup_times"],cfg.raw["durations"]).passed
        @test_throws ErrorException VerifyFigureRedesign.verify_axial_pulse_row(dir,pulse_file,1,
            merge(sampled,(;destination="rest")),rest,reference,(.35,0.),roots,ctx.roles,
            nothing,cfg.raw["followup_times"],cfg.raw["durations"])
        herald=ctx.search.equilibria[ctx.roles["herald"]].state
        held=FigureDetails.S.observe(ctx,herald,cfg;reference=ctx,retain=true,tight=true,cycles=true)
        FigureDetails.S.save_observation(dir,"herald_held",held)
        @test VerifyFigureRedesign.verify_observation(dir,dir,"herald_held",dir,dir,reference,
            (.35,0.),"compatible","herald",herald).passed
        released=FigureDetails.S.context(reference,0.,0.,cfg;tight=true,grid=41)
        FigureDetails.S.save_context(joinpath(dir,"released"),released)
        withdrawn=FigureDetails.S.observe(released,herald,cfg;reference=ctx,retain=true,tight=true,cycles=true)
        FigureDetails.S.save_observation(dir,"herald_release",withdrawn)
        @test VerifyFigureRedesign.verify_observation(dir,dir,"herald_release",joinpath(dir,"released"),
            dir,reference,(0.,0.),withdrawn.status,withdrawn.destination,herald).passed
        @test_throws ErrorException VerifyFigureRedesign.verify_observation(dir,dir,"herald_release",
            joinpath(dir,"released"),dir,reference,(0.,0.),withdrawn.status,"seizure",herald)
        held=VerifyFigureRedesign.integrate(rest,reference,.35,0.,[0.,1.])
        @test length(held.u)==2
        @test maximum(abs.(last(held.u).-rest))<1e-8
        path=joinpath(dir,"trace.csv")
        FigureDetails.CSV.write(path,[(;time=t,E=x[1],I=x[2]) for (t,x) in zip(held.t,held.u)])
        @test VerifyFigureRedesign.compare(path,reference,.35,0.,dir).file=="trace.csv"
        pulses=[(;axis="E_increase",amplitude=.5,duration=100.),
            (;axis="I_increase",amplitude=3.,duration=100.)]
        @test FigureDetails.switching(ctx,cfg,pulses;output=joinpath(dir,"fixed_switching"))=="retained"
        checks=VerifyFigureRedesign.verify_switching(dir,dir,reference,[.35,0.],"fixed_switching","retained")
        @test length(checks)==4
        @test all(r->r.passed,checks)
        unavailable_dir=joinpath(dir,"false_unavailable")
        mkpath(unavailable_dir)
        FigureDetails.S.write_record(joinpath(unavailable_dir,"summary.toml"),
            Dict("status"=>"sources_unavailable","legs"=>Any[]))
        @test_throws ErrorException VerifyFigureRedesign.verify_switching(dir,dir,reference,[.35,0.],
            "false_unavailable","sources_unavailable")
        fixed_dir=joinpath(dir,"fixed_switching")
        fixed_summary=joinpath(fixed_dir,"summary.toml")
        saved_summary=read(fixed_summary,String)
        altered=FigureDetails.TOML.parsefile(fixed_summary)
        altered["legs"][1]["amplitude"]=.75
        FigureDetails.S.write_record(fixed_summary,altered)
        @test_throws ErrorException VerifyFigureRedesign.verify_switching(dir,dir,reference,[.35,0.],
            "fixed_switching","retained")
        write(fixed_summary,saved_summary)
        @test VerifyFigureRedesign.verify_pulse_trace(dir,fixed_dir,"cycle_1_leg_1",dir,
            reference,(.35,0.),.85,0.,rest,100.,"compatible","active").passed
        continuation=joinpath(fixed_dir,"cycle_1_leg_1.csv")
        original=read(continuation,String)
        continuation_rows=collect(FigureDetails.CSV.File(continuation))
        shifted=VerifyFigureRedesign.integrate([first(continuation_rows).E+.01,
            first(continuation_rows).I],reference,.35,0.,getproperty.(continuation_rows,:time))
        FigureDetails.CSV.write(continuation,[(;time=t,E=x[1],I=x[2]) for (t,x) in
            zip(shifted.t,shifted.u)])
        @test VerifyFigureRedesign.compare(continuation,reference,.35,0.,dir).passed
        @test !VerifyFigureRedesign.verify_pulse_trace(dir,fixed_dir,"cycle_1_leg_1",dir,
            reference,(.35,0.),.85,0.,rest,100.,"compatible","active").passed
        write(continuation,original)
        pulse_file=joinpath(dir,"fixed_switching","cycle_1_leg_1_pulse.csv")
        lines=readlines(pulse_file);fields=split(lines[2],',')
        fields[2]=string(parse(Float64,fields[2])+.01)
        lines[2]=join(fields,",");write(pulse_file,join(lines,"\n")*"\n")
        @test !all(r->r.passed,VerifyFigureRedesign.verify_switching(dir,dir,reference,[.35,0.],
            "fixed_switching","retained"))
        failed_pulses=[merge(pulses[1],(;amplitude=0.)),pulses[2]]
        failed=FigureDetails.switching(ctx,cfg,failed_pulses;output=joinpath(dir,"failed_switching"))
        @test failed!="retained"
        failed_checks=VerifyFigureRedesign.verify_switching(dir,dir,reference,[.35,0.],"failed_switching",failed)
        @test length(failed_checks)==1
        @test all(r->r.passed,failed_checks)
        failed_dir=joinpath(dir,"failed_switching")
        failed_summary=joinpath(failed_dir,"cycle_1_leg_1_summary.toml")
        failed_phase=joinpath(failed_dir,"cycle_1_leg_1.toml")
        original_summary=read(failed_summary,String);original_phase=read(failed_phase,String)
        spoofed_summary=FigureDetails.TOML.parsefile(failed_summary)
        spoofed_phase=FigureDetails.TOML.parsefile(failed_phase)
        spoofed_summary["destination"]="active"
        spoofed_phase["destination"]=ctx.roles["active"]
        FigureDetails.S.write_record(failed_summary,spoofed_summary)
        FigureDetails.S.write_record(failed_phase,spoofed_phase)
        @test !VerifyFigureRedesign.verify_pulse_trace(dir,failed_dir,"cycle_1_leg_1",dir,
            reference,(.35,0.),.35,0.,rest,100.,"compatible","active").passed
        write(failed_summary,original_summary);write(failed_phase,original_phase)
        failed_file=joinpath(dir,"failed_switching","cycle_1_leg_1.csv")
        lines=readlines(failed_file);fields=split(lines[2],',')
        fields[2]=string(parse(Float64,fields[2])+.01)
        lines[2]=join(fields,",");write(failed_file,join(lines,"\n")*"\n")
        @test !all(r->r.passed,VerifyFigureRedesign.verify_switching(dir,dir,reference,[.35,0.],
            "failed_switching",failed))
        rm(failed_file)
        @test_throws ErrorException VerifyFigureRedesign.verify_switching(dir,dir,reference,[.35,0.],
            "failed_switching",failed)
        source=ctx.search.equilibria[ctx.roles["seizure"]].state
        outcome=FigureDetails.S.observe(ctx,source,cfg;reference=ctx,retain=true,tight=true,cycles=true)
        FigureDetails.S.save_observation(dir,"original_seizure",outcome)
        report=(;status=outcome.status,destination=outcome.destination,
            final_E=outcome.final[1],final_I=outcome.final[2])
        @test VerifyFigureRedesign.verify_intervention(dir,dir,reference,[.35,0.],report,source).passed
        @test !VerifyFigureRedesign.verify_intervention(dir,dir,reference,[.35,0.],
            merge(report,(;final_E=report.final_E+.01)),source).passed
        @test_throws ErrorException VerifyFigureRedesign.verify_intervention(dir,dir,reference,[.35,0.],
            merge(report,(;destination="rest")),source)
        seizure_file=joinpath(dir,"original_seizure.csv")
        times=getproperty.(collect(FigureDetails.CSV.File(seizure_file)),:time)
        alternative=VerifyFigureRedesign.integrate(rest,reference,.35,0.,times)
        FigureDetails.CSV.write(seizure_file,[(;time=t,E=x[1],I=x[2]) for (t,x) in
            zip(alternative.t,alternative.u)])
        @test VerifyFigureRedesign.compare(seizure_file,reference,.35,0.,dir).passed
        @test !VerifyFigureRedesign.verify_intervention(dir,dir,reference,[.35,0.],report,source).passed
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
