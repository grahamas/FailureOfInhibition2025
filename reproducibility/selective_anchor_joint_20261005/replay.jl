# Replay the selective anchor's switching, induction, and threshold observations in Julia.
length(ARGS) == 1 || error("usage: julia --project=. replay.jl NEW_OUTPUT_DIRECTORY")
const ROOT = normpath(joinpath(@__DIR__, "../.."))
const OUT = abspath(ARGS[1])
ispath(OUT) && error("output path already exists: $OUT")
mkpath(OUT)

include(joinpath(ROOT, "scripts/run_narrative_study.jl"))
using .NarrativeStudy

const NS = NarrativeStudy
const config = NS.load_config(joinpath(ROOT, "experiments/narrative_study.toml"))
const base = Dict{String,Any}(
    "family" => "figure4", "e_to_e" => 17.0, "i_to_e" => 13.0,
    "e_to_i" => 19.0, "i_to_i" => 6.0, "theta_off" => 8.0,
    "tau_ratio" => 0.2, "B_E" => 0.35)

baseline = NS.context(base, 41; tight=true)
println("baseline roles=", baseline.roles); flush(stdout)
witnesses = Dict(
    "rest_to_active" => Dict("target"=>"E", "amplitude"=>0.5, "duration"=>100.0),
    "active_to_rest" => Dict("target"=>"I", "amplitude"=>3.0, "duration"=>100.0))
mkpath(joinpath(OUT, "baseline_switching"))
repeat_ok = NS.repeat_switching(baseline, witnesses, config,
    joinpath(OUT, "baseline_switching"); tight=true)
println("repeat_switching=", repeat_ok); flush(stdout)
inductions = Dict{String,Any}[]
for (source, target, amplitude, duration) in (("rest","E",1.0,100.0),
    ("active","I",4.5,100.0))
    initial = baseline.search.equilibria[baseline.roles[source]].state
    trial = NS.pulse(baseline,initial,target,amplitude,duration,config;retain=true,tight=true)
    if trial.phase !== nothing
        name = "induce_" * source * "_to_" * trial.destination
        NS.IR.save_phase(OUT,name,trial.phase)
        NS.write_trajectory_csv(joinpath(OUT,name*"_pulse.csv"),trial.solution)
    end
    println("induce source=",source," target=",target," amplitude=",amplitude,
        " duration=",duration," status=",trial.status," destination=",trial.destination)
    push!(inductions,Dict("source"=>source,"target"=>target,"amplitude"=>amplitude,
        "duration"=>duration,"status"=>trial.status,"destination"=>trial.destination,
        "final"=>trial.final,"horizon"=>trial.horizon))
    flush(stdout)
end

interventions = Dict{String,Any}[]
for theta in (8.0,8.25,8.5,8.75,9.0,10.0,12.0)
    grid = theta==8.75 ? 41 : 21
    discovered = NS.context(merge(base,Dict("theta_off"=>theta)),grid;tight=true)
    changed = merge(discovered,(;roles=NS.NarrativeModels.match_roles(
        baseline.search,baseline.roles,discovered.search)))
    original = baseline.search.equilibria[baseline.roles["seizure"]].state
    phase = NS.IR.observe_phase(changed.model,changed.search,original,config;
        retain=theta==8.75,tight=true)
    theta==8.75 && NS.IR.save_phase(OUT,"theta_off_8.75_from_seizure",phase)
    outcome = NS.phase_row(phase,changed.roles)
    println("theta=",theta," roles=",changed.roles," seizure_source_destination=",
        outcome.destination," status=",outcome.status," horizon=",outcome.horizon)
    row = Dict{String,Any}("theta_off"=>theta,"grid"=>grid,"root_count"=>length(changed.search.equilibria),
        "sink_count"=>length(NS.IR.sink_indices(changed.search)),"roles"=>changed.roles,
        "seizure_source_destination"=>outcome.destination,"seizure_source_status"=>outcome.status,
        "seizure_source_final"=>phase.final,"seizure_source_horizon"=>outcome.horizon)
    switch_trials = Dict{String,Any}[]
    if all(k->haskey(changed.roles,k),("rest","active"))
        for (source,target,amplitude,duration) in (("rest","E",0.5,100.0),
            ("active","I",3.0,100.0))
            initial = changed.search.equilibria[changed.roles[source]].state
            trial = NS.pulse(changed,initial,target,amplitude,duration,config;tight=true)
            println("theta=",theta," switch=",source,"->",trial.destination,
                " status=",trial.status)
            push!(switch_trials,Dict("source"=>source,"target"=>target,
                "amplitude"=>amplitude,"duration"=>duration,
                "status"=>trial.status,"destination"=>trial.destination))
        end
    end
    row["fixed_switch_trials"] = switch_trials
    if theta==8.75 && length(switch_trials)==2
        directory = joinpath(OUT,"theta_off_8.75_switching")
        mkpath(directory)
        row["repeat_switching"] = NS.repeat_switching(changed,witnesses,config,directory;tight=true)
        println("theta=",theta," repeat_switching=",row["repeat_switching"])
    end
    push!(interventions,row)
    flush(stdout)
end

NS.write_record(joinpath(OUT,"summary.toml"),Dict(
    "purpose"=>"selective anchor joint switching, induction, and threshold intervention",
    "julia_version"=>string(VERSION),"baseline_parameters"=>base,
    "baseline_roles"=>baseline.roles,"baseline_grid"=>41,
    "baseline_repeat_switching"=>repeat_ok,"induction_trials"=>inductions,
    "threshold_interventions"=>interventions,
    "source_sha256"=>Dict(path=>NS.Evidence.file_hash(joinpath(ROOT,path))
        for path in vcat([joinpath("src",name) for name in readdir(joinpath(ROOT,"src"))
            if endswith(name,".jl")],
            ["scripts/run_narrative_study.jl","scripts/narrative_models.jl",
             "scripts/run_input_release_study.jl","scripts/input_release_models.jl",
             "scripts/run_basin_rescue_study.jl","scripts/run_minimal_experiment.jl",
             "experiments/narrative_study.toml",
             "Project.toml","Manifest.toml"])),
    "replay_sha256"=>NS.Evidence.file_hash(@__FILE__),
    "limits"=>"Finite-window Julia trajectories and sampled threshold points; no global completeness or biological certification"))

repeat_ok || error("baseline repeated switching did not replay")
all(trial["status"]=="compatible" &&
    trial["destination"]==(trial["source"]=="rest" ? "herald" : "seizure")
    for trial in inductions) || error("a high-state induction did not replay")
all(length(row["fixed_switch_trials"])==2 &&
    all(trial["status"]=="compatible" &&
        trial["destination"]==(trial["source"]=="rest" ? "active" : "rest")
        for trial in row["fixed_switch_trials"]) for row in interventions) ||
    error("a sampled switching pulse did not replay")
by_theta = Dict(row["theta_off"]=>row for row in interventions)
by_theta[8.5]["seizure_source_destination"]=="seizure" ||
    error("theta_off=8.5 control did not replay")
by_theta[8.75]["seizure_source_destination"]=="herald" ||
    error("theta_off=8.75 redirection did not replay")
get(by_theta[8.75],"repeat_switching",false) ||
    error("theta_off=8.75 repeated switching did not replay")
println("Julia replay passed; summary: ",joinpath(OUT,"summary.toml"))
