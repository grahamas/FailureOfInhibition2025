using Test, TOML, SHA
if !isdefined(@__MODULE__, :write_report)
    include(joinpath(@__DIR__, "..", "scripts", "atlas", "report.jl"))
end

@testset "Atlas evidence report" begin
    mktempdir() do directory
        outside = joinpath(directory, "outside.toml")
        write(outside, "schema_version = 1\n")
        output = joinpath(directory, "output"); mkdir(output)
        symlink(outside, joinpath(output, "state.toml"))
        @test_throws ArgumentError atlas_report_data(output)
    end
    roots = [Dict("state"=>[0.1,0.2],"stability"=>"Attracting","spectrum"=>"RealSpectrum")]
    points = [Dict("id"=>id,"signature"=>sig,"roots"=>roots) for (id,sig) in (("a","same"),("b","same"),("c","other"))]
    @test length(atlas_region_components(points,[])) == 3
    path = Dict("id"=>"ab","from"=>"a","to"=>"b","validated"=>true,"signature_preserved"=>true,"status"=>"completed")
    @test length(atlas_region_components(points,[path])) == 3
    @test length(atlas_region_components(points,[merge(path,Dict("sample_ids"=>["a","b"]))])) == 3
    @test length(atlas_region_components(points,[merge(path,Dict("to"=>"c"))])) == 3
    @test length(atlas_region_components(points,[merge(path,Dict("validated"=>false))])) == 3
    five_points = [Dict("id"=>id,"signature"=>"same","roots"=>roots) for id in ("a","m1","m2","m3","b")]
    sampled_path = merge(path, Dict("sample_ids"=>["a","m1","m2","m3","b"]))
    @test atlas_region_components(five_points,[sampled_path]) == [["a","b","m1","m2","m3"]]
    @test length(atlas_region_components(five_points,[merge(sampled_path,Dict("sample_ids"=>["a","m1","m1","m1","b"]))])) == 5
    @test length(atlas_region_components(five_points,[merge(sampled_path,Dict("sample_ids"=>["a","missing","b"]))])) == 5
    @test length(atlas_region_components(five_points,[merge(sampled_path,Dict("sample_ids"=>["b","a"]))])) == 5
    @test length(atlas_region_components([merge(p,Dict("eligible_for_regions"=>p["id"]!="m2")) for p in five_points],[sampled_path])) == 4
    @test length(atlas_region_components([merge(p,Dict("signature"=>p["id"]=="m2" ? "different" : "same")) for p in five_points],[sampled_path])) == 5
    @test isempty(atlas_region_components([merge(points[1], Dict("roots"=>[]))],[]))
    @test isempty(atlas_region_components([merge(points[1], Dict("signature"=>"Eturn:unresolved"))],[]))
    plot_io = IOBuffer()
    projection_points = [Dict("source_job"=>"projection","parameters"=>Dict("B"=>Float64(i),"r"=>Float64(i)),"roots"=>roots) for i in 1:2]
    _report_scientific_plots(plot_io,Dict("points"=>projection_points,"sections"=>[]))
    @test occursin("varying parameter coordinates: B, r",String(take!(plot_io)))
    @test_throws ArgumentError atlas_region_components([points[1],points[1]],[])
    mktempdir() do directory
        mkpath(joinpath(directory,"results"))
        jobs = [Dict("id"=>"1","kind"=>"discovery","status"=>"completed","elapsed_seconds"=>2.0,"result_file"=>"results/1.toml"),
            Dict("id"=>"2","kind"=>"cycles","status"=>"timed_out","elapsed_seconds"=>5.0,"error"=>"<script>alert(1)</script>"),
            Dict("id"=>"3","kind"=>"discovery","status"=>"pending","result_file"=>"results/not-yet-written.toml")]
        open(joinpath(directory,"state.toml"),"w") do io; TOML.print(io,Dict("schema_version"=>1,"mode"=>"benchmark","jobs"=>jobs)); end
        open(joinpath(directory,"config.toml"),"w") do io; TOML.print(io,Dict("version"=>1)); end
        events = [Dict("id"=>"<unsafe>","status"=>"candidate"),
            Dict("kind"=>"local_neighborhood_witness","validated"=>true,"witness_point_ids"=>["a","m1"]),
            Dict("kind"=>"local_neighborhood_witness","validated"=>true,"witness_point_ids"=>["a"]),
            Dict("kind"=>"hopf","validated"=>true,"bracket_point_ids"=>["a","b"]),
            Dict("kind"=>"bt","validated"=>true,"witness_point_ids"=>["m2"])]
        evidence = Dict("points"=>vcat(five_points,[points[3]]),"paths"=>[sampled_path],"events"=>events,
            "cycles"=>[Dict("validated"=>true,"witness_point_ids"=>["m3"])],
            "sections"=>[Dict("parameter_pair"=>["A","a"],"section_samples"=>[Dict("status"=>"sampled","parameters"=>Dict("A"=>8.0,"a"=>1.0))],"transition_curves"=>[Dict("points"=>[Dict("parameters"=>Dict("A"=>9.0,"a"=>1.1),"classification"=>"fold_candidate"),Dict("parameters"=>Dict("A"=>8.5,"a"=>1.15),"classification"=>"trace_zero_without_positive_determinant")])])])
        open(joinpath(directory,"results","1.toml"),"w") do io; TOML.print(io,Dict("status"=>"completed","evidence"=>evidence)); end
        jobs[1]["result_hash"] = bytes2hex(sha256(read(joinpath(directory,"results","1.toml"))))
        open(joinpath(directory,"state.toml"),"w") do io; TOML.print(io,Dict("schema_version"=>1,"mode"=>"benchmark","jobs"=>jobs)); end
        destination = write_report(directory)
        html = read(destination,String)
        @test occursin("&lt;script&gt;",html)
        @test !occursin("<script>",html)
        @test occursin("&lt;unsafe&gt;",html)
        @test occursin("CompletenessNotCertified",html)
        @test occursin("<svg",html)
        @test occursin("<polygon data-marker=\"trace-zero-no-positive-determinant\"",html)
        @test occursin("gray diamonds are trace-zero points without a resolved positive determinant",html)
        @test occursin("A (dimensionless)",html)
        @test occursin("href=\"results/1.toml\"",html)
        base_csv = read(joinpath(directory,"base_cases.csv"),String)
        @test count(line -> startswith(line,"\"local_neighborhood_witness\""),split(base_csv,'\n')) == 2
        @test count(line -> startswith(line,"\"transition_witness\""),split(base_csv,'\n')) == 2
        @test count(line -> startswith(line,"\"periodic_witness\""),split(base_csv,'\n')) == 1
        @test !any(line -> startswith(line,"\"transition_witness\",\"m2\""),split(base_csv,'\n'))
        coverage = TOML.parsefile(joinpath(directory,"coverage.toml"))
        @test coverage["observed_components"] == 2
        @test coverage["pending_jobs"] == 1
        @test coverage["issues"] == 2
        @test coverage["jobs_with_issues"] == 2
        @test coverage["base_case_role_rows"] == 7
        @test coverage["unique_base_case_points"] == 5
        @test count(issue -> issue["job"]=="3",atlas_report_data(directory).issues) == 1
        @test !any(issue -> issue["job"]=="3" && issue["status"]=="missing_result",atlas_report_data(directory).issues)
        @test _report_safe_file(directory,"../state.toml") === nothing
        @test _report_safe_file(directory,joinpath(directory,"state.toml")) === nothing
        @test atlas_timing_rows(jobs)[1]["completed_samples"] == 0 # cycles first
        @test occursin("region_representative",read(joinpath(directory,"base_cases.csv"),String))
        before = read(destination)
        @test read(write_report(directory)) == before
        open(joinpath(directory,"results","1.toml"),"a") do io; println(io); end
        @test isempty(atlas_report_data(directory).records["points"])
        @test any(issue -> issue["status"]=="unverified_result",atlas_report_data(directory).issues)
        write_report(directory)
        @test TOML.parsefile(joinpath(directory,"coverage.toml"))["observed_components"] == 0
        @test !occursin("region_representative",read(joinpath(directory,"base_cases.csv"),String))
        delete!(jobs[1],"result_hash")
        open(joinpath(directory,"state.toml"),"w") do io; TOML.print(io,Dict("schema_version"=>1,"mode"=>"benchmark","jobs"=>jobs)); end
        @test isempty(atlas_report_data(directory).records["points"])
        @test any(issue -> occursin("no recorded",get(issue,"detail","")),atlas_report_data(directory).issues)
    end
    mktempdir() do directory
        open(joinpath(directory,"state.toml"),"w") do io; TOML.print(io,Dict("jobs"=>[])); end
        @test isfile(write_report(directory))
        @test TOML.parsefile(joinpath(directory,"coverage.toml"))["observed_points"] == 0
    end
    for filename in ("index.html", "base_cases.csv", "timings.csv", "coverage.toml")
        mktempdir() do directory
            output = joinpath(directory,"report"); mkdir(output)
            open(joinpath(output,"state.toml"),"w") do io; TOML.print(io,Dict("jobs"=>[])); end
            generated = ("index.html", "base_cases.csv", "timings.csv", "coverage.toml")
            for name in generated; write(joinpath(output,name),"existing $name"); end
            outside = joinpath(directory,"outside"); write(outside,"outside remains unchanged")
            rm(joinpath(output,filename)); symlink(outside,joinpath(output,filename))
            @test_throws ArgumentError write_report(output)
            @test read(outside,String) == "outside remains unchanged"
            @test islink(joinpath(output,filename))
            @test all(name -> name == filename || read(joinpath(output,name),String) == "existing $name",generated)
        end
    end
    mktempdir() do directory
        path = joinpath(directory,"target"); write(path,"original")
        @test_throws ErrorException _report_atomic_write(io -> (write(io,"incomplete"); error("writer failed")),path)
        @test read(path,String) == "original"
        @test readdir(directory) == ["target"]
    end
end

function report_pilot_fixture(directory; control_status="passed", recovery_passed=true, changes=1, audit_status="agreement_observed")
    mkpath(joinpath(directory,"results"))
    roots=[Dict("state"=>[0.1,0.2],"stability"=>"Attracting","spectrum"=>"RealSpectrum")]
    point(id)=Dict("id"=>id,"signature"=>"resolved","roots"=>roots,"parameters"=>Dict("A"=>1.0))
    manifest=Dict("control"=>1,"training"=>1,"training_followup"=>24,"audit"=>1,"heldout"=>1,"recovery"=>1)
    selection=Dict("entries"=>[Dict("point_id"=>"training","signature"=>"resolved")],"recovery_routes"=>[Dict("cohort"=>"shell1/parameter/mask0","selection_rule"=>"Parameter-only nearest frozen base; no heldout inventory or stability used","base_point_id"=>"training")],"frozen_before_heldout"=>true,"source_role"=>"training","scope"=>"grouping hypothesis")
    stream=IOBuffer();TOML.print(stream,selection;sorted=true)
    metadata=Dict("manifest_counts"=>manifest,"manifest"=>Dict("training_indices"=>[1,2],"heldout_indices"=>[3],"holdout_design"=>"Contiguous deterministic accepted-index blocks"),"stage"=>"complete","base_selection"=>selection,"base_selection_hash"=>bytes2hex(sha256(take!(stream))))
    evidence=Dict{String,Any}(
        "control"=>Dict("controls"=>[Dict("control"=>"periodic","status"=>control_status,"scope"=>"protected_actual_model_control")],"points"=>[point("control")]),
        "training"=>Dict("points"=>[point("training")]),
        "audit"=>Dict("audits"=>[Dict("cohort"=>"shell1/parameter/mask0","comparisons"=>[Dict("pair"=>["grid5_loose","grid9_loose"],"status"=>audit_status,"classification_resolved"=>true,"inventory_match"=>audit_status=="agreement_observed"),Dict("pair"=>["grid9_loose","grid9_tight"],"status"=>audit_status,"classification_resolved"=>true,"inventory_match"=>audit_status=="agreement_observed")],"training_repeat_comparison"=>Dict("status"=>"agreement_observed","classification_resolved"=>true,"inventory_match"=>true),"continuation_pair"=>Dict("comparison_status"=>"shared_prefix_consistent","comparison_tolerance"=>1e-6,"shared_prefix_count"=>12,"shared_prefix_max_state_difference"=>0.0,"depth_extensions"=>[Dict("direction"=>direction,"extension_point_count"=>0,"new_candidate_count"=>0,"extension_stability_changes"=>0,"deep_termination"=>"parameter_boundary","shallow_termination"=>"parameter_boundary") for direction in (-1,1)]))]),
        "heldout"=>Dict("points"=>[point("heldout")]),
        "recovery"=>Dict("recoveries"=>[Dict("target_cohort"=>"shell1/parameter/mask0","target_job_id"=>"heldout","base_point_id"=>"training","status"=>recovery_passed ? "sampled_tracked_target_coverage" : "tracked_coverage_unresolved","tracked_target_coverage"=>recovery_passed,"tracked_observed_endpoint_coverage"=>recovery_passed,"heldout_inventory_available"=>true,"sampled_inventory_preserved"=>changes==0,"independent_endpoint_solved"=>true,"saved_target_root_count"=>1,"observed_target_root_count"=>1,"matched_target_roots"=>recovery_passed ? 1 : 0,"samples"=>[Dict("id"=>"sample$i") for i in 1:17],"changes"=>[Dict("first_point_id"=>"sample$i","second_point_id"=>"sample$(i+1)") for i in 1:changes])]))
    jobs=Dict{String,Any}[]
    for role in ("control","training","audit","heldout","recovery")
        relative="results/$role.toml"
        open(joinpath(directory,relative),"w") do io;TOML.print(io,Dict("status"=>"completed","evidence"=>evidence[role]));end
        push!(jobs,Dict("id"=>role,"kind"=>role,"status"=>"completed","elapsed_seconds"=>0.1,"payload"=>Dict("pilot_role"=>role,"pilot_stage"=>role,"pilot_cohort"=>"shell1/parameter/mask0","pilot_sample_index"=>role=="heldout" ? 3 : 1),"result_file"=>relative,"result_hash"=>bytes2hex(sha256(read(joinpath(directory,relative))))))
    end
    state=Dict("mode"=>"pilot","pilot"=>metadata,"jobs"=>jobs)
    open(joinpath(directory,"state.toml"),"w") do io;TOML.print(io,state);end
    state
end

@testset "Pilot report questions and denominators" begin
    mktempdir() do directory
        state=report_pilot_fixture(directory;changes=2)
        summary=atlas_pilot_summary(atlas_report_data(directory))
        @test summary["recommendations"] == ["explore"]
        @test summary["recovery_planned_targets"] == 1
        @test summary["recovery_tracked_targets"] == 1
        @test summary["recoveries"][1]["structural_changes_observed"] == 2
        @test summary["recoveries"][1]["sampled_signature_change_brackets"][1]["first_path_fraction"]==0.0
        @test summary["recoveries"][1]["sampled_signature_change_brackets"][1]["second_path_fraction"]==1/16
        @test summary["recovery_unresolved_or_unexecuted_targets"] == 0
        @test first(filter(r->r["role"]=="training_followup",summary["role_totals"]))["quota_semantics"] == "optional_capacity"
        write_report(directory)
        @test isfile(joinpath(directory,"pilot_summary.toml"))
        @test TOML.parsefile(joinpath(directory,"coverage.toml"))["observed_points"] == 2 # method control excluded
        @test !occursin("region_representative",read(joinpath(directory,"base_cases.csv"),String))
        @test occursin("Method controls (not discovery)",read(joinpath(directory,"index.html"),String))
        @test occursin("not novelty counts",read(joinpath(directory,"index.html"),String))
        # A queued held-out job keeps its assigned denominator and is not
        # counted a second time as a missing scientific result.
        heldout=only(filter(j->j["id"]=="heldout",state["jobs"]))
        heldout["status"]="pending"
        open(joinpath(directory,"state.toml"),"w") do io;TOML.print(io,state);end
        summary=atlas_pilot_summary(atlas_report_data(directory))
        @test "reallocate" in summary["recommendations"]
        @test only(filter(r->r["role"]=="heldout",summary["role_totals"]))["planned_jobs"] == 1
        @test count(i->i["job"]=="heldout",atlas_report_data(directory).issues) == 1
    end
    mktempdir() do directory
        report_pilot_fixture(directory;control_status="failed",recovery_passed=false,audit_status="disagreement_or_unresolved")
        summary=atlas_pilot_summary(atlas_report_data(directory))
        @test "repair" in summary["recommendations"]
        @test "reassess" in summary["recommendations"]
        @test !("explore" in summary["recommendations"])
        @test summary["recovery_unresolved_or_unexecuted_targets"] == 1
        @test any(reason->occursin("not evidence of cycle absence",reason),summary["reason_codes"])
    end
    mktempdir() do directory
        state=report_pilot_fixture(directory)
        delete!(state["pilot"],"manifest_counts")
        state["jobs"][1]["payload"]="malformed metadata"
        open(joinpath(directory,"state.toml"),"w") do io;TOML.print(io,state);end
        summary=atlas_pilot_summary(atlas_report_data(directory))
        @test summary["recovery_planned_targets"] == "not_recorded"
        @test summary["recovery_unresolved_or_unexecuted_targets"] == "unknown_planned_denominator"
        @test !("explore" in summary["recommendations"])
    end
end

function update_report_pilot_result!(edit, directory, role)
    state=TOML.parsefile(joinpath(directory,"state.toml"))
    job=only(filter(j->j["id"]==role,state["jobs"]))
    path=joinpath(directory,job["result_file"])
    result=TOML.parsefile(path);edit(result)
    open(path,"w") do io;TOML.print(io,result);end
    job["result_hash"]=bytes2hex(sha256(read(path)))
    open(joinpath(directory,"state.toml"),"w") do io;TOML.print(io,state);end
end

@testset "Pilot report limits and malformed evidence" begin
    mktempdir() do directory
        report_pilot_fixture(directory)
        update_report_pilot_result!(directory,"audit") do result
            depth=result["evidence"]["audits"][1]["continuation_pair"]
            depth["depth_extensions"][1]["new_candidate_count"]=2
            depth["depth_extensions"][1]["deep_termination"]="step_limit"
        end
        summary=atlas_pilot_summary(atlas_report_data(directory))
        @test "reallocate" in summary["recommendations"]
        @test !("repair" in summary["recommendations"])
        @test !("reassess" in summary["recommendations"])
        @test summary["paired_audits"][1]["extension_candidate_count"]==2
        @test summary["paired_audits"][1]["depth_comparison_status"]=="shared_prefix_consistent"
        @test any(reason->occursin("not_depth_saturation",reason),summary["reason_codes"])
        update_report_pilot_result!(directory,"audit") do result
            result["evidence"]["audits"][1]["comparisons"][1]["classification_resolved"]=false
        end
        @test "reallocate" in atlas_pilot_summary(atlas_report_data(directory))["recommendations"]
        update_report_pilot_result!(directory,"audit") do result
            result["evidence"]["audits"]="malformed evidence"
        end
        summary=atlas_pilot_summary(atlas_report_data(directory))
        @test "missing_or_duplicate_audit_scientific_evidence" in summary["reason_codes"]
        @test !("explore" in summary["recommendations"])
    end
    mktempdir() do directory
        report_pilot_fixture(directory)
        update_report_pilot_result!(directory,"recovery") do result
            result["evidence"]["recoveries"][1]["saved_target_root_count"]=0
        end
        summary=atlas_pilot_summary(atlas_report_data(directory))
        @test summary["recovery_tracked_targets"]==0
        @test summary["recovery_unresolved_or_unexecuted_targets"]==1
        @test summary["recoveries"][1]["reported_tracked_target_coverage"]===true
        @test summary["recoveries"][1]["endpoint_coverage_evidence_complete"]===false
        @test !("explore" in summary["recommendations"])
    end
    mktempdir() do directory
        state=report_pilot_fixture(directory)
        state["pilot"]["base_selection"]["frozen_before_heldout"]=false
        open(joinpath(directory,"state.toml"),"w") do io;TOML.print(io,state);end
        summary=atlas_pilot_summary(atlas_report_data(directory))
        @test !summary["frozen_base_selection"]["hash_verified"]
        @test "reassess" in summary["recommendations"]
    end
    mktempdir() do directory
        report_pilot_fixture(directory)
        marker="RAW_ATTEMPT_MUST_NOT_BE_DUPLICATED"
        update_report_pilot_result!(directory,"training") do result
            result["evidence"]["points"][1]["attempts"]=[Dict("reason"=>repeat(marker,10)) for _ in 1:100]
        end
        write_report(directory)
        @test !occursin(marker,read(joinpath(directory,"index.html"),String))
        @test !occursin(marker,read(joinpath(directory,"pilot_summary.toml"),String))
        @test occursin("see raw evidence",read(joinpath(directory,"index.html"),String))
    end
end

@testset "Pilot assigned followups and continuation failures remain open" begin
    mktempdir() do directory
        state=report_pilot_fixture(directory)
        relative="results/optional-followup.toml"
        open(joinpath(directory,relative),"w") do io;TOML.print(io,Dict("status"=>"failed","evidence"=>Dict()));end
        push!(state["jobs"],Dict("id"=>"optional-followup","kind"=>"nexus","status"=>"failed","elapsed_seconds"=>0.1,"payload"=>Dict("pilot_role"=>"training_followup","pilot_cohort"=>"shell1/parameter/mask0"),"result_file"=>relative,"result_hash"=>bytes2hex(sha256(read(joinpath(directory,relative))))))
        open(joinpath(directory,"state.toml"),"w") do io;TOML.print(io,state);end
        summary=atlas_pilot_summary(atlas_report_data(directory))
        @test "reallocate" in summary["recommendations"]
        @test !("explore" in summary["recommendations"])
        @test any(reason->occursin("assigned_optional_training_followup_unresolved",reason),summary["reason_codes"])
    end
    for termination in ("minimum_step","unknown")
        mktempdir() do directory
            report_pilot_fixture(directory)
            update_report_pilot_result!(directory,"audit") do result
                result["evidence"]["audits"][1]["continuation_pair"]["depth_extensions"][1]["deep_termination"]=termination
            end
            summary=atlas_pilot_summary(atlas_report_data(directory))
            @test "reassess" in summary["recommendations"]
            @test !("explore" in summary["recommendations"])
        end
    end
end

@testset "Pilot unknown recovery fields never establish target coverage" begin
    for (key,value) in (("status","unknown"),("tracked_observed_endpoint_coverage",false),("heldout_inventory_available",false),("saved_target_root_count",2))
        mktempdir() do directory
            report_pilot_fixture(directory)
            update_report_pilot_result!(directory,"recovery") do result
                result["evidence"]["recoveries"][1][key]=value
            end
            summary=atlas_pilot_summary(atlas_report_data(directory))
            @test summary["recovery_tracked_targets"]==0
            @test !("explore" in summary["recommendations"])
        end
    end
    mktempdir() do directory
        state=report_pilot_fixture(directory)
        duplicate=deepcopy(only(filter(j->j["id"]=="recovery",state["jobs"])))
        duplicate["id"]="duplicate-recovery"
        push!(state["jobs"],duplicate)
        state["pilot"]["manifest_counts"]["recovery"]=2
        open(joinpath(directory,"state.toml"),"w") do io;TOML.print(io,state);end
        summary=atlas_pilot_summary(atlas_report_data(directory))
        @test summary["recovery_evaluated_targets"]==1
        @test summary["recovery_tracked_targets"]==1
        @test "reassess" in summary["recommendations"]
        @test !("explore" in summary["recommendations"])
    end
end

@testset "Pilot expanded declared blocks retain per-index identities" begin
    mktempdir() do directory
        state=report_pilot_fixture(directory)
        manifest=state["pilot"]["manifest_counts"]
        merge!(manifest,Dict("control"=>3,"training"=>512,"audit"=>112,"heldout"=>256,"recovery"=>64))
        state["pilot"]["manifest"]["training_indices"]=[1,2,3,4]
        state["pilot"]["manifest"]["heldout_indices"]=[5,6]
        only(filter(j->j["id"]=="heldout",state["jobs"]))["payload"]["pilot_sample_index"]=5
        for (old_id,new_id,index) in (("heldout","heldout-six",6),("audit","audit-three",3),("recovery","recovery-six",6))
            original=only(filter(j->j["id"]==old_id,state["jobs"]))
            copied=deepcopy(original);copied["id"]=new_id;copied["payload"]["pilot_sample_index"]=index
            copied["result_file"]="results/$new_id.toml"
            result=TOML.parsefile(joinpath(directory,original["result_file"]))
            old_id=="heldout" && (result["evidence"]["points"][1]["id"]=new_id)
            old_id=="recovery" && (result["evidence"]["recoveries"][1]["target_job_id"]="heldout-six")
            open(joinpath(directory,copied["result_file"]),"w") do io;TOML.print(io,result);end
            copied["result_hash"]=bytes2hex(sha256(read(joinpath(directory,copied["result_file"]))))
            push!(state["jobs"],copied)
        end
        state["pilot"]["base_selection"]["recovery_routes"]=[Dict("cohort"=>"shell1/parameter/mask0","sample_index"=>index,"base_point_id"=>"training","selection_rule"=>"parameter_only") for index in (5,6)]
        stream=IOBuffer();TOML.print(stream,state["pilot"]["base_selection"];sorted=true)
        state["pilot"]["base_selection_hash"]=bytes2hex(sha256(take!(stream)))
        open(joinpath(directory,"state.toml"),"w") do io;TOML.print(io,state);end
        summary=atlas_pilot_summary(atlas_report_data(directory))
        @test summary["recovery_planned_targets"]==64
        @test summary["recovery_evaluated_targets"]==2
        @test summary["recovery_tracked_targets"]==2
        @test sort([row["target_sample_index"] for row in summary["recoveries"]])==[5,6]
        @test sort([row["sample_index"] for row in summary["paired_audits"]])==[1,3]
        @test [row["sample_index"] for row in summary["frozen_base_selection"]["recovery_routes"]]==[5,6]
        write_report(directory)
        @test occursin("64 prespecified targets out of 256 planned held-out points",read(joinpath(directory,"index.html"),String))
    end
end

@testset "Pilot classification uncertainty is distinct from inventory disagreement" begin
    mktempdir() do directory
        report_pilot_fixture(directory)
        update_report_pilot_result!(directory,"audit") do result
            comparison=result["evidence"]["audits"][1]["comparisons"][1]
            comparison["classification_resolved"]=false
            comparison["status"]="disagreement_or_unresolved"
        end
        summary=atlas_pilot_summary(atlas_report_data(directory))
        @test "reallocate" in summary["recommendations"]
        @test !("repair" in summary["recommendations"])
        @test !("reassess" in summary["recommendations"])
        @test summary["paired_audits"][1]["inventory_disagreement_comparisons"]==0
        update_report_pilot_result!(directory,"audit") do result
            result["evidence"]["audits"][1]["comparisons"][1]["status"]="unknown"
        end
        @test "reassess" in atlas_pilot_summary(atlas_report_data(directory))["recommendations"]
    end
    mktempdir() do directory
        report_pilot_fixture(directory)
        update_report_pilot_result!(directory,"heldout") do result
            result["status"]="unresolved"
        end
        @test atlas_pilot_summary(atlas_report_data(directory))["recovery_tracked_targets"]==0
    end
end

@testset "Pilot resolved-signature disagreements do not establish reliability" begin
    for repeat in (false,true)
        mktempdir() do directory
            report_pilot_fixture(directory)
            update_report_pilot_result!(directory,"audit") do result
                audit=result["evidence"]["audits"][1]
                comparison=repeat ? audit["training_repeat_comparison"] : audit["comparisons"][1]
                comparison["status"]="disagreement_or_unresolved"
                comparison["inventory_match"]=true
                comparison["classification_resolved"]=true
            end
            summary=atlas_pilot_summary(atlas_report_data(directory))
            @test "reassess" in summary["recommendations"]
            @test !("explore" in summary["recommendations"])
        end
    end
end

@testset "Dispersed design separation is cohort-local and position-bound" begin
    designs=[Dict("cohort"=>"A","indices"=>[1,4,7,9,20,30],"training_indices"=>[1,4,7,9],"heldout_indices"=>[20,30],"audit_indices"=>[1,7]),
        Dict("cohort"=>"B","indices"=>[20,40,50,60,1,70],"training_indices"=>[20,40,50,60],"heldout_indices"=>[1,70],"audit_indices"=>[20,50])]
    manifest=Dict("selection_algorithm"=>"SHA-256 lexicographic rank, numeric-index tie-break","selection_seed"=>"pilot-v1","selection_pool"=>collect(1:256),"selection_pool_bounds"=>[1,256],"cohort_designs"=>designs)
    jobs=[Dict("payload"=>Dict("pilot_role"=>role,"pilot_cohort"=>cohort,"pilot_sample_index"=>index,"pilot_design_position"=>position)) for (role,cohort,index,position) in (("training","A",1,1),("heldout","A",20,5),("training","B",20,1),("heldout","B",1,5),("audit","A",7,3))]
    @test _report_pilot_design(manifest,jobs)["verified"] # same ordinal may cross roles in another cohort
    @test _report_pilot_design(manifest,jobs)["selection_seed"]=="pilot-v1"
    malformed=deepcopy(jobs);malformed[2]["payload"]["pilot_design_position"]=1
    @test !_report_pilot_design(manifest,malformed)["verified"]
    malformed=deepcopy(jobs);malformed[2]["payload"]["pilot_sample_index"]=1
    @test !_report_pilot_design(manifest,malformed)["verified"]
    malformed=deepcopy(manifest);malformed["cohort_designs"][1]["heldout_indices"]=[1,30]
    @test !_report_pilot_design(malformed,jobs)["verified"]
    malformed=deepcopy(manifest);push!(malformed["cohort_designs"],deepcopy(designs[1]))
    @test !_report_pilot_design(malformed,jobs)["verified"]
end
