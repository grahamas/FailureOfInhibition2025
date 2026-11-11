module AtlasTargetedTests
using Test,TOML,SHA,LinearAlgebra,FailureOfInhibition2025
include(joinpath(@__DIR__,"..","scripts","atlas","core.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","events.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","science.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","pilot.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","targeted.jl"))
# In-memory synthetic source packet for pure manifest/selection tests. Production
# supplies hash-verified targeted_read_result from the protocol module.
targeted_read_result(packet,id)=deepcopy(packet["fixture_results"][id])
function fixture_packet()
    p=Dict{String,Any}("A"=>0.,"B"=>0.,"C"=>0.,"D"=>0.,"a"=>0.,"b"=>0.,"h"=>2.,"r"=>1.)
    point(id,params)=Dict("id"=>id,"parameters"=>deepcopy(params),"signature"=>"fixture_signature","roots"=>[Dict("state"=>[0.3,0.25])])
    jobs=Any[];results=Dict{String,Any}();cohorts=Any[]
    add(id,payload,result)=begin push!(jobs,Dict("id"=>id,"payload"=>payload));results[id]=result end
    for (index,name) in enumerate(("bt","cusp","hopf"))
        id="job-"*lpad(string(index),6,'0')
        add(id,Dict("kind"=>name=="hopf" ? "periodic" : "nexus","pilot_role"=>"control","pilot_control"=>name),Dict())
    end
    for (id,chart) in (("job-000414","parameter"),("job-000509","equilibrium"))
        payload=Dict("pilot_cohort"=>id,"pilot_chart"=>chart,"pilot_sample_index"=>17)
        add(id,payload,Dict("error"=>"fixture_singular_solver"))
        push!(cohorts,Dict("cohort"=>id,"samples"=>[Dict("accepted_index"=>17,"parameters"=>p,"chart_coordinates"=>[0.,0.,0.,0.,0.,0.,2.])]))
    end
    bases=Any[]
    for index in 1:67
        id="job-"*lpad(string(100+index),6,'0');q=copy(p);q["a"]=index/10
        base=Dict("point_id"=>id,"source_job"=>id,"parameters"=>q,"signature"=>"fixture_signature","roots"=>[Dict("state"=>[0.3,0.25])])
        push!(bases,base);add(id,Dict(),Dict())
    end
    samples=[point("old-$i",p) for i in 1:17]
    row=Dict("samples"=>samples,"changes"=>Any[],"tracks"=>Any[],"tracked_target_coverage"=>false,
        "sampled_inventory_preserved"=>false,"observed_target_root_count"=>1,"matched_target_roots"=>0)
    for id in ("job-000929","job-000941","job-000953","job-000959","job-000946","job-000954")
        target=point(id*"-target",p);base=deepcopy(bases[1])
        add(id,Dict("base"=>base,"target"=>target,"pilot_target_id"=>id*"-target"),Dict("evidence"=>Dict("recoveries"=>[row])))
    end
    for id in ("job-000872","job-000876","job-000877")
        q=copy(p);q["a"]=0.21
        add(id,Dict(),Dict("evidence"=>Dict("points"=>[point(id,q)])))
    end
    for id in ("job-000584","job-000585","job-000598","job-000608","job-000613","job-000627","job-000637")
        pair=Dict("bounds"=>[0.,1.],"initial_state"=>[0.3,0.25],"shallow"=>Any[],"deep"=>Any[])
        add(id,Dict("training_point"=>point(id,p)),Dict("evidence"=>Dict("audits"=>[Dict("continuation_pair"=>pair)])))
    end
    Dict("state"=>Dict("jobs"=>jobs,"pilot"=>Dict("manifest"=>Dict("cohort_designs"=>cohorts),"base_selection"=>Dict("entries"=>bases))),"fixture_results"=>results)
end
@testset "Fixed targeted manifest and parameter-only routing" begin
    packet=fixture_packet();before=deepcopy(packet)
    jobs=build_targeted_jobs(packet,Dict())
    @test length(jobs)==22
    @test [job["target_role"] for job in jobs]==vcat(fill("control",3),fill("diagnosis",2),fill("recovery_refinement",4),fill("multiroot",3),fill("transition",3),fill("audit_extension",7))
    @test all(get(job,"pilot_role","")=="control" for job in jobs[1:3])
    @test all(job["config_overrides"]["balance_atol"]==1e-11 for job in jobs)
    @test jobs[5]["known_chart_coordinates"]==[0.,0.,2.]
    @test all(job["base"]["point_id"]=="job-000102" for job in jobs[10:12])
    @test packet==before
    changed=deepcopy(packet)
    for id in ("job-000872","job-000876","job-000877")
        changed["fixture_results"][id]["evidence"]["points"][1]["signature"]="different_observed_signature"
    end
    @test [job["base"]["point_id"] for job in build_targeted_jobs(changed,Dict())[10:12]]==[job["base"]["point_id"] for job in jobs[10:12]]
    @test TOML.parse(sprint(io->TOML.print(io,Dict("jobs"=>jobs))))["jobs"][6]["original_level17"]["grid_size"]==17
    @test targeted_scientific_policy()["tracking_grids"]==[33,65,129]
end
@testset "Singular-grid diagnosis retains attempts and known root" begin
    p=Dict{String,Any}("A"=>7.03125,"B"=>0.,"C"=>0.,"D"=>0.,"a"=>1.2677485756974058,"b"=>-9.955393718707331,"h"=>11.112014199781825,"r"=>1.)
    job=Dict{String,Any}("kind"=>"targeted_diagnosis","id"=>"diagnosis-test","source_job_id"=>"job-000509","parameters"=>p,
        "known_chart_coordinates"=>[2.0315552216378627,9.955393718707331,11.112014199781825],"source_error"=>"first-pilot SingularException")
    result=execute_targeted_job(job,Dict{String,Any}());row=only(result["evidence"]["targeted_diagnoses"])
    @test [variant["grid_size"] for variant in row["variants"]]==[5,9,13]
    @test [length(variant["point"]["attempts"]) for variant in row["variants"]]==[26,82,170]
    @test all(!isempty(variant["point"]["roots"]) for variant in row["variants"])
    @test sum(variant["singular_solver_attempt_count"] for variant in row["variants"])>0
    @test row["known_chart_seed_included"]
    @test row["analytic_diagnostics"]["equilibrium_unique"]
    @test (result["status"]=="completed")==row["inventory_agreement"]
    @test TOML.parse(sprint(io->TOML.print(io,result)))["status"]==result["status"]
end
@testset "Nested tracking keeps origin and independent inventories separate" begin
    config=Dict{String,Any}("balance_atol"=>1e-11,"event_atol"=>1e-9,"seed_grid"=>9)
    p=Dict{String,Any}("A"=>0.,"B"=>0.,"C"=>0.,"D"=>0.,"a"=>0.,"b"=>0.,"h"=>2.,"r"=>1.)
    q=copy(p);q["a"]=0.2;q["b"]=0.1
    base=atlas_point(p,config;id="base");target=atlas_point(q,config;id="target")
    job=Dict{String,Any}("kind"=>"targeted_recovery","id"=>"nested-test","source_job_id"=>"fixture","target_job_id"=>"fixture-target","target_role"=>"recovery_refinement",
        "base"=>Dict("point_id"=>"base","parameters"=>p,"roots"=>base["roots"]),"target"=>target)
    result=execute_targeted_job(job,config);row=only(result["evidence"]["targeted_recoveries"])
    @test [level["grid_size"] for level in row["levels"]]==[33,65,129]
    @test all(level["base_origin_target_coverage"] for level in row["levels"])
    @test row["base_origin_target_coverage"]
    @test row["sampled_inventory_preserved"]
    @test row["analytic_diagnostics"]["uncoupled_unique_attracting_path"]
    @test row["inventory_locations"]==129
    @test length(row["independent_points"])==129
    @test all(length(point["attempts"])==81 for point in row["independent_points"])
    pairwise=filter(edge->edge["kind"]=="finite_same_signature_samples",row["adjacency"]["edges"])
    @test length(pairwise)==128
    @test all(!edge["validated"] for edge in pairwise)
    whole=only(filter(edge->edge["kind"]=="sampled_inventory_preserved",row["adjacency"]["edges"]))
    @test length(whole["sample_ids"])==129 && whole["validated"]
    @test !whole["continuous_coverage_certified"]
    @test all(level["target_origin_reverse"]["origin"]=="target_origin_reverse" for level in row["levels"])
    @test !isempty(TOML.parse(sprint(io->TOML.print(io,result))))
    # Target has three equilibria while the original base has one. Reverse
    # target-seeded tracks cannot turn one forward track into coverage of three.
    three=equilibrium_chart(12.,0.,0.,0.,-log(2.),0.,log(9.))
    low=copy(three["params"]);low["A"]=7.
    low_point=atlas_point(low,config;id="low-base");target_point=atlas_point(three["params"],config;id="three-target")
    @test length(low_point["roots"])==1
    @test length(target_point["roots"])==3
    multi=merge(job,Dict("target_role"=>"multiroot","base"=>Dict("point_id"=>"low-base","parameters"=>low,"roots"=>low_point["roots"]),"target"=>target_point))
    output=execute_targeted_job(multi,config);record=only(output["evidence"]["targeted_recoveries"])
    @test !record["base_origin_target_coverage"]
    @test record["observed_target_root_count"]==3
    @test record["terminal_grid13_assessed"]
    @test record["terminal_grid_comparison"]["status"]=="agreement_observed"
    @test all(length(level["base_origin_forward"]["endpoint_clean_tracks"])<=1 for level in record["levels"])
    @test all(level["target_origin_reverse"]["initial_track_count"]==3 for level in record["levels"])
    @test output["status"]=="unresolved"
end
@testset "Triangular scalar fold windows are diagnostics, not detour runs" begin
    p=Dict{String,Any}("A"=>8.4375,"B"=>0.,"C"=>0.5376,"D"=>0.,"a"=>2.585299626609738,"b"=>5.607116425868363,"h"=>0.2957017184817085,"r"=>1.)
    q=Dict{String,Any}("A"=>7.03125,"B"=>0.,"C"=>0.6528,"D"=>0.,"a"=>3.693838405049577,"b"=>0.9181371761819133,"h"=>6.9277532385684015,"r"=>1.)
    diagnostics=targeted_analytic_path(p,q);windows=diagnostics["scalar_fold_windows"]
    @test length(windows)==2
    @test all(window->window["original_sample_bracket"]==[3,4],windows)
    @test all(window->window["threshold_residual"]<=1e-11&&window["iterations"]<=64,windows)
    @test all(!window["certified"] for window in windows)
    @test diagnostics["alternative_detour_not_run"]
    a=copy(p);a["A"]=12.;a["a"]=targeted_fold_threshold(12.,"minimum")
    b=copy(a);b["a"]+=0.001
    endpoints=targeted_analytic_path(a,b)["scalar_fold_windows"]
    left=only(filter(window->window["turn"]=="minimum",endpoints))
    @test left["t"]==0.0
    @test left["threshold_residual"]==0.0
    @test left["iterations"]==0
    c=copy(a);c["a"]-=0.001
    right=only(filter(window->window["turn"]=="minimum",targeted_analytic_path(c,a)["scalar_fold_windows"]))
    @test right["t"]==1.0
    @test right["threshold_residual"]==0.0
    @test targeted_scientific_policy()["scalar_fold_iteration_cap"]==64
end
end
