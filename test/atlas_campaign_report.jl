module CampaignReportTests
using Test,TOML,SHA
include(joinpath(@__DIR__,"..","scripts","atlas","report.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","campaign_report.jl"))
function fixture(;n=0,status="pending",result_status=status,role="inventory",round=1)
    jobs=[Dict("id"=>"j$i","status"=>status,"payload"=>Dict("campaign_role"=>role,"campaign_round"=>round,"campaign_shell"=>1,"campaign_chart"=>"parameter","boundary_mask"=>0)) for i in 1:n]
    results=Dict{String,Any}()
    for job in jobs
        status=="pending" && continue
        id=job["id"]
        results[id]=Dict("status"=>result_status,"source_result"=>"results/$id.toml","evidence"=>Dict("points"=>[Dict("id"=>"p$id","signature"=>"one_attractor","roots"=>[Dict("state"=>[0.1,0.2])])]))
    end
    (;jobs,results,state=Dict("campaign"=>Dict("stage"=>"round1_inventory","bases_frozen"=>false)),manifest=Dict(),config=Dict(),issues=Dict{String,Any}[])
end
@testset "Empty, failed-control, partial, and finished execution" begin
    empty=campaign_report_summary(fixture())
    @test empty["status"]["resolved"]==0
    @test empty["completeness"]=="CompletenessNotCertified"
    data=fixture(n=20);data.jobs[1]["status"]="failed"
    summary=campaign_report_summary(data)
    @test summary["status"]["pending"]==19
    @test summary["status"]["attempted"]==1
    @test summary["status"]["resolved"]==0
    @test length(summary["unresolved_frontier"])==20
    partial=fixture(n=2,status="completed",result_status="unresolved")
    summary=campaign_report_summary(partial)
    @test summary["status"]["execution_completed"]==2
    @test summary["status"]["resolved"]==0
    @test isempty(summary["graph"]["sampled_components"])
    complete=campaign_report_summary(fixture(n=2,status="completed"))
    @test complete["status"]["resolved"]==2
    @test length(complete["graph"]["sampled_components"])==2 # equal signature never connects
    @test isempty(complete["graph"]["edges"])
end
@testset "No held-out leakage and explicit components" begin
    training=fixture(n=129,status="completed")
    ids=["pj$i" for i in 1:129]
    edge=Dict("id"=>"path","from"=>first(ids),"to"=>last(ids),"sample_ids"=>ids,"status"=>"completed","validated"=>true,"signature_preserved"=>true,"forward_tracking_unambiguous"=>true,"audit_agreement"=>true,"tracking_guard_failures"=>String[])
    training.results["j1"]["evidence"]["paths"]=[edge]
    summary=campaign_report_summary(training)
    @test length(summary["graph"]["sampled_components"])==1
    @test summary["graph"]["edges"][1]["kind"]=="sampled_structural_preservation"
    edge["audit_agreement"]=false
    @test length(campaign_report_summary(training)["graph"]["sampled_components"])==129
    @test campaign_report_summary(training)["graph"]["edges"][1]["kind"]=="uncertainty"
    held=campaign_report_summary(fixture(n=2,status="completed",role="validation_inventory",round=4))
    @test isempty(held["graph"]["sampled_components"])
    @test all(r->r["saved_resolved_inventory_records"]==0,held["round_novelty"])
end
@testset "Presentation discloses bounded claims and raw source links" begin
    data=fixture(n=1,status="completed")
    data.results["j1"]["evidence"]["points"][1]["id"]="<script>"
    summary=campaign_report_summary(data)
    io=IOBuffer();_campaign_render_html(io,summary,data);html=String(take!(io))
    @test occursin("&lt;script&gt;",html)
    @test !occursin("<summary>inventory · <script>",html)
    @test occursin("CompletenessNotCertified",html)
    @test occursin("projected domains differ",html)
    @test occursin("missing-region probability",html)
    @test occursin("results/j1.toml",html)
    @test occursin("Frozen before validation: false",html)
    @test occursin("256 held-out",html)
    @test !haskey(summary["graph"]["nodes"][1],"roots") # do not duplicate raw numerical tuples
end
@testset "Graph endpoints link only to established evidence nodes" begin
    data=fixture(n=1,status="completed")
    summary=campaign_report_summary(data)
    push!(summary["graph"]["edges"],Dict("kind"=>"uncertainty","from"=>"pj1","to"=>"missing-witness"))
    io=IOBuffer();_campaign_render_html(io,summary,data);html=String(take!(io))
    @test occursin("From evidence</a>",html)
    @test occursin("To evidence: unresolved endpoint (missing-witness)",html)
    @test !occursin("href=#"*_campaign_anchor("missing-witness"),html)
    @test !occursin("From inventory</a>",html)
end
@testset "Unresolved classification and direct audit exclusion" begin
    data=fixture(n=1,status="completed")
    data.results["j1"]["evidence"]["points"][1]["signature"]="unresolved_spectrum"
    summary=campaign_report_summary(data)
    @test summary["status"]["resolved"]==0
    @test isempty(summary["graph"]["sampled_components"])
    data=fixture(n=1,status="completed")
    push!(data.jobs,Dict("id"=>"audit","status"=>"failed","payload"=>Dict("campaign_role"=>"audit","training_source_job"=>"j1")))
    summary=campaign_report_summary(data)
    @test isempty(summary["graph"]["sampled_components"])
    @test !summary["graph"]["nodes"][1]["eligible"]
    data.state["campaign"]["base_selection"]=Dict("entries"=>[Dict("point_id"=>"pj1","members"=>["pj1"],"roots"=>[Dict("raw_numerics"=>999)],"parameters"=>Dict("A"=>1.0))])
    @test !haskey(campaign_report_summary(data)["bases"]["entries"][1],"roots")
end
@testset "Completed execution retains explicit unresolved scientific records" begin
    function path_packet(record)
        data=fixture(n=1,status="completed",role="component_path")
        data.results["j1"]["evidence"]["paths"]=[merge(Dict{String,Any}("id"=>"interval1","from"=>"pj1","to"=>"pj1","status"=>"signature_change_observed","crosscheck_agreement"=>true,"validated"=>false,"continuous_coverage_certified"=>false,"completeness"=>"CompletenessNotCertified"),record)]
        data
    end
    clean=path_packet(Dict())
    summary=campaign_report_summary(clean)
    @test summary["status"]["execution_completed"]==1
    @test summary["status"]["fully_resolved"]==1
    @test summary["status"]["completed_with_unresolved"]==0
    @test isempty(summary["unresolved_frontier"])
    @test summary["graph"]["edges"][1]["kind"]=="sampled_transition"
    for (fields,reason) in ((Dict("tracking_guard_failures"=>["missing_root"]),"tracking_guard_failures"),
                            (Dict("audit_agreement"=>false),"audit_disagreement_or_unresolved"),
                            (Dict("localization_stop_reason"=>"split_cap"),"localization_split_cap"),
                            (Dict("localization_stop_reason"=>"inventory_cap"),"localization_inventory_cap"),
                            (Dict("corrector_status"=>"unresolved_corrector_cap"),"unresolved_corrector_cap"))
        data=path_packet(fields);summary=campaign_report_summary(data)
        @test summary["status"]["execution_completed"]==1
        @test summary["status"]["fully_resolved"]==0
        @test summary["status"]["completed_with_unresolved"]==1
        @test summary["graph"]["nodes"][1]["eligible"] # execution eligibility unchanged
        @test any(row->get(row,"record_id","")=="interval1" && get(row,"reason","")==reason && row["source_job"]=="j1" && row["source_result"]=="results/j1.toml",summary["unresolved_frontier"])
        io=IOBuffer();_campaign_render_html(io,summary,data);html=String(take!(io))
        @test occursin(reason,html)
        @test occursin("completed with unresolved evidence",html)
    end
    transition=path_packet(Dict())
    push!(transition.results["j1"]["evidence"]["paths"],Dict("id"=>"whole-path","kind"=>"unknown_connection","status"=>"unresolved","validated"=>false,"audit_agreement"=>true,"forward_tracking_unambiguous"=>true,"tracking_guard_failures"=>String[]))
    @test campaign_report_summary(transition)["status"]["fully_resolved"]==1
    @test isempty(campaign_report_summary(transition)["unresolved_frontier"])
    transition.results["j1"]["evidence"]["paths"][1]["crosscheck_agreement"]=false
    @test campaign_report_summary(transition)["status"]["fully_resolved"]==0
    # A candidate is unresolved only when its saved diagnostics say so.
    clean.results["j1"]["evidence"]["events"]=[Dict("id"=>"trace","kind"=>"trace_zero","status"=>"candidate","validated"=>false)]
    @test campaign_report_summary(clean)["status"]["fully_resolved"]==1
    clean.results["j1"]["evidence"]["events"][1]["status"]="unresolved_corrector"
    @test campaign_report_summary(clean)["status"]["fully_resolved"]==0
end
@testset "Frozen inputs are mandatory" begin
    mktempdir() do output
        for name in ("state.toml","config.toml","campaign_manifest.toml")
            write(joinpath(output,name),"")
        end
        @test_throws ArgumentError campaign_report_data(output)
        write(joinpath(output,"index.html"),"previous")
        @test_throws ArgumentError write_campaign_report(output)
        @test read(joinpath(output,"index.html"),String)=="previous"
    end
end
end

module CampaignReportContractTests
using Test
const R=getfield(parentmodule(@__MODULE__),:CampaignReportTests)
const container=parentmodule(@__MODULE__)
function packet(jobs,results)
    (;jobs,results,state=Dict(),manifest=Dict(),config=Dict(),issues=[])
end
@testset "Event labels and incident intervals use adapter fields" begin
    data=R.fixture(n=2,status="completed",role="component_path")
    data.results["j1"]["evidence"]["events"]=[Dict("id"=>"fold","kind"=>"fold","validated"=>true,"status"=>"completed","source_interval_id"=>"interval"),Dict("id"=>"trace","kind"=>"trace_zero","validated"=>false,"status"=>"candidate","source_interval_id"=>"interval")]
    data.results["j1"]["evidence"]["paths"]=[Dict("id"=>"interval","from"=>"pj1","to"=>"pj2","status"=>"crosscheck_disagreement","crosscheck_agreement"=>false,"linked_event_ids"=>["fold","trace"])]
    graph=R.campaign_evidence_graph(data)
    witnesses=filter(n->n["node_type"]=="event_witness",graph["nodes"])
    @test Set(n["classification"] for n in witnesses)==Set(["numerically_validated_fold_witness","trace_zero_candidate"])
    sides=filter(e->e["kind"]=="event_side_context",graph["edges"])
    @test length(sides)==4
    @test all(e->e["status"]=="unresolved_incident_interval" && !e["interval_validation_inferred_from_event"],sides)
end
if isdefined(container,:AtlasCampaignScienceTests)
    const E=getfield(container,:AtlasCampaignScienceTests)
    @testset "Actual adapter outputs populate compact campaign report" begin
        p=E.parameters();q=copy(p);q["a"]=.2
        left=E.point("left",p);right=E.point("right",q)
        path_result=E.execute_campaign_job(Dict("kind"=>"campaign_path","id"=>"path","from"=>left,"to"=>right),Dict();point_solver=E.fake_point,track_solver=E.fake_tracks)
        jobs=[Dict("id"=>id,"status"=>"completed","payload"=>Dict("campaign_role"=>id=="path" ? "component_path" : "inventory","campaign_round"=>1)) for id in ("left","right","path")]
        results=Dict(id=>Dict("status"=>"completed","source_result"=>"results/$id.toml","evidence"=>id=="path" ? path_result["evidence"] : Dict("points"=>[id=="left" ? left : right])) for id in ("left","right","path"))
        summary=R.campaign_report_summary(packet(jobs,results))
        @test first(path_result["evidence"]["paths"])["validated"]
        @test length(summary["graph"]["sampled_components"])==1
        @test summary["status"]["fully_resolved"]==3
        @test length(only(summary["graph"]["sampled_components"]))==129
        @test count(e->e["kind"]=="fixed_parameter_inventory_audit",summary["graph"]["edges"])==4
        base=E.point("base",p);base["point_id"]="base"
        target=E.point("held",p;roots=[E.root(.3),E.root(.4)])
        result=E.execute_campaign_job(Dict("kind"=>"campaign_recovery","id"=>"recovery","base"=>base,"target_parameters"=>p,"campaign_target_job"=>"held-job","validation_evidence"=>Dict("points"=>[target]),"validation_inventory_status"=>"completed"),Dict();point_solver=E.fake_point,track_solver=E.fake_tracks)
        recovery_job=Dict("id"=>"recovery","status"=>result["status"],"payload"=>Dict("campaign_role"=>"validation_recovery","campaign_round"=>4,"campaign_mask"=>15,"campaign_target_job"=>"held-job"))
        summary=R.campaign_report_summary(packet([recovery_job],Dict("recovery"=>Dict("status"=>result["status"],"source_result"=>"results/recovery.toml","evidence"=>result["evidence"]))))
        recovery=only(summary["recoveries"])
        @test recovery["target_job_id"]=="held-job"
        @test recovery["observed_target_root_count"]==2
        @test recovery["matched_target_roots"]==1
        @test !recovery["forward_target_coverage"]
        @test recovery["sampled_structural_preservation"]=="not_assessed"
        @test summary["coverage_ledger"][1]["boundary_mask"]=="15"
        @test isempty(summary["graph"]["sampled_components"])
    end
else
    @testset "Actual adapter integration requires science fixture module" begin
        @test_skip false
    end
end
if isdefined(container,:AtlasCampaignTests)
    const C=getfield(container,:AtlasCampaignTests)
    @testset "Actual scheduler strata retain boundary masks" begin
        manifest=C.fixture_manifest()
        samples=C.campaign_sample_jobs(manifest,1)
        sample=first(filter(p->p["campaign_mask"]==15,samples))
        job=Dict("id"=>"scheduled","status"=>"pending","payload"=>sample)
        ledger=only(R.campaign_report_summary(packet([job],Dict()))["coverage_ledger"])
        @test ledger["boundary_mask"]=="15"
        @test ledger["chart"]==sample["campaign_chart"]
    end
else
    @testset "Actual scheduler integration requires scheduler fixture module" begin
        @test_skip false
    end
end
end

# Exercise execute_job's real evidence wrapping with a deterministic saved-event
# producer. No numerical solver is run by this fixture.
module CampaignReportScientificWrapperTests
using Test
include(joinpath(@__DIR__,"..","scripts","atlas","science.jl"))
execute_event_job(job,config)=deepcopy(job["saved_event"])
const R=getfield(parentmodule(@__MODULE__),:CampaignReportTests)
function wrapped_summary(kind,record)
    result=execute_job(Dict{String,Any}("id"=>"scientific","kind"=>kind,"saved_event"=>record),Dict{String,Any}())
    job=Dict("id"=>"scientific","status"=>result["status"],"payload"=>Dict("campaign_role"=>kind=="section" ? "nexus_section" : "proposal"))
    data=(;jobs=[job],results=Dict("scientific"=>Dict("status"=>result["status"],"source_result"=>"results/scientific.toml","evidence"=>result["evidence"])),state=Dict(),config=Dict(),manifest=Dict(),issues=[])
    R.campaign_report_summary(data)
end
@testset "Real scientific wrapper preserves section and cycle uncertainties" begin
    for termination in ("unresolved_event_rank","unresolved_corrector","unresolved_saved_tuple_residual")
        raw=Dict("status"=>"sampled","section_samples"=>Any[],"transition_curves"=>[Dict("event"=>"fold","direction"=>1,"termination"=>termination)])
        summary=wrapped_summary("section",raw)
        @test summary["status"]["execution_completed"]==1
        @test summary["status"]["fully_resolved"]==0
        @test any(row->get(row,"reason","")=="termination:"*termination && row["record_id"]=="sections/1/transition_curves/1" && row["source_result"]=="results/scientific.toml",summary["unresolved_frontier"])
    end
    for termination in ("step_limit","parameter_boundary","initial_point_not_on_event_locus")
        @test wrapped_summary("section",Dict("status"=>"sampled","section_samples"=>Any[],"transition_curves"=>[Dict("termination"=>termination)]))["status"]["fully_resolved"]==1
    end
    for sample in (Dict("status"=>"sampled","equilibria"=>Any[]),Dict("status"=>"sampled","equilibria"=>[Dict("stability"=>"StabilityUnresolved")]),Dict("status"=>"outside_admissible_domain"))
        @test wrapped_summary("section",Dict("status"=>"sampled","section_samples"=>[sample]))["status"]["fully_resolved"]==0
    end
    for orbit in (Dict("validation"=>"PeriodicOrbitUnresolved"),Dict("validation"=>"NumericallyValidatedPeriodicOrbit","stability"=>"PeriodicOrbitStabilityUnresolved"))
        @test wrapped_summary("periodic",Dict("status"=>"numerically_validated","orbit"=>orbit))["status"]["fully_resolved"]==0
    end
    for termination in ("initial_equilibrium_unresolved","initial_orbit_unresolved","initial_phase_unresolved","initial_tangent_unresolved","minimum_step")
        @test wrapped_summary("periodic",Dict("status"=>"numerically_validated","branches"=>[Dict("termination"=>termination)]))["status"]["fully_resolved"]==0
    end
end
@testset "Replacement summaries and refinement scope" begin
    job=Dict("id"=>"replacement","status"=>"completed","payload"=>Dict("campaign_role"=>"replacement"))
    function diagnostics(evidence)
        R._campaign_unresolved_diagnostics(job,Dict("status"=>"completed","source_result"=>"results/replacement.toml","evidence"=>evidence))
    end
    for branch in (Dict("finite_outcome"=>"unresolved_96_step_limit"),Dict("prefix_comparison"=>Dict("status"=>"shared_prefix_disagreement_or_unresolved")),Dict("points"=>[Dict("stability"=>"StabilityUnresolved")]))
        @test !isempty(diagnostics(Dict("targeted_continuations"=>[Dict("branches"=>[branch])])) )
    end
    @test !isempty(diagnostics(Dict("targeted_transitions"=>[Dict("initial_brackets"=>[Dict("source_endpoint_inventories_reproduced"=>false,"source_endpoint_comparisons"=>[Dict("status"=>"disagreement_or_unresolved")])])])) )
    @test !isempty(diagnostics(Dict("targeted_recoveries"=>[Dict("terminal_grid_comparison"=>Dict("status"=>"disagreement_or_unresolved"))])))
    @test !isempty(diagnostics(Dict("targeted_recoveries"=>[Dict("independent_points"=>[Dict("id"=>"independent","signature"=>"unresolved_search","roots"=>Any[])])])) )
    record=Dict{String,Any}("base_origin_target_coverage"=>true,"levels"=>[Dict("grid_size"=>33,"base_origin_target_coverage"=>false),Dict("grid_size"=>129,"base_origin_target_coverage"=>true)],"original_level17"=>Dict("status"=>"unresolved"),"attempts"=>[Dict("status"=>"unresolved")])
    @test isempty(diagnostics(Dict("targeted_recoveries"=>[record])))
    record["levels"][2]["reverse_target_origin_base_coverage"]=false
    rows=diagnostics(Dict("targeted_recoveries"=>[record]))
    @test length(rows)==1
    @test rows[1]["reason"]=="reverse_target_origin_base_coverage_not_established"
    @test rows[1]["record_id"]=="targeted_recoveries/1/levels/2"
    @test record["base_origin_target_coverage"] # reporting does not alter forward evidence
    @test isempty(diagnostics(Dict("events"=>[Dict("kind"=>"fold","validated"=>false,"status"=>"candidate")])))
    @test !isempty(diagnostics(Dict("events"=>[Dict("kind"=>"fold","validated"=>false,"status"=>"candidate","coefficients"=>Dict("resolved"=>false,"reason"=>"unresolved_zero_eigenvalue_pairing"))])))
end
end
