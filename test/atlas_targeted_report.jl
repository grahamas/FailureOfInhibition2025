module TargetedReportTests
using Test,TOML,SHA
include(joinpath(@__DIR__,"..","scripts","atlas","report.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","targeted_report.jl"))

point(id;signature="resolved",roots=1)=Dict{String,Any}("id"=>id,"signature"=>signature,"parameters"=>Dict("A"=>1.0),"roots"=>[Dict("state"=>[0.1,0.2],"residual"=>1e-12) for _ in 1:roots])
function graph_record(;n=3,preserved=false)
    points=[point("p$i") for i in 1:n]
    edges=Dict{String,Any}[]
    Dict{String,Any}("sampled_inventory_preserved"=>preserved,"adjacency"=>Dict("nodes"=>points,"edges"=>edges))
end

@testset "Targeted inventory evidence is not signature connectivity" begin
    record=graph_record(n=2)
    graph=targeted_graph(record;source_job="case")
    @test isempty(graph["edges"])
    @test length(graph["display_groups"])==2
    push!(record["adjacency"]["edges"],Dict("from"=>"p1","to"=>"p2","kind"=>"inventory_sample_segment","status"=>"finite_same_signature_samples","validated"=>false))
    graph=targeted_graph(record;source_job="case")
    @test graph["edges"][1]["kind"]=="finite_interval"
    @test length(graph["display_groups"])==2
    record["adjacency"]["edges"][1]["kind"]="sampled_inventory_preserved"
    record["adjacency"]["edges"][1]["status"]="completed"
    record["sampled_inventory_preserved"]=true
    @test targeted_graph(record)["edges"][1]["kind"]!="sampled_inventory_link" # two endpoints insufficient
    full=graph_record(n=129,preserved=true)
    edge=Dict("from"=>"p1","to"=>"p129","kind"=>"sampled_inventory_preserved","status"=>"completed","validated"=>true,"sample_ids"=>["p$i" for i in 1:129],"declared_sample_count"=>129)
    push!(full["adjacency"]["edges"],edge)
    graph=targeted_graph(full;source_job="full")
    @test graph["edges"][1]["kind"]=="sampled_inventory_link"
    @test length(graph["display_groups"])==3
    @test sort(vcat([g["point_ids"] for g in graph["display_groups"]]...))==sort(["p$i" for i in 1:129])
    full["adjacency"]["nodes"][64]["signature"]="unresolved"
    @test targeted_graph(full)["edges"][1]["kind"]!="sampled_inventory_link"
    @test_throws ArgumentError targeted_graph(Dict("points"=>[point("dup"),point("dup")]))
end

@testset "Tracking origins and change evidence remain distinct" begin
    record=graph_record(n=3)
    for (from,to,origin) in (("p1","p2","base_origin_forward"),("p3","p2","target_origin_reverse"))
        push!(record["adjacency"]["edges"],Dict("from"=>from,"to"=>to,"kind"=>"tracked_root_link","origin"=>origin,"status"=>"clean_sampled_corrector_link","validated"=>true,"track_clean"=>true,"track"=>1,"from_state"=>[0.1,0.2],"to_state"=>[0.1,0.2]))
    end
    graph=targeted_graph(record)
    @test getindex.(graph["edges"],"kind")==["forward_track","reverse_track"]
    record["adjacency"]["edges"][1]["track_clean"]=false
    @test targeted_graph(record)["edges"][1]["kind"]=="unresolved"
    record["adjacency"]["edges"][2]["status"]="unknown"
    @test targeted_graph(record)["edges"][2]["kind"]=="unresolved"
    push!(record["adjacency"]["edges"],Dict("from"=>"p1","to"=>"missing","kind"=>"sampled_inventory_change","status"=>"completed"))
    @test !targeted_graph(record)["edges"][3]["endpoint_inventories_saved"]
    @test targeted_graph(record)["edges"][3]["kind"]=="unresolved"
    interval=Dict("from"=>"p1","to"=>"p2","kind"=>"inventory_sample_segment","status"=>"signature_change_observed","crosscheck_agreement"=>true)
    @test _targeted_edge_kind(interval)=="change_bracket"
    interval["crosscheck_agreement"]=false
    @test _targeted_edge_kind(interval)=="unresolved"
end

function synthetic_targeted_data()
    record=graph_record(n=3)
    for (from,to,origin) in (("p1","p2","base_origin_forward"),("p3","p2","target_origin_reverse"))
        push!(record["adjacency"]["edges"],Dict("from"=>from,"to"=>to,"kind"=>"tracked_root_link","origin"=>origin,"status"=>"clean_sampled_corrector_link","validated"=>true,"track_clean"=>true,"track"=>1,"from_state"=>[0.1,0.2],"to_state"=>[0.1,0.2]))
    end
    merge!(record,Dict("target_job_id"=>"old-target","base_point_id"=>"old-base","finest_level"=>129,"base_origin_target_coverage"=>false,"independent_endpoint_solved"=>true,"terminal_inventory_agreement"=>true,"levels"=>[Dict("grid_size"=>n,"base_origin_target_coverage"=>false,"reverse_target_origin_base_coverage"=>true) for n in (33,65,129)]))
    jobs=[Dict("id"=>"recovery","status"=>"unresolved","payload"=>Dict("target_role"=>"recovery_refinement")),Dict("id"=>"pending","status"=>"pending","payload"=>Dict("target_role"=>"multiroot"))]
    results=Dict("recovery"=>Dict("status"=>"unresolved","source_result"=>"results/recovery.toml","evidence"=>Dict("targeted_recoveries"=>[record])))
    records=Dict("points"=>Dict{String,Any}[])
    (;jobs,pilot_results=results,records,issues=[Dict("job"=>"pending","status"=>"pending")],state=Dict(),manifest=Dict())
end

@testset "Outcome-selected summary and schematic rendering" begin
    data=synthetic_targeted_data();summary=targeted_report_summary(data)
    @test summary["route_denominator"]==2
    @test summary["evaluated_recovery_routes"]==1
    @test summary["forward_covered_routes"]==0 # reverse success is not forward recovery
    @test summary["recoveries"][1]["reverse_target_origin_base_coverage"]===true
    @test occursin("Outcome-selected",summary["study_scope"])
    io=IOBuffer();_targeted_map_svg(io,summary["graphs"][1]);svg=String(take!(io))
    @test occursin("marker-end",svg)
    @test occursin("8 4",svg) # reverse links have a redundant dash encoding
    @test occursin("schematic",svg)
    @test !occursin("disconnected",svg)
    summary["diagnoses"]=_targeted_compact_diagnosis.(summary["diagnoses"])
    summary["transitions"]=_targeted_compact_transition.(summary["transitions"])
    summary["continuations"]=_targeted_compact_continuation.(summary["continuations"])
    summary["controls"]=Dict{String,Any}[]
    io=IOBuffer();_targeted_render_html(io,summary,data);html=String(take!(io))
    @test occursin("not a fresh held-out",html)
    @test occursin("Reverse success",html)
    @test occursin("not full root inventories",html) || occursin("separately from inventory adjacency",html)
    @test occursin("targeted_graphs.toml",html)
    escaped=graph_record(n=1);escaped["adjacency"]["nodes"][1]["id"]="<script>"
    io=IOBuffer();_targeted_map_svg(io,targeted_graph(escaped;source_job="<unsafe>"));html=String(take!(io))
    @test occursin("&lt;script&gt;",html)
    @test !occursin("<script>",html)
end

@testset "Targeted report input guards fail closed" begin
    mktempdir() do output
        manifest=Dict("jobs"=>[])
        open(joinpath(output,"targeted_manifest.toml"),"w") do io;TOML.print(io,manifest);end
        state=Dict("protocol"=>"targeted_followup_v1","targeted_manifest_hash"=>_targeted_digest(manifest),"jobs"=>[])
        open(joinpath(output,"state.toml"),"w") do io;TOML.print(io,state);end
        @test_throws ArgumentError targeted_report_data(output) # verifier is required, never silently skipped
        state["targeted_manifest_hash"]="tampered"
        open(joinpath(output,"state.toml"),"w") do io;TOML.print(io,state);end
        @test_throws ArgumentError write_targeted_report(output)
        @test !isfile(joinpath(output,"index.html"))
        outside=joinpath(output,"outside");write(outside,"untouched")
        symlink(outside,joinpath(output,"targeted_graphs.toml"))
        @test_throws ArgumentError write_targeted_report(output)
        @test read(outside,String)=="untouched"
    end
end
@testset "Typed witnesses and incident-interval uncertainty" begin
    data=synthetic_targeted_data();graph=targeted_report_summary(data)["graphs"][1]
    roots=[node for node in graph["nodes"] if node["node_type"]=="root_witness"]
    @test length(roots)==4
    @test Set(node["origin"] for node in roots)==Set(["base_origin_forward","target_origin_reverse"])
    @test all(link->!get(link,"inventory_membership_asserted",false),filter(link->link["kind"]=="parameter_context",graph["witness_edges"]))
    record=graph_record(n=2)
    record["events"]=[Dict("id"=>"candidate","state"=>[0.1,0.2],"parameters"=>Dict("A"=>1.0),"kind"=>"fold","classification"=>"numerically_validated_fold","validated"=>true,"event_condition_residual"=>1e-12,"raw_estimate_ids"=>["estimate1"])]
    push!(record["adjacency"]["edges"],Dict("from"=>"p1","to"=>"p2","kind"=>"inventory_sample_segment","status"=>"crosscheck_disagreement","crosscheck_agreement"=>false,"linked_event_ids"=>["candidate"]))
    graph=targeted_graph(record;source_job="mixed")
    @test graph["edges"][1]["kind"]=="unresolved"
    @test count(node->node["node_type"]=="event_witness",graph["nodes"])==1
    @test all(link->link["status"]=="unresolved_incident_interval",filter(link->link["kind"]=="event_side_context",graph["witness_edges"]))
    @test !any(link->get(link,"interval_validation_inferred_from_event",false),graph["witness_edges"])
    io=IOBuffer();_targeted_map_svg(io,graph);@test occursin("candidate",String(take!(io)))
    record["events"][1]["state"]=[NaN,0.2]
    @test isempty(targeted_graph(record)["witness_nodes"])
    diagnosis=Dict("source_job"=>"diagnose","source_result"=>"results/diagnose.toml","known_chart_seed_included"=>true)
    @test _targeted_compact_diagnosis(diagnosis)["known_chart_seed_included"]===true
    transition=Dict("source_job"=>"event","source_result"=>"results/event.toml","events"=>[Dict("kind"=>"fold","event_condition_residual"=>1e-12)],"event_estimates"=>[Dict(),Dict()])
    compact=_targeted_compact_transition(transition)
    @test compact["events"][1]["kind"]=="fold"
    @test compact["events"][1]["event_condition_residual"]==1e-12
    @test compact["raw_event_estimate_count"]==2
end

@testset "Map preview preserves decisive middle excursions" begin
    record=graph_record(n=100)
    for (i,node) in enumerate(record["adjacency"]["nodes"])
        node["signature"]=i<=75 ? "three_roots" : i<=80 ? "five_roots" : i<=90 ? "three_roots" : "one_root"
    end
    for i in 1:99
        changed=record["adjacency"]["nodes"][i]["signature"]!=record["adjacency"]["nodes"][i+1]["signature"]
        push!(record["adjacency"]["edges"],Dict("from"=>"p$i","to"=>"p$(i+1)","kind"=>changed ? "sampled_inventory_change" : "finite_same_signature_samples","status"=>"sampled_endpoints_only"))
    end
    io=IOBuffer();_targeted_map_svg(io,targeted_graph(record));svg=String(take!(io))
    for id in ("p75","p76","p80","p81","p90","p91")
        @test occursin(id,svg)
    end
    @test occursin("omitted only from this preview",svg)
end

end

module TargetedReportIntegrationTests
using Test,TOML
const fixture_parent=parentmodule(@__MODULE__)
if isdefined(fixture_parent,:TargetedEngineTests)
    const E=getfield(fixture_parent,:TargetedEngineTests)
    if !isdefined(E,:write_targeted_report)
        Base.include(E,joinpath(@__DIR__,"..","scripts","atlas","report.jl"))
        Base.include(E,joinpath(@__DIR__,"..","scripts","atlas","targeted_report.jl"))
    end
    @testset "Real verifier and combined-cap report before scheduler metadata" begin
        mktempdir() do parent
            first=E.first_packet(joinpath(parent,"first"))
            cap=32*1024^2
            prepared,effective=E.targeted_prepare(E.targeted_config(),first,cap)
            config=merge(prepared,Dict("mode"=>"pilot","storage_bytes"=>effective))
            output=joinpath(parent,"second")
            state=E.initialize_atlas(output,"pilot",config,10.0)
            for key in ("source_packet_retained_bytes","combined_storage_cap_bytes","new_packet_cap_bytes")
                delete!(get(state,"targeted",Dict()),key)
            end
            E.save_atlas_state(output,state)
            @test isfile(E.write_report(output))
            summary=TOML.parsefile(joinpath(output,"targeted_summary.toml"))
            @test summary["storage"]["first_packet_reserved_bytes"]==E.storage_owned_bytes(first)
            @test summary["storage"]["combined_cap_bytes"]==cap
            @test summary["storage"]["new_packet_cap_bytes"]==effective
            @test summary["storage"]["combined_retained_bytes"]<=cap
            @test sum(row["assigned"] for row in summary["progress"])==22
            @test summary["evaluated_recovery_routes"]==0
            @test summary["forward_covered_routes"]==0
            @test isempty(TOML.parsefile(joinpath(output,"targeted_graphs.toml"))["graphs"])
            original=read(joinpath(output,"index.html"))
            write(joinpath(first,"changed"),"original packet mutation")
            @test_throws ArgumentError E.write_report(output)
            @test read(joinpath(output,"index.html"))==original
        end
    end
else
    @testset "Targeted report integration requires the engine fixture" begin
        @test_skip false # Full suite loads atlas_targeted_engine.jl before this file.
    end
end
end
