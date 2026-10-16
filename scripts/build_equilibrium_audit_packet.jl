"""Package an auditable role inventory and diagnostic plot arrays; preserve legacy labels."""
module EquilibriumAuditPacket
using FailureOfInhibition2025
import CSV,TOML
include("run_equilibrium_audit.jl")
const Q=EquilibriumAuditStudy
const A=Q.A
const IOChecks=Q.IOChecks

function label_audit(details)
    cases=TOML.parsefile(joinpath(details,"selection.toml"))["cases"]
    rows=NamedTuple[]
    for id in ("selective_withdrawal","threshold_to_active")
        p=only(filter(p->p["id"]==id,cases));base=joinpath(details,"interventions",id)
        old=A.search_view(A.retained_context(read(joinpath(base,"context.toml"),String))["equilibria"])
        oldroles=TOML.parsefile(joinpath(base,"roles.toml"))
        for r in CSV.File(joinpath(base,"summary.csv");stringtype=String)
            dir=joinpath(base,r.axis*"_"*string(r.value))
            records=A.retained_context(read(joinpath(dir,"context.toml"),String))["equilibria"]
            roles=TOML.parsefile(joinpath(dir,"roles.toml"));search=A.search_view(records)
            outcome=TOML.parsefile(joinpath(dir,"original_seizure.toml"))
            outcome["status"]=="compatible" || continue
            k=outcome["destination"]
            resolved=A.destination_label(search,roles,k;reference_search=old,reference_roles=oldroles)
            r.destination==resolved.label || error("intervention destination translation mismatch")
            push!(rows,(;id,axis=r.axis,value=r.value,root=k,local_role=something(findfirst(==(k),roles),"unassigned"),
                reported_role=r.destination,origin=resolved.origin,reference_root=resolved.reference_root,E=records[k]["state"][1],I=records[k]["state"][2]))
        end
    end
    rows
end

function build(audit,continuation,fold_checks,output)
    IOChecks.verify_manifest(audit);IOChecks.verify_manifest(continuation)
    IOChecks.verify_manifest(fold_checks)
    fold_summary=TOML.parsefile(joinpath(fold_checks,"summary.toml"))
    fold_summary["completed"] || error("fold checks incomplete")
    fold_summary["source_sha256"]==IOChecks.digest(joinpath(@__DIR__,"check_equilibrium_fold_matches.jl")) || error("fold-check source changed")
    fold_summary["continuation_manifest_sha256"]==IOChecks.digest(joinpath(continuation,"checksums.toml")) || error("fold checks used a different continuation")
    ispath(output) && error("packet output exists")
    summary=TOML.parsefile(joinpath(continuation,"summary.toml"))
    summary["completed"] || error("continuation incomplete")
    summary["audit_manifest_sha256"]==IOChecks.digest(joinpath(audit,"checksums.toml")) || error("continuation used a different inventory")
    summary["source_sha256"]==IOChecks.digest(joinpath(@__DIR__,"continue_equilibrium_audit.jl")) || error("continuation driver changed")
    meta=TOML.parsefile(joinpath(audit,"audit.toml"))
    for (name,hash) in meta["source_code"]
        IOChecks.digest(joinpath(@__DIR__,name))==hash || error("inventory source changed")
    end
    mkpath(output);mkpath(joinpath(output,"inventory"))
    for name in ("audit.toml","summary.csv","seeds.csv","seed_parameters.toml")
        cp(joinpath(audit,name),joinpath(output,name))
    end
    for name in ("checks.csv","stops.csv","connections.csv","summary.toml")
        cp(joinpath(continuation,name),joinpath(output,"continuation_"*name))
    end
    for name in ("checks.csv","attempts.csv","summary.toml")
        cp(joinpath(fold_checks,name),joinpath(output,"fold_"*name))
    end
    inventory=collect(CSV.File(joinpath(audit,"inventory.csv");stringtype=String))
    # Lossless chunks stay below jj's default per-file limit; their union is the full inventory.
    chunks=NamedTuple[]
    for (scope,id) in sort(unique((r.scope,r.id) for r in inventory))
        selected=filter(r->r.scope==scope && r.id==id,inventory)
        for (part,lo) in enumerate(1:1500:length(selected))
            subset=selected[lo:min(lo+1499,length(selected))]
            name=scope*"_"*id*"_$part.csv"
            path=joinpath(output,"inventory",name);CSV.write(path,subset)
            run(`gzip -n $path`)
            push!(chunks,(;file="inventory/"*name*".gz",rows=length(subset),scope,id))
        end
    end
    CSV.write(joinpath(output,"inventory_chunks.csv"),chunks)
    contexts=joinpath(output,"contexts.csv");cp(joinpath(audit,"contexts.csv"),contexts);run(`gzip -n $contexts`)
    # Context manifest is large; use similarly bounded, lossless CSV chunks.
    if filesize(contexts*".gz")>=1_000_000
        rm(contexts*".gz")
        allcontexts=collect(CSV.File(joinpath(audit,"contexts.csv")))
        for (part,lo) in enumerate(1:10000:length(allcontexts))
            path=joinpath(output,"contexts_$part.csv");CSV.write(path,allcontexts[lo:min(lo+9999,length(allcontexts))]);run(`gzip -n $path`)
        end
    end
    details=joinpath(Q.ROOT,meta["details"])
    CSV.write(joinpath(output,"intervention_labels.csv"),label_audit(details))
    phase_labels=NamedTuple[]
    cyclebase=joinpath(details,"cases/input_dependent/baseline_3")
    cycle_roots=A.retained_context(read(joinpath(cyclebase,"context.toml"),String))["equilibria"]
    cycle_roles=TOML.parsefile(joinpath(cyclebase,"roles.toml"))
    for name in filter(n->startswith(n,"cycle_1_phase_"),readdir(cyclebase))
        path=joinpath(cyclebase,name,"pulses/pulses.csv")
        trials=collect(CSV.File(path;stringtype=String))
        for destination in sort(unique(r.destination for r in trials))
            index=startswith(destination,"equilibrium_") ? parse(Int,split(destination,'_')[end]) : 0
            local_role=index>0 ? something(findfirst(==(index),cycle_roles),"unassigned") : destination
            push!(phase_labels,(;source=name,destination,local_role,root=index,
                E=index>0 ? cycle_roots[index]["state"][1] : NaN,I=index>0 ? cycle_roots[index]["state"][2] : NaN,
                observations=count(r->r.destination==destination,trials),source_sha256=IOChecks.digest(path)))
        end
    end
    CSV.write(joinpath(output,"phase_labels.csv"),phase_labels)
    # Time-scale scans reuse fixed equilibrium coordinates; keep these spectrum
    # observations separate from counts of independently archived root contexts.
    time_labels=NamedTuple[]
    for name in readdir(joinpath(details,"timescales"))
        dir=joinpath(details,"timescales",name)
        p=TOML.parsefile(joinpath(dir,"parameters.toml"));be=p["B_E"];bi=p["B_I"]
        baseline=A.retained_context(read(joinpath(dir,"context.toml"),String))["equilibria"]
        path=joinpath(dir,"equilibria.csv");observed=collect(CSV.File(path;stringtype=String))
        for ratio in sort(unique(r.ratio for r in observed))
            group=sort(filter(r->r.ratio==ratio,observed);by=r->r.root)
            rec=deepcopy(baseline)
            for row in group
                rec[row.root]["stability"]["classification"]=row.stability=="Unresolved" ? "StabilityUnresolved" : row.stability
            end
            explained,_=A.explain(A.M.model(merge(p,Dict("tau_ratio"=>ratio)),be,bi),rec)
            for (row,description) in zip(group,explained)
                abs(row.alpha-description.alpha)<1e-10 || error("time-scale spectrum changed")
                push!(time_labels,(;id=p["id"],ratio,root=row.root,E=row.E,I=row.I,recorded_stability=row.stability,alpha=row.alpha,
                    local_role=description.role,reasons=row.stability=="Unresolved" ? "stability_unresolved" : description.reasons,
                    descriptor=description.descriptor,source_sha256=IOChecks.digest(path)))
            end
        end
    end
    CSV.write(joinpath(output,"timescale_labels.csv"),time_labels)
    seeds=collect(CSV.File(joinpath(audit,"seeds.csv");stringtype=String))
    entries=TOML.parsefile(joinpath(audit,"seed_parameters.toml"))["seeds"]
    nulls=NamedTuple[];roots=NamedTuple[]
    for seed in seeds
        entry=only(filter(e->e["seed"]==seed.seed,entries));m=A.M.model(entry["parameters"],seed.B_E,seed.B_I);c=m.coupling
        folder=joinpath(continuation,"seed_$(seed.seed)")
        rec=A.retained_context(read(joinpath(folder,"context.toml"),String))["equilibria"]
        explained,_=A.explain(m,rec)
        append!(roots,[merge((;seed=seed.seed),r) for r in explained])
        for name in readdir(folder)
            endswith(name,".csv") || continue
            cp(joinpath(folder,name),joinpath(output,"seed_$(seed.seed)_"*name))
        end
        for u in range(-c.i_to_e+seed.B_E,c.e_to_e+seed.B_E;length=2001)
            e=A.M.occupancy(m.excitatory.response,u);i=(c.e_to_e*e+seed.B_E-u)/c.i_to_e
            0<=i<=.55 && push!(nulls,(;seed=seed.seed,population="E",E=e,I=i))
        end
        for v in range(-c.i_to_i+seed.B_I,c.e_to_i+seed.B_I;length=2001)
            i=A.M.occupancy(m.inhibitory.response,v);e=(v+c.i_to_i*i-seed.B_I)/c.e_to_i
            0<=e<=.55 && push!(nulls,(;seed=seed.seed,population="I",E=e,I=i))
        end
    end
    for seed in seeds
        CSV.write(joinpath(output,"seed_$(seed.seed)_nullclines.csv"),filter(r->r.seed==seed.seed,nulls))
    end
    CSV.write(joinpath(output,"seed_roots.csv"),roots)
    cp(joinpath(details,"cases/input_dependent/baseline_3/orbit_1.csv"),joinpath(output,"cycle.csv"))
    for id in ("selective_withdrawal","threshold_to_active")
        cp(joinpath(details,"interventions",id,"summary.csv"),joinpath(output,id*"_interventions.csv"))
    end
    # Preserve all plotted-map root samples, with intrinsic descriptors and original labels side by side.
    for id in ("selective_withdrawal","input_dependent")
        path=joinpath(output,"map_"*id*".csv")
        rows=filter(r->r.scope=="detail_geometry" && r.id==id && r.stability=="Attracting",inventory)
        CSV.write(path,[(;B_E=r.B_E,B_I=r.B_I,root=r.root,E=r.E,I=r.I,role=r.role,descriptor=r.descriptor,reasons=r.reasons) for r in rows])
        run(`gzip -n $path`)
    end
    sources=Dict{String,String}(TOML.parsefile(joinpath(details,"metadata.toml"))["source_sha256"])
    for name in ("run_equilibrium_audit.jl","equilibrium_role_audit.jl","continue_equilibrium_audit.jl",
        "check_equilibrium_fold_matches.jl","build_equilibrium_audit_packet.jl","verify_figure_redesign.jl","figure_packet_io.jl")
        relative="scripts/"*name;sources[relative]=IOChecks.digest(joinpath(Q.ROOT,relative))
    end
    for (relative,hash) in sources
        IOChecks.digest(joinpath(Q.ROOT,relative))==hash || error("scientific source changed during packaging: $relative")
    end
    Q.write_record(joinpath(output,"packet.toml"),Dict("sources"=>sources,"inventory_rows"=>length(inventory),
        "audit_manifest_sha256"=>IOChecks.digest(joinpath(audit,"checksums.toml")),
        "continuation_manifest_sha256"=>IOChecks.digest(joinpath(continuation,"checksums.toml")),
        "fold_manifest_sha256"=>IOChecks.digest(joinpath(fold_checks,"checksums.toml")),
        "role_policy"=>"Existing roles unchanged; intrinsic descriptors and contextual eligibility are separate",
        "completeness"=>"CompletenessNotCertified"))
    for (dir,_,names) in walkdir(output),name in names
        path=joinpath(dir,name);filesize(path)<1_048_576 || error("oversized packet file: $path")
    end
    IOChecks.write_manifest(output)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==4 || error("usage: build_equilibrium_audit_packet.jl AUDIT CONTINUATION FOLD_CHECKS NEW_OUTPUT")
    EquilibriumAuditPacket.build(abspath.(ARGS)...)
end
