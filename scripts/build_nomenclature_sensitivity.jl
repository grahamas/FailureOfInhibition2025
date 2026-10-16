"""Build a read-only reanalysis of the verified unassigned-equilibrium packet."""
module NomenclatureSensitivity
using FailureOfInhibition2025, LinearAlgebra
import CSV, TOML
include("equilibrium_nomenclature.jl")
include("figure_packet_io.jl")
include("input_response_models.jl")
const N = EquilibriumNomenclature
const IOChecks = FigurePacketIO
const ROOT = normpath(joinpath(@__DIR__, ".."))
const COHORTS = ("screen", "detail_geometry")
const LOW_Q = (.05, .15, .25)
const LOW_I_Q = (.25, .50, .75)
const HIGH_E_Q = (.35, .50, .65)
const HIGH_I_Q = (.50, .75, .90)
readrows(p) = collect(CSV.File(p; stringtype=String))
gzrows(p) = open(io -> collect(CSV.File(io; stringtype=String)), `gzip -dc $p`)
write_record(p, x) = open(io -> TOML.print(io, x; sorted=true), p, "w")
function compressed(p, rows)
    CSV.write(p, rows)
    run(`gzip -n $p`)
end
function chunks(out, stem, rows; size=1500)
    for (part, start) in enumerate(1:size:length(rows))
        compressed(joinpath(out, "$(stem)_$part.csv"), rows[start:min(start+size-1, length(rows))])
    end
end
function observation(r)
    (;context=String(r.context), scope=String(r.scope), id=String(r.id), root=Int(r.root),
      axis=String(coalesce(r.axis, "")), value=Float64(r.value), B_E=Float64(r.B_E), B_I=Float64(r.B_I),
      E=Float64(r.E), I=Float64(r.I), role=String(r.role), stability=String(r.stability),
      near_singular=Bool(r.near_singular), arm=String(r.arm), I_side=String(r.I_side),
      alpha=Float64(r.alpha), lambda_imag=Float64(r.lambda1_imag))
end
eligible(r) = r.stability == "Attracting" && !r.near_singular

function distributions(out, cohort, rows)
    qrows = NamedTuple[]
    for field in (:E, :I, :joint_activity)
        values = sort([field == :joint_activity ? max(r.E, r.I) : getproperty(r, field) for r in rows])
        # Every distinct observed coordinate is retained, including exact numerical zeros.
        exact = [(;value=x, cumulative=searchsortedlast(values, x), total=length(values)) for x in unique(values)]
        chunks(out, "cdf_$(cohort)_$field", exact; size=10000)
        for q in 0:100
            x = N.observed_quantile(values, q/100)
            push!(qrows, (;cohort, field=String(field), percentile=q, value=x,
                cumulative=searchsortedlast(values, x), total=length(values)))
        end
    end
    CSV.write(joinpath(out, "quantiles_$cohort.csv"), qrows)
    qrows
end

function surfaces(out, cohort, rows, qs)
    xs = [r.value for r in qs if r.field == "E"]
    ys = [r.value for r in qs if r.field == "I"]
    E, I = getproperty.(rows, :E), getproperty.(rows, :I)
    rest = N.lower_counts(E, I, xs, ys)
    fail = filter(r -> r.I_side == "descending", rows)
    below = N.lower_counts(getproperty.(fail, :E), getproperty.(fail, :I), xs, ys)
    failI = sort(getproperty.(fail, :I))
    # Inclusive high boundaries via a reversed-coordinate lower rectangle.
    high = reverse(N.lower_counts(-E, -I, reverse(-xs), reverse(-ys)); dims=(1,2))
    result = NamedTuple[]
    for (family, matrix) in (("rest", rest), ("seizure", [searchsortedlast(failI, ys[j])-below[i,j] for i in eachindex(xs), j in eachindex(ys)]), ("herald", high))
        for i in eachindex(xs), j in eachindex(ys)
            push!(result, (;cohort, family, E_percentile=i-1, I_percentile=j-1,
                E_boundary=xs[i], I_boundary=ys[j], count=matrix[i,j], total=length(rows)))
        end
    end
    chunks(out, "surfaces_$cohort", result; size=10000)
end

function activity_probes(out, cohort, rows)
    # Readable occupancy-scale probes supplement percentile plots. Snap to an
    # observed value and retain both requested and actual boundaries explicitly.
    es,is=sort(unique(getproperty.(rows,:E))),sort(unique(getproperty.(rows,:I)))
    nearest(xs,x)=xs[argmin(abs.(xs.-x))]
    result=NamedTuple[]
    for (family,ep,ip) in (("rest",(.001,.01,.02,.05),(.001,.01,.02,.05)),
                           ("seizure",(.01,.02,.05),(.001,.01,.05)),
                           ("herald",(.25,.4,.45),(.1,.3,.45)))
        for requested_E in ep,requested_I in ip
            e,i=nearest(es,requested_E),nearest(is,requested_I)
            member(r)=family=="rest" ? r.E<=e && r.I<=i : family=="seizure" ?
                r.E>e && r.I<=i && r.I_side=="descending" : r.E>=e && r.I>=i
            selected=filter(member,rows)
            push!(result,(;cohort,family,requested_E,requested_I,E_boundary=e,I_boundary=i,
                count=length(selected),total=length(rows),lower=count(r->r.arm=="lower",selected),
                middle=count(r->r.arm=="middle",selected),upper=count(r->r.arm=="upper",selected),
                no_turning_points=count(r->r.arm=="no_turning_points",selected),
                old_rest=count(r->r.role=="rest",selected),old_active=count(r->r.role=="active",selected),
                old_herald=count(r->r.role=="herald",selected),old_seizure=count(r->r.role=="seizure",selected),
                old_unassigned=count(r->r.role=="unassigned",selected)))
        end
    end
    CSV.write(joinpath(out,"activity_probes_$cohort.csv"),result)
end

function scenarios(rows)
    es, is = sort(getproperty.(rows, :E)), sort(getproperty.(rows, :I))
    result = NamedTuple[]
    for re in LOW_Q, ri in LOW_I_Q, se in LOW_Q, si in LOW_I_Q, he in HIGH_E_Q, hi in HIGH_I_Q
        b = N.Boundaries(N.observed_quantile(es,re), N.observed_quantile(is,ri),
            N.observed_quantile(es,se), N.observed_quantile(is,si), N.observed_quantile(es,he), N.observed_quantile(is,hi))
        push!(result, (;scenario=length(result)+1, rest_E_q=re, rest_I_q=ri, seizure_E_q=se,
            seizure_I_q=si, herald_E_q=he, herald_I_q=hi,
            rest_E=b.rest_E, rest_I=b.rest_I, seizure_E=b.seizure_E, seizure_I=b.seizure_I,
            herald_E=b.herald_E, herald_I=b.herald_I))
    end
    result
end
boundary(s) = N.Boundaries(s.rest_E,s.rest_I,s.seizure_E,s.seizure_I,s.herald_E,s.herald_I)
names(m) = join((String(k) for k in keys(m) if getproperty(m,k)), "|")

function compare_scenarios(out, cohort, rows, intervention)
    ss = scenarios(rows)
    CSV.write(joinpath(out, "scenarios_$cohort.csv"), ss)
    counts = NamedTuple[]; translations = NamedTuple[]; inter = NamedTuple[]
    support = zeros(Int, length(rows), 4)
    roles = sort(unique(r.role for r in rows))
    for s in ss
        b = boundary(s)
        mm = [N.memberships(r.E,r.I,r.I_side,b) for r in rows]
        for (k,m) in enumerate(mm), j in 1:4
            support[k,j] += m[j]
        end
        ns = [sum(values(m)) for m in mm]
        push!(counts, (;cohort, scenario=s.scenario, total=length(rows),
            rest=count(m->m.rest,mm), seizure=count(m->m.seizure,mm), herald=count(m->m.herald,mm),
            partial_failure=count(m->m.partial_failure,mm), overlap=count(>(1),ns), gap=count(==(0),ns)))
        for old in roles
            indices = findall(r->r.role==old,rows)
            for (j, family) in enumerate(("rest","seizure","herald","partial_failure"))
                push!(translations,(;cohort,scenario=s.scenario,old_role=old,family,
                    members=count(k->mm[k][j],indices),old_total=length(indices)))
            end
        end
        for r in intervention
            m = N.memberships(r.E,r.I,r.I_side,b)
            push!(inter,merge((;boundary_cohort=cohort,scenario=s.scenario),r,
                (;memberships=names(m),overlap=sum(values(m))>1,gap=sum(values(m))==0)))
        end
    end
    CSV.write(joinpath(out,"scenario_counts_$cohort.csv"),counts)
    chunks(out,"old_roles_$cohort",translations;size=10000)
    chunks(out,"interventions_$cohort",inter;size=3000)
    records = [merge(r,(;rest_scenarios=support[k,1],seizure_scenarios=support[k,2],
        herald_scenarios=support[k,3],partial_scenarios=support[k,4],scenarios=length(ss))) for (k,r) in enumerate(rows)]
    chunks(out,"membership_support_$cohort",records)
    counts
end

function dynamics(out, cohort, rows, cases)
    result = NamedTuple[]; maxerror = 0.
    for r in rows
        r.arm in ("lower","middle") || continue
        model = InputResponseModels.model(cases[r.id],r.B_E,r.B_I)
        j = zeros(2,2);point_jacobian!(j,[r.E,r.I],model,0.)
        err = abs(maximum(real.(eigvals(j))) - r.alpha)
        err < 1e-9 || error("spectrum reconstruction mismatch")
        maxerror = max(maxerror,err)
        push!(result,merge(r,(;joint_activity=max(r.E,r.I),J_EE=j[1,1],J_EI=j[1,2],J_IE=j[2,1],J_II=j[2,2],
            trace=tr(j),determinant=det(j),linear_decay_ms=-1/r.alpha)))
    end
    chunks(out,"dynamics_$cohort",result)
    # Lowest joint activity per circuit and arm; no cutoff is called rest.
    representatives = [first(sort(filter(r->r.id==id && r.arm==arm,result);by=r->(r.joint_activity,r.context,r.root)))
        for (id,arm) in sort(unique((r.id,r.arm) for r in result))]
    CSV.write(joinpath(out,"representatives_$cohort.csv"),representatives)
    (;dynamics_records=length(result),middle=count(r->r.arm=="middle",result),max_spectrum_error=maxerror,
        smallest_middle_joint_activity=minimum(r.joint_activity for r in result if r.arm=="middle"),
        middle_nonpositive_J_EE=count(r->r.arm=="middle" && r.J_EE<=0,result))
end

function build(source, out)
    IOChecks.verify_manifest(source)
    packet = TOML.parsefile(joinpath(source,"packet.toml"))
    for (file,hash) in packet["sources"]
        IOChecks.digest(joinpath(ROOT,file)) == hash || error("audit source changed: $file")
    end
    ispath(out) && error("destination exists")
    index = readrows(joinpath(source,"inventory_chunks.csv"))
    rows = mapreduce(vcat,index) do c
        data = observation.(gzrows(joinpath(source,c.file)))
        length(data)==c.rows || error("chunk row count mismatch")
        data
    end
    length(rows)==packet["inventory_rows"] || error("inventory total mismatch")
    length(unique((r.context,r.root) for r in rows))==length(rows) || error("duplicate root identity")
    context_rows = reduce(vcat,[gzrows(joinpath(source,n)) for n in readdir(source) if startswith(n,"contexts_")])
    sum(r.roots for r in context_rows)==length(rows) || error("context/root total mismatch")
    bycontext = Dict{String,Int}()
    for r in rows;bycontext[r.context]=get(bycontext,r.context,0)+1;end
    all(r->get(bycontext,r.context,0)==r.roots,context_rows) || error("per-context count mismatch")
    cases = Dict(p["id"]=>p for p in TOML.parsefile(joinpath(ROOT,"reproducibility/figure_redesign_20261006/cases.toml"))["cases"])
    intervention = map(readrows(joinpath(source,"intervention_labels.csv"))) do old
        r = only(filter(r->r.scope=="intervention" && r.id==old.id && r.axis==old.axis && r.value==old.value && r.root==old.root,rows))
        r.E==old.E && r.I==old.I && r.role==old.local_role || error("destination identity mismatch")
        merge((;id=String(old.id),axis=String(old.axis),value=old.value,root=old.root,E=r.E,I=r.I,
            local_role=r.role,reported_role=String(old.reported_role),origin=String(old.origin)),
            (;I_side=r.I_side,stability=r.stability,near_singular=r.near_singular))
    end
    mkpath(out)
    CSV.write(joinpath(out,"intervention_destinations.csv"),intervention)
    inventory_summary = [(;scope, stability, near_singular, records=count(r->r.scope==scope && r.stability==stability && r.near_singular==near_singular,rows))
        for (scope,stability,near_singular) in sort(unique((r.scope,r.stability,r.near_singular) for r in rows))]
    CSV.write(joinpath(out,"inventory_summary.csv"),inventory_summary)
    meta = Dict{String,Any}("completed"=>false,"source_manifest_sha256"=>IOChecks.digest(joinpath(source,"checksums.toml")),
        "source"=>relpath(source,ROOT),"inventory_rows"=>length(rows),"contexts"=>length(context_rows),
        "interventions"=>length(intervention),"policy"=>"Exploratory memberships only; no adopted cutoff or changed legacy role",
        "completeness"=>"CompletenessNotCertified", "sources"=>Dict(relpath(f,ROOT)=>IOChecks.digest(f) for f in
            [@__FILE__,joinpath(@__DIR__,"equilibrium_nomenclature.jl"),joinpath(@__DIR__,"figure_packet_io.jl"),joinpath(ROOT,"Project.toml"),joinpath(ROOT,"Manifest.toml")]))
    for cohort in COHORTS
        selected = filter(r->r.scope==cohort && eligible(r),rows)
        println(cohort,": ",length(selected)," attracting nonsingular records");flush(stdout)
        qs = distributions(out,cohort,selected)
        surfaces(out,cohort,selected,qs)
        activity_probes(out,cohort,selected)
        counts = compare_scenarios(out,cohort,selected,intervention)
        ds = dynamics(out,cohort,selected,cases)
        meta[cohort] = merge(Dict("records"=>length(selected),"scenarios"=>length(counts),
            "overlap_min"=>minimum(r.overlap for r in counts),"overlap_max"=>maximum(r.overlap for r in counts),
            "gap_min"=>minimum(r.gap for r in counts),"gap_max"=>maximum(r.gap for r in counts)),Dict(string(k)=>v for (k,v) in pairs(ds)))
    end
    # Preserve branch evidence separately: a branch connection does not define a role.
    for name in ("seeds.csv","seed_roots.csv","continuation_connections.csv","fold_checks.csv")
        cp(joinpath(source,name),joinpath(out,name))
    end
    meta["completed"]=true
    write_record(joinpath(out,"study.toml"),meta)
    IOChecks.write_manifest(out)
    println("Sensitivity packet complete: ",out)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("usage: build_nomenclature_sensitivity.jl AUDIT_PACKET NEW_OUTPUT")
    NomenclatureSensitivity.build(abspath.(ARGS)...)
end
