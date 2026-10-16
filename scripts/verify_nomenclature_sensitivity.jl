"""Check saved sensitivities by direct predicates, independently of prefix counting."""
module VerifyNomenclatureSensitivity
import CSV,TOML
include("figure_packet_io.jl")
const IOChecks=FigurePacketIO
rows(p,n)=collect(CSV.File(joinpath(p,n);stringtype=String))
function chunks(p,prefix)
    files=sort(filter(n->startswith(n,prefix*"_") && endswith(n,".csv.gz"),readdir(p)))
    reduce(vcat,[open(io->collect(CSV.File(io;stringtype=String)),`gzip -dc $(joinpath(p,n))`) for n in files])
end
function memberships(e,i,descending,s)
    (e<=s.rest_E && i<=s.rest_I,
     e>s.seizure_E && i<=s.seizure_I && descending,
     e>=s.herald_E && i>=s.herald_I,
     e>=s.herald_E && i>s.seizure_I && i<s.herald_I && descending)
end
function verify(p,source)
    IOChecks.verify_manifest(p);IOChecks.verify_manifest(source)
    meta=TOML.parsefile(joinpath(p,"study.toml"))
    meta["source_manifest_sha256"]==IOChecks.digest(joinpath(source,"checksums.toml")) || error("wrong source")
    for (file,hash) in meta["sources"]
        IOChecks.digest(joinpath(@__DIR__,"..",file))==hash || error("changed analysis source")
    end
    source_index=rows(source,"inventory_chunks.csv")
    checks=0;root_records=0;intervention_records=0
    destinations=Dict((r.id,r.axis,r.value,r.root)=>r for r in rows(source,"intervention_labels.csv"))
    for cohort in ("screen","detail_geometry")
        data=chunks(p,"membership_support_$cohort")
        # Exact root identity and retained classifications against the source packet.
        original=Dict((String(r.context),Int(r.root))=>r for c in source_index if c.scope==cohort
            for r in open(io->collect(CSV.File(io;stringtype=String)),`gzip -dc $(joinpath(source,c.file))`)
            if r.stability=="Attracting" && !r.near_singular)
        length(data)==length(original) || error("cohort coverage changed")
        Set((r.context,r.root) for r in data)==Set(keys(original)) || error("root identity coverage changed")
        root_records+=length(data)
        for r in data
            o=original[(r.context,r.root)]
            (r.E,r.I,r.role,r.arm,r.I_side,r.alpha)==(o.E,o.I,o.role,o.arm,o.I_side,o.alpha) || error("root changed")
        end
        # Typed arrays keep the independent direct enumeration inexpensive.
        E=Float64[r.E for r in data];I=Float64[r.I for r in data];D=Bool[r.I_side=="descending" for r in data]
        support=zeros(Int,length(data),4)
        ss=rows(p,"scenarios_$cohort.csv");saved=rows(p,"scenario_counts_$cohort.csv")
        length(ss)==length(saved)==729 || error("scenario coverage changed")
        for (s,c) in zip(ss,saved)
            c.scenario==s.scenario || error("scenario ordering changed")
            b=NamedTuple{(:rest_E,:rest_I,:seizure_E,:seizure_I,:herald_E,:herald_I)}(Float64.((s.rest_E,s.rest_I,s.seizure_E,s.seizure_I,s.herald_E,s.herald_I)))
            counts=zeros(Int,4);overlap=0;gap=0
            for k in eachindex(E)
                m=memberships(E[k],I[k],D[k],b)
                for j in 1:4;counts[j]+=m[j];support[k,j]+=m[j];end
                overlap+=sum(m)>1;gap+=sum(m)==0
            end
            Tuple(counts)==(c.rest,c.seizure,c.herald,c.partial_failure) && overlap==c.overlap && gap==c.gap || error("scenario count mismatch")
            checks+=1
        end
        for (k,r) in enumerate(data)
            Tuple(support[k,:])==(r.rest_scenarios,r.seizure_scenarios,r.herald_scenarios,r.partial_scenarios) || error("root support mismatch")
        end
        # Direct enumeration on a 10×10 subgrid, with repeated/equality boundaries.
        for r in chunks(p,"surfaces_$cohort")
            r.E_percentile in (0,1,5,10,25,50,75,90,99,100) && r.I_percentile in (0,1,5,10,25,50,75,90,99,100) || continue
            count=0;e=Float64(r.E_boundary);i=Float64(r.I_boundary)
            for k in eachindex(E)
                count+=r.family=="rest" ? E[k]<=e && I[k]<=i : r.family=="seizure" ? E[k]>e && I[k]<=i && D[k] : E[k]>=e && I[k]>=i
            end
            count==r.count || error("surface mismatch");checks+=1
        end
        for r in rows(p,"activity_probes_$cohort.csv")
            e=Float64(r.E_boundary);i=Float64(r.I_boundary)
            e in E && i in I || error("probe not an observed value")
            indices=[k for k in eachindex(E) if r.family=="rest" ? E[k]<=e && I[k]<=i : r.family=="seizure" ? E[k]>e && I[k]<=i && D[k] : E[k]>=e && I[k]>=i]
            length(indices)==r.count || error("activity probe mismatch")
            count(k->data[k].arm=="middle",indices)==r.middle || error("probe geometry mismatch")
            checks+=1
        end
        for field in ("E","I","joint_activity")
            values=sort(field=="E" ? E : field=="I" ? I : max.(E,I))
            cdf=sort(chunks(p,"cdf_$(cohort)_$field");by=r->r.value)
            [r.value for r in cdf]==unique(values) || error("CDF support changed")
            all(r->r.cumulative==searchsortedlast(values,r.value) && r.total==length(values),cdf) || error("CDF counts changed")
        end
        intervention=chunks(p,"interventions_$cohort")
        length(intervention)==length(destinations)*length(ss) || error("intervention coverage changed")
        length(Set((r.scenario,r.id,r.axis,r.value,r.root) for r in intervention))==length(intervention) || error("duplicate intervention record")
        intervention_records+=length(intervention)
        for r in intervention
            old=destinations[(r.id,r.axis,r.value,r.root)]
            (r.E,r.I,r.local_role,r.reported_role,r.origin)==(old.E,old.I,old.local_role,old.reported_role,old.origin) || error("intervention provenance changed")
            s=ss[r.scenario];m=memberships(r.E,r.I,r.I_side=="descending",s)
            labels=join([("rest","seizure","herald","partial_failure")[j] for j in 1:4 if m[j]],"|")
            labels==coalesce(r.memberships,"") && r.overlap==(sum(m)>1) && r.gap==(sum(m)==0) || error("intervention mismatch")
        end
        println(cohort,": exact root identities, scenario counts, support, CDFs, interventions, and sampled surfaces agree")
    end
    println("Verified ",checks," direct aggregate comparisons; all ",root_records," root membership-support records and ",intervention_records," intervention/scenario records")
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("usage: verify_nomenclature_sensitivity.jl PACKET AUDIT_PACKET")
    VerifyNomenclatureSensitivity.verify(abspath.(ARGS)...)
end
