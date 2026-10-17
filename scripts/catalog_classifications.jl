"""Exploratory catalog views. Candidate names are proposals, not production roles."""
module CatalogClassifications
include("equilibrium_catalog.jl")
const C = EquilibriumCatalog

"""Ordered low/high occupancy boundaries; equality is low at the low edge, high at the high edge."""
struct Policy
    id::String
    low_E::Float64
    low_I::Float64
    high_E::Float64
    high_I::Float64
    function Policy(id, low_E, low_I, high_E, high_I)
        !isempty(id) && all(isfinite, (low_E,low_I,high_E,high_I)) &&
            0 <= low_E < high_E <= .5 && 0 <= low_I < high_I <= .5 ||
            throw(ArgumentError("require named, finite, ordered occupancy boundaries"))
        new(id,low_E,low_I,high_E,high_I)
    end
end
# Trial settings spanning readable boundaries; these are not estimated physiological thresholds.
const POLICIES = (Policy("narrow_endpoints",.005,.001,.475,.45),
                  Policy("reference",.01,.01,.45,.30),
                  Policy("broad_endpoints",.02,.02,.40,.20))
const LABELS = ("Descriptive remainder", "Rest candidate", "Seizure candidate",
                "Herald candidate", "High-E / intermediate-I failure branch")
const BANDS = ("low", "intermediate", "high")
band(x,low,high) = x <= low ? 1 : x >= high ? 3 : 2
activity_code(d,p::Policy) = 3*(band(d.E,p.low_E,p.high_E)-1)+band(d.I,p.low_I,p.high_I)
activity_name(code) = "$(BANDS[div(code-1,3)+1]) E / $(BANDS[mod(code-1,3)+1]) I"

"""Disjoint endpoint proposals, independent of coexisting roots, stability, and E arm.

0 retains descriptive coordinates; 1 rest; 2 seizure; 3 herald; 4 intermediate-I
failure branch. A low-I rising/peak/unresolved response is not labeled seizure.
Herald may occur on either response branch. No rule turns a root into an attractor.
"""
function candidate(d,p::Policy)
    d.E <= p.low_E && d.I <= p.low_I && return 1
    d.E > p.low_E && d.I <= p.low_I && d.I_branch == 2 && return 2
    d.E >= p.high_E && d.I >= p.high_I && return 3
    d.E >= p.high_E && p.low_I < d.I < p.high_I && d.I_branch == 2 && return 4
    0
end
eligible(d) = d.stability == 1 && !d.near_singular
function annotation(d)
    a,b,c = (candidate(d,p) for p in POLICIES)
    (;context_id=d.context_id,root_id=d.root_id,narrow=a,reference=b,broad=c,
      activity=activity_code(d,POLICIES[2]),agreement=a==b==c)
end

"""Summarize discovered attracting, nonsingular roots without choosing one representative.

Empty sets have missing extrema, never zero-valued activity. Counts preserve
multiplicity. These are sampled equilibrium repertoires, not basin or cycle maps.
"""
function summarize(context, ds)
    sinks=filter(eligible,ds)
    labels=[candidate(d,POLICIES[2]) for d in sinks]
    bounds(field,op)=isempty(sinks) ? missing : op(getproperty(d,field) for d in sinks)
    (;context_id=context.context_id,scope=context.scope,case_id=context.case_id,
      B_E=context.B_E,B_I=context.B_I,roots=length(ds),sinks=length(sinks),
      rest=count(==(1),labels),seizure=count(==(2),labels),herald=count(==(3),labels),
      partial=count(==(4),labels),descriptive=count(==(0),labels),
      middle=count(d->d.E_arm==2,sinks),descending=count(d->d.I_branch==2,sinks),
      complex=count(d->d.omega>0,sinks),
      E_min=bounds(:E,minimum),E_max=bounds(:E,maximum),
      I_min=bounds(:I,minimum),I_max=bounds(:I,maximum),
      alpha_min=bounds(:lambda_max_real,minimum),alpha_max=bounds(:lambda_max_real,maximum))
end

function build(directory,out)
    catalog=C.read_catalog(directory)
    ispath(out) && error("output exists");mkpath(out)
    data=C.Descriptor[];C.foreach_descriptor(d->push!(data,d),catalog)
    C.write_gzip(joinpath(out,"annotations.csv"),annotation.(data))
    grouped=Dict(id=>C.Descriptor[] for id in keys(catalog.contexts))
    counts=Dict{Tuple{String,String,Int,Int,Bool},Int}()
    for d in data
        push!(grouped[d.context_id],d)
        for p in POLICIES
            k=(C.context(catalog,d).scope,p.id,candidate(d,p),Int(d.stability),d.near_singular)
            counts[k]=get(counts,k,0)+1
        end
    end
    summaries=[summarize(catalog.contexts[id],grouped[id]) for id in sort(collect(keys(grouped)))]
    @assert sum(s.roots for s in summaries)==length(data)
    @assert all(s->s.rest+s.seizure+s.herald+s.partial+s.descriptive==s.sinks,summaries)
    C.write_gzip(joinpath(out,"contexts.csv"),summaries)
    C.CSV.write(joinpath(out,"counts.csv"),[(;scope=k[1],policy=k[2],candidate=k[3],stability=k[4],near_singular=k[5],count=v) for (k,v) in sort(collect(counts);by=first)])
    meta=Dict("completed"=>true,"schema_version"=>1,"roots"=>length(data),"contexts"=>length(summaries),
        "catalog_manifest_sha256"=>C.IOChecks.digest(joinpath(directory,"checksums.toml")),
        "implementation_sha256"=>C.IOChecks.digest(@__FILE__),
        "catalog_reader_sha256"=>C.IOChecks.digest(joinpath(@__DIR__,"equilibrium_catalog.jl")),
        "completeness"=>"CompletenessNotCertified", "classification_status"=>"exploratory proposals only",
        "labels"=>Dict(string(i-1)=>s for (i,s) in enumerate(LABELS)),
        "activity_codes"=>Dict(string(i)=>activity_name(i) for i in 1:9),
        "policies"=>[Dict(string(k)=>getfield(p,k) for k in fieldnames(Policy)) for p in POLICIES],
        "selection"=>"Portraits separate screen and detail_geometry. Input maps use detail_geometry for the two Figure 2 circuits.",
        "agreement"=>"Same candidate code under all three declared policies; not probability or agreement over all possible boundaries.",
        "sampling"=>"Adaptive samples and overlapping cohorts; counts are records, not prevalence. No interpolation, branch identity, cycles, or basin information.")
    C.write_toml(joinpath(out,"study.toml"),meta);C.IOChecks.write_manifest(out)
    println("Classified $(length(data)) roots in $(length(summaries)) contexts")
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("usage: catalog_classifications.jl CATALOG NEW_OUTPUT")
    CatalogClassifications.build(abspath.(ARGS)...)
end
