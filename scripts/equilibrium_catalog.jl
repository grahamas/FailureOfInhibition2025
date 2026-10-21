"""Compact, label-independent equilibrium descriptors and a verified packet reader."""
module EquilibriumCatalog
import CSV, TOML
include("figure_packet_io.jl")
const IOChecks = FigurePacketIO
const SCHEMA_VERSION = 1
const E_ARMS = Dict("no_turning_points"=>0, "lower"=>1, "middle"=>2, "upper"=>3,
                    "turning_point"=>4, "unresolved"=>255)
const I_BRANCHES = Dict("unresolved"=>0, "ascending"=>1, "descending"=>2, "peak"=>3)
const STABILITY = Dict("StabilityUnresolved"=>0, "Attracting"=>1, "Saddle"=>2, "Repelling"=>3)
const FIELDS = (:context_id,:root_id,:E_arm,:I_branch,:E,:I,:lambda_max_real,
                :lambda_min_real,:omega,:stability,:near_singular)

"""Float64 coordinates and complete two-dimensional eigenvalue spectrum.

Identifiers are packet-local. Arm/branch and stability codebooks are separate
from biological names. Near-singularity and unresolved stability are preserved,
never inferred again from rounded display values.
"""
struct Descriptor
    context_id::UInt32
    root_id::UInt16
    E_arm::UInt8
    I_branch::UInt8
    E::Float64
    I::Float64
    lambda_max_real::Float64
    lambda_min_real::Float64
    omega::Float64
    stability::UInt8
    near_singular::Bool
    function Descriptor(context_id,root_id,E_arm,I_branch,E,I,a,b,w,stability,near_singular)
        context_id isa Integer && 0<context_id<=typemax(UInt32) || throw(ArgumentError("invalid context ID"))
        root_id isa Integer && 0<root_id<=typemax(UInt16) || throw(ArgumentError("invalid root ID"))
        all(x->x isa Integer,(E_arm,I_branch,stability)) && E_arm in values(E_ARMS) && I_branch in values(I_BRANCHES) && stability in values(STABILITY) || throw(ArgumentError("unknown descriptor code"))
        all(x->x isa Real && isfinite(x),(E,I,a,b,w)) || throw(ArgumentError("nonfinite descriptor"))
        0<=E<=.5 && 0<=I<=.5 || throw(ArgumentError("occupancy outside equilibrium bounds"))
        a>=b && w>=0 && (w==0 || a==b) || throw(ArgumentError("invalid real-system spectrum"))
        near_singular isa Bool || throw(ArgumentError("near_singular must be Boolean"))
        new(context_id,root_id,E_arm,I_branch,E,I,a,b,w,stability,near_singular)
    end
end

"""Encode an audit root, rejecting spectra that cannot be represented losslessly."""
function encode(r,context_id)
    a,b = ComplexF64(r.lambda1_real,r.lambda1_imag),ComplexF64(r.lambda2_real,r.lambda2_imag)
    pair=sort([a,b];by=z->(real(z),imag(z)))
    all(z->isfinite(real(z)) && isfinite(imag(z)),pair) || throw(ArgumentError("nonfinite spectrum"))
    real_pair=all(z->imag(z)==0,pair)
    (real_pair || (real(pair[1])==real(pair[2]) && imag(pair[1])==-imag(pair[2]))) || throw(ArgumentError("spectrum is not a real pair or conjugate pair"))
    Descriptor(context_id,r.root,E_ARMS[r.arm],I_BRANCHES[r.I_side],r.E,r.I,
        real(pair[2]),real(pair[1]),abs(imag(pair[2])),STABILITY[r.stability],r.near_singular)
end

"""Recover eigenvalues in ascending (real, imaginary) order; zero imaginary parts are canonical +0.0."""
function spectrum(d::Descriptor)
    d.omega==0 ? (complex(d.lambda_min_real,0.),complex(d.lambda_max_real,0.)) :
        (complex(d.lambda_min_real,-d.omega),complex(d.lambda_max_real,d.omega))
end
record(d::Descriptor) = NamedTuple{FIELDS}(Tuple(getfield(d,k) for k in FIELDS))
from_record(r) = Descriptor((getproperty(r,k) for k in FIELDS)...)
name(codes,code) = only(k for (k,v) in codes if v==code)

"""Expand codes for analysis; this supplies no biological label."""
function expand(d::Descriptor)
    eig=spectrum(d)
    (;context_id=d.context_id,root=d.root_id,arm=name(E_ARMS,d.E_arm),I_side=name(I_BRANCHES,d.I_branch),
      E=d.E,I=d.I,lambda1_real=real(eig[1]),lambda1_imag=imag(eig[1]),
      lambda2_real=real(eig[2]),lambda2_imag=imag(eig[2]),
      stability=name(STABILITY,d.stability),near_singular=d.near_singular)
end

"""Short display only; numerical storage is never rounded."""
function descriptor_text(d::Descriptor;digits=4)
    small(x)=string(round(x;sigdigits=digits))
    flag=d.near_singular ? "; near singular" : ""
    "E$(d.E_arm)/I$(d.I_branch) | E=$(small(d.E)), I=$(small(d.I)) | "*
        "$(name(STABILITY,d.stability)); alpha=$(small(d.lambda_max_real)) ms^-1$flag"
end

"""Lossless three-layer view of one record; identity is local to its catalog.

Activity is continuous. Structure and dynamics preserve source codes and spectrum.
Interpretation is deliberately deferred, independently of activity and companions.
"""
function layers(d::Descriptor)
    (;identity=(;context_id=d.context_id,root_id=d.root_id),
      activity=(;E=d.E,I=d.I),
      structure_dynamics=(;E_arm=d.E_arm,I_branch=d.I_branch,stability=d.stability,
          near_singular=d.near_singular,lambda_max_real=d.lambda_max_real,
          lambda_min_real=d.lambda_min_real,omega=d.omega),
      interpretation=(;status=:deferred,label=nothing))
end

"""Readable layered description; rounding is display-only, omega is angular frequency."""
function layered_text(d::Descriptor;digits=4)
    x=layers(d);small(v)=string(round(v;sigdigits=digits))
    arm=name(E_ARMS,d.E_arm);side=name(I_BRANCHES,d.I_branch)
    mode=d.omega==0 ? "real spectrum" : "complex spectrum"
    flag=d.near_singular ? "; near singular" : ""
    "E=$(small(x.activity.E)), I=$(small(x.activity.I)) | E arm: $arm; I response: $side | " *
        "$(name(STABILITY,d.stability)); $mode; Re λ=($(small(d.lambda_min_real)), $(small(d.lambda_max_real))) ms⁻¹; " *
        "ω=$(small(d.omega)) rad/ms$flag | biological interpretation deferred"
end

read_csv(path) = collect(CSV.File(path;stringtype=String))
read_gzip(path) = open(io->collect(CSV.File(io;stringtype=String)),`gzip -dc $path`)
function write_gzip(path,rows)
    CSV.write(path,rows);run(`gzip -n $path`)
end
write_toml(path,x) = open(io->TOML.print(io,x;sorted=true),path,"w")

struct Catalog{C,P}
    directory::String
    metadata::Dict{String,Any}
    contexts::C
    parameters::P
    root_chunks::Vector{Tuple{String,Int}}
end

"""Visit one decoded descriptor at a time; only one compressed chunk is loaded."""
function foreach_descriptor(f,catalog::Catalog)
    for (file,n) in catalog.root_chunks
        rows=read_gzip(joinpath(catalog.directory,file))
        length(rows)==n || error("root chunk count mismatch")
        for row in rows
            Tuple(propertynames(row))==FIELDS || error("unexpected root columns")
            f(from_record(row))
        end
    end
    nothing
end

"""Context and parameter lookup; intervention overrides are explicit and independent of names."""
context(catalog::Catalog,d::Descriptor) = catalog.contexts[d.context_id]
function context_parameters(catalog::Catalog,d::Descriptor)
    c=context(catalog,d);p=copy(catalog.parameters[c.case_id])
    if !isempty(c.axis)
        haskey(p,c.axis) && isfinite(c.value) || error("invalid parameter override")
        p[c.axis]=c.value
    end
    (;parameters=p,B_E=c.B_E,B_I=c.B_I)
end

"""Open a standalone catalog, checking hashes, identities, codes, and context counts.

The original source packet is not required for reading. It is required for the
separate source round-trip verification. Source paths in contexts are provenance
strings and are never opened by this reader.
"""
function read_catalog(directory)
    IOChecks.verify_manifest(directory)
    meta=TOML.parsefile(joinpath(directory,"catalog.toml"))
    meta["schema_version"]==SCHEMA_VERSION && meta["completed"] || error("unsupported or incomplete catalog")
    meta["E_arms"]==E_ARMS && meta["I_branches"]==I_BRANCHES && meta["stability_codes"]==STABILITY || error("codebook mismatch")
    idx=read_csv(joinpath(directory,"chunks.csv"))
    length(unique(r.file for r in idx))==length(idx) || error("duplicate chunk index")
    all(r->r.kind in ("contexts","roots"),idx) || error("unknown chunk kind")
    expected=Set(vcat(["catalog.toml","chunks.csv","parameters.toml"],[IOChecks.checked_name(r.file) for r in idx]))
    Set(keys(IOChecks.files(directory)))==expected || error("unexpected catalog files")
    contexts=Dict{UInt32,NamedTuple}()
    for r in idx
        r.kind=="contexts" || continue
        entries=read_gzip(joinpath(directory,r.file));length(entries)==r.rows || error("context chunk count mismatch")
        for c in entries
            id=UInt32(c.context_id);id>0 && !haskey(contexts,id) || error("duplicate or zero context ID")
            isfinite(c.B_E) && isfinite(c.B_I) && c.roots>=0 || error("invalid context")
            occursin(r"^[0-9a-f]{64}$",c.source_sha256) || error("invalid source digest")
            contexts[id]=merge(NamedTuple(c),(;axis=String(coalesce(c.axis,""))))
        end
    end
    length(contexts)==meta["contexts"] || error("context total mismatch")
    length(Set(c.context for c in values(contexts)))==length(contexts) || error("duplicate original context")
    registry=TOML.parsefile(joinpath(directory,"parameters.toml"))["cases"]
    IOChecks.digest(joinpath(directory,"parameters.toml"))==meta["parameter_registry_sha256"] || error("parameter registry digest mismatch")
    parameters=Dict(p["id"]=>p for p in registry)
    length(parameters)==length(registry) || error("duplicate case ID")
    all(c->haskey(parameters,c.case_id),values(contexts)) || error("missing case parameters")
    roots=Tuple{String,Int}[(r.file,r.rows) for r in idx if r.kind=="roots"]
    catalog=Catalog(String(directory),meta,contexts,parameters,roots)
    seen=Set{Tuple{UInt32,UInt16}}();counts=Dict(id=>0 for id in keys(contexts))
    foreach_descriptor(catalog) do d
        haskey(contexts,d.context_id) || error("dangling context reference")
        key=(d.context_id,d.root_id);key in seen && error("duplicate root identity");push!(seen,key)
        counts[d.context_id]+=1
    end
    length(seen)==meta["roots"] || error("root total mismatch")
    all(counts[id]==c.roots for (id,c) in contexts) || error("per-context root count mismatch")
    # Validate overrides even for root-free contexts.
    for c in values(contexts)
        isempty(c.axis) || (haskey(parameters[c.case_id],c.axis) && isfinite(c.value)) || error("invalid context override")
    end
    catalog
end
end
