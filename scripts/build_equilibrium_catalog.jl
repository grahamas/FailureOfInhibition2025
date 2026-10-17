"""Build a compact catalog from the frozen audit; source packet is never modified."""
module BuildEquilibriumCatalog
include("equilibrium_catalog.jl")
const C=EquilibriumCatalog
const ROOT=normpath(joinpath(@__DIR__,".."))
const DEFAULT_SOURCE=joinpath(ROOT,"reproducibility/equilibrium_audit_20261016/data")

function write_chunks(out,kind,rows;size=10000)
    index=NamedTuple[]
    for (part,start) in enumerate(1:size:length(rows))
        name="$(kind)_$(lpad(part,3,'0')).csv"
        group=rows[start:min(start+size-1,length(rows))]
        C.write_gzip(joinpath(out,name),group)
        push!(index,(;kind,file=name*".gz",rows=length(group)))
    end
    index
end

function build(source,out;chunk_size=10000)
    chunk_size>0 || throw(ArgumentError("chunk size must be positive"))
    ispath(out) && error("destination exists")
    C.IOChecks.verify_manifest(source)
    upstream=C.TOML.parsefile(joinpath(source,"packet.toml"))
    registry="reproducibility/figure_redesign_20261006/cases.toml"
    C.IOChecks.digest(joinpath(ROOT,registry))==upstream["sources"][registry] || error("source parameter registry changed")
    oldcontexts=reduce(vcat,[C.read_gzip(joinpath(source,n)) for n in sort(readdir(source)) if startswith(n,"contexts_") && endswith(n,".csv.gz")])
    sort!(oldcontexts;by=r->r.context)
    length(unique(r.context for r in oldcontexts))==length(oldcontexts) || error("duplicate source context")
    length(oldcontexts)==C.TOML.parsefile(joinpath(source,"audit.toml"))["contexts"] || error("source context total mismatch")
    ids=Dict(r.context=>UInt32(k) for (k,r) in enumerate(oldcontexts))
    contexts=[(;context_id=ids[r.context],context=r.context,scope=r.scope,case_id=r.id,
        B_E=r.B_E,B_I=r.B_I,axis=coalesce(r.axis,""),value=r.value,roots=r.roots,
        source=r.source,source_sha256=r.source_sha256) for r in oldcontexts]
    index=C.read_csv(joinpath(source,"inventory_chunks.csv"))
    roots=C.Descriptor[]
    for chunk in index
        rows=C.read_gzip(joinpath(source,C.IOChecks.checked_name(chunk.file)))
        length(rows)==chunk.rows || error("source chunk count mismatch")
        append!(roots,[C.encode(r,ids[r.context]) for r in rows])
    end
    length(roots)==upstream["inventory_rows"] || error("source root total mismatch")
    sort!(roots;by=r->(r.context_id,r.root_id))
    mkpath(out)
    chunks=vcat(write_chunks(out,"contexts",contexts;size=chunk_size),
                write_chunks(out,"roots",C.record.(roots);size=chunk_size))
    C.CSV.write(joinpath(out,"chunks.csv"),chunks)
    cp(joinpath(ROOT,registry),joinpath(out,"parameters.toml"))
    source_bytes=sum(filesize(joinpath(source,r.file)) for r in index)
    metadata=Dict("schema_version"=>C.SCHEMA_VERSION,"completed"=>true,
        "roots"=>length(roots),"contexts"=>length(contexts),"chunk_size"=>chunk_size,
        "source_manifest_sha256"=>C.IOChecks.digest(joinpath(source,"checksums.toml")),
        "parameter_registry_sha256"=>C.IOChecks.digest(joinpath(out,"parameters.toml")),
        "source_root_inventory_bytes"=>source_bytes,"completeness"=>"CompletenessNotCertified",
        "E_arms"=>C.E_ARMS,"I_branches"=>C.I_BRANCHES,"stability_codes"=>C.STABILITY,
        "units"=>Dict("E"=>"occupancy","I"=>"occupancy","eigenvalues"=>"ms^-1"),
        "conventions"=>Dict("identifiers"=>"packet-local context ID plus original within-context root index",
            "spectrum"=>"ordered real parts and nonnegative conjugate frequency; imaginary zero is canonical +0.0",
            "geometry"=>"existing E-nullcline-arm convention and inhibitory response branch; not biological labels",
            "parameters"=>"case registry plus optional single-axis context override; B_E and B_I are separate",
            "uncertainty"=>"source stability and near-singular status retained without reclassification"),
        "sources"=>Dict(relpath(f,ROOT)=>C.IOChecks.digest(f) for f in
            [@__FILE__,joinpath(@__DIR__,"equilibrium_catalog.jl"),joinpath(@__DIR__,"figure_packet_io.jl")]))
    C.write_toml(joinpath(out,"catalog.toml"),metadata)
    C.IOChecks.write_manifest(out)
    C.read_catalog(out)
    bytes=sum(filesize(joinpath(out,n)) for n in readdir(out))
    # This gate covers dictionaries, parameters and metadata as well as roots.
    bytes < source_bytes/2 || error("catalog exceeds half the source root inventory: $bytes / $source_bytes bytes")
    println("Catalog: $(length(roots)) roots, $(length(contexts)) contexts; $bytes bytes total; $(round(100bytes/source_bytes;digits=2))% of source root inventory")
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)<=2 || error("usage: build_equilibrium_catalog.jl [AUDIT_PACKET [NEW_OUTPUT]]")
    source=isempty(ARGS) ? BuildEquilibriumCatalog.DEFAULT_SOURCE : abspath(ARGS[1])
    output=length(ARGS)<2 ? joinpath(BuildEquilibriumCatalog.ROOT,"output/equilibrium_catalog_20261016") : abspath(ARGS[2])
    BuildEquilibriumCatalog.build(source,output)
end
