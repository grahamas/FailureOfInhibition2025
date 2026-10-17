"""Compare a compact catalog with every original audit root and context."""
module VerifyEquilibriumCatalog
include("equilibrium_catalog.jl")
const C=EquilibriumCatalog
function verify(directory,source)
    catalog=C.read_catalog(directory)
    C.IOChecks.verify_manifest(source)
    catalog.metadata["source_manifest_sha256"]==C.IOChecks.digest(joinpath(source,"checksums.toml")) || error("wrong source packet")
    lookup=Dict(c.context=>id for (id,c) in catalog.contexts)
    descriptors=Dict{Tuple{UInt32,UInt16},C.Descriptor}()
    C.foreach_descriptor(catalog) do d
        descriptors[(d.context_id,d.root_id)]=d
    end
    checked=0
    for chunk in C.read_csv(joinpath(source,"inventory_chunks.csv"))
        for r in C.read_gzip(joinpath(source,C.IOChecks.checked_name(chunk.file)))
            key=(lookup[r.context],UInt16(r.root));d=pop!(descriptors,key)
            x=C.expand(d)
            # Compare retained source values directly; do not call the encoder.
            for field in (:E,:I,:arm,:I_side,:stability,:near_singular,:lambda1_real,:lambda1_imag,:lambda2_real,:lambda2_imag)
                isequal(getproperty(x,field),getproperty(r,field)) || error("round-trip mismatch: $(r.context), $(r.root), $field")
            end
            c=C.context(catalog,d)
            (c.B_E,c.B_I,c.scope,c.case_id,c.source_sha256)==(r.B_E,r.B_I,r.scope,r.id,r.source_sha256) || error("root/context mismatch")
            checked+=1
        end
    end
    isempty(descriptors) && checked==catalog.metadata["roots"] || error("root coverage mismatch")
    context_count=0
    for file in sort(readdir(source))
        startswith(file,"contexts_") && endswith(file,".csv.gz") || continue
        for r in C.read_gzip(joinpath(source,file))
            c=catalog.contexts[lookup[r.context]]
            for field in (:scope,:B_E,:B_I,:value,:roots,:source,:source_sha256)
                isequal(getproperty(c,field),getproperty(r,field)) || error("context round-trip mismatch: $field")
            end
            c.axis==coalesce(r.axis,"") && c.case_id==r.id || error("context identity mismatch")
            context_count+=1
        end
    end
    context_count==length(catalog.contexts) || error("context coverage mismatch")
    bytes=sum(filesize(joinpath(directory,n)) for n in readdir(directory))
    bytes<catalog.metadata["source_root_inventory_bytes"]/2 || error("size acceptance failed")
    println("Exact Float64/code/status round-trip: $checked roots and $context_count contexts; $bytes bytes including dictionary and metadata")
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("usage: verify_equilibrium_catalog.jl CATALOG AUDIT_PACKET")
    VerifyEquilibriumCatalog.verify(abspath.(ARGS)...)
end
