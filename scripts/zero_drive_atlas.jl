#!/usr/bin/env julia
module ZeroDriveAtlas

using TOML, SHA, LinearAlgebra

# CLI coordination must stay lightweight: package loading happens only in the
# isolated worker, where start_atlas_worker enforces the setup deadline.
const ATLAS_LIGHTWEIGHT_COORDINATOR = abspath(PROGRAM_FILE) == (@__FILE__) &&
    (isempty(ARGS) || first(ARGS) != "__worker")

include("atlas/core.jl")
include("atlas/events.jl")
include("atlas/science.jl")
include("atlas/engine.jl")
include("atlas/report.jl")

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        ZeroDriveAtlas.atlas_main(ARGS)
    catch exception
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        exit(1)
    end
end
