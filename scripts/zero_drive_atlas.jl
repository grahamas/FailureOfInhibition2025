#!/usr/bin/env julia
module ZeroDriveAtlas

using TOML, SHA, LinearAlgebra

# Campaign coordination stays lightweight. Numerical packages load only in the
# isolated worker or the explicitly requested bounded calibration command.
const ATLAS_LIGHTWEIGHT_COORDINATOR = abspath(PROGRAM_FILE) == (@__FILE__) &&
    (isempty(ARGS) || !(first(ARGS) in ("__worker","calibrate")))

include("atlas/core.jl")
include("atlas/events.jl")
include("atlas/science.jl")
include("atlas/pilot.jl")
include("atlas/targeted.jl")
include("atlas/targeted_events.jl")
include("atlas/campaign_science.jl")
include("atlas/storage.jl")
include("atlas/targeted_protocol.jl")
include("atlas/engine.jl")
include("atlas/calibration.jl")
include("atlas/campaign.jl")
include("atlas/report.jl")
include("atlas/targeted_report.jl")
include("atlas/campaign_report.jl")

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
