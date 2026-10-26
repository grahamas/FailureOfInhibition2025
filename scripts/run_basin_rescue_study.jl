# Command-line entry point for basin_rescue_core.jl.
include("basin_rescue_core.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    exit(BasinRescueStudy.main())
end
