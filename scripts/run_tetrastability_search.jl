# Command-line entry point for tetrastability_core.jl.
include("tetrastability_core.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(TetrastabilityExperiment.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
