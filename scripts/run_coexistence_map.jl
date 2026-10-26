# Command-line entry point for coexistence_core.jl.
include("coexistence_core.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(CoexistenceExperiment.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
