# Command-line entry point for pulse_experiment_core.jl.
include("pulse_experiment_core.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(PulseExperiment.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
