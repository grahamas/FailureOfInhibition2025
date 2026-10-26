# Command-line entry point for FigureDetails.
include("figure_details_core.jl")
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS) in (4,5) || error("usage: run_figure_details.jl CONFIG SCREEN OUTPUT STAGE [CASE]")
    FigureDetails.run(ARGS[1:4]...;case_id=length(ARGS)==5 ? ARGS[5] : nothing)
end
