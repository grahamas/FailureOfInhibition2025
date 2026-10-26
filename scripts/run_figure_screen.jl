# Command-line entry point for FigureScreen.
include("figure_screen_core.jl")
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS) in (2,3) || error("usage: run_figure_screen.jl CONFIG OUTPUT [CASE_LIMIT]")
    FigureScreen.run(ARGS[1],ARGS[2];limit=length(ARGS)==3 ? parse(Int,ARGS[3]) : nothing)
end
