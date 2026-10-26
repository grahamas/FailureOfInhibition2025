# Command-line entry point for InputResponseStudy.
include("input_response_core.jl")
if abspath(PROGRAM_FILE)==@__FILE__;InputResponseStudy.main();end
