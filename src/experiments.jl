"""
Experiment management utilities for organizing and reproducing computational experiments.

This module provides functions to:
- Create timestamped experiment directories
- Save reproducibility metadata (git commit, parameters, etc.)
- Organize plots and data outputs consistently
"""

import Dates
import JSON
import Plots
import CSV
import DataFrames: DataFrame

"""
    create_experiment_dir(; base_dir="experiments", experiment_name="experiment", timestamp=now())

Create a timestamped directory for experiment outputs.

# Arguments
- `base_dir`: Base directory for all experiments (default: "experiments")
- `experiment_name`: Name of the experiment (default: "experiment")
- `timestamp`: Timestamp to use (default: current time)

# Returns
- Path to the created experiment directory

# Examples
```julia
# Create directory: experiments/my_simulation_2025-12-30_21-45-00/
exp_dir = create_experiment_dir(experiment_name="my_simulation")

# Create directory in custom location
exp_dir = create_experiment_dir(base_dir="/path/to/results", experiment_name="analysis")
```
"""
function create_experiment_dir(; base_dir="experiments", experiment_name="experiment", timestamp=Dates.now())
    # Format timestamp as YYYY-MM-DD_HH-MM-SS
    timestamp_str = Dates.format(timestamp, "yyyy-mm-dd_HH-MM-SS")
    
    # Create directory name: experiment_name_timestamp
    dir_name = "$(experiment_name)_$(timestamp_str)"
    exp_dir = joinpath(base_dir, dir_name)
    
    # Create the directory
    mkpath(exp_dir)
    
    return exp_dir
end

"""
    get_git_commit()

Get the current git commit hash and status.

# Returns
- Dictionary with keys:
  - `:commit`: Current commit hash
  - `:is_dirty`: Whether there are uncommitted changes
  - `:error`: Error message if git command failed (otherwise nothing)

# Examples
```julia
git_info = get_git_commit()
println("Commit: ", git_info[:commit])
println("Dirty: ", git_info[:is_dirty])
```
"""
function get_git_commit()
    result = Dict{Symbol, Any}()
    
    try
        # Get commit hash
        commit_output = read(`git rev-parse HEAD`, String)
        result[:commit] = strip(commit_output)
        
        # Check if repository is dirty (has uncommitted changes)
        status_output = read(`git status --porcelain`, String)
        result[:is_dirty] = !isempty(strip(status_output))
        result[:error] = nothing
    catch e
        # If git command fails (e.g., not in a git repo), return error
        result[:commit] = "unknown"
        result[:is_dirty] = false
        result[:error] = string(e)
    end
    
    return result
end

"""
    save_experiment_metadata(exp_dir; params=nothing, description="", additional_info=Dict())

Save experiment metadata including git commit, timestamp, and parameters.

# Arguments
- `exp_dir`: Experiment directory path
- `params`: Optional model parameters to save
- `description`: Optional text description of the experiment
- `additional_info`: Optional dictionary of additional metadata to save

# Returns
- Path to the saved metadata file (JSON format)

# Examples
```julia
exp_dir = create_experiment_dir(experiment_name="sim1")
metadata_file = save_experiment_metadata(
    exp_dir,
    params=params,
    description="Testing traveling wave formation",
    additional_info=Dict("author" => "researcher1", "notes" => "Initial run")
)
```
"""
function save_experiment_metadata(exp_dir; params=nothing, description="", additional_info=Dict())
    metadata = Dict{String, Any}()
    
    # Add timestamp
    metadata["timestamp"] = string(Dates.now())
    
    # Add git information
    git_info = get_git_commit()
    metadata["git_commit"] = git_info[:commit]
    metadata["git_is_dirty"] = git_info[:is_dirty]
    if git_info[:error] !== nothing
        metadata["git_error"] = git_info[:error]
    end
    
    # Add description
    if !isempty(description)
        metadata["description"] = description
    end
    
    # Add parameters if provided
    if params !== nothing
        # Convert parameters to a JSON-serializable format
        params_dict = Dict{String, Any}()
        
        # Extract parameter values using fieldnames
        for field in fieldnames(typeof(params))
            value = getfield(params, field)
            
            # Convert to string for complex types
            if isa(value, Tuple) || isa(value, Number) || isa(value, String)
                params_dict[string(field)] = value
            else
                params_dict[string(field)] = string(value)
            end
        end
        
        metadata["parameters"] = params_dict
    end
    
    # Add any additional information
    for (key, value) in additional_info
        metadata[string(key)] = value
    end
    
    # Save to JSON file
    metadata_file = joinpath(exp_dir, "metadata.json")
    open(metadata_file, "w") do io
        JSON.print(io, metadata, 2)  # Pretty print with 2-space indentation
    end
    
    return metadata_file
end

"""
    save_plot(fig, exp_dir, filename; format=:png, kwargs...)

Save a plot to the experiment directory.

# Arguments
- `fig`: Plots.jl figure object
- `exp_dir`: Experiment directory path
- `filename`: Filename for the plot (without extension)
- `format`: Output format (default: :png, options: :png, :pdf, :svg)
- `kwargs...`: Additional keyword arguments passed to `savefig()`

# Returns
- Path to the saved plot file

# Examples
```julia
exp_dir = create_experiment_dir(experiment_name="plots")
p = plot(1:10, sin.(1:10))
save_plot(p, exp_dir, "sine_wave")  # Saves to exp_dir/sine_wave.png
save_plot(p, exp_dir, "sine_wave", format=:pdf)  # Saves as PDF
```
"""
function save_plot(fig, exp_dir, filename; format=:png, kwargs...)
    # Add extension if not present
    if !occursin(".", filename)
        filename = "$(filename).$(format)"
    end
    
    # Create full path
    filepath = joinpath(exp_dir, filename)
    
    # Save the plot
    Plots.savefig(fig, filepath; kwargs...)
    
    return filepath
end

"""
    save_experiment_results(data, exp_dir, filename; format=:csv, kwargs...)

Save experiment results (DataFrame or solution) to the experiment directory.

# Arguments
- `data`: Data to save (DataFrame, ODE solution, or dictionary)
- `exp_dir`: Experiment directory path
- `filename`: Filename for the results (without extension)
- `format`: Output format (default: :csv)
- `kwargs...`: Additional keyword arguments

# Returns
- Path to the saved results file

# Examples
```julia
exp_dir = create_experiment_dir(experiment_name="results")

# Save simulation results
sol = solve_model(A₀, tspan, params)
save_experiment_results(sol, exp_dir, "simulation", params=params)

# Save DataFrame directly
df = DataFrame(time=1:10, value=rand(10))
save_experiment_results(df, exp_dir, "data")
```
"""
function save_experiment_results(data, exp_dir, filename; format=:csv, params=nothing, kwargs...)
    # Add extension if not present
    if !occursin(".", filename)
        filename = "$(filename).$(format)"
    end
    
    # Create full path
    filepath = joinpath(exp_dir, filename)
    
    # Handle different data types
    if format == :csv
        if isa(data, DataFrame)
            # Save DataFrame directly
            CSV.write(filepath, data; kwargs...)
        elseif hasfield(typeof(data), :u) && hasfield(typeof(data), :t)
            # Looks like an ODE solution - use existing save function
            # Import the save_simulation_results from simulate.jl
            save_simulation_results(data, filepath; params=params)
        else
            error("Unsupported data type for CSV format: $(typeof(data))")
        end
    elseif format == :json
        # Save as JSON
        open(filepath, "w") do io
            JSON.print(io, data, 2)
        end
    else
        error("Unsupported format: $(format)")
    end
    
    return filepath
end

"""
    ExperimentContext

A context manager for organizing experiment outputs.

# Fields
- `dir`: Experiment directory path
- `name`: Experiment name
- `timestamp`: Creation timestamp

# Examples
```julia
# Create experiment context
exp = ExperimentContext("my_experiment")

# Save metadata
save_experiment_metadata(exp.dir, params=params, description="Test run")

# Save plots and data
p = plot(1:10, sin.(1:10))
save_plot(p, exp.dir, "sine_wave")

# Access directory path
println("Results saved to: ", exp.dir)
```
"""
struct ExperimentContext
    dir::String
    name::String
    timestamp::Dates.DateTime
    
    function ExperimentContext(name::String; base_dir="experiments")
        timestamp = Dates.now()
        dir = create_experiment_dir(base_dir=base_dir, experiment_name=name, timestamp=timestamp)
        new(dir, name, timestamp)
    end
end

# Helper method to get full path for a file
Base.joinpath(exp::ExperimentContext, parts...) = joinpath(exp.dir, parts...)
