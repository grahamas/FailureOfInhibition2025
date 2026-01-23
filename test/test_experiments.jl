using Test
using FailureOfInhibition2025
using Dates
using JSON
using Plots
using DataFrames
using CSV

@testset "Experiment Utilities" begin
    
    @testset "create_experiment_dir" begin
        # Test basic functionality
        test_timestamp = DateTime(2025, 12, 30, 21, 45, 0)
        exp_dir = create_experiment_dir(
            base_dir="test_experiments",
            experiment_name="test_exp",
            timestamp=test_timestamp
        )
        
        @test isdir(exp_dir)
        @test occursin("test_exp_2025-12-30_21-45-00", exp_dir)
        
        # Clean up
        rm(exp_dir; recursive=true)
        rm("test_experiments"; recursive=true)
        
        # Test with default parameters
        exp_dir2 = create_experiment_dir(experiment_name="another_test")
        @test isdir(exp_dir2)
        @test occursin("another_test", exp_dir2)
        
        # Clean up
        rm(exp_dir2; recursive=true)
        rm("experiments"; recursive=true)
    end
    
    @testset "get_git_commit" begin
        git_info = get_git_commit()
        
        @test haskey(git_info, :commit)
        @test haskey(git_info, :is_dirty)
        @test haskey(git_info, :error)
        
        # Should return a valid commit hash (not "unknown" since we're in a git repo)
        @test git_info[:commit] != "unknown"
        @test isa(git_info[:is_dirty], Bool)
    end
    
    @testset "save_experiment_metadata" begin
        # Create a temporary experiment directory
        exp_dir = create_experiment_dir(
            base_dir="test_experiments",
            experiment_name="metadata_test"
        )
        
        # Test basic metadata saving
        metadata_file = save_experiment_metadata(
            exp_dir,
            description="Test experiment",
            additional_info=Dict("test_key" => "test_value")
        )
        
        @test isfile(metadata_file)
        @test occursin("metadata.json", metadata_file)
        
        # Read and verify the metadata
        metadata = JSON.parsefile(metadata_file)
        @test haskey(metadata, "timestamp")
        @test haskey(metadata, "git_commit")
        @test haskey(metadata, "git_is_dirty")
        @test haskey(metadata, "description")
        @test metadata["description"] == "Test experiment"
        @test haskey(metadata, "test_key")
        @test metadata["test_key"] == "test_value"
        
        # Clean up
        rm(exp_dir; recursive=true)
        rm("test_experiments"; recursive=true)
    end
    
    @testset "save_experiment_metadata with parameters" begin
        # Create test parameters
        lattice = PointLattice()
        conn = ScalarConnectivity(1.0)
        connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1))
        
        params = WilsonCowanParameters{1}(
            α = (1.0,),
            β = (1.0,),
            τ = (1.0,),
            connectivity = connectivity,
            nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
            stimulus = nothing,
            lattice = lattice,
            pop_names = ("E",)
        )
        
        # Create experiment directory and save metadata with parameters
        exp_dir = create_experiment_dir(
            base_dir="test_experiments",
            experiment_name="param_test"
        )
        
        metadata_file = save_experiment_metadata(exp_dir, params=params)
        
        @test isfile(metadata_file)
        
        # Verify parameters are in metadata
        metadata = JSON.parsefile(metadata_file)
        @test haskey(metadata, "parameters")
        @test haskey(metadata["parameters"], "α")
        @test haskey(metadata["parameters"], "τ")
        
        # Clean up
        rm(exp_dir; recursive=true)
        rm("test_experiments"; recursive=true)
    end
    
    @testset "save_plot" begin
        # Create a simple plot
        p = plot(1:10, sin.(1:10), title="Test Plot")
        
        # Create experiment directory
        exp_dir = create_experiment_dir(
            base_dir="test_experiments",
            experiment_name="plot_test"
        )
        
        # Save the plot
        plot_file = save_plot(p, exp_dir, "test_plot")
        
        @test isfile(plot_file)
        @test occursin("test_plot", plot_file)
        @test occursin(".png", plot_file)
        
        # Test with different format
        plot_file_pdf = save_plot(p, exp_dir, "test_plot_pdf", format=:pdf)
        @test isfile(plot_file_pdf)
        @test occursin(".pdf", plot_file_pdf)
        
        # Clean up
        rm(exp_dir; recursive=true)
        rm("test_experiments"; recursive=true)
    end
    
    @testset "save_experiment_results DataFrame" begin
        # Create a test DataFrame
        df = DataFrame(
            time = 1:10,
            value = rand(10)
        )
        
        # Create experiment directory
        exp_dir = create_experiment_dir(
            base_dir="test_experiments",
            experiment_name="results_test"
        )
        
        # Save the DataFrame
        results_file = save_experiment_results(df, exp_dir, "test_results")
        
        @test isfile(results_file)
        @test occursin("test_results", results_file)
        @test occursin(".csv", results_file)
        
        # Verify we can read it back
        df_read = CSV.read(results_file, DataFrame)
        @test size(df_read) == size(df)
        @test names(df_read) == names(df)
        
        # Clean up
        rm(exp_dir; recursive=true)
        rm("test_experiments"; recursive=true)
    end
    
    @testset "save_experiment_results ODE solution" begin
        # Create a simple model and solve it
        lattice = PointLattice()
        conn = ScalarConnectivity(0.5)
        connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1))
        
        params = WilsonCowanParameters{1}(
            α = (1.0,),
            β = (1.0,),
            τ = (1.0,),
            connectivity = connectivity,
            nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
            stimulus = nothing,
            lattice = lattice,
            pop_names = ("E",)
        )
        
        A₀ = reshape([0.1], 1, 1)
        tspan = (0.0, 10.0)
        sol = solve_model(A₀, tspan, params, saveat=0.5)
        
        # Create experiment directory
        exp_dir = create_experiment_dir(
            base_dir="test_experiments",
            experiment_name="solution_test"
        )
        
        # Save the solution
        results_file = save_experiment_results(sol, exp_dir, "simulation", params=params)
        
        @test isfile(results_file)
        @test occursin("simulation", results_file)
        @test occursin(".csv", results_file)
        
        # Verify we can read it back
        df_read = CSV.read(results_file, DataFrame)
        @test "time" in names(df_read)
        @test "E" in names(df_read)
        
        # Clean up
        rm(exp_dir; recursive=true)
        rm("test_experiments"; recursive=true)
    end
    
    @testset "ExperimentContext" begin
        # Create an experiment context
        exp = ExperimentContext("context_test", base_dir="test_experiments")
        
        @test isdir(exp.dir)
        @test exp.name == "context_test"
        @test isa(exp.timestamp, DateTime)
        @test occursin("context_test", exp.dir)
        
        # Test joinpath helper
        test_path = joinpath(exp, "test_file.txt")
        @test occursin(exp.dir, test_path)
        @test occursin("test_file.txt", test_path)
        
        # Clean up
        rm(exp.dir; recursive=true)
        rm("test_experiments"; recursive=true)
    end
    
end
