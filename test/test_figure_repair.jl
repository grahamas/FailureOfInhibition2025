using Test
import TOML
include(joinpath(@__DIR__,"..","scripts","repair_figure_serialization.jl"))

@testset "Figure source repair preserves numerical identity" begin
    R=RepairFigureSerialization
    current=read(joinpath(R.ROOT,"scripts/run_figure_details.jl"),String)
    prior=replace(current,r"                attempts=\[Dict.*\n                S.write_record\(joinpath\(dir,\"attempts_.*\n"=>
        "                S.write_record(joinpath(dir,\"attempts_\$(branch.direction).toml\"),Dict(\"attempts\"=>branch.attempts))\n")
    @test prior!=current
    function fixture(dir,source)
        path=joinpath(dir,"source/scripts/run_figure_details.jl");mkpath(dirname(path));write(path,source)
        open(joinpath(dir,"metadata.toml"),"w") do io
            TOML.print(io,Dict("source_sha256"=>Dict("scripts/run_figure_details.jl"=>R.digest(path))))
        end
    end
    mktempdir() do dir
        fixture(dir,prior);R.repair(dir)
        @test read(joinpath(dir,"source/scripts/run_figure_details.jl"),String)==current
        @test read(joinpath(dir,"serialization_repair/previous_driver.jl"),String)==prior
        @test TOML.parsefile(joinpath(dir,"metadata.toml"))["serialization_repair"]["current_driver_sha256"]==R.digest(joinpath(R.ROOT,"scripts/run_figure_details.jl"))
    end
    mktempdir() do dir
        fixture(dir,replace(prior,"max_steps=250"=>"max_steps=251"))
        @test_throws ErrorException R.repair(dir)
        @test !ispath(joinpath(dir,"serialization_repair"))
    end
    old_replay=replace(current,r"^    metadata\[\"replay_from_artifact_directory\"\]=.*\n"m=>"")
    @test old_replay!=current
    mktempdir() do dir
        fixture(dir,old_replay);R.repair(dir;kind="replay")
        @test read(joinpath(dir,"source/scripts/run_figure_details.jl"),String)==current
        @test read(joinpath(dir,"replay_repair/previous_driver.jl"),String)==old_replay
    end
end
