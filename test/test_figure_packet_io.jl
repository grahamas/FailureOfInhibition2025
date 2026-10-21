using Test
import TOML
include(joinpath(@__DIR__,"..","scripts","figure_packet_io.jl"))
@testset "Complete figure packet manifest" begin
    mktempdir() do d
        mkpath(joinpath(d,"nested"));write(joinpath(d,"nested","data.csv"),"time,E,I\n0,0,0\n")
        FigurePacketIO.write_manifest(d);original=read(joinpath(d,"checksums.toml"))
        @test length(FigurePacketIO.verify_manifest(d)["files"])==1
        FigurePacketIO.write_manifest(d);@test read(joinpath(d,"checksums.toml"))==original
        write(joinpath(d,"extra"),"x");@test_throws ErrorException FigurePacketIO.verify_manifest(d)
        rm(joinpath(d,"extra"));write(joinpath(d,"nested","data.csv"),"changed")
        @test_throws ErrorException FigurePacketIO.verify_manifest(d)
        rm(joinpath(d,"nested","data.csv"));@test_throws ErrorException FigurePacketIO.verify_manifest(d)
        for name in ("../escape","/absolute","nested\\data.csv","a/./b","a//b","C:/absolute")
            @test_throws ErrorException FigurePacketIO.checked_name(name)
        end
        @test FigurePacketIO.portable_name("nested\\data.csv",'\\')=="nested/data.csv"
        @test_throws ErrorException FigurePacketIO.portable_name("nested\\..\\escape",'\\')
        @test_throws ErrorException FigurePacketIO.portable_name("C:\\absolute",'\\')
        rm(joinpath(d,"checksums.toml"));@test_throws ErrorException FigurePacketIO.verify_manifest(d)
        if !Sys.iswindows()
            write(joinpath(d,"real"),"x");symlink(joinpath(d,"real"),joinpath(d,"link"))
            @test_throws ErrorException FigurePacketIO.write_manifest(d)
        end
    end
end

@testset "Verification binds completed stage contents" begin
    mktempdir() do root
        stage=joinpath(root,"cases","example");mkpath(stage)
        mkpath(joinpath(stage,"nested"))
        data=joinpath(stage,"nested","trajectory.csv")
        write(data,"time,E,I\n0,0,0\n")
        marker=joinpath(stage,"done.toml")
        function complete()
            open(marker,"w") do io
                TOML.print(io,Dict("files"=>Dict("nested/trajectory.csv"=>FigurePacketIO.digest(data))))
            end
        end
        complete()
        first_record=FigurePacketIO.completed_stage_manifests(root,["cases/example"])
        write(data,"time,E,I\n0,1,0\n")
        @test_throws ErrorException FigurePacketIO.completed_stage_manifests(root,["cases/example"])
        complete()
        @test FigurePacketIO.completed_stage_manifests(root,["cases/example"])!=first_record
        write(joinpath(stage,"extra.csv"),"unrecorded")
        @test_throws ErrorException FigurePacketIO.completed_stage_manifests(root,["cases/example"])
        rm(joinpath(stage,"extra.csv"))
        @test_throws ErrorException FigurePacketIO.completed_stage_manifests(root,["cases/example","cases/example"])
    end
end
