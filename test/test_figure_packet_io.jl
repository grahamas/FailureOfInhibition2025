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
        rm(joinpath(d,"checksums.toml"));@test_throws ErrorException FigurePacketIO.verify_manifest(d)
        if !Sys.iswindows()
            write(joinpath(d,"real"),"x");symlink(joinpath(d,"real"),joinpath(d,"link"))
            @test_throws ErrorException FigurePacketIO.write_manifest(d)
        end
    end
end
