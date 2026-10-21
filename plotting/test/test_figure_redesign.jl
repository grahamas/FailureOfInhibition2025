include(joinpath(@__DIR__,"..","..","scripts","render_figure_redesign.jl"))
@testset "Figure redesign semantic selections" begin
    R=FigureRedesignRenderer
    @test R.repertoire_label("Attracting:rest|Repelling:active|Attracting:seizure|Saddle:unassigned")=="R+S"
    @test R.repertoire_label("Repelling:active")=="no sink found"
    @test R.repertoire_label("Attracting:unassigned|Attracting:unassigned")=="U+U"
    cycles=[(;direction=-1,index=1,ratio=4.4),(;direction=-1,index=2,ratio=3.9),
        (;direction=1,index=1,ratio=3.5)]
    stops=[(;direction=-1,termination="minimum_step"),(;direction=1,termination="parameter_boundary")]
    endpoint=R.decreasing_endpoint(cycles,stops)
    @test endpoint.point.index==2
    @test endpoint.termination=="minimum_step"
    @test R.termination_label(endpoint.termination)=="minimum step"
    mktempdir() do packet
        timescales=joinpath(packet,R.PERIODIC_DIR);mkpath(timescales)
        CSV.write(joinpath(timescales,"cycles.csv"),cycles)
        CSV.write(joinpath(timescales,"stops.csv"),stops)
        @test occursin("minimum step",R.continuation_caption(packet))
        root=joinpath(packet,"cases","input_dependent","baseline_3")
        for phase in (0.,.5)
            d=joinpath(root,"cycle_1_phase_$phase","pulses");mkpath(d)
            CSV.write(joinpath(d,"pulses.csv"),[
                (;axis="I_increase",duration=100.,amplitude=1.,phase,destination=phase==0 ? "rest" : "seizure"),
                (;axis="E_increase",duration=100.,amplitude=1.,phase,destination="rest")])
        end
        points,choice=R.phase_data(packet)
        @test choice==("I_increase",100.)
        @test Set(x.phase for x in points)==Set((0.,.5))
        @test length(unique(x.destination for x in points))==2
    end
end
