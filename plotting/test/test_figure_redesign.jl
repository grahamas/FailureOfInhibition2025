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
        mkpath(root)
        write(joinpath(root,"summary.toml"),"B_E = 2.0\nB_I = 1.0\nvalidated_cycles = 1\n")
        write(joinpath(root,"orbit_1.toml"),"validation = \"NumericallyValidatedPeriodicOrbit\"\n")
        for phase in (0.,.5)
            d=joinpath(root,"cycle_1_phase_$phase","pulses");mkpath(d)
            CSV.write(joinpath(d,"pulses.csv"),[
                (;axis="I_increase",duration=100.,amplitude=1.,phase,destination=phase==0 ? "rest" : "seizure"),
                (;axis="E_increase",duration=100.,amplitude=1.,phase,destination="rest")])
        end
        other=joinpath(packet,"cases","input_dependent","baseline_1");mkpath(other)
        write(joinpath(other,"summary.toml"),"B_E = 0.5\nB_I = 0.0\nvalidated_cycles = 1\n")
        for phase in (0.,.5)
            d=joinpath(other,"cycle_1_phase_$phase","pulses");mkpath(d)
            CSV.write(joinpath(d,"pulses.csv"),[(;axis="E_increase",duration=50.,amplitude=a,phase,
                destination=phase==0 ? "rest" : "seizure") for a in (1.,2.)])
        end
        points,choice=R.phase_data(packet)
        @test choice==("I_increase",100.)
        @test Set(x.phase for x in points)==Set((0.,.5))
        @test length(unique(x.destination for x in points))==2
        write(joinpath(root,"summary.toml"),"B_E = 2.0\nB_I = 1.0\nvalidated_cycles = 2\n")
        @test_throws ErrorException R.phase_data(packet)
    end
end

@testset "Mechanism basin display switches and legacy packets" begin
    R=FigureRedesignRenderer
    mktempdir() do packet
        roots=[(;root=1,condition="failure",E=.1,I=.1,stability="Attracting",role="active")]
        @test R.mechanism_overlays(packet,"failure",roots).basins===nothing
        @test R.mechanism_overlays(packet,"failure",roots;show_separatrices=true).separatrices===nothing
        folder=joinpath(packet,"mechanism","overlays");mkpath(folder)
        entry=Dict("shape"=>[2,2],"basin_options"=>Dict("grid_points"=>2,"bounds"=>[[0.,.5],[0.,.5]]))
        R.NullclineOverlays.write_record(joinpath(folder,"metadata.toml"),Dict("conditions"=>Dict("failure"=>entry)))
        CSV.write(joinpath(folder,"failure_basins.csv"),[
            (;condition="failure",E_index=e,I_index=i,E=(e-1)*.5,I=(i-1)*.5,
              label=e==i ? 1 : 0,status=e==i ? "compatible" : "unresolved") for e in 1:2 for i in 1:2])
        data=R.mechanism_overlays(packet,"failure",roots)
        @test data.basins.labels==[1 0;0 1]
        @test R.mechanism_overlays(packet,"failure",roots;show_basins=false).basins===nothing
        fig=R.Figure();ax=R.Axis(fig[1,1])
        R.draw_mechanism_overlays!(ax,data,roots)
        @test length(ax.scene.plots)==1
        @test R.pale_role_color("active")!=R.pale_role_color("seizure")
        @test R.pale_role_color("active").g>.8
        # A rendered smoke artifact proves the layer is accepted by the plotting backend.
        R.save(joinpath(packet,"basin_smoke.png"),fig)
        @test filesize(joinpath(packet,"basin_smoke.png"))>0
    end
end
