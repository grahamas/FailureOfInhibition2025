using Test, TOML, SHA
if !isdefined(@__MODULE__, :write_report)
    include(joinpath(@__DIR__, "..", "scripts", "atlas", "report.jl"))
end

@testset "Atlas evidence report" begin
    mktempdir() do directory
        outside = joinpath(directory, "outside.toml")
        write(outside, "schema_version = 1\n")
        output = joinpath(directory, "output"); mkdir(output)
        symlink(outside, joinpath(output, "state.toml"))
        @test_throws ArgumentError atlas_report_data(output)
    end
    roots = [Dict("state"=>[0.1,0.2],"stability"=>"Attracting","spectrum"=>"RealSpectrum")]
    points = [Dict("id"=>id,"signature"=>sig,"roots"=>roots) for (id,sig) in (("a","same"),("b","same"),("c","other"))]
    @test length(atlas_region_components(points,[])) == 3
    path = Dict("id"=>"ab","from"=>"a","to"=>"b","validated"=>true,"signature_preserved"=>true,"status"=>"completed")
    @test length(atlas_region_components(points,[path])) == 3
    @test length(atlas_region_components(points,[merge(path,Dict("sample_ids"=>["a","b"]))])) == 3
    @test length(atlas_region_components(points,[merge(path,Dict("to"=>"c"))])) == 3
    @test length(atlas_region_components(points,[merge(path,Dict("validated"=>false))])) == 3
    five_points = [Dict("id"=>id,"signature"=>"same","roots"=>roots) for id in ("a","m1","m2","m3","b")]
    sampled_path = merge(path, Dict("sample_ids"=>["a","m1","m2","m3","b"]))
    @test atlas_region_components(five_points,[sampled_path]) == [["a","b","m1","m2","m3"]]
    @test length(atlas_region_components(five_points,[merge(sampled_path,Dict("sample_ids"=>["a","m1","m1","m1","b"]))])) == 5
    @test length(atlas_region_components(five_points,[merge(sampled_path,Dict("sample_ids"=>["a","missing","b"]))])) == 5
    @test length(atlas_region_components(five_points,[merge(sampled_path,Dict("sample_ids"=>["b","a"]))])) == 5
    @test length(atlas_region_components([merge(p,Dict("eligible_for_regions"=>p["id"]!="m2")) for p in five_points],[sampled_path])) == 4
    @test length(atlas_region_components([merge(p,Dict("signature"=>p["id"]=="m2" ? "different" : "same")) for p in five_points],[sampled_path])) == 5
    @test isempty(atlas_region_components([merge(points[1], Dict("roots"=>[]))],[]))
    @test isempty(atlas_region_components([merge(points[1], Dict("signature"=>"Eturn:unresolved"))],[]))
    @test_throws ArgumentError atlas_region_components([points[1],points[1]],[])
    mktempdir() do directory
        mkpath(joinpath(directory,"results"))
        jobs = [Dict("id"=>"1","kind"=>"discovery","status"=>"completed","elapsed_seconds"=>2.0,"result_file"=>"results/1.toml"),
            Dict("id"=>"2","kind"=>"cycles","status"=>"timed_out","elapsed_seconds"=>5.0,"error"=>"<script>alert(1)</script>"),
            Dict("id"=>"3","kind"=>"discovery","status"=>"pending")]
        open(joinpath(directory,"state.toml"),"w") do io; TOML.print(io,Dict("schema_version"=>1,"mode"=>"benchmark","jobs"=>jobs)); end
        open(joinpath(directory,"config.toml"),"w") do io; TOML.print(io,Dict("version"=>1)); end
        evidence = Dict("points"=>vcat(five_points,[points[3]]),"paths"=>[sampled_path],"events"=>[Dict("id"=>"<unsafe>","status"=>"candidate")],
            "sections"=>[Dict("parameter_pair"=>["A","a"],"section_samples"=>[Dict("status"=>"sampled","parameters"=>Dict("A"=>8.0,"a"=>1.0))],"transition_curves"=>[Dict("points"=>[Dict("parameters"=>Dict("A"=>9.0,"a"=>1.1),"classification"=>"fold_candidate")])])])
        open(joinpath(directory,"results","1.toml"),"w") do io; TOML.print(io,Dict("status"=>"completed","evidence"=>evidence)); end
        jobs[1]["result_hash"] = bytes2hex(sha256(read(joinpath(directory,"results","1.toml"))))
        open(joinpath(directory,"state.toml"),"w") do io; TOML.print(io,Dict("schema_version"=>1,"mode"=>"benchmark","jobs"=>jobs)); end
        destination = write_report(directory)
        html = read(destination,String)
        @test occursin("&lt;script&gt;",html)
        @test !occursin("<script>",html)
        @test occursin("&lt;unsafe&gt;",html)
        @test occursin("CompletenessNotCertified",html)
        @test occursin("<svg",html)
        @test occursin("A (dimensionless)",html)
        @test occursin("href=\"results/1.toml\"",html)
        @test !occursin("transition_witness",read(joinpath(directory,"base_cases.csv"),String))
        coverage = TOML.parsefile(joinpath(directory,"coverage.toml"))
        @test coverage["observed_components"] == 2
        @test coverage["pending_jobs"] == 1
        @test coverage["issues"] == 2
        @test _report_safe_file(directory,"../state.toml") === nothing
        @test _report_safe_file(directory,joinpath(directory,"state.toml")) === nothing
        @test atlas_timing_rows(jobs)[1]["completed_samples"] == 0 # cycles first
        @test occursin("region_representative",read(joinpath(directory,"base_cases.csv"),String))
        before = read(destination)
        @test read(write_report(directory)) == before
        open(joinpath(directory,"results","1.toml"),"a") do io; println(io); end
        @test isempty(atlas_report_data(directory).records["points"])
        @test any(issue -> issue["status"]=="unverified_result",atlas_report_data(directory).issues)
        write_report(directory)
        @test TOML.parsefile(joinpath(directory,"coverage.toml"))["observed_components"] == 0
        @test !occursin("region_representative",read(joinpath(directory,"base_cases.csv"),String))
        delete!(jobs[1],"result_hash")
        open(joinpath(directory,"state.toml"),"w") do io; TOML.print(io,Dict("schema_version"=>1,"mode"=>"benchmark","jobs"=>jobs)); end
        @test isempty(atlas_report_data(directory).records["points"])
        @test any(issue -> occursin("no recorded",get(issue,"detail","")),atlas_report_data(directory).issues)
    end
    mktempdir() do directory
        open(joinpath(directory,"state.toml"),"w") do io; TOML.print(io,Dict("jobs"=>[])); end
        @test isfile(write_report(directory))
        @test TOML.parsefile(joinpath(directory,"coverage.toml"))["observed_points"] == 0
    end
    for filename in ("index.html", "base_cases.csv", "timings.csv", "coverage.toml")
        mktempdir() do directory
            output = joinpath(directory,"report"); mkdir(output)
            open(joinpath(output,"state.toml"),"w") do io; TOML.print(io,Dict("jobs"=>[])); end
            generated = ("index.html", "base_cases.csv", "timings.csv", "coverage.toml")
            for name in generated; write(joinpath(output,name),"existing $name"); end
            outside = joinpath(directory,"outside"); write(outside,"outside remains unchanged")
            rm(joinpath(output,filename)); symlink(outside,joinpath(output,filename))
            @test_throws ArgumentError write_report(output)
            @test read(outside,String) == "outside remains unchanged"
            @test islink(joinpath(output,filename))
            @test all(name -> name == filename || read(joinpath(output,name),String) == "existing $name",generated)
        end
    end
    mktempdir() do directory
        path = joinpath(directory,"target"); write(path,"original")
        @test_throws ErrorException _report_atomic_write(io -> (write(io,"incomplete"); error("writer failed")),path)
        @test read(path,String) == "original"
        @test readdir(directory) == ["target"]
    end
end
