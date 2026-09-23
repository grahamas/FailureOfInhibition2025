include(joinpath(@__DIR__, "..", "scripts", "claim_catalogue.jl"))

@testset "Draft manuscript claims catalogue" begin
    path = joinpath(@__DIR__, "..", "claims", "catalogue.toml")
    catalogue = ClaimCatalogue.load_catalogue(path)
    expected_ids = [
        "equal-slope-nonmonotonic",
        "independent-slopes-and-fits",
        "monotone-ordering",
        "figure3-anchor",
        "general-multistability",
        "rescue",
        "cycles",
        "intervention-superiority",
        "scope",
    ]
    expected_locations = Dict(
        "equal-slope-nonmonotonic" => [(264, 267)],
        "independent-slopes-and-fits" => [(279, 279), (307, 307)],
        "monotone-ordering" => [(282, 296)],
        "figure3-anchor" => [(360, 360)],
        "general-multistability" => [(340, 346), (375, 375)],
        "rescue" => [(350, 352)],
        "cycles" => [(377, 377)],
        "intervention-superiority" => [(379, 379)],
        "scope" => [(326, 326), (437, 439)],
    )
    @test catalogue["state"] == "draft"
    @test [claim["id"] for claim in catalogue["claims"]] == expected_ids
    @test Dict(claim["id"] => [(location["line_start"], location["line_end"])
        for location in claim["locations"]] for claim in catalogue["claims"]) ==
        expected_locations
    @test catalogue["manuscript"]["revision"] ==
        "ed18f283f2c947d6a4710989664b2a05dcc74447"
    @test Set(evidence["id"] for evidence in catalogue["evidence"]) == Set(
        Iterators.flatten(claim["evidence_ids"] for claim in catalogue["claims"]))

    invalid = deepcopy(catalogue)
    invalid["claims"][1]["disposition"] = "proven"
    @test_throws ArgumentError ClaimCatalogue.validate_catalogue(invalid)

    invalid = deepcopy(catalogue)
    invalid["claims"][2]["id"] = invalid["claims"][1]["id"]
    @test_throws ArgumentError ClaimCatalogue.validate_catalogue(invalid)

    invalid = deepcopy(catalogue)
    invalid["claims"][1]["evidence_ids"] = ["missing-evidence"]
    @test_throws ArgumentError ClaimCatalogue.validate_catalogue(invalid)

    invalid = deepcopy(catalogue)
    invalid["claims"][1]["locations"][1]["line_end"] = 0
    @test_throws ArgumentError ClaimCatalogue.validate_catalogue(invalid)

    invalid = deepcopy(catalogue)
    invalid["claims"][1]["dispositon"] = invalid["claims"][1]["disposition"]
    @test_throws ArgumentError ClaimCatalogue.validate_catalogue(invalid)

    invalid = deepcopy(catalogue)
    invalid["state"] = "author_reviewed"
    @test_throws ArgumentError ClaimCatalogue.validate_catalogue(invalid)
    for claim in invalid["claims"]
        claim["author_review"] = "accepted"
    end
    @test ClaimCatalogue.validate_catalogue(invalid) === invalid
end
