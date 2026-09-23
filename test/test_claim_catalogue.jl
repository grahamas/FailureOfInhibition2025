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
    @test catalogue["schema_version"] == 2
    @test [claim["id"] for claim in catalogue["claims"]] == expected_ids
    @test Dict(claim["id"] => [(location["line_start"], location["line_end"])
        for location in claim["locations"]] for claim in catalogue["claims"]) ==
        expected_locations
    @test catalogue["manuscript"]["revision"] ==
        "ed18f283f2c947d6a4710989664b2a05dcc74447"
    @test Set(evidence["id"] for evidence in catalogue["evidence"]) == Set(
        Iterators.flatten(claim["evidence_ids"] for claim in catalogue["claims"]))
    expected_decisions = Dict(
        "equal-slope-nonmonotonic" => ("validated", "accepted"),
        "independent-slopes-and-fits" => ("not_validated", "accepted"),
        "monotone-ordering" => ("validated", "accepted"),
        "figure3-anchor" => ("validated_narrower", "accepted"),
        "general-multistability" => ("not_validated", "pending"),
        "rescue" => ("counterexample", "accepted"),
        "cycles" => ("not_validated", "pending"),
        "intervention-superiority" => ("not_validated", "accepted"),
        "scope" => ("not_validated", "accepted"),
    )
    @test Dict(claim["id"] => (claim["disposition"], claim["author_review"])
        for claim in catalogue["claims"]) == expected_decisions
    expected_actions = Dict(
        "equal-slope-nonmonotonic" => "Retain the claim with the midpoint and symmetry qualification.",
        "independent-slopes-and-fits" => "Remove the independent-slope and superior-fit assertion from the Results; do not add an asymmetric response or fitting study in this revision.",
        "monotone-ordering" => "Retain the same-model, same-parameter equilibrium-ordering result and its stated exclusions.",
        "figure3-anchor" => "Revise the Figure 3 statement to the anchor-specific finite-search observation; do not infer addition, prevalence, or exhaustive attractor counts.",
        "general-multistability" => "Keep the general tristability or tetrastability assertion pending while the declared tetrastability search is run; revise only after author review of that result.",
        "rescue" => "Split the statement: retain the nonnegative-drive obstruction and the protocol-specific negative-E counterexample, and remove both universal rescue clauses.",
        "cycles" => "Keep the cycle amplitude or frequency assertion pending until candidate-specific orbit evidence exists; do not assert interictal spikes without an observation mapping.",
        "intervention-superiority" => "Move this statement to the Discussion as a theoretical hypothesis, preserving that no functional or clinical ranking has been validated.",
        "scope" => "Keep only the non-spatial fixed-point and traveling-wave analogy in the Discussion; remove functional cortical-state and clinical conclusions.",
    )
    @test Dict(claim["id"] => claim["author_action"] for claim in catalogue["claims"]) ==
        expected_actions

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
    invalid["claims"][1]["author_action"] = "  "
    @test_throws ArgumentError ClaimCatalogue.validate_catalogue(invalid)

    invalid = deepcopy(catalogue)
    delete!(invalid["claims"][1], "author_action")
    @test_throws ArgumentError ClaimCatalogue.validate_catalogue(invalid)

    invalid = deepcopy(catalogue)
    invalid["schema_version"] = 1
    @test_throws ArgumentError ClaimCatalogue.validate_catalogue(invalid)

    invalid = deepcopy(catalogue)
    invalid["state"] = "author_reviewed"
    @test_throws ArgumentError ClaimCatalogue.validate_catalogue(invalid)
    for claim in invalid["claims"]
        claim["author_review"] = "accepted"
    end
    @test ClaimCatalogue.validate_catalogue(invalid) === invalid
end
