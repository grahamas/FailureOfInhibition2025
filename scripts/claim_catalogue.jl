"""Schema validation for the draft manuscript claims catalogue."""
module ClaimCatalogue

import TOML

const STATES = Set(("draft", "author_reviewed"))
const DISPOSITIONS = Set(("validated", "validated_narrower", "counterexample", "not_validated"))
const AUTHOR_REVIEWS = Set(("pending", "accepted", "changes_requested"))
const EVIDENCE_CATEGORIES = Set(("analytic", "numerical", "scope"))
const ROOT_KEYS = ("schema_version", "catalogue_id", "state", "manuscript", "evidence", "claims")
const MANUSCRIPT_KEYS = ("repository", "revision", "path")
const EVIDENCE_KEYS = ("id", "category", "summary")
const CLAIM_KEYS = ("id", "statement", "locations", "disposition",
    "evidence_ids", "limitation", "author_review")

function require_keys(table, expected_keys, label)
    missing = filter(key -> !haskey(table, key), expected_keys)
    isempty(missing) || throw(ArgumentError(
        "$label is missing required keys: $(join(missing, ", "))"))
end

function require_exact_keys(table, expected_keys, label)
    require_keys(table, expected_keys, label)
    extra = setdiff(collect(keys(table)), collect(expected_keys))
    isempty(extra) || throw(ArgumentError(
        "$label has unknown keys: $(join(sort!(extra), ", "))"))
end

function required_string(table, key, label)
    value = table[key]
    value isa AbstractString && !isempty(strip(value)) ||
        throw(ArgumentError("$label.$key must be a nonempty string"))
    return value
end

function required_id(table, key, label)
    value = required_string(table, key, label)
    occursin(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", value) ||
        throw(ArgumentError("$label.$key must be a lowercase hyphenated identifier"))
    return value
end

function require_unique(values, label)
    length(unique(values)) == length(values) ||
        throw(ArgumentError("$label must be unique"))
end

function validate_catalogue(raw)
    raw isa AbstractDict || throw(ArgumentError("catalogue must be a TOML table"))
    require_exact_keys(raw, ROOT_KEYS, "catalogue")
    raw["schema_version"] === 1 ||
        throw(ArgumentError("catalogue.schema_version must be integer 1"))
    required_id(raw, "catalogue_id", "catalogue")
    state = required_string(raw, "state", "catalogue")
    state in STATES || throw(ArgumentError("unknown catalogue.state: $state"))
    manuscript = raw["manuscript"]
    manuscript isa AbstractDict || throw(ArgumentError("manuscript must be a table"))
    require_exact_keys(manuscript, MANUSCRIPT_KEYS, "manuscript")
    required_string(manuscript, "repository", "manuscript")
    revision = required_string(manuscript, "revision", "manuscript")
    occursin(r"^[0-9a-f]{40}$", revision) ||
        throw(ArgumentError("manuscript.revision must be a full lowercase Git SHA"))
    required_string(manuscript, "path", "manuscript")

    evidence = raw["evidence"]
    evidence isa AbstractVector && !isempty(evidence) ||
        throw(ArgumentError("catalogue.evidence must be a nonempty array of tables"))
    evidence_ids = String[]
    for (index, item) in enumerate(evidence)
        label = "evidence[$index]"
        item isa AbstractDict || throw(ArgumentError("$label must be a table"))
        require_exact_keys(item, EVIDENCE_KEYS, label)
        push!(evidence_ids, required_id(item, "id", label))
        category = required_string(item, "category", label)
        category in EVIDENCE_CATEGORIES ||
            throw(ArgumentError("unknown $label.category: $category"))
        required_string(item, "summary", label)
    end
    require_unique(evidence_ids, "evidence IDs")

    claims = raw["claims"]
    claims isa AbstractVector && !isempty(claims) ||
        throw(ArgumentError("catalogue.claims must be a nonempty array of tables"))
    claim_ids = String[]
    for (index, claim) in enumerate(claims)
        label = "claims[$index]"
        claim isa AbstractDict || throw(ArgumentError("$label must be a table"))
        require_exact_keys(claim, CLAIM_KEYS, label)
        push!(claim_ids, required_id(claim, "id", label))
        required_string(claim, "statement", label)
        required_string(claim, "limitation", label)
        disposition = required_string(claim, "disposition", label)
        disposition in DISPOSITIONS ||
            throw(ArgumentError("unknown $label.disposition: $disposition"))
        author_review = required_string(claim, "author_review", label)
        author_review in AUTHOR_REVIEWS ||
            throw(ArgumentError("unknown $label.author_review: $author_review"))

        locations = claim["locations"]
        locations isa AbstractVector && !isempty(locations) ||
            throw(ArgumentError("$label.locations must be a nonempty array"))
        for (location_index, location) in enumerate(locations)
            location_label = "$label.locations[$location_index]"
            location isa AbstractDict ||
                throw(ArgumentError("$location_label must be a table"))
            require_exact_keys(location, ("line_start", "line_end"), location_label)
            line_start = location["line_start"]
            line_end = location["line_end"]
            line_start isa Integer && !(line_start isa Bool) && line_start > 0 ||
                throw(ArgumentError("$location_label.line_start must be a positive integer"))
            line_end isa Integer && !(line_end isa Bool) && line_end >= line_start ||
                throw(ArgumentError("$location_label.line_end must be an integer at or after line_start"))
        end

        references = claim["evidence_ids"]
        references isa AbstractVector && !isempty(references) &&
            all(reference -> reference isa AbstractString, references) ||
            throw(ArgumentError("$label.evidence_ids must be a nonempty string array"))
        require_unique(references, "$label evidence references")
        unknown = setdiff(references, evidence_ids)
        isempty(unknown) || throw(ArgumentError(
            "$label references unknown evidence IDs: $(join(unknown, ", "))"))
    end
    require_unique(claim_ids, "claim IDs")
    state == "author_reviewed" && any(claim -> claim["author_review"] != "accepted", claims) &&
        throw(ArgumentError("author_reviewed catalogue requires every claim to be accepted"))
    return raw
end

function load_catalogue(path::AbstractString)
    return validate_catalogue(TOML.parsefile(path))
end

function main(args=ARGS)
    length(args) <= 1 || throw(ArgumentError("usage: claim_catalogue.jl [PATH]"))
    path = isempty(args) ? joinpath(@__DIR__, "..", "claims", "catalogue.toml") : only(args)
    catalogue = load_catalogue(path)
    println("Catalogue schema valid: $(length(catalogue["claims"])) claim rows and $(length(catalogue["evidence"])) evidence entries")
    return 0
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(ClaimCatalogue.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
