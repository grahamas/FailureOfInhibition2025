import SHA
import TOML

# The file format contains data only. Loading never invokes Julia code or establishes proof.
const _CERTIFICATE_FORMAT_VERSION = 1
const _CERTIFICATE_SOURCE_FILES = (
    "src/responses.jl", "src/drives.jl", "src/point_model.jl",
    "src/certification.jl", "src/certification_io.jl",
)
const _CERTIFICATE_DEPENDENCY_FILES = ("Project.toml", "Manifest.toml")

for relative_path in _CERTIFICATE_DEPENDENCY_FILES
    Base.include_dependency(joinpath(dirname(@__DIR__), relative_path))
end

function _certificate_source_hashes()
    Dict(relative_path => bytes2hex(SHA.sha256(read(joinpath(dirname(@__DIR__), relative_path))))
         for relative_path in _CERTIFICATE_SOURCE_FILES)
end

# This value is captured with the methods when Julia loads or precompiles the
# package, rather than being inferred later from potentially edited disk files.
const _CERTIFICATE_LOADED_SOURCE_HASHES = _certificate_source_hashes()

function _certificate_dependency_hashes()
    Dict(relative_path => bytes2hex(SHA.sha256(read(joinpath(dirname(@__DIR__), relative_path))))
         for relative_path in _CERTIFICATE_DEPENDENCY_FILES)
end
const _CERTIFICATE_LOADED_DEPENDENCY_HASHES = _certificate_dependency_hashes()

function _certificate_check_loaded_dependencies(expected=_CERTIFICATE_LOADED_DEPENDENCY_HASHES)
    current = _certificate_dependency_hashes()
    current == expected || throw(ArgumentError(
        "certification Project/Manifest files differ from the loaded context; restart Julia before certifying or replaying"))
    return current
end

function _certificate_check_loaded_sources(expected=_CERTIFICATE_LOADED_SOURCE_HASHES)
    current = _certificate_source_hashes()
    current == expected || throw(ArgumentError(
        "certification source files differ from the loaded code; restart Julia before certifying or replaying"))
    return current
end

function _certificate_provenance(model=nothing)
    provenance = Dict{String,String}(
        "julia_version" => string(VERSION),
        "IntervalArithmetic_version" => string(Base.pkgversion(IntervalArithmetic)),
        "rounding" => "IntervalArithmetic directed outward rounding; BigFloat endpoints",
    )
    merge!(provenance, _certificate_check_loaded_sources())
    merge!(provenance, _certificate_check_loaded_dependencies())
    if model !== nothing
        io = IOBuffer()
        TOML.print(io, _certificate_encode_model(model); sorted=true)
        provenance["model_sha256"] = bytes2hex(SHA.sha256(take!(io)))
        # Positive timescales and drive interpretation do not change the balance
        # equation. Numeric encodings remain type sensitive, including tonic zero.
        encoded_model = _certificate_encode_model(model)
        problem = Dict(
            "excitatory_response" => encoded_model["excitatory"]["response"],
            "inhibitory_response" => encoded_model["inhibitory"]["response"],
            "coupling" => encoded_model["coupling"],
            "tonic_drive" => [_certificate_encode_number(x) for x in _certification_drive(model)],
            "domain_convention" => "[0,1/2]^2; closed E,I domain",
            "balance_convention" => "g_X=-X+(1-X)*F_X(u_X)",
        )
        TOML.print(io, problem; sorted=true)
        provenance["equilibrium_problem_sha256"] = bytes2hex(SHA.sha256(take!(io)))
    end
    return provenance
end

function _certificate_encode_number(value::Real)
    isfinite(value) || throw(ArgumentError("certificate numbers must be finite"))
    if value isa BigFloat
        bits = precision(value)
        significand, exponent_value = setprecision(BigFloat, bits) do
            fraction, exponent_value = frexp(value)
            (iszero(value) ? BigInt(0) : BigInt(ldexp(fraction, bits)), exponent_value)
        end
        return Dict("type" => "BigFloat", "significand" => string(significand),
                    "exponent" => exponent_value - bits, "precision" => bits,
                    "negative_zero" => iszero(value) && signbit(value))
    elseif value isa Union{Float32,Float64}
        integer = value isa Float32 ? reinterpret(UInt32, value) : reinterpret(UInt64, value)
        return Dict("type" => string(typeof(value)), "bits" => string(integer; base=16))
    elseif value isa Integer && !(value isa Bool)
        return Dict("type" => string(typeof(value)), "value" => string(value))
    end
    throw(ArgumentError("certificate numbers must be integers, Float32, Float64, or BigFloat"))
end

const _CERTIFICATE_INTEGER_TYPES = Dict(
    "Int8" => Int8, "Int16" => Int16, "Int32" => Int32, "Int64" => Int64,
    "Int128" => Int128, "UInt8" => UInt8, "UInt16" => UInt16,
    "UInt32" => UInt32, "UInt64" => UInt64, "UInt128" => UInt128, "BigInt" => BigInt,
)

function _certificate_decode_number(data)
    kind = data["type"]
    value = if kind == "BigFloat"
        bits = data["precision"]
        bits isa Integer && !(bits isa Bool) && 2 <= bits <= 1_048_576 ||
            throw(ArgumentError("invalid BigFloat precision"))
        exponent_value = data["exponent"]
        exponent_value isa Integer && !(exponent_value isa Bool) ||
            throw(ArgumentError("invalid BigFloat exponent"))
        text = data["significand"]
        text isa String && ncodeunits(text) <= 400_000 && occursin(r"^-?(0|[1-9][0-9]*)$", text) ||
            throw(ArgumentError("invalid BigFloat significand"))
        significand = parse(BigInt, text)
        ndigits(abs(significand); base=2) <= bits ||
            throw(ArgumentError("BigFloat significand exceeds stored precision"))
        negative_zero = data["negative_zero"]
        negative_zero isa Bool || throw(ArgumentError("invalid zero sign"))
        negative_zero && !iszero(significand) && throw(ArgumentError("invalid zero sign"))
        setprecision(BigFloat, Int(bits)) do
            result = ldexp(BigFloat(significand), Int(exponent_value))
            !iszero(significand) && iszero(result) &&
                throw(ArgumentError("BigFloat exponent underflows the supported range"))
            negative_zero ? -zero(result) : result
        end
    elseif kind == "Float32"
        reinterpret(Float32, parse(UInt32, data["bits"]; base=16))
    elseif kind == "Float64"
        reinterpret(Float64, parse(UInt64, data["bits"]; base=16))
    elseif haskey(_CERTIFICATE_INTEGER_TYPES, kind)
        parse(_CERTIFICATE_INTEGER_TYPES[kind], data["value"])
    else
        throw(ArgumentError("unsupported certificate numeric type"))
    end
    isfinite(value) || throw(ArgumentError("certificate numbers must be finite"))
    return value
end

_certificate_encode_box(box) = Dict("lower" => [_certificate_encode_number(x) for x in box.lower],
                                    "upper" => [_certificate_encode_number(x) for x in box.upper])
function _certificate_decode_box(data)
    length(data["lower"]) == length(data["upper"]) == 2 ||
        throw(ArgumentError("certificate boxes must have two coordinates"))
    lower = Tuple(_certificate_decode_number.(data["lower"]))
    upper = Tuple(_certificate_decode_number.(data["upper"]))
    all(x -> x isa BigFloat, (lower..., upper...)) ||
        throw(ArgumentError("certificate box endpoints must be BigFloat"))
    all(lower .<= upper) || throw(ArgumentError("reversed certificate box endpoints"))
    return CertificationBox(lower, upper)
end

function _certificate_encode_response(response)
    response isa Union{LogisticResponse,FailureOfInhibitionResponse} ||
        throw(ArgumentError("unsupported certificate response"))
    return Dict("type" => string(nameof(typeof(response))),
                "parameters" => Dict(string(f) => _certificate_encode_number(getfield(response, f))
                                     for f in fieldnames(typeof(response))))
end

function _certificate_decode_response(data)
    parameters = data["parameters"]
    if data["type"] == "LogisticResponse"
        return LogisticResponse(; slope=_certificate_decode_number(parameters["slope"]),
                                threshold=_certificate_decode_number(parameters["threshold"]))
    elseif data["type"] == "FailureOfInhibitionResponse"
        return FailureOfInhibitionResponse(; slope=_certificate_decode_number(parameters["slope"]),
            onset_threshold=_certificate_decode_number(parameters["onset_threshold"]),
            failure_threshold=_certificate_decode_number(parameters["failure_threshold"]))
    end
    throw(ArgumentError("unsupported certificate response"))
end

function _certificate_encode_model(model::PointModelParameters)
    drive = model.drive
    encoded_drive = if drive isa NoDrive
        Dict("type" => "NoDrive")
    elseif drive isa PiecewiseConstantDrive && isempty(drive.pulses)
        Dict("type" => "PiecewiseConstantDrive",
             "baseline" => [_certificate_encode_number(x) for x in drive.baseline],
             "interpretation" => string(drive.interpretation))
    else
        throw(ArgumentError("certificates require an autonomous, pulse-free drive"))
    end
    population(p) = Dict("timescale" => _certificate_encode_number(p.timescale),
                         "response" => _certificate_encode_response(p.response))
    return Dict("excitatory" => population(model.excitatory),
                "inhibitory" => population(model.inhibitory), "drive" => encoded_drive,
                "coupling" => Dict(string(f) => _certificate_encode_number(getfield(model.coupling, f))
                                   for f in fieldnames(typeof(model.coupling))))
end

function _certificate_decode_model(data)
    population(p) = PopulationParameters(; timescale=_certificate_decode_number(p["timescale"]),
                                         response=_certificate_decode_response(p["response"]))
    drive_data = data["drive"]
    drive = if drive_data["type"] == "NoDrive"
        NoDrive()
    elseif drive_data["type"] == "PiecewiseConstantDrive"
        interpretation = drive_data["interpretation"] == "AfferentExcitation" ? AfferentExcitation :
                         drive_data["interpretation"] == "AbstractIntervention" ? AbstractIntervention :
                         throw(ArgumentError("unsupported drive interpretation"))
        length(drive_data["baseline"]) == 2 || throw(ArgumentError("drive baseline requires E and I"))
        PiecewiseConstantDrive(; baseline=Tuple(_certificate_decode_number.(drive_data["baseline"])),
                                 pulses=(), interpretation)
    else
        throw(ArgumentError("unsupported certificate drive"))
    end
    coupling = data["coupling"]
    return PointModelParameters(; excitatory=population(data["excitatory"]),
        inhibitory=population(data["inhibitory"]), drive,
        coupling=PointCoupling(; (f => _certificate_decode_number(coupling[string(f)])
                                  for f in (:e_to_e, :i_to_e, :e_to_i, :i_to_i))...))
end

"""
    write_equilibrium_certificate(path, certificate)

Write a versioned TOML proof artifact with exact binary numeric inputs, proof
boxes, witnesses, search limits, and source/dependency hashes. The full model hash
retains timescales; the equilibrium problem hash omits positive timescales and
drive interpretation, uses tonic drive values, and remains numeric-type sensitive.
Writing preserves
unresolved results and does not certify them. Only supported autonomous point
models can be archived. No Julia Serialization or executable expressions are used.
"""
function write_equilibrium_certificate(path, certificate)
    options = certificate.options
    data = Dict{String,Any}(
        "format" => "FailureOfInhibition2025 equilibrium certificate",
        "version" => _CERTIFICATE_FORMAT_VERSION,
        "model" => _certificate_encode_model(certificate.model),
        "options" => Dict(string(f) => _certificate_encode_number(getfield(options, f))
                          for f in fieldnames(typeof(options))),
        "domain" => _certificate_encode_box(certificate.domain),
        "status" => string(certificate.status),
        "has_equilibrium_count" => certificate.equilibrium_count !== nothing,
        "equilibrium_count" => something(certificate.equilibrium_count, 0),
        "processed_boxes" => certificate.processed_boxes,
        "evaluations" => certificate.evaluations,
        "elapsed_seconds" => _certificate_encode_number(certificate.elapsed_seconds),
        "provenance" => certificate.provenance,
        "roots" => [Dict("witness" => _certificate_encode_box(r.witness),
                         "enclosure" => _certificate_encode_box(r.enclosure),
                         "precision" => r.precision,
                         "norm_bound" => _certificate_encode_number(r.norm_bound)) for r in certificate.roots],
        "nodes" => [Dict("box" => _certificate_encode_box(n.box), "parent" => n.parent,
                         "children" => collect(n.children), "depth" => n.depth,
                         "precision" => n.precision, "kind" => string(n.kind),
                         "root_id" => n.root_id, "reason" => string(n.reason)) for n in certificate.nodes],
    )
    open(path, "w") do io
        TOML.print(io, data; sorted=true)
    end
    return path
end

"""
    read_equilibrium_certificate(path)

Decode a versioned, data-only TOML equilibrium artifact. Parameters and endpoints
retain their stored integer or binary floating-point values and precision. Reading
a recorded `certified` status does not establish a proof; call
[`verify_equilibrium_certificate`](@ref) to replay it against current sources.
"""
function read_equilibrium_certificate(path)
    data = TOML.parsefile(path)
    data["format"] == "FailureOfInhibition2025 equilibrium certificate" &&
        data["version"] == _CERTIFICATE_FORMAT_VERSION ||
        throw(ArgumentError("unsupported equilibrium certificate format/version"))
    fields = (:maximum_boxes, :maximum_depth, :maximum_evaluations, :maximum_records,
              :time_limit, :maximum_precision)
    options = CertificationOptions(; (f => _certificate_decode_number(data["options"][string(f)]) for f in fields)...)
    roots = CertificationRoot[
        CertificationRoot(_certificate_decode_box(r["witness"]), _certificate_decode_box(r["enclosure"]),
                          r["precision"], _certificate_decode_number(r["norm_bound"])) for r in data["roots"]]
    nodes = CertificationNode[]
    for n in data["nodes"]
        length(n["children"]) == 2 || throw(ArgumentError("certificate nodes require two child indices"))
        push!(nodes, CertificationNode(_certificate_decode_box(n["box"]), n["parent"],
            Tuple(n["children"]), n["depth"], n["precision"], Symbol(n["kind"]), n["root_id"], Symbol(n["reason"])))
    end
    status = Symbol(data["status"])
    status in (:certified, :unresolved) || throw(ArgumentError("invalid certificate status"))
    data["has_equilibrium_count"] isa Bool || throw(ArgumentError("invalid equilibrium count flag"))
    count = data["has_equilibrium_count"] ? data["equilibrium_count"] : nothing
    return EquilibriumCertificate(_certificate_decode_model(data["model"]), options,
        _certificate_decode_box(data["domain"]), roots, nodes, status, count,
        data["processed_boxes"], data["evaluations"], _certificate_decode_number(data["elapsed_seconds"]),
        Dict{String,String}(data["provenance"]))
end

"""
    verify_equilibrium_certificate(certificate)::Bool

Require matching current source, Julia, Project, and Manifest provenance, then
replay every interval witness and coverage step. A stored success status, root
count, or successful file read alone is never proof. Incomplete, malformed,
tampered, unsupported, or source-mismatched certificates return `false`.
"""
function verify_equilibrium_certificate(certificate)::Bool
    try
        certificate.model isa PointModelParameters || return false
        certificate.status == :certified || return false
        certificate.provenance == _certificate_provenance(certificate.model) || return false
        field, jacobian = _certification_callbacks(certificate.model)
        return _verify_certificate(field, jacobian, certificate)
    catch exception
        exception isa InterruptException && rethrow()
        return false
    end
end
