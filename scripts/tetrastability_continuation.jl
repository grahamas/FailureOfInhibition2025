"""Tetrastability protocol functions included by run_tetrastability_search.jl."""
function _replace_axis(cell, axis, value)
    return merge(cell, NamedTuple{(axis,)}((value,)))
end

function _continuation_options(config, axis; smoke=false)
    table = config.continuation
    width = getproperty(config.axes, axis).upper - getproperty(config.axes, axis).lower
    return ContinuationOptions(initial_step=table.initial_step,
        minimum_step=table.minimum_step, maximum_step=table.maximum_step,
        max_steps=smoke ? table.smoke_max_steps : table.max_steps,
        max_corrector_iters=table.max_corrector_iters,
        corrector_atol=table.corrector_atol,
        parameter_difference_step=table.parameter_difference_step,
        parameter_scale=width, rank_atol=table.rank_atol, rank_rtol=table.rank_rtol)
end

function continue_candidate!(state, config, confirmation, output; smoke=false)
    confirmation.confirmed || return
    cell = confirmation.cell
    final_grid = last(config.confirmation_grid_points)
    final_item = confirmation.refined[(:failure_of_inhibition, final_grid)]
    equilibrium_options = refined_options(config.equilibrium_options,
        config.confirmation.tolerance_factor)
    stability_options = refined_options(config.stability_options,
        config.confirmation.tolerance_factor)
    for track in confirmation.assessment.quality_tracks
        track_index = track.track_id
        root_index = track.matches[final_grid]
        initial_state = final_item.result.equilibria[root_index].state
        for axis in SEARCH_AXES
            bounds_table = getproperty(config.axes, axis)
            bounds = (bounds_table.lower, bounds_table.upper)
            initial_parameter = getproperty(cell, axis)
            factory = parameter -> models_for_cell(config,
                _replace_axis(cell, axis, parameter)).failure_of_inhibition
            branch_id = "$(confirmation.candidate_id)_root$(track_index)_$(axis)"
            try
                result = continue_equilibria(factory, initial_state, initial_parameter;
                    parameter_bounds=bounds, options=_continuation_options(config, axis; smoke),
                    equilibrium_options, stability_options)
                Evidence.write_toml(joinpath(output, "contexts", "continuations", branch_id * ".toml"),
                    Dict("candidate_id" => confirmation.candidate_id,
                        "root_track" => track_index, "axis" => string(axis),
                        "initial_parameter" => initial_parameter, "parameter_bounds" => bounds,
                        "initial_solve" => result.initial_solve,
                        "negative" => result.negative, "positive" => result.positive,
                        "options" => result.options, "completeness" => result.completeness))
                for branch in (result.negative, result.positive)
                    _append_rows(joinpath(output, "branches.csv"), [(;
                        candidate_id=confirmation.candidate_id, root_track=track_index,
                        axis=string(axis), direction=branch.direction,
                        points=length(branch.points), attempts=length(branch.attempts),
                        screening_candidates=length(branch.candidates),
                        termination=string(branch.termination))])
                    point_rows = [begin
                        point = branch.points[index]
                        (; candidate_id=confirmation.candidate_id, root_track=track_index,
                            axis=string(axis), direction=branch.direction, point=index,
                            parameter=point.parameter, E=point.state[1], I=point.state[2],
                            residual_norm=point.equilibrium_attempt.residual_norm,
                            tangent_parameter=point.tangent[3],
                            stability=string(point.stability.classification),
                            spectral_abscissa=point.stability.spectral_abscissa,
                            trace=point.stability.trace, determinant=point.stability.determinant)
                    end for index in eachindex(branch.points)]
                    _append_rows(joinpath(output, "continuation_points.csv"), point_rows)
                    attempt_rows = [(; candidate_id=confirmation.candidate_id,
                        root_track=track_index, axis=string(axis), direction=branch.direction,
                        attempt=index, step=attempt.step, iterations=attempt.iterations,
                        arclength_residual=attempt.arclength_residual,
                        status=string(attempt.status), accepted=attempt.accepted)
                        for (index, attempt) in enumerate(branch.attempts)]
                    _append_rows(joinpath(output, "continuation_attempts.csv"), attempt_rows)
                    screening_rows = [begin
                        first_point = branch.points[candidate.first_point]
                        second_point = branch.points[candidate.second_point]
                        (; candidate_id=confirmation.candidate_id, root_track=track_index,
                            axis=string(axis), direction=branch.direction,
                            kind=string(candidate.kind), first_point=candidate.first_point,
                            second_point=candidate.second_point,
                            lower=min(first_point.parameter, second_point.parameter),
                            upper=max(first_point.parameter, second_point.parameter))
                    end for candidate in branch.candidates]
                    _append_rows(joinpath(output, "continuation_candidates.csv"), screening_rows)
                end
            catch error
                error isa InterruptException && rethrow()
                push!(state.failures, branch_id)
                Evidence.write_toml(joinpath(output, "contexts", "continuations",
                    branch_id * "_failure.toml"), Evidence.error_record(error))
            end
        end
    end
end
