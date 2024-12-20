import libcasm.monte.sampling as sampling


def test_SelectedEventFunctionParams():
    params = sampling.SelectedEventFunctionParams()
    assert isinstance(params, sampling.SelectedEventFunctionParams)

    # test collect and do_not_collect
    params.collect("event_type")
    assert len(params.function_names) == 1
    assert "event_type" in params.function_names

    params.do_not_collect("event_type")
    assert len(params.function_names) == 0


def test_SelectedEventFunctionParams_get_parameters():
    params = sampling.SelectedEventFunctionParams()
    assert isinstance(params, sampling.SelectedEventFunctionParams)

    params.collect(
        name="dE_activated",
        initial_begin=0.01,
        bin_width=1.0,
        max_size=10,
        spacing="linear",
    )
    data = params.get_parameters("dE_activated")
    # print(xtal.pretty_json(data))
    assert "initial_begin" in data
    assert "bin_width" in data
    assert "max_size" in data
    assert "spacing" in data


def test_collect_hop_correlations():
    params = sampling.SelectedEventFunctionParams()
    assert isinstance(params, sampling.SelectedEventFunctionParams)

    assert params.correlations_data_params is None

    params.collect_hop_correlations(
        jumps_per_position_sample=10,
    )
    assert isinstance(params.correlations_data_params, sampling.CorrelationsDataParams)
    assert params.correlations_data_params.jumps_per_position_sample == 10

    data = params.correlations_data_params.to_dict()
    # print(xtal.pretty_json(data))

    correlations_data_params_in = sampling.CorrelationsDataParams.from_dict(data)
    assert isinstance(correlations_data_params_in, sampling.CorrelationsDataParams)

    params.do_not_collect_hop_correlations()
    assert params.correlations_data_params is None

    params.collect_hop_correlations(**data)
    assert isinstance(params.correlations_data_params, sampling.CorrelationsDataParams)
    assert params.correlations_data_params.jumps_per_position_sample == 10


def test_SelectedEventFunctionParams_to_and_from_dict():
    params = sampling.SelectedEventFunctionParams()

    params.collect("event_type")
    params.collect(
        name="dE_activated",
        initial_begin=0.01,
        bin_width=1.0,
        max_size=10,
        spacing="linear",
    )
    params.collect_hop_correlations(
        jumps_per_position_sample=10,
    )
    # print(params)

    # test to_dict and from_dict
    data = params.to_dict()
    # print(xtal.pretty_json(data))
    assert isinstance(data, dict)

    params_in = sampling.SelectedEventFunctionParams.from_dict(data)
    # print(params_in)
    assert isinstance(params_in, sampling.SelectedEventFunctionParams)
    assert len(params.function_names) == 2
    assert "event_type" in params.function_names
    assert "dE_activated" in params.function_names
    assert isinstance(params.correlations_data_params, sampling.CorrelationsDataParams)
    assert params.correlations_data_params.jumps_per_position_sample == 10
