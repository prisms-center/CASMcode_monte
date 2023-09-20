// bindings are local to the module they are included in by default

py::bind_map<std::map<std::string, bool>>(m, "BooleanValueMap");
py::bind_map<std::map<std::string, double>>(m, "ScalarValueMap");
py::bind_map<std::map<std::string, Eigen::VectorXd>>(m, "VectorValueMap");
py::bind_map<std::map<std::string, Eigen::MatrixXd>>(m, "MatrixValueMap");
py::bind_map<monte::SamplerMap>(m, "SamplerMap",
                                R"pbdoc(
    SamplerMap stores :class:`~libcasm.monte.Sampler` by name of the sampled quantity

    Notes
    -----
    SamplerMap is a Dict[str, :class:`~libcasm.monte.Sampler`]-like object.
    )pbdoc");

py::bind_map<monte::StateSamplingFunctionMap>(m, "StateSamplingFunctionMap",
                                              R"pbdoc(
    StateSamplingFunctionMap stores :class:`~libcasm.monte.StateSamplingFunction` by name of the sampled quantity.

    Notes
    -----
    StateSamplingFunctionMap is a Dict[str, :class:`~libcasm.monte.StateSamplingFunction`]-like object.
    )pbdoc");

py::bind_map<monte::jsonStateSamplingFunctionMap>(
    m, "jsonStateSamplingFunctionMap",
    R"pbdoc(
    jsonStateSamplingFunctionMap stores :class:`~libcasm.monte.jsonStateSamplingFunction` by name of the sampled quantity.

    Notes
    -----
    jsonStateSamplingFunctionMap is a Dict[str, :class:`~libcasm.monte.jsonStateSamplingFunction`]-like object.
    )pbdoc");

py::bind_map<monte::RequestedPrecisionMap>(m, "RequestedPrecisionMap",
                                           R"pbdoc(
    RequestedPrecisionMap stores :class:`~libcasm.monte.RequestedPrecision` with :class:`~libcasm.monte.SamplerComponent` keys.

    Notes
    -----
    RequestedPrecisionMap is a Dict[:class:`~libcasm.monte.SamplerComponent`, :class:`~libcasm.monte.RequestedPrecision`]-like object.
    )pbdoc");

py::bind_map<monte::EquilibrationResultMap>(m, "EquilibrationResultMap",
                                            R"pbdoc(
    EquilibrationResultMap stores :class:`~libcasm.monte.IndividualEquilibrationResult` by :class:`~libcasm.monte.SamplerComponent`

    Notes
    -----
    EquilibrationResultMap is a Dict[:class:`~libcasm.monte.SamplerComponent`, :class:`~libcasm.monte.IndividualEquilibrationResult`]-like object.
    )pbdoc");

py::bind_map<monte::ConvergenceResultMap<statistics_type>>(
    m, "ConvergenceResultMap",
    R"pbdoc(
    ConvergenceResultMap stores :class:`~libcasm.monte.IndividualConvergenceResult` by :class:`~libcasm.monte.SamplerComponent`

    Notes
    -----
    ConvergenceResultMap is a Dict[:class:`~libcasm.monte.SamplerComponent`, :class:`~libcasm.monte.IndividualConvergenceResult`]-like object.
    )pbdoc");