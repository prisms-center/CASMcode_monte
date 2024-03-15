// bindings are local to the module they are included in by default

py::bind_map<std::map<std::string, bool>>(m, "BooleanValueMap");
py::bind_map<std::map<std::string, double>>(m, "ScalarValueMap");
py::bind_map<std::map<std::string, Eigen::VectorXd>>(m, "VectorValueMap");
py::bind_map<std::map<std::string, Eigen::MatrixXd>>(m, "MatrixValueMap");
