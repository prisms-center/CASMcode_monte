from typing import Optional

import libcasm.monte.sampling._monte_sampling as _sampling


class RequestedPrecisionConstructor:
    """Sets requested precision level for equilibration and convergence

    Parameters
    ----------
    sampling_functions: _sampling.StateSamplingFunctionMap`
        State sampling function dictionary, which specifies the
        quantities that may be converged.
    completion_check_params: _sampling.CompletionCheckParams`
        Completion check parameters to set requested_precision.

    """

    def __init__(
        self,
        sampling_functions: _sampling.StateSamplingFunctionMap,
        completion_check_params: _sampling.CompletionCheckParams,
    ):
        self.sampling_functions = sampling_functions
        self.completion_check_params = completion_check_params

    def set_precision(
        self,
        quantity: str,
        abs: Optional[float] = None,
        rel: Optional[float] = None,
        component_name: Optional[list[str]] = None,
        component_index: Optional[list[int]] = None,
    ):
        """Set requested precision level for equilibration and convergence

        Allows setting absolute or relative precision to the specified level for
        the specified quantities. By default, all components are converged to
        the same level. If `component_name` or `component_index` are specified,
        then only the specified components are requested to converge to that level.

        Parameters
        ----------
        quantity: str
            The name of the quantity to be converged. Must match
            a state sampling function name.
        abs: Optional[float]=None
            The requested absolute convergence level
        rel: Optional[float]=None,
            The requested relative convergence level
        component_name: Optional[list[str]]=None
            The name of components to converge. Must be in the
            `component_names` of the state sampling function for
            the named quantity.
        component_index: Optional[list[int]]=None
            The indices of components to converge.

        Returns
        -------
        self: RequestedPrecisionConstructor
            To allow chaining multiple calls, `self` is returned
        """
        if quantity not in self.sampling_functions:
            raise Exception(f"{quantity} is not in sampling_functions")
        f = self.sampling_functions[quantity]

        if rel is None and abs is None:
            raise Exception("No abs or rel precision specified")

        if component_index is None:
            component_index = []

        if component_name is not None:
            for n in component_name:
                if n not in f.component_names:
                    raise Exception(f"{n} is not a component of {quantity}")
                component_index.append(f.component_names.index(n))

        component_index = list(set(component_index))

        if len(component_index) == 0:
            component_index = list(range(len(f.component_names)))

        for i in component_index:
            key = _sampling.SamplerComponent(
                sampler_name=quantity,
                component_name=f.component_names[i],
                component_index=i,
            )
            req = _sampling.RequestedPrecision(abs=abs, rel=rel)
            self.completion_check_params.requested_precision[key] = req

        return self


def converge(
    sampling_functions: _sampling.StateSamplingFunctionMap,
    completion_check_params: _sampling.CompletionCheckParams,
):
    """Helper for setting completion_check_params.requested_precision

    Example usage:

    .. code-block:: Python

        converge(sampling_functions, completion_check_params)
            .set_precision("formation_energy", abs=0.001)
            .set_precision("parametric_composition", abs=0.001, component_name=["a"])
            .set_precision("corr", abs=0.001, component_index=[0,1,2,3])

    Allows setting absolute or relative precision to the specified level for
    the specified quantities. By default, all components are converged to
    the same level. If `component_name` or `component_index` are specified,
    then only the specified components are requested to converge to that level.

    Parameters
    ----------
    sampling_functions: _sampling.StateSamplingFunctionMap`
        State sampling function dictionary
    completion_check_params: _sampling.CompletionCheckParams
        Completion check parameters to set requested_precision

    Returns
    -------
    rpc: RequestedPrecisionConstructor
        A RequestedPrecisionConstructor
    """
    return RequestedPrecisionConstructor(
        sampling_functions,
        completion_check_params,
    )
