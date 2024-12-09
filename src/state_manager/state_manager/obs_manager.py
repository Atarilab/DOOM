import logging
from typing import Any, Callable, Dict, Optional


class ObsTerm:
    """
    A descriptor for defining state observations with flexible parameter support.
    This is used to define an observation term for each mode/controller.
    """

    def __init__(
        self,
        func: Callable = None,
        params: Optional[Dict[str, Any]] = None,
        include: bool = True,
    ):
        """
        Initialize an observation term.

        :param func: Function to compute the observation
        :param params: Optional dictionary of parameters to pass to the function
        :param include: Whether to include the observation in the method compute()
        """
        self._func = func
        self._params = params or {}
        self._dependencies = None
        self._include = include

        # Analyze function signature to determine dependencies
        if func:
            self._dependencies = self._extract_dependencies(func)

    def _extract_dependencies(self, func: Callable) -> Optional[list[str]]:
        """
        Extract state dependencies from the observation function.

        :param func: Observation function
        :return: List of required state keys
        """
        import inspect

        # Get function signature
        sig = inspect.signature(func)
        parameters = list(sig.parameters.keys())

        # If first parameter is 'states', it needs full state
        if parameters and parameters[0] == "states":
            return None  # Indicates full state access

        # Try to extract type hints
        annotations = func.__annotations__

        # If no type hints, return None
        if not annotations:
            return None

        # Look for explicit dependencies in type hints
        return list(annotations.get("states", {}).keys())

    def __call__(self, full_state: Dict[str, Any]) -> Any:
        """
        Compute the observation from the full state.

        :param full_state: Complete state dictionary
        :return: Computed observation
        """
        if self._func is None:
            raise ValueError("No observation function defined")

        # Determine how to call the function
        if self._dependencies is None:
            # If no specific dependencies, pass full state and params
            return self._func(full_state, **self._params)

        # Extract only the required dependencies
        state_subset = {dep: full_state.get(dep) for dep in self._dependencies}

        # Call with subset state and params
        return self._func(state_subset, **self._params)


class ObservationManager:
    """
    Manages a collection of state observations with dependency tracking.
    This is responsible for computing the observations for each mode/controller from the states computed
    by the StateManager.

    Note that the order of the observations upon registrations matters as it will be maintained for computing the observations
    and can be directly passed to your controller.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize the State Observation Manager.

        :param logger: Optional logger for tracking observations
        """
        self._observations: Dict[str, ObsTerm] = {}
        self._all_observations: Dict[str, Any] = {}
        self.logger = logger or logging.getLogger(__name__)

    def register(self, name: str, obs_term: ObsTerm):
        """
        Register a new state observation.

        :param name: Name of the observation
        :param obs_term: ObsTerm instance defining the observation
        """
        if name in self._observations:
            self.logger.warning(f"Overwriting existing observation: {name}")

        self._observations[name] = obs_term

    def compute(self, combined_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute all registered observations from the combined state.

        :param combined_state: Complete state dictionary
        :return: Dictionary of computed observations
        """
        self._all_observations.clear()
        observations = {}

        for name, obs_term in self._observations.items():
            try:
                computed_obs = obs_term(combined_state)
                self._all_observations[name] = computed_obs

                # Only include in returned observations if include is True
                if obs_term._include:
                    observations[name] = computed_obs
            except Exception as e:
                self.logger.error(f"Error computing observation {name}: {e}")

        return observations

    def get_observation(self, name: str) -> Any:
        """
        Retrieve a specific observation, including those not returned by compute().

        :param name: Name of the observation
        :return: The computed observation
        """
        return self._all_observations.get(name)
