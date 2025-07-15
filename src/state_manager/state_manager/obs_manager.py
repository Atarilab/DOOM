import logging
from typing import Any, Callable, Dict, Optional, Union, Tuple
import torch


class ObsTerm:
    """
    A descriptor for defining state observations with flexible parameter support.
    This is used to define an observation term for each mode/controller.
    """

    def __init__(
        self,
        func: Optional[Callable] = None,
        params: Optional[Dict[str, Any]] = None,
        include: bool = True,
        obs_dim: int = 0,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize an observation term.

        :param func: Function to compute the observation
        :param params: Optional dictionary of parameters to pass to the function
        :param include: Whether to include the observation in the method compute()
        :param obs_dim: Dimension/shape of the observation tensor
        :param dtype: Data type for the preallocated tensor
        :param device: Device for the preallocated tensor
        """
        self._func = func
        self._params = params or {}
        self._dependencies = None
        self._include = include
        self._obs_dim = obs_dim
        self._dtype = dtype or torch.float32
        self._device = device
        self._preallocated_tensor = None

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

    def preallocate_tensor(self):
        """
        Preallocate the tensor for this observation term.
        """
        if self._obs_dim > 0:
            self._preallocated_tensor = torch.zeros(
                self._obs_dim, 
                dtype=self._dtype, 
                device=self._device
            )
        

    def __call__(self, full_state: Dict[str, Any]) -> Any:
        """
        Compute the observation from the full state.

        :param full_state: Complete state dictionary
        :return: Computed observation
        """
        if self._func is None:
            raise ValueError("No observation function defined")

        # Add device and dtype to params if available
        call_params = self._params.copy()
        if self._device is not None:
            call_params['device'] = self._device
        if self._dtype is not None:
            call_params['dtype'] = self._dtype

        # Determine how to call the function
        if self._dependencies is None:
            # If no specific dependencies, pass full state and params
            computed_obs = self._func(full_state, **call_params)
        else:
            # Extract only the required dependencies
            state_subset = {dep: full_state.get(dep) for dep in self._dependencies}
            # Call with subset state and params
            computed_obs = self._func(state_subset, **call_params)

        # If we have a preallocated tensor and the computed observation is a tensor,
        # copy the data instead of creating a new tensor
        if (self._preallocated_tensor is not None and 
            isinstance(computed_obs, torch.Tensor)):
            self._preallocated_tensor.copy_(computed_obs)
            return self._preallocated_tensor
        
        return computed_obs


class ObservationManager:
    """
    Manages a collection of observations with dependency tracking.
    This is responsible for computing the observations for each mode/controller from the states computed
    by the StateManager.

    Note that the order of the observations upon registrations matters as it will be maintained for computing the observations
    and can be directly passed to your controller/policy.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, device: Optional[torch.device] = None):
        """
        Initialize the observation manager.

        Args:
            logger: Optional logger for debugging
            device: Device for preallocated tensors (e.g., torch.device('cuda'))
        """
        self.logger = logger
        self.device = device
        self.obs_terms = {}
        self.obs_order = []
        
        # Preallocated full observation tensor
        self.full_obs_tensor = None
        self.full_obs_dim = 0
        self.obs_start_indices = {}  # Track start indices for each observation

    def register(self, name: str, obs_term: ObsTerm):
        """
        Register a new state observation and preallocate its tensor if dimension is specified.

        :param name: Name of the observation
        :param obs_term: ObsTerm instance defining the observation
        """
        if name in self.obs_terms:
            if self.logger:
                self.logger.warning(f"Overwriting existing observation: {name}")

        self.obs_terms[name] = obs_term
        
        # Preallocate tensor if obs_dim is specified
        if obs_term._obs_dim > 0:
            obs_term.preallocate_tensor()
            
            # Track start index for this observation in the full tensor
            self.obs_start_indices[name] = self.full_obs_dim
            self.full_obs_dim += obs_term._obs_dim

    def preallocate_full_tensor(self, batch_size: int = 1):
        """
        Preallocate the full observation tensor after all observations are registered.
        
        :param batch_size: Batch size for the observation tensor (default: 1)
        """
        if self.full_obs_dim > 0:
            self.full_obs_tensor = torch.zeros(
                (batch_size, self.full_obs_dim),
                dtype=torch.float32,
                device=self.device
            )
            if self.logger:
                self.logger.info(f"Preallocated full observation tensor with shape: ({batch_size}, {self.full_obs_dim})")

    def compute(self, combined_state: Dict[str, Any], batch_idx: int = 0) -> Dict[str, Any]:
        """
        Compute all registered observations from the combined state.

        :param combined_state: Complete state dictionary
        :param batch_idx: Batch index for the observation tensor (default: 0)
        :return: Dictionary of computed observations
        """
        self.obs_order.clear()
        observations = {}

        for name, obs_term in self.obs_terms.items():
            try:
                computed_obs = obs_term(combined_state)
                self.obs_order.append(name)

                # Only include in returned observations if include is True
                if obs_term._include:
                    observations[name] = computed_obs
                    self.full_obs_tensor[batch_idx, self.obs_start_indices[name]:self.obs_start_indices[name] + obs_term._obs_dim] = computed_obs
                        
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error computing observation {name}: {e}")

        return observations

    def compute_full_tensor(self, combined_state: Dict[str, Any], batch_idx: int = 0) -> torch.Tensor:
        """
        Compute the full concatenated observation tensor efficiently using pre-allocated buffers.

        :param combined_state: Complete state dictionary
        :param batch_idx: Batch index for the observation tensor (default: 0)
        :return: Full concatenated observation tensor
        """
        if self.full_obs_tensor is None:
            raise ValueError("Full observation tensor not preallocated. Call preallocate_full_tensor() first.")

        # Compute all observations (this will also fill the preallocated tensor)
        self.compute(combined_state, batch_idx)
        
        return self.full_obs_tensor

    def get_observation(self, name: str) -> Any:
        """
        Retrieve a specific observation, including those not returned by compute().

        :param name: Name of the observation
        :return: The computed observation
        """
        return self.obs_terms.get(name)

    def get_full_obs_dim(self) -> int:
        """
        Get the total dimension of the full observation tensor.
        
        :return: Total observation dimension
        """
        return self.full_obs_dim

    def get_obs_start_index(self, name: str) -> Optional[int]:
        """
        Get the start index of a specific observation in the full tensor.
        
        :param name: Name of the observation
        :return: Start index in the full tensor, or None if not found
        """
        return self.obs_start_indices.get(name)
