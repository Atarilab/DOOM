from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch


class ObservationHistoryStorage:
    def __init__(
        self, num_envs: int, policy_architecture: str, num_obs: int, max_length: int, device: torch.device = "cpu"
    ):
        """
        Initialize a FIFO queue for state history, starting with zeros at initialization.

        Args:
            num_envs (int): Number of environments.
            policy_architecture (str): Policy architecture.
            num_obs (int): Number of observations per environment.
            max_length (int): Maximum length of the state history for each environment.
            device (torch.device): Device to store the buffer (e.g., "cuda" or "cpu").
        """
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.max_length = max_length
        self.policy_architecture = policy_architecture
        self.device = device

        # Initialize the buffer with zeros of shape (num_envs, num_obs * max_length)
        self.buffer = torch.zeros((num_envs, num_obs * max_length), device=device)

    def add(self, observation: torch.Tensor):
        """
        Add a new observation to the buffer. Perform FIFO replacement.

        Args:
            observation (torch.Tensor): The new observation to add.
                                         Should have shape `(num_envs, num_obs)`.
        """
        if observation.shape != (self.num_envs, self.num_obs):
            raise ValueError(
                f"Observation shape must be ({self.num_envs}, {self.num_obs}). Current shape: ({observation.shape})"
            )

        # Shift the buffer to make space for the new observation
        self.buffer[:, : -self.num_obs] = self.buffer[:, self.num_obs :].clone()

        # Add the new observation at the end
        self.buffer[:, -self.num_obs :] = observation

    def get(self) -> torch.Tensor:
        """
        Get the current state history.

        Returns:
            torch.Tensor: Observation tensor with correct shape for policy.
        """
        obs = self.buffer.detach().clone()
        if self.policy_architecture == "recurrent":
            return obs
        else:
            return obs.unsqueeze(0)

    def reset(self, done: torch.Tensor):
        """Reset the buffer for environments that are done.

        Args:
            done (torch.Tensor): mask of dones.
        """

        done_indices = torch.nonzero(done == 1)
        self.buffer[done_indices] = 0.0


def reorder_robot_states(states: np.ndarray, origin_order: List[str], target_order: List[str]) -> np.ndarray:
    """
    Reorder robot states based on origin and target leg orders.

    Args:
        states (np.ndarray): Input states to be reordered
        origin_order (List[str]): Original leg order
        target_order (List[str]): Desired leg order

    Returns:
        np.ndarray: Reordered states
    """
    # Convert input to NumPy array if it's not already
    states = np.asarray(states)

    # Validate input lengths (4 for feet states and 12 for joint states)
    if states.size not in {4, 12}:
        raise ValueError(f"Expected 4 or 12 states, got {states.size}")

    if len(origin_order) != 4 or len(target_order) != 4:
        raise ValueError("Both origin and target orders must be lists of 4 legs")

    num_legs = 4
    # Determine number of entries per leg
    entries_per_leg = states.size // num_legs

    # Create reordering indices using NumPy
    reorder_indices = np.concatenate(
        [
            np.arange(
                origin_order.index(leg) * entries_per_leg,
                origin_order.index(leg) * entries_per_leg + entries_per_leg,
            )
            for leg in target_order
        ]
    )

    # Reorder states using advanced NumPy indexing
    return states[reorder_indices]


def create_joint_mapping(list_a: list, list_b: list) -> list:
    """
    Creates a mapping from list A to list B.

    Args:
        list_a (list): Source list of joint names
        list_b (list): Target list of joint names

    Returns:
        list: List of indices where each element i gives the index in list_b
                that corresponds to list_a[i]. Returns -1 if not found.
    """
    mapping = []
    for item in list_a:
        try:
            idx = list_b.index(item)
            mapping.append(idx)
        except ValueError:
            # Item not found in list_b
            mapping.append(-1)
    return mapping

def tensorify(data: Union[np.ndarray, torch.Tensor, list, float, tuple], 
                  dtype: torch.dtype = torch.float32,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Helper function to ensure data is a torch tensor with consistent dtype and device.
    
    :param data: Input data (numpy array, torch tensor, list, or scalar)
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device (if None, keeps original device)
    :return: torch.Tensor with specified dtype and device
    """
    if isinstance(data, torch.Tensor):
        tensor = data.to(dtype=dtype)    
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).to(dtype=dtype)
    else: # Handle lists, scalars, etc.
        tensor = torch.tensor(data, dtype=dtype)
    
    if device is not None:
        tensor = tensor.to(device)
    return tensor

class EMAFilter:
    def __init__(self, alpha: float, action_dim: int):
        self.alpha = alpha
        self.filtered_value = np.zeros(action_dim)
        self.is_first_action = True

    def filter(self, new_value: np.ndarray) -> np.ndarray:
        if self.is_first_action:
            self.filtered_value = new_value
            self.is_first_action = False
        else:
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value
