from typing import Any, Dict, Optional

import torch

from utils.helpers import EMAFilter
from utils.math import unscale_transform


class ActionTerm:
    def __init__(self, configs: Dict[str, Any]):
        self.configs = configs

    def process_actions(self, actions: torch.Tensor):
        pass


class JointPositionAction(ActionTerm):
    def __init__(
        self,
        configs: Dict[str, Any],
        action_scale: torch.Tensor,
        actions_mapping: torch.Tensor,
        default_joint_pos: Optional[torch.Tensor] = None,
    ):
        super().__init__(configs)
        controller_config = configs.get("controller_config", {})
        self.clip_actions = controller_config.get("clip_actions", False)
        self.action_dim = controller_config.get("action_dim", 12)
        self.device = controller_config.get("device", "cpu")
        self.raw_action = torch.zeros(self.action_dim, dtype=torch.float32, device=self.device)
        self.action_scale = action_scale
        self.offset = (
            default_joint_pos
            if default_joint_pos is not None
            else torch.zeros(self.action_dim, dtype=torch.float32, device=self.device)
        )
        self.actions_mapping = actions_mapping
        # self.filtered_action = EMAFilter(configs.get("filtered_action_alpha", 0.9), self.action_dim)

    def process_actions(self, actions: torch.Tensor):
        # Apply exponential moving average filter to smooth actions
        raw_action = actions
        if self.clip_actions:
            raw_action = torch.clip(actions, -self.clip_actions, self.clip_actions)

        self.raw_action.copy_(raw_action)
        joint_pos_targets = (raw_action * self.action_scale + self.offset)[self.actions_mapping]
        return joint_pos_targets


class EMAJointPositionAction(JointPositionAction):
    def __init__(
        self,
        configs: Dict[str, Any],
        action_scale: torch.Tensor,
        actions_mapping: torch.Tensor,
        default_joint_pos: Optional[torch.Tensor] = None,
    ):
        super().__init__(configs, action_scale, actions_mapping, default_joint_pos)
        self.filtered_action = EMAFilter(configs.get("action_filter_alpha", 0.5), self.action_dim)

    def process_actions(self, actions: torch.Tensor):
        # Apply exponential moving average filter to smooth actions
        raw_action = actions
        if self.clip_actions:
            raw_action = torch.clip(actions, -self.clip_actions, self.clip_actions)

        self.raw_action.copy_(raw_action)
        filtered_action = self.filtered_action.filter(raw_action)
        joint_pos_targets = (filtered_action * self.action_scale + self.offset)[self.actions_mapping]
        return joint_pos_targets


class JointPositionToLimitsAction(JointPositionAction):
    def __init__(
        self,
        configs: Dict[str, Any],
        action_scale: torch.Tensor,
        actions_mapping: torch.Tensor,
        default_joint_pos: Optional[torch.Tensor] = None,
        soft_joint_pos_limits: Optional[torch.Tensor] = None,
        joint_ids: Optional[torch.Tensor] = None,
    ):
        """
        Initialize JointPositionToLimitsAction.

        Args:
            configs: Configuration dictionary
            action_scale: Scale factor for actions. Shape (action_dim,).
            actions_mapping: Mapping from action indices to joint indices. Shape (action_dim,).
            default_joint_pos: Default joint positions. Shape (num_joints,).
            soft_joint_pos_limits: Soft joint position limits. Shape (2, num_joints) where [0] is lower and [1] is upper.
            joint_ids: Joint indices that correspond to the actions. Shape (action_dim,).
        """
        super().__init__(configs, action_scale, actions_mapping, default_joint_pos)
        controller_config = configs.get("controller_config", {})

        self.action_scale = action_scale
        self._joint_ids = joint_ids if joint_ids is not None else actions_mapping

        # Initialize rescale_to_limits configuration
        self.cfg_rescale_to_limits = controller_config.get("rescale_to_limits", False)

        # Store soft joint position limits
        if soft_joint_pos_limits is not None:
            # Extract limits for the relevant joints
            # soft_joint_pos_limits shape: (2, num_joints) where [0] is lower, [1] is upper
            self._soft_joint_pos_limits = soft_joint_pos_limits[:, self._joint_ids]
        else:
            self._soft_joint_pos_limits = None
            # Disable rescale_to_limits if limits are not provided
            if self.cfg_rescale_to_limits:
                self.cfg_rescale_to_limits = False

        # Initialize storage tensors
        self.raw_action = torch.zeros_like(action_scale)
        self._processed_actions = torch.zeros_like(action_scale)

    def process_actions(self, actions: torch.Tensor):
        """Process actions following the reference pattern."""
        # store the raw actions
        self.raw_action[:] = actions

        self._processed_actions = self.raw_action * self.action_scale

        # when scale is 0, track default joint positions
        if isinstance(self.action_scale, float):
            if self.action_scale == 0.0:
                # use default joint positions when scale is 0
                self._processed_actions = self.offset[zero_scale_mask]
        elif isinstance(self.action_scale, torch.Tensor):
            # when scale is a tensor, use default positions where scale is 0
            zero_scale_mask = self.action_scale == 0.0
            if zero_scale_mask.any():
                self._processed_actions[zero_scale_mask] = self.offset[zero_scale_mask]

        # rescale the position targets if configured
        # this is useful when the input actions are in the range [-1, 1]
        if self.cfg_rescale_to_limits and self._soft_joint_pos_limits is not None:
            # clip to [-1, 1]
            actions = self._processed_actions.clamp(-1.0, 1.0)
            # rescale within the joint limits
            actions = unscale_transform(
                actions,
                self._soft_joint_pos_limits[0],  # lower limits
                self._soft_joint_pos_limits[1],  # upper limits
            )
            self._processed_actions[:] = actions[:]

        # Apply the actions_mapping to get final joint position targets
        # Map processed actions to joint space
        joint_pos_targets = self._processed_actions[self.actions_mapping]
        return joint_pos_targets


class EMAJointPositionToLimitsAction(JointPositionToLimitsAction):
    def __init__(
        self,
        configs: Dict[str, Any],
        action_scale: torch.Tensor,
        actions_mapping: torch.Tensor,
        default_joint_pos: Optional[torch.Tensor] = None,
        soft_joint_pos_limits: Optional[torch.Tensor] = None,
        joint_ids: Optional[torch.Tensor] = None,
    ):
        super().__init__(configs, action_scale, actions_mapping, default_joint_pos, soft_joint_pos_limits, joint_ids)
        controller_config = configs.get("controller_config", {})
        # EMA coefficient (alpha)
        self._alpha = controller_config.get("action_filter_alpha", 0.5)
        # Initialize previous applied actions (for moving average)
        self._prev_applied_actions = torch.zeros_like(action_scale)

    def process_actions(self, actions: torch.Tensor):
        """Process actions with EMA filtering following the reference pattern."""
        # apply affine transformations (and rescale if configured)
        super().process_actions(actions)

        # set position targets as moving average
        ema_actions = self._alpha * self._processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions

        # clamp the targets to joint limits
        if self._soft_joint_pos_limits is not None:
            self._processed_actions[:] = torch.clamp(
                ema_actions,
                min=self._soft_joint_pos_limits[0],  # lower limits
                max=self._soft_joint_pos_limits[1],  # upper limits
            )
        else:
            self._processed_actions[:] = ema_actions[:]

        # update previous targets
        self._prev_applied_actions[:] = self._processed_actions[:]

        # Apply the actions_mapping to get final joint position targets
        # Map processed actions to joint space
        joint_pos_targets = self._processed_actions[self.actions_mapping]
        return joint_pos_targets
