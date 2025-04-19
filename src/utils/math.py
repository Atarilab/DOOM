from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

GRAVITY_DIR = torch.tensor([0, 0, -1.0])  # Standard gravity in the Z direction


def quaternion_to_euler(q, order="wxyz") -> np.ndarray:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q: The quaternion. Shape is (4,).
        order: The convention/order of the quaternion in the arguments.
               Defaults to 'wxyz'.

    Returns:
        The corresponding euler angles. Shape is (3,)
    """
    if order == "wxyz":
        w, x, y, z = q
    elif order == "xyzw":
        x, y, z, w = q
    else:
        raise ValueError(f"Unknown quaternion order: {order}")

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor, order="wxyz") -> torch.Tensor:
    """
    Rotate a vector by the inverse of a quaternion along the last dimension of q and v.

    :param q: The quaternion. Shape is (..., 4).
    :param v: The vector in (x, y, z). Shape is (..., 3).
    :param order: The convention/order of the quaternion in the arguments. Defaults to 'wxyz'

    :return : The rotated vector in (x, y, z). Shape is (..., 3).
    """
    if order == "wxyz":
        q_w = q[..., 0]
        q_vec = q[..., 1:]
    else:
        q_w = q[..., -1]
        q_vec = q[..., :-1]

    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)

    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def quat_to_rotmatrix(q: np.ndarray, order="wxyz") -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    :param q: The quaternion. Shape is (4,).
    :param order: The convention/order of the quaternion in the arguments. Defaults to 'wxyz'

    :return : The corresponding rotation matrix. Shape is (3,3)
    """
    if order == "wxyz":
        q = q[[1, 2, 3, 0]]

    rot_matrix = R.from_quat(q).as_matrix()
    return rot_matrix
