from __future__ import annotations

import torch

GRAVITY_DIR = torch.tensor([0, 0, -1.0])  # Standard gravity in the Z direction

def quaternion_to_euler(q: torch.Tensor, order: str = "wxyz") -> torch.Tensor:
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
        w, x, y, z = q[0], q[1], q[2], q[3]
    elif order == "xyzw":
        x, y, z, w = q[0], q[1], q[2], q[3]
    else:
        raise ValueError(f"Unknown quaternion order: {order}")

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if torch.abs(sinp) >= 1:
        pitch = torch.sign(sinp) * torch.pi / 2  # Use 90 degrees if out of range
    else:
        pitch = torch.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw])

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

def quat_to_rotmatrix(q: torch.Tensor, order: str = "wxyz") -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
    :param q: The quaternion. Shape is (4,).
    :param order: The convention/order of the quaternion in the arguments. Defaults to 'wxyz'

    :return : The corresponding rotation matrix. Shape is (3,3)
    """
    # Normalize quaternion to handle zero norm cases
    norm = torch.norm(q)
    if norm < 1e-8:
        # If quaternion is zero or very small, return identity matrix
        return torch.eye(3, dtype=q.dtype, device=q.device)
    
    q = q / norm
    
    if order == "wxyz":
        w, x, y, z = q[0], q[1], q[2], q[3]
    else:  # xyzw
        x, y, z, w = q[0], q[1], q[2], q[3]

    # Convert quaternion to rotation matrix
    # Using the standard formula for quaternion to rotation matrix conversion
    rot_matrix = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)]),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)]),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
    ])
    
    return rot_matrix

def euler_to_quaternion(roll: float, pitch: float, yaw: float, order: str = "wxyz") -> torch.Tensor:
    """
    Convert Euler angles to quaternion.

    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians
        order: The convention/order of the quaternion in the output. Defaults to 'wxyz'

    Returns:
        The corresponding quaternion. Shape is (4,)
    """
    # Convert Euler angles to quaternion using the standard formula
    # Roll (x), Pitch (y), Yaw (z) -> Quaternion (w, x, y, z)
    
    # Convert floats to tensors for torch operations
    roll_tensor = torch.tensor(roll, dtype=torch.float32)
    pitch_tensor = torch.tensor(pitch, dtype=torch.float32)
    yaw_tensor = torch.tensor(yaw, dtype=torch.float32)
    
    cr = torch.cos(roll_tensor * 0.5)
    sr = torch.sin(roll_tensor * 0.5)
    cp = torch.cos(pitch_tensor * 0.5)
    sp = torch.sin(pitch_tensor * 0.5)
    cy = torch.cos(yaw_tensor * 0.5)
    sy = torch.sin(yaw_tensor * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    if order == "wxyz":
        return torch.stack([w, x, y, z])
    else:  # xyzw
        return torch.stack([x, y, z, w])

@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[:, 0:1], -q[:, 1:]), dim=-1).view(shape)

@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

@torch.jit.script
def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (N, 4).

    Returns:
        The inverse quaternion in (w, x, y, z). Shape is (N, 4).
    """
    return normalize(quat_conjugate(q))

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)

@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)



@torch.jit.script
def combine_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t12: torch.Tensor | None = None, q12: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Combine transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{02} = T_{01} \times T_{12}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
        t12: Position of frame 2 w.r.t. frame 1. Shape is (N, 3).
            Defaults to None, in which case the position is assumed to be zero.
        q12: Quaternion orientation of frame 2 w.r.t. frame 1 in (w, x, y, z). Shape is (N, 4).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 0.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    # compute orientation
    if q12 is not None:
        q02 = quat_mul(q01, q12)
    else:
        q02 = q01
    # compute translation
    if t12 is not None:
        t02 = t01 + quat_apply(q01, t12)
    else:
        t02 = t01

    return t02, q02


@torch.jit.script
def subtract_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor | None = None, q02: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Subtract transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{12} = T_{01}^{-1} \times T_{02}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
        t02: Position of frame 2 w.r.t. frame 0. Shape is (N, 3).
            Defaults to None, in which case the position is assumed to be zero.
        q02: Quaternion orientation of frame 2 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 1.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    # compute orientation
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    # compute translation
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12


@torch.jit.script
def pose_diff(pos1: torch.Tensor, quat1: torch.Tensor, pos2: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    """
    Optimized JIT-compiled version of goal pose difference computation.
    Uses vectorized operations and avoids intermediate tensor allocations.
    """
    # Compute quaternion difference: quat_mul(object_quat_w, quat_conjugate(goal_quat_w))
    # Optimized quaternion conjugate: negate x, y, z components
    quat2_conj = torch.cat([quat2[0:1], -quat2[1:]], dim=0)
    
    # Vectorized quaternion multiplication using torch operations
    # Reshape to (1, 4) for batch operations
    q1 = quat1.unsqueeze(0)  # (1, 4)
    q2 = quat2_conj.unsqueeze(0)  # (1, 4)
    
    # Extract components
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    # Optimized quaternion multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    
    quat_diff = torch.stack([
        qq - ww + (z1 - y1) * (y2 - z2),  # w
        qq - xx + (x1 + w1) * (x2 + w2),  # x
        qq - yy + (w1 - x1) * (y2 + z2),  # y
        qq - zz + (z1 + y1) * (w2 - x2)   # z
    ], dim=1).squeeze(0)  # Remove batch dimension
    
    # Compute position difference
    pos_diff = pos2 - pos1
    
    # Concatenate results
    return torch.cat([pos_diff, quat_diff], dim=0)


@torch.jit.script
def pos_diff(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    """
    Optimized position difference computation.
    Computes pos2 - pos1 efficiently.
    
    Args:
        pos1: First position tensor (..., 3)
        pos2: Second position tensor (..., 3)
    
    Returns:
        Position difference tensor (..., 3)
    """
    return pos2 - pos1


@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to Quaternions.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians). Shape is (N,).
        pitch: Rotation around y-axis (in radians). Shape is (N,).
        yaw: Rotation around z-axis (in radians). Shape is (N,).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qw, qx, qy, qz], dim=-1)


@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)

@torch.jit.script
def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Computes the rotation difference between two quaternions.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        Angular error between input quaternions in radians.
    """
    quat_diff = quat_mul(q1, quat_conjugate(q2))
    return torch.norm(axis_angle_from_quat(quat_diff), dim=-1)