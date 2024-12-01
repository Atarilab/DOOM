import numpy as np

def quaternion_to_euler(q) -> np.ndarray:
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        x, y, z, w = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return [roll, pitch, yaw]


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v.

    Args:
        q: The quaternion in (x, y, z, w). Shape is (4,).
        v: The vector in (x, y, z). Shape is (3,).

    Returns:
        The rotated vector in (x, y, z). Shape is (3,).
    """
    # Extract quaternion components
    q_w = q[-1]
    q_vec = q[:-1]

    # Compute the terms for the rotation
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * 2.0 * q_w
    c = q_vec * np.dot(q_vec, v) * 2.0
    
    return a - b + c