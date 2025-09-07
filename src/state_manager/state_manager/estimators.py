import threading

import torch

from utils.helpers import tensorify
from utils.math import quat_to_rotmatrix, quaternion_to_euler


class VelocityEstimator:
    def __init__(self, alpha=0.1, position_noise=0.01, velocity_noise=0.1, method="finite_diff", device="cuda:0"):
        """
        Initialize velocity estimator with multiple estimation methods.

        Args:
            alpha: Smoothing factor for velocity estimation
            position_noise: Process noise for position
            velocity_noise: Process noise for velocity
            method: Estimation method ('ekf' or 'finite_diff')
            device: Device for torch tensors
        """
        self.method = method
        self.alpha = alpha
        self.device = device

        # EKF specific initialization
        if method == "ekf":
            # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
            self.state = torch.zeros(12, dtype=torch.float32, device=device)
            self.last_quaternion = None
            self.covariance = torch.eye(12, dtype=torch.float32, device=device) * 1000

            # Process noise covariance
            self.Q = torch.diag(torch.tensor([
                position_noise,
                position_noise,
                position_noise,  # position
                velocity_noise,
                velocity_noise,
                velocity_noise,  # linear velocity
                0.1,
                0.1,
                0.1,  # orientation
                0.1,
                0.1,
                0.1,  # angular velocity
            ], dtype=torch.float32, device=device))

            # Measurement noise covariance
            self.R = torch.diag(torch.tensor([
                position_noise,
                position_noise,
                position_noise,  # position
                0.1,
                0.1,
                0.1,  # orientation
            ], dtype=torch.float32, device=device))

        # Finite differencing specific initialization
        elif method == "finite_diff":
            self.last_position = None
            self.last_quaternion = None
            self.smoothed_linear_velocity = torch.zeros(3, dtype=torch.float32, device=device)
            self.smoothed_angular_velocity = torch.zeros(3, dtype=torch.float32, device=device)

        self.last_timestamp = None
        self._lock = threading.Lock()

    def _compute_angular_velocity(self, q1, q2, dt):
        """
        Compute angular velocity using skew-symmetric matrix method. Better than finite differencing
        for calculating angular velocity.

        :param q1: Previous quaternion (torch tensor)
        :param q2: Current quaternion (torch tensor)
        :param dt: Time delta

        :return: Angular velocity vector (torch tensor)
        """
        # Convert quaternions to rotation matrices
        R1 = tensorify(quat_to_rotmatrix(q1), dtype=torch.float32, device=self.device)
        R2 = tensorify(quat_to_rotmatrix(q2), dtype=torch.float32, device=self.device)

        # Compute relative rotation matrix
        R_diff = torch.matmul(R2, R1.T)

        # Extract angular velocity from skew-symmetric matrix
        try:
            # Compute matrix logarithm using torch operations
            # For rotation matrices, we can use eigenvalue decomposition
            eigenvalues, eigenvectors = torch.linalg.eig(R_diff)
            
            # Compute log of eigenvalues
            log_eigenvalues = torch.log(eigenvalues)
            
            # Reconstruct the matrix logarithm
            omega_skew = torch.matmul(torch.matmul(eigenvectors, torch.diag(log_eigenvalues)), torch.linalg.inv(eigenvectors))
            
            # Extract angular velocity from skew-symmetric matrix and divide by dt
            angular_velocity = torch.tensor([
                omega_skew[2, 1].real, 
                omega_skew[0, 2].real, 
                omega_skew[1, 0].real
            ], dtype=torch.float32, device=self.device) / dt
        except Exception:
            # Fallback to zero angular velocity if computation fails
            angular_velocity = torch.zeros(3, dtype=torch.float32, device=self.device)

        return angular_velocity

    def finite_diff_update(self, position, quaternion, timestamp, logger=None):
        """
        Finite differencing method for linear velocity estimation with smoothing. Skew-symmetric method for angular
        velocity estimation with smoothing

        :param position: Current 3D position [x, y, z] (torch tensor)
        :param quaternion: Orientation quaternion [w, x, y, z] (torch tensor)
        :param timestamp: Current timestamp

        :return: Estimated [linear_velocities, angular_velocities] (torch tensors)
        """
        with self._lock:
            # First measurement
            if self.last_position is None:
                self.last_position = position
                self.last_quaternion = quaternion
                self.last_timestamp = timestamp
                return self.smoothed_linear_velocity, self.smoothed_angular_velocity

            # Calculate time delta
            dt = timestamp - self.last_timestamp

            if dt <= 0:
                return self.smoothed_linear_velocity, self.smoothed_angular_velocity

            # Linear velocity using finite differencing
            raw_linear_velocity = (position - self.last_position) / dt
            # Angular velocity calculation
            raw_angular_velocity = self._compute_angular_velocity(self.last_quaternion, quaternion, dt)

            # Exponential smoothing for both linear and angular velocities
            self.smoothed_linear_velocity = (
                self.alpha * raw_linear_velocity + (1 - self.alpha) * self.smoothed_linear_velocity
            )

            self.smoothed_angular_velocity = (
                self.alpha * raw_angular_velocity + (1 - self.alpha) * self.smoothed_angular_velocity
            )

            # Update last known state
            self.last_position = position
            self.last_quaternion = quaternion
            self.last_timestamp = timestamp

            return self.smoothed_linear_velocity, self.smoothed_angular_velocity

    def ekf_update(self, position, quaternion, timestamp, logger=None):
        """
        Extended Kalman Filter velocity estimation.

        :param position: Current 3D position [x, y, z] (torch tensor)
        :param quaternion: Orientation quaternion [w, x, y, z] (torch tensor)
        :param timestamp: Current timestamp
        :return: Estimated [linear_velocities, angular_velocities] (torch tensors)
        """
        with self._lock:
            # Calculate time delta
            if self.last_timestamp is None:
                self.last_timestamp = timestamp
                self.state[:3] = position
                self.state[6:9] = tensorify(quaternion_to_euler(quaternion, order="wxyz"), dtype=torch.float32, device=self.device)
                self.last_quaternion = quaternion
                return torch.zeros(3, dtype=torch.float32, device=self.device), torch.zeros(3, dtype=torch.float32, device=self.device)

            dt = timestamp - self.last_timestamp

            if dt <= 0:
                return self.state[3:6], self.state[9:]

            # Prediction step
            F = torch.eye(12, dtype=torch.float32, device=self.device)
            F[0:3, 3:6] = torch.eye(3, dtype=torch.float32, device=self.device) * dt
            F[6:9, 9:12] = torch.eye(3, dtype=torch.float32, device=self.device) * dt

            # Predict state and covariance
            predicted_state = torch.matmul(F, self.state)
            predicted_covariance = torch.matmul(torch.matmul(F, self.covariance), F.T) + self.Q

            # Measurement update
            H = torch.zeros((6, 12), dtype=torch.float32, device=self.device)
            H[:3, :3] = torch.eye(3, dtype=torch.float32, device=self.device)  # Position
            H[3:6, 6:9] = torch.eye(3, dtype=torch.float32, device=self.device)  # Orientation
            # Compute Euler angles from current quaternion
            current_euler = tensorify(quaternion_to_euler(quaternion, order="wxyz"), dtype=torch.float32, device=self.device)

            # Angular velocity estimation
            raw_angular_velocity = self._compute_angular_velocity(self.last_quaternion, quaternion, dt)

            # Innovation
            innovation = torch.cat([position - predicted_state[:3], current_euler - predicted_state[6:9]])

            # Innovation covariance
            S = torch.matmul(torch.matmul(H, predicted_covariance), H.T) + self.R

            # Kalman gain
            K = torch.matmul(torch.matmul(predicted_covariance, H.T), torch.inverse(S))

            # Update state and covariance
            self.state = predicted_state + torch.matmul(K, innovation)
            self.state[9:] = raw_angular_velocity  # Update angular velocity
            self.covariance = torch.matmul((torch.eye(12, dtype=torch.float32, device=self.device) - torch.matmul(K, H)), predicted_covariance)
            self.last_quaternion = quaternion

            # Update timestamp
            self.last_timestamp = timestamp

            return self.state[3:6], self.state[9:]

    def update(self, position, quaternion, timestamp, logger=None):
        """
        Unified update method for velocity estimation.

        :param position: Current 3D position [x, y, z] (torch tensor)
        :param quaternion: Orientation quaternion [x, y, z, w] (torch tensor)
        :param timestamp: Current timestamp
        :return: Estimated [linear_velocities, angular_velocities] (torch tensors)
        """
        if self.method == "ekf":
            return self.ekf_update(position, quaternion, timestamp, logger)
        elif self.method == "finite_diff":
            return self.finite_diff_update(position, quaternion, timestamp, logger)
        else:
            raise ValueError(f"Unknown estimation method: {self.method}")
