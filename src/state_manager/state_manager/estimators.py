import threading

import numpy as np
from scipy.linalg import logm

from utils.math import quat_to_rotmatrix, quaternion_to_euler


class VelocityEstimator:
    def __init__(self, alpha=0.1, position_noise=0.01, velocity_noise=0.1, method="finite_diff"):
        """
        Initialize velocity estimator with multiple estimation methods.

        Args:
            alpha: Smoothing factor for velocity estimation
            position_noise: Process noise for position
            velocity_noise: Process noise for velocity
            method: Estimation method ('ekf' or 'finite_diff')
        """
        self.method = method
        self.alpha = alpha

        # EKF specific initialization
        if method == "ekf":
            # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
            self.state = np.zeros(12)
            self.last_quaternion = None
            self.covariance = np.eye(12) * 1000

            # Process noise covariance
            self.Q = np.diag(
                [
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
                ]
            )

            # Measurement noise covariance
            self.R = np.diag(
                [
                    position_noise,
                    position_noise,
                    position_noise,  # position
                    0.1,
                    0.1,
                    0.1,  # orientation
                ]
            )

        # Finite differencing specific initialization
        elif method == "finite_diff":
            self.last_position = None
            self.last_quaternion = None
            self.smoothed_linear_velocity = np.zeros(3)
            self.smoothed_angular_velocity = np.zeros(3)

        self.last_timestamp = None
        self._lock = threading.Lock()

    def _compute_angular_velocity(self, q1, q2, dt):
        """
        Compute angular velocity using skew-symmetric matrix method. Better than finite differencing
        for calculating angular velocity.

        :param q1: Previous quaternion
        :param q2: Current quaternion
        :param dt: Time delta

        :return: Angular velocity vector
        """
        # Convert quaternions to rotation matrices

        R1 = quat_to_rotmatrix(q1)
        R2 = quat_to_rotmatrix(q2)

        # Compute relative rotation matrix
        R_diff = R2 @ R1.T

        # Extract angular velocity from skew-symmetric matrix
        try:
            omega_skew = logm(R_diff) / dt
            angular_velocity = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
        except Exception:
            # Fallback to zero angular velocity if computation fails
            angular_velocity = np.zeros(3)

        return angular_velocity

    def finite_diff_update(self, position, quaternion, timestamp, logger=None):
        """
        Finite differencing method for linear velocity estimation with smoothing. Skew-symmetric method for angular
        velocity estimation with smoothing

        :param position: Current 3D position [x, y, z]
        :param quaternion: Orientation quaternion [w, x, y, z]
        :param timestamp: Current timestamp

        :return: Estimated [linear_velocities, angular_velocities]
        """
        with self._lock:
            # First measurement
            if self.last_position is None:
                self.last_position = position
                self.last_quaternion = quaternion
                self.last_timestamp = timestamp
                return np.zeros(3), np.zeros(3)

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

        :param position: Current 3D position [x, y, z]
        :param quaternion: Orientation quaternion [w, x, y, z]
        :param timestamp: Current timestamp
        :return: Estimated [linear_velocities, angular_velocities]
        """
        with self._lock:
            # Calculate time delta
            if self.last_timestamp is None:
                self.last_timestamp = timestamp
                self.state[:3] = position
                self.state[6:9] = quaternion_to_euler(quaternion, order="wxyz")
                self.last_quaternion = quaternion
                return np.zeros(3), np.zeros(3)

            dt = timestamp - self.last_timestamp

            if dt <= 0:
                return self.state[3:6], self.state[9:]

            # Prediction step
            F = np.eye(12)
            F[0:3, 3:6] = np.eye(3) * dt
            F[6:9, 9:12] = np.eye(3) * dt

            # Predict state and covariance
            predicted_state = F @ self.state
            predicted_covariance = F @ self.covariance @ F.T + self.Q

            # Measurement update
            H = np.zeros((6, 12))
            H[:3, :3] = np.eye(3)  # Position
            H[3:6, 6:9] = np.eye(3)  # Orientation
            # Compute Euler angles from current quaternion
            current_euler = quaternion_to_euler(quaternion, order="wxyz")

            # Angular velocity estimation
            raw_angular_velocity = self._compute_angular_velocity(self.last_quaternion, quaternion, dt)

            # Innovation
            innovation = np.concatenate([position - predicted_state[:3], current_euler - predicted_state[6:9]])

            # Innovation covariance
            S = H @ predicted_covariance @ H.T + self.R

            # Kalman gain
            K = predicted_covariance @ H.T @ np.linalg.inv(S)

            # Update state and covariance
            self.state = predicted_state + K @ innovation
            self.state[9:] = raw_angular_velocity  # Update angular velocity
            self.covariance = (np.eye(12) - K @ H) @ predicted_covariance
            self.last_quaternion = quaternion

            # Update timestamp
            self.last_timestamp = timestamp

            return self.state[3:6], self.state[9:]

    def update(self, position, quaternion, timestamp, logger=None):
        """
        Unified update method for velocity estimation.

        :param position: Current 3D position [x, y, z]
        :param quaternion: Orientation quaternion [x, y, z, w]
        :param timestamp: Current timestamp
        :return: Estimated [linear_velocities, angular_velocities]
        """
        if self.method == "ekf":
            return self.ekf_update(position, quaternion, timestamp, logger)
        elif self.method == "finite_diff":
            return self.finite_diff_update(position, quaternion, timestamp, logger)
        else:
            raise ValueError(f"Unknown estimation method: {self.method}")
