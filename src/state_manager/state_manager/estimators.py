import numpy as np
import threading

from utils.math import quaternion_to_euler

class VelocityEstimator:
    def __init__(self, alpha=0.1, position_noise=0.01, velocity_noise=0.1):
        """
        Initialize Extended Kalman Filter for velocity estimation.
        
        :param alpha: Smoothing factor for velocity estimation
        :param position_noise: Process noise for position
        :param velocity_noise: Process noise for velocity
        """
        # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        self.state = np.zeros(12)
        self.covariance = np.eye(12) * 1000  # Large initial uncertainty
        
        # Process noise covariance
        self.Q = np.diag([
            position_noise, position_noise, position_noise,  # position
            velocity_noise, velocity_noise, velocity_noise,  # linear velocity
            0.1, 0.1, 0.1,  # orientation
            0.1, 0.1, 0.1   # angular velocity
        ])
        
        # Measurement noise covariance
        self.R = np.diag([
            position_noise, position_noise, position_noise,  # position
            0.1, 0.1, 0.1   # orientation
        ])
        
        self.last_timestamp = None
        self._lock = threading.Lock()
    
    def ekf_update(self, position, quaternion, timestamp):
        """
        Perform Extended Kalman Filter update step.
        
        :param position: Current 3D position [x, y, z]
        :param quaternion: Orientation quaternion [x, y, z, w]
        :param timestamp: Current timestamp
        :return: Estimated [linear_velocities, angular_velocities]
        """
        with self._lock:
            # Calculate time delta
            if self.last_timestamp is None:
                self.last_timestamp = timestamp
                self.state[:3] = position
                self.state[6:9] = quaternion_to_euler(quaternion)
                return np.zeros(6)
            
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
            current_euler = quaternion_to_euler(quaternion)
            
            # Innovation
            innovation = np.concatenate([
                position - predicted_state[:3],
                current_euler - predicted_state[6:9]
            ])
            
            # Innovation covariance
            S = H @ predicted_covariance @ H.T + self.R
            
            # Kalman gain
            K = predicted_covariance @ H.T @ np.linalg.inv(S)
            
            # Update state and covariance
            self.state = predicted_state + K @ innovation
            self.covariance = (np.eye(12) - K @ H) @ predicted_covariance
            
            # Update timestamp
            self.last_timestamp = timestamp
            
            return self.state[3:6], self.state[9:]