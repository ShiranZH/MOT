import numpy as np

class KalmanFilter(object):
    def __init__(self, dt, ux, uy, std_acc, std_meax, std_meay):
        # Define sampling time
        self.dt = dt

        # Define the control input variables
        self.u = np.matrix([[ux], [uy]])

        # Initial State
        self.x = np.matrix([[0], [0], [0], [0]])

        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt ** 2) / 2, 0],
                            [0, (self.dt ** 2) / 2],
                            [self.dt, 0],
                            [0, self.dt]])

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0],
                            [0, (self.dt ** 4) / 4, 0, (self.dt ** 3) / 2],
                            [(self.dt ** 3) / 2, 0, self.dt ** 2, 0],
                            [0, (self.dt ** 3) / 2, 0, self.dt ** 2]]) * std_acc ** 2

        # Initial Measurement Noise Covariance
        self.R = np.matrix([[std_meax ** 2, 0],
                            [0, std_meay ** 2]])

        # Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    def getxy(self):
        return self.x[0:2]

    def predict(self):
        # Update time state
        # x_k =A*x_(k-1) + B*u_(k-1)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        # P = A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.x[0:2]


    def update(self, z):
        # S = H*P*H' + R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H' * inv(H*P*H' + R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))

        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P
        return self.x[0:2]


