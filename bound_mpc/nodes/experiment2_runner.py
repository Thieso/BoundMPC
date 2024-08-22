import rclpy
import threading
from rclpy.node import Node
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from bound_mpc.utils import get_default_path, get_default_weights
from bound_mpc_msg.srv import MPCParams, Trajectory
from bound_mpc.utils import create_traj_msg
from bound_mpc.Logger import Logger
from bound_mpc.Plotter import Plotter
from bound_mpc.RobotModel import RobotModel


class ExperimentRunner(Node):
    def __init__(self):
        super().__init__("experiment_runner")

        # Initial joint configuration
        q0 = np.zeros((7, 1))
        q0[1] = 0
        q0[3] = -np.pi / 1.8
        q0[5] = np.pi/2 - np.pi/1.8
        q0 = q0.flatten()
        dq0 = np.zeros(7)

        # Initial Cartesian pose
        robot_model = RobotModel()
        p0fk, _, _ = robot_model.forward_kinematics(q0, dq0)
        p0 = p0fk[:3]
        r0 = R.from_rotvec(p0fk[3:])

        weights = get_default_weights()
        p_via, r_via, p_limits, r_limits, bp1_list, br1_list, s, e_p_min, e_r_min, e_p_max, e_r_max = get_default_path(p0, r0, 5)

        save_data = False
        show_plots = False
        path = "/data"
        tail = "default"
        t0 = 0

        params = MPCParams.Request()
        params.n = 10
        params.dt = 0.1
        params.weights = weights.tolist()
        params.build = True
        params.simulate = False
        params.experiment = False
        params.use_acados = False
        params.learning_based = False
        params.real_time = False
        params.nr_segs = 4

        # Set the reference path using the via points
        r1 = R.from_euler('XYZ', [np.pi/2, 0, 0]) * r0
        r2 = R.from_euler('XYZ', [0, 0, -np.pi/3]) * r1
        r3 = R.from_euler('XYZ', [0, 0, np.pi/2.01]) * R.from_euler('XYZ', [np.pi/2, 0, 0]) * R.from_euler('XYZ', [0, 0, -np.pi/2]) * r1
        r4 = R.from_euler('XYZ', [0, 0, np.pi/2]) * R.from_euler('XYZ', [np.pi/2, 0, 0]) * R.from_euler('XYZ', [0, 0, -np.pi/2]) * r1
        r_via = [
            r0.as_matrix(),
            r1.as_matrix(),
            r2.as_matrix(),
            r3.as_matrix(),
            r4.as_matrix(),
        ]
        p_via = [
            p0,
            p0 + np.array([-0.2, -0.0, 0.1]),
            p0 + np.array([-0.6, -0.6, 0.1]),
            p0 + np.array([-0.8, -0.5, -0.2]),
            p0 + np.array([-0.8, -0.5, -0.5])
        ]
        p_lower = [
            np.array([-1.0, -1.0]),
            np.array([-0.01, -1.0]),
            np.array([-1.0, -1.0]),
            np.array([-0.1, -0.1]),
            np.array([-0.1, -0.1])
        ]
        p_upper = [
            np.array([1.0, 1.0]),
            np.array([0.01, 1.0]),
            np.array([1.0, 1.0]),
            np.array([0.1, 0.1]),
            np.array([0.1, 0.1])
        ]
        p_limits = [p_lower, p_upper]
        r_lower = [
            np.array([-1.0, -1.0]),
            np.array([-0.11, -0.11]),
            np.array([-1.0, -1.0]),
            np.array([-0.1, -0.1]),
            np.array([-0.1, -0.1])
        ]
        r_upper = [
            np.array([1.0, 1.0]),
            np.array([0.11, 0.11]),
            np.array([1.0, 1.0]),
            np.array([0.1, 0.1]),
            np.array([0.1, 0.1])
        ]
        r_limits = [r_lower, r_upper]
        bp1_list = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0])
        ]
        br1_list = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0])
        ]

        # Send the parameters to the MPC
        srv = self.create_client(MPCParams, '/mpc/set_params')
        while not srv.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Parameter service not available, waiting again...')
        future = srv.call_async(params)
        rclpy.spin_until_future_complete(self, future)

        # Create Logger
        logger = Logger(t0, params.n)

        try:
            # Publish Trajectory
            msg = create_traj_msg(p_via, r_via, p_limits, r_limits, bp1_list,
                                  br1_list, s, e_p_min, e_r_min, e_p_max,
                                  e_r_max, p0fk, q0)
            traj_srv = self.create_client(Trajectory, '/mpc/set_trajectory')
            traj_srv.wait_for_service()
            future = traj_srv.call_async(msg)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()

            # Start the logging
            logger.start()

            # Wait for logger to receive first message
            while logger.phi < 1e-6:
                rclpy.spin_once(logger)
                time.sleep(0.01)

            # Wait for the robot to finish
            while logger.phi_max - logger.phi > 0.01:
                # while True:
                rclpy.spin_once(logger)
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt")
            pass

        # Stop the logging thread
        logger.stop()

        # Plot stuff
        plotter = Plotter(logger,
                          params,
                          t0,
                          p_via,
                          r_via,
                          path=path,
                          tail=tail,
                          save_data=save_data,
                          x_values='path')

        if show_plots:
            plt.show()


def main():
    rclpy.init()
    node = ExperimentRunner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
