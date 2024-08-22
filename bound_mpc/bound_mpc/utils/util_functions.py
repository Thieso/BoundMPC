import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from bound_mpc_msg.msg import Vector
from bound_mpc_msg.srv._trajectory import Trajectory_Request
from .lie_functions import rodrigues_matrix
from ..BoundMPC.jerk_trajectory_casadi import (calcAngle, calcVelocity,
                                               calcAcceleration)


def compute_initial_rot_errors(pr, pr_ref, dp_ref, br1, br2):
    tauc = R.from_rotvec(pr).as_matrix()
    taud = R.from_rotvec(pr_ref).as_matrix()
    dtau_init = R.from_matrix(tauc @ taud.T).as_rotvec()

    norm_ref = np.linalg.norm(dp_ref)
    if norm_ref > 1e-4:
        dp_normed = dp_ref / norm_ref
    else:
        dp_normed = np.array([0, 1.0, 0])
    r01 = np.zeros((3, 3))
    r01[:, 0] = br2
    r01[:, 1] = dp_normed
    r01[:, 2] = br1
    dtau_01 = r01.T @ R.from_rotvec(dtau_init).as_matrix() @ r01
    eul = R.from_matrix(dtau_01).as_euler('zyx')
    dtau_init_orth2 = eul[2] * br2
    dtau_init_orth1 = eul[0] * br1
    dtau_init_par = eul[1] * dp_normed

    return [dtau_init, dtau_init_par, dtau_init_orth1, dtau_init_orth2]


def create_traj_msg(p_via,
                    r_via,
                    p_limits,
                    r_limits,
                    bp1_list,
                    br1_list,
                    s,
                    e_p_min,
                    e_r_min,
                    e_p_max,
                    e_r_max,
                    p0fk,
                    q0,
                    update=False):
    msg = Trajectory_Request()
    msg.update = update
    for i in range(len(p_via)):
        vec = Vector()
        vec.x = p_via[i].tolist()
        msg.p_via.append(vec)
        vec = Vector()
        vec.x = R.from_matrix(r_via[i]).as_rotvec().tolist()
        msg.r_via.append(vec)

        vec = Vector()
        vec.x = p_limits[0][i].tolist()
        msg.p_upper.append(vec)
        vec = Vector()
        vec.x = p_limits[1][i].tolist()
        msg.p_lower.append(vec)

        vec = Vector()
        vec.x = r_limits[0][i].tolist()
        msg.r_upper.append(vec)
        vec = Vector()
        vec.x = r_limits[1][i].tolist()
        msg.r_lower.append(vec)

        vec = Vector()
        vec.x = bp1_list[i].tolist()
        msg.bp1.append(vec)
        vec = Vector()
        vec.x = br1_list[i].tolist()
        msg.br1.append(vec)
    msg.s.x = s
    msg.e_p_min.x = e_p_min
    msg.e_r_min.x = e_r_min
    msg.e_p_max.x = e_p_max
    msg.e_r_max.x = e_r_max
    msg.p0.x = p0fk.tolist()
    msg.q0.x = q0.tolist()
    return msg


def integrate_rotation_reference(pr_ref, omega, phi0, phi1):
    """ Integrate the rotation reference by using the constant angular velocity
    omega over the interval phi1 - phi0.
    """
    r0 = R.from_rotvec(pr_ref).as_matrix()
    omega_norm = np.linalg.norm(omega)
    if omega_norm > 1e-4:
        dr = rodrigues_matrix(omega/omega_norm, (phi1 - phi0)*omega_norm)
        r1 = dr @ r0
    else:
        r1 = r0
    return R.from_matrix(r1).as_rotvec()


def project_position_bounds(corners_3d, p_ref, dp_ref, pidx,
                            exact_points=True):
    """ Project the positions bounds, given as corners of a rectangle, to the
    principle axis planes.
    """
    p = np.zeros(4)
    p_l = np.empty((corners_3d.shape[0], 2))
    p_u = np.empty((corners_3d.shape[0], 2))
    for i in range(corners_3d.shape[0]):
        # Get velocity orthogonal to the path in the plane
        vi = dp_ref[i, pidx] / np.linalg.norm(dp_ref[i, pidx])
        v_proj = np.array([vi[1], -vi[0]])
        # Project the corners onto the plane and onto the orthogonal velocity
        p[0] = np.dot(corners_3d[i, :3][pidx], v_proj)
        p[1] = np.dot(corners_3d[i, 3:6][pidx], v_proj)
        p[2] = np.dot(corners_3d[i, 6:9][pidx], v_proj)
        p[3] = np.dot(corners_3d[i, 9:12][pidx], v_proj)
        # Find the projected points by choosing the corner that is the most far
        # away or the highest projection on the orthogonal velocity vector
        if exact_points:
            idx = np.argmin(p)
            p_l[i, :] = corners_3d[i, 3*idx:3*idx+3][pidx]
            idx = np.argmax(p)
            p_u[i, :] = corners_3d[i, 3*idx:3*idx+3][pidx]
        else:
            p_l[i, :] = np.min(p) * v_proj
            p_u[i, :] = np.max(p) * v_proj
    # Add the reference
    p_l += p_ref[:, pidx]
    p_u += p_ref[:, pidx]
    return p_l, p_u


def move_robot_kinematic(robot_pub, t_ros, q_new):
    """ Move the robot kinematically by just publishing the new joint state
    for Rviz (Only for visualization purposes).
    """
    robot_state_msg = JointState()
    robot_link_names = [
        "iiwa_joint_1", "iiwa_joint_2", "iiwa_joint_3", "iiwa_joint_4",
        "iiwa_joint_5", "iiwa_joint_6", "iiwa_joint_7"
    ]
    robot_state_msg.name = robot_link_names
    robot_state_msg.header.frame_id = "r1/world"
    robot_state_msg.header.stamp = t_ros
    robot_state_msg.position = q_new.tolist()
    robot_state_msg.velocity = np.zeros((7, )).tolist()
    robot_pub.publish(robot_state_msg)


def integrate_joint(model, jerk_matrix, q, dq, ddq, dt):
    qn = calcAngle(jerk_matrix, dt, q, dq, ddq, dt)
    dqn = calcVelocity(jerk_matrix, dt, dq, ddq, dt)
    ddqn = calcAcceleration(jerk_matrix, dt, ddq, dt)
    pn_lie, jac_fk, djac_fk = model.forward_kinematics(qn, dqn)
    ddjac_fk = model.ddjacobian_fk(q, dq, ddq)
    vn = jac_fk @ dqn
    an = djac_fk @ dqn + jac_fk @ ddqn
    jn = ddjac_fk @ dqn + 2 * djac_fk @ ddqn + jac_fk @ ddqn
    return (qn, dqn, ddqn, pn_lie, vn, an, jn)
