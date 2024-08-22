import numpy as np


def get_default_path(p0, r0, nr_segs=2):
    """
    Get necessary specifications for a default path.
    """
    # Position and orienation via points
    p_via = [p0] * nr_segs
    r_via = [r0.as_matrix()] * nr_segs

    # Position bounds
    p_lower = [np.array([-1.0, -1.0])] * nr_segs
    p_upper = [np.array([1.0, 1.0])] * nr_segs
    p_limits = [p_lower, p_upper]

    # Orientation bounds
    r_lower = [np.array([-1.0, -1.0])] * nr_segs
    r_upper = [np.array([1.0, 1.0])] * nr_segs
    r_limits = [r_lower, r_upper]

    # Desired first basis vector
    bp1_list = [np.array([0.0, 0.0, 1.0])] * nr_segs
    br1_list = [np.array([0.0, 0.0, 1.0])] * nr_segs

    # Slope
    s = [0.0] * nr_segs

    # Minimum position error
    e_p_min = [0.01] * nr_segs

    # Minimum orientation error
    e_r_min = [15 * np.pi / 180] * nr_segs

    # Maximum error bounds
    e_p_max = [0.20] * nr_segs
    e_r_max = [45 * np.pi / 180] * nr_segs

    return p_via, r_via, p_limits, r_limits, bp1_list, br1_list, s, e_p_min, e_r_min, e_p_max, e_r_max


def get_default_weights():
    w_p = 1000.0
    w_r = 1.0
    dw_p = 0.1
    dw_r = 0.1
    w_speed = 0.5
    w_phi = 8
    w_dphi = 5
    w_ddphi = 4
    w_dddphi = 0.5
    scal = 8 / w_phi
    w_phi *= scal
    w_dphi *= scal
    w_ddphi *= scal
    w_dddphi *= scal
    w_a = 0.05
    w_q = 0.01
    w_dq = 0.01
    w_ddq = 0.001
    w_jerk = 0.0001
    w_term = 10

    weights = np.array([
        w_p, w_r, dw_p, dw_r, w_speed, w_a, w_phi, w_dphi, w_ddphi,
        w_dddphi, w_q, w_dq, w_ddq, w_jerk, w_term
    ])
    return weights
