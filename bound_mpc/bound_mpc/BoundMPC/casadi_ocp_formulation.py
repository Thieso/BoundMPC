import casadi as ca
import numpy as np
from .bound_mpc_functions import (reference_function, error_function,
                                  decomp_function, objective_function,
                                  integration_function,
                                  get_current_and_next_segment)


def setup_optimization_problem(N, nr_joints, nr_segs, dt, u_min, u_max, ut_min,
                               ut_max, q_lim_lower, q_lim_upper, dq_lim_lower,
                               dq_lim_upper, solver_opts):
    """Build the optimization problem using symbolic variables such that it
    can be easily used later on by just inputting the new state.
    """
    path_following = False
    nr_u = nr_joints + 1
    # Initial jerk values
    jerk_current = ca.SX.sym('jerk_current', 1, nr_joints)
    jerk_phi_current = ca.SX.sym('jerk_phi_current', 1, 1)

    # Desired path parameter state
    x_phi_d = ca.SX.sym('x phi_desired', 3)
    phi_max = ca.SX.sym('max path parameter', 1)
    dphi_max = ca.SX.sym('max path parameter', 1)

    # Desired joint config
    qd = ca.SX.sym('q desired', nr_joints)

    # Reference trajectory parameters
    phi_switch = ca.SX.sym('path parameter switch', nr_segs + 1)
    p_ref = ca.SX.sym('linear ref position', nr_segs, 6)
    dp_ref = ca.SX.sym('linear ref velocity', nr_segs, 6)
    dp_normed_ref = ca.SX.sym('norm of orientation reference', nr_segs, 3)
    bp1 = ca.SX.sym('orthogonal error basis 1', nr_segs, 3)
    bp2 = ca.SX.sym('orthogonal error basis 2', nr_segs, 3)
    br1 = ca.SX.sym('orthogonal error basis 1r', nr_segs, 3)
    br2 = ca.SX.sym('orthogonal error basis 2r', nr_segs, 3)
    v1 = ca.SX.sym('v1', nr_segs, 3)
    v2 = ca.SX.sym('v2', nr_segs, 3)
    v3 = ca.SX.sym('v3', nr_segs, 3)

    # Error computation variables
    dtau_init = ca.SX.sym('initial lie space error', 3)
    dtau_init_par = ca.SX.sym('initial lie space error par', 3, nr_segs)
    dtau_init_orth1 = ca.SX.sym('initial lie space error orth1', 3, nr_segs)
    dtau_init_orth2 = ca.SX.sym('initial lie space error orth2', 3, nr_segs)
    jac_dtau_r = ca.SX.sym('right jacobian at initial error', 3, 3)
    jac_dtau_l = ca.SX.sym('left jacobian at initial error', 3, 3)

    # Error bound parametrization
    a4 = ca.SX.sym('parameter 4 error function', nr_segs + 1, 9)
    a3 = ca.SX.sym('parameter 3 error function', nr_segs + 1, 9)
    a2 = ca.SX.sym('parameter 2 error function', nr_segs + 1, 9)
    a1 = ca.SX.sym('parameter 1 error function', nr_segs + 1, 9)
    a0 = ca.SX.sym('parameter 0 error function', nr_segs + 1, 9)

    # Objective function weights
    weights = ca.SX.sym('cost weights', 15)

    # Initialize variables
    w = []
    lbw = []
    ubw = []
    J = 0
    g = []
    g_names = []
    lbg = []
    ubg = []
    uk_prev = ca.vertcat(jerk_current.T, jerk_phi_current.T)

    # initial state
    qk = ca.SX.sym('q_0', nr_joints)
    dqk = ca.SX.sym('dq_0', nr_joints)
    ddqk = ca.SX.sym('ddq_0', nr_joints)
    phik = ca.SX.sym('phi_0', 1)
    dphik = ca.SX.sym('dphi_0', 1)
    ddphik = ca.SX.sym('ddphi_0', 1)
    p0 = ca.SX.sym('p_0', 6)
    pk = p0
    v0 = ca.SX.sym('v_0', 6)
    vprev = v0
    init_state = ca.vertcat(qk, dqk, ddqk, phik, dphik, ddphik)
    i_omega_0 = p0[3:]
    i_omega_ref_0 = ca.SX.sym('i_omega_ref_0', 3)
    phik_prev = phik

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        uk = ca.SX.sym(f'u_{k}', nr_u)
        w += [uk]
        lbw += [u_min] * nr_joints
        ubw += [u_max] * nr_joints
        lbw += [ut_min]
        ubw += [ut_max]

        # Integrate dynamical system
        x_new = integration_function(nr_joints,
                                     nr_u,
                                     dt,
                                     q=qk,
                                     dq=dqk,
                                     ddq=ddqk,
                                     p_rot=pk[3:],
                                     phi=phik,
                                     dphi=dphik,
                                     ddphi=ddphik,
                                     u=uk,
                                     u_prev=uk_prev)
        # uk_prev = u_control
        uk_prev = uk
        q_new = x_new['q_new']
        dq_new = x_new['dq_new']
        ddq_new = x_new['ddq_new']
        phi_new = x_new['phi_new']
        dphi_new = x_new['dphi_new']
        ddphi_new = x_new['ddphi_new']

        # New state
        p_new = x_new['p_new']
        v_new = x_new['v_new']
        qk = ca.SX.sym(f'q_{k+1}', nr_joints)
        dqk = ca.SX.sym(f'dq_{k+1}', nr_joints)
        ddqk = ca.SX.sym(f'ddq_{k+1}', nr_joints)
        phik = ca.SX.sym(f'phi_{k+1}', 1)
        dphik = ca.SX.sym(f'dphi_{k+1}', 1)
        ddphik = ca.SX.sym(f'ddphi_{k+1}', 1)
        pk = ca.SX.sym(f'i_omega_{k+1}', 6)
        vk = ca.SX.sym(f'v_{k+1}', 6)
        w += [qk]
        lbw += q_lim_lower
        ubw += q_lim_upper
        w += [dqk]
        lbw += dq_lim_lower
        ubw += dq_lim_upper
        w += [ddqk]
        lbw += [-np.inf] * nr_joints
        ubw += [np.inf] * nr_joints
        w += [pk]
        lbw += [-np.inf] * 6
        ubw += [np.inf] * 6
        w += [vk]
        lbw += [-np.inf] * 6
        ubw += [np.inf] * 6
        w += [phik]
        lbw += [0.0]
        ubw += [np.inf]
        w += [dphik]
        lbw += [-np.inf]
        ubw += [np.inf]
        w += [ddphik]
        lbw += [-np.inf]
        ubw += [np.inf]

        # Compute reference trajectory
        reference = reference_function(dp_ref=dp_ref,
                                       p_ref=p_ref,
                                       phi_switch=phi_switch,
                                       phi=phik,
                                       phi_prev=phik_prev,
                                       bp1=bp1,
                                       bp2=bp2,
                                       br1=br1,
                                       br2=br2,
                                       v1=v1,
                                       v2=v2,
                                       v3=v3,
                                       dp_normed_ref=dp_normed_ref,
                                       a4=a4,
                                       a3=a3,
                                       a2=a2,
                                       a1=a1,
                                       a0=a0,
                                       nr_segs=nr_segs)
        phik_prev = phik
        p_d = reference['p_d']
        dp_d = reference['dp_d'].T
        bp1_current = reference['bp1_current']
        bp2_current = reference['bp2_current']
        br1_current = reference['br1_current']
        br2_current = reference['br2_current']
        v1_current = reference['v1_current']
        v2_current = reference['v2_current']
        v3_current = reference['v3_current']
        dp_normed_d = reference['dp_normed_d'].T

        # cond1 = ca.if_else(phik_prev < phi_switch[1], phik - phi_switch[1], 0)
        # via_const = ca.if_else(phik > phi_switch[1], cond1, 0)
        # g += [via_const**2]
        # g_names += ["Tangential orientation error"]
        # lbg += [0]
        # ubg += [0.001]

        # Compute errors
        errors = error_function(p=pk,
                                v=vk,
                                p_ref=p_d,
                                dp_ref=dp_d,
                                dp_normed_ref=dp_normed_d,
                                dphi=dphik,
                                i_omega_0=i_omega_0,
                                i_omega_ref_0=i_omega_ref_0,
                                dtau_init=dtau_init,
                                dtau_init_par=dtau_init_par.T,
                                dtau_init_orth1=dtau_init_orth1.T,
                                dtau_init_orth2=dtau_init_orth2.T,
                                br1=br1_current,
                                br2=br2_current,
                                jac_dtau_l=jac_dtau_l,
                                jac_dtau_r=jac_dtau_r,
                                phi=phik,
                                phi_switch=phi_switch,
                                v1=v1_current,
                                v2=v2_current,
                                v3=v3_current,
                                nr_segs=nr_segs)
        e_p_park = errors['e_p_par']
        e_p = errors['e_p']
        de_p = errors['de_p']
        # de_p_park = errors['de_p_par']
        e_r = errors['e_r']
        de_r = errors['de_r']
        e_r_park = errors['e_r_par']
        e_r_orth1k = errors['e_r_orth1']
        e_r_orth2k = errors['e_r_orth2']

        # Numerically differentiated velocity as acceleration
        ak = (vk - vprev) / dt
        # ak = acceleration_ee(qk, dqk, ddqk)
        v_ref = dphik * dp_d
        a_ref = ddphik * dp_d

        # Increment objective value
        # A sigmoid function is used to be able to use the full error at the
        # end of the path which avoid oscillation and increases solver
        # speed.
        sigm = 1 / (1 + ca.exp(-100 * (phik - (phi_max - 0.02))))
        if path_following:
            e_p_obj = e_p
            e_r_obj = e_r
        else:
            e_p_obj = sigm * e_p + (1 - sigm) * e_p_park
            e_r_obj = sigm * e_r + (1 - sigm) * e_r_park
            # e_p_obj = e_p_park
            # e_r_obj = e_r_park
        x_phi = ca.vertcat(phik, dphik, ddphik)
        J = J + objective_function(nr_joints,
                                   nr_u,
                                   x_phi=x_phi,
                                   e_p_par=e_p_obj,
                                   de_p=de_p,
                                   e_r_par=e_r_obj,
                                   de_r=de_r,
                                   x_phi_d=x_phi_d,
                                   v=vk,
                                   v_ref=v_ref,
                                   a=ak,
                                   a_ref=a_ref,
                                   u=uk,
                                   q=qk,
                                   dq=dqk,
                                   ddq=ddqk,
                                   qd=qd,
                                   weights=weights)
        vprev = vk

        # -----------------------------------------------------------------
        # CONSTRAINTS
        # -----------------------------------------------------------------

        # Dynamical system constraint
        g += [q_new - qk]
        g_names += ["Dynamical System"] * nr_joints
        lbg += [0] * nr_joints
        ubg += [0] * nr_joints
        g += [dq_new - dqk]
        g_names += ["Dynamical System"] * nr_joints
        lbg += [0] * nr_joints
        ubg += [0] * nr_joints
        g += [ddq_new - ddqk]
        g_names += ["Dynamical System"] * nr_joints
        lbg += [0] * nr_joints
        ubg += [0] * nr_joints
        g += [p_new - pk]
        g_names += ["Dynamical System"]
        lbg += [0] * 6
        ubg += [0] * 6
        g += [v_new - vk]
        g_names += ["Dynamical System"]
        lbg += [0] * 6
        ubg += [0] * 6
        g += [phi_new - phik]
        g_names += ["Dynamical System"]
        lbg += [0]
        ubg += [0]
        g += [dphi_new - dphik]
        g_names += ["Dynamical System"]
        lbg += [0]
        ubg += [0]
        g += [ddphi_new - ddphik]
        g_names += ["Dynamical System"]
        lbg += [0]
        ubg += [0]

        # Path parameter constraint (Can not be a state constraint because
        # the bound cannot be a symbolic variable in casadi).
        g += [phik - phi_max]
        g_names += ["Path parameter"]
        lbg += [-np.inf]
        ubg += [0]
        g += [dphik - dphi_max]
        g_names += ["Path velocity"]
        lbg += [-np.inf]
        ubg += [0]

        # Bounds for the tangential orientation error
        e_r_proj = ca.dot(dp_normed_d, e_r_park)
        r_par_bound = reference["r_par_bound"]
        g += [e_r_proj**2 - r_par_bound**2]
        g_names += ["Tangential orientation error"]
        lbg += [-np.inf]
        ubg += [0]

        # Orthogonal position error bounds
        decomp = decomp_function(e=e_p,
                                 e_off=reference['e_p_off'],
                                 b1=bp1_current,
                                 b2=bp2_current,
                                 p_lower=reference['bound_lower'][:2],
                                 p_upper=reference['bound_upper'][:2])
        g += [decomp['constraint0']]
        g += [decomp['constraint1']]
        for i in range(2):
            g_names += ["Orthogonal position error"]
            lbg += [-np.inf]
            ubg += [0]

        # Orthogonal orientation error bounds
        e_r_proj1 = ca.dot(br1_current, e_r_orth1k)
        e_r_proj2 = ca.dot(br2_current, e_r_orth2k)
        bound = (reference['bound_upper'][2:] - reference['bound_lower'][2:])/2
        g += [(e_r_proj1 - reference['e_r_off'][0])**2 - bound[0]**2]
        g_names += ["Orthogonal orientation error"]
        lbg += [-np.inf]
        ubg += [0]
        g += [(e_r_proj2 - reference['e_r_off'][1])**2 - bound[1]**2]
        g_names += ["Orthogonal orientation error"]
        lbg += [-np.inf]
        ubg += [0]

        # Terminal cost
        # if k == N - 1:
        #     J = J + weights[14] * ca.sumsqr(ca.dot(e_p, bp1_current) - reference['e_p_off'][0])
        #     J = J + weights[14] * ca.sumsqr(ca.dot(e_p, bp2_current) - reference['e_p_off'][1])
        #     J = J + weights[14] * ca.sumsqr(e_r_proj1 - reference['e_r_off'][0])
        #     J = J + weights[14] * ca.sumsqr(e_r_proj2 - reference['e_r_off'][1])
        #     J = J + weights[14] * ca.sumsqr(e_r_park)


    # Create an NLP solver
    params = ca.vertcat(init_state, p0, v0, i_omega_ref_0, dtau_init,
                        dtau_init_par.reshape((-1, 1)),
                        dtau_init_orth1.reshape((-1, 1)),
                        dtau_init_orth2.reshape((-1, 1)), x_phi_d,
                        jerk_current.T, jerk_phi_current, phi_switch,
                        jac_dtau_r.reshape(
                            (-1, 1)), jac_dtau_l.reshape((-1, 1)),
                        p_ref.reshape((-1, 1)), dp_ref.reshape((-1, 1)),
                        dp_normed_ref.reshape((-1, 1)), bp1.reshape((-1, 1)),
                        bp2.reshape((-1, 1)), br1.reshape((-1, 1)),
                        br2.reshape((-1, 1)), a4.reshape((-1, 1)),
                        a3.reshape((-1, 1)), a2.reshape((-1, 1)),
                        a1.reshape((-1, 1)), a0.reshape(
                            (-1, 1)), weights, phi_max, dphi_max,
                        v1.reshape((-1, 1)), v2.reshape((-1, 1)),
                        v3.reshape((-1, 1)), qd)

    prob = {
        'f': J,
        'x': ca.vertcat(*w),
        'g': ca.vertcat(*g),
        'p': params
    }
    lbu = lbw
    ubu = ubw
    lbg = lbg
    ubg = ubg
    g_names = g_names
    solver = ca.nlpsol('solver', 'ipopt', prob, solver_opts)

    return solver, lbu, ubu, lbg, ubg, g_names
