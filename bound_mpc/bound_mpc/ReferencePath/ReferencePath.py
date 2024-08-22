import numpy as np
from scipy.spatial.transform import Rotation as R


class ReferencePath():
    """
    Reference Path class
    """

    def __init__(self,
                 p,
                 r,
                 p_limit,
                 r_limit,
                 bp1,
                 br1,
                 s,
                 e_p_min,
                 e_r_min,
                 e_p_max,
                 e_r_max,
                 nr_segs=2,
                 phi_bias=0):
        # Position and orientation points
        self.p = p
        self.r = r
        l_traj = len(self.p)

        # Number of linear segments
        self.nr_segs = nr_segs

        # Init a bias path parameter in case of replanning
        self.phi_bias = phi_bias

        # Switched flag indicates whether the next path element is used
        self.switched = True

        # Bound parameters
        self.s = s
        self.e_p_min = e_p_min
        self.e_r_min = e_r_min
        self.e_p_max = e_p_max
        self.e_r_max = e_r_max
        for i in range(nr_segs-1):
            self.s.append(self.s[-1])
            self.e_p_min.append(self.e_p_min[-1])
            self.e_r_min.append(self.e_r_min[-1])
            self.e_p_max.append(self.e_p_max[-1])
            self.e_r_max.append(self.e_r_max[-1])

        # Position and orientation limits
        self.p_lower = p_limit[0]
        self.p_upper = p_limit[1]
        self.r_lower = r_limit[0]
        self.r_upper = r_limit[1]
        for i in range(nr_segs-1):
            self.p_lower.append(self.p_lower[-1])
            self.p_upper.append(self.p_upper[-1])
            self.r_lower.append(self.r_lower[-1])
            self.r_upper.append(self.r_upper[-1])

        # Which sector of the path is currently used
        self.sector = 0

        self.dr = []
        self.iw = [np.zeros(3)]
        self.dr_limit = []
        for i in range(1, l_traj):
            drot = R.from_matrix(self.r[i] @ self.r[i-1].T).as_rotvec()
            self.dr.append(drot)
            self.iw.append(self.iw[i-1] + self.dr[i-1])
        for i in range(nr_segs-1):
            self.dr.append(np.array([1, 1, 1]))
            self.iw.append(self.iw[-1])
            self.r.append(self.r[-1])

        self.dp = []
        for i in range(1, l_traj):
            self.dp.append(self.p[i] - self.p[i-1])
            if np.linalg.norm(self.dp[-1]) < 1e-3:
                if i > 1:
                    self.dp[-1] = self.dp[-2]
                else:
                    self.dp[-1] = np.array([0, 1, 0])
        for i in range(nr_segs-1):
            self.p.append(self.p[-1])
            self.dp.append(self.dp[-1])

        # Compute the switching points and the arc length based on the length
        self.phi = [0]
        l = []
        l_total = 0
        for i in range(1, l_traj):
            li = np.linalg.norm(self.p[i] - self.p[i-1])
            # If there is no change in position than there is most likely change
            # in orientation and then we need some path parameter there
            if np.linalg.norm(li) < 1e-3:
                li = np.linalg.norm(self.dr[i-1]) / np.pi
            l.append(li)
            l_total += li
        for i in range(l_traj-1):
            self.phi.append(l[i])
        for i in range(nr_segs-1):
            self.phi.append(1)
        self.phi_max = l_total + self.phi_bias

        # Scale the angular velocity to match it to the desired phi values
        for i in range(l_traj):
            self.dr[i] = self.dr[i] / self.phi[i+1]

        # Basis vectors for orthogonal plane
        self.bp1 = bp1
        self.br1 = br1
        self.bp2 = []
        self.br2 = []
        for i in range(len(self.bp1)):
            dp_normed = self.dp[i] / np.linalg.norm(self.dp[i])
            self.bp1[i] = self.gram_schmidt_u(dp_normed, self.bp1[i], np.eye(3))

            orth_check = self.bp1[i] @ self.dp[i]
            if np.abs(orth_check) > 1e-6:
                print(f"[WARNING] Pos Basis vector {i} not orthogonal on path")
            if np.linalg.norm(self.bp1[i]) < 1e-3:
                print(f"[WARNING] Pos Basis vector {i} is too close to direction")

            self.bp1[i] = self.bp1[i] / np.linalg.norm(self.bp1[i])
            self.bp2.append(np.cross(dp_normed, self.bp1[i]))

        for i in range(len(self.bp1)):
            norm_dr = np.linalg.norm(self.dr[i])
            if norm_dr > 1e-4:
                omega = self.dr[i] / norm_dr
            else:
                omega = np.array([0, 1.0, 0])
            self.br1[i] = self.gram_schmidt_u(omega, self.br1[i], np.eye(3))

            orth_check = self.br1[i] @ self.dr[i]
            if np.abs(orth_check) > 1e-6:
                print(f"[WARNING] Rot Basis vector {i} not orthogonal on path")
            if np.linalg.norm(self.br1[i]) < 1e-3:
                print(f"[WARNING] Rot Basis vector {i} is too close to direction")

            self.br1[i] = self.br1[i] / np.linalg.norm(self.br1[i])
            self.br2.append(np.cross(omega, self.br1[i]))

        for i in range(nr_segs-1):
            self.bp1.append(self.bp1[-1])
            self.br1.append(self.br1[-1])
            self.bp2.append(self.bp2[-1])
            self.br2.append(self.br2[-1])

        # Initialize the reference parametrization
        self.pd = np.zeros((6, self.nr_segs))
        self.dpd = np.zeros((6, self.nr_segs))
        self.dpd_normed = np.zeros((3, self.nr_segs))
        self.ddpd = np.zeros((6, self.nr_segs))
        self.asymm_lower = np.zeros((4, self.nr_segs))
        self.asymm_upper = np.zeros((4, self.nr_segs))
        self.phi_switch = np.ones((self.nr_segs+1,)) * self.phi_bias

        for i in range(self.nr_segs):
            self.set_point(i)
        self.compute_normed_velocity()

    def find_largest_proj(self, b_prev, b, p1, p2):
        sign = np.sign(np.dot(b_prev, b))
        max_proj = np.max(np.abs((np.dot(p1, b), np.dot(p2, b))))
        return sign * max_proj

    def compute_normed_velocity(self):
        for i in range(self.nr_segs):
            norm = np.linalg.norm(self.dpd[3:, i])
            if norm > 1e-4:
                self.dpd_normed[:, i] = self.dpd[3:, i] / np.linalg.norm(self.dpd[3:, i])
            else:
                self.dpd_normed[:, i] = np.array([0, 1.0, 0])

    def set_point(self, idx):
        self.pd[:3, idx] = self.p[self.sector+idx]
        self.pd[3:, idx] = self.iw[self.sector+idx]
        # self.dpd[:3, idx] = self.dp[self.sector+idx] / self.phi[self.sector+idx+1]
        self.dpd[:3, idx] = self.dp[self.sector+idx] / np.linalg.norm(self.dp[self.sector+idx])
        self.dpd[3:, idx] = self.dr[self.sector+idx]
        self.asymm_lower[:2, idx] = self.p_lower[self.sector+idx]
        self.asymm_lower[2:, idx] = self.r_lower[self.sector+idx]
        self.asymm_upper[:2, idx] = self.p_upper[self.sector+idx]
        self.asymm_upper[2:, idx] = self.r_upper[self.sector+idx]
        self.phi_switch[idx+1] = np.array(self.phi).cumsum()[self.sector+idx+1] + self.phi_bias

    def update(self, phi_current):
        # If the current phi is larger than the switching phi, update the
        # parameters
        if phi_current <= self.phi_switch[1]:
            self.switched = False
        while phi_current > self.phi_switch[1]:
            self.switched = True
            self.sector += 1

            # Shift the segements to the front
            for i in range(self.nr_segs-1):
                self.pd[:, i] = np.copy(self.pd[:, i+1])
                self.dpd[:, i] = np.copy(self.dpd[:, i+1])
                self.asymm_lower[:, i] = np.copy(self.asymm_lower[:, i+1])
                self.asymm_upper[:, i] = np.copy(self.asymm_upper[:, i+1])
                self.phi_switch[i] = np.copy(self.phi_switch[i+1])
            self.phi_switch[self.nr_segs-1] = np.copy(self.phi_switch[self.nr_segs]) + self.phi_bias

            # Set new segment
            self.set_point(self.nr_segs - 1)

            # Compute the normed velocity
            self.compute_normed_velocity()

    def compute_phis(self):
        self.phi_switch = np.array(self.phi).cumsum()[self.sector]
        if self.sector+2 < len(self.phi):
            self.phi_switch_next = np.array([sum(self.phi[:self.sector+2])])
        else:
            self.phi_switch_next = np.array([self.phi_max])

    def get_parameters(self, phi_current):
        self.update(phi_current)
        return self.pd, self.dpd_normed, self.dpd, self.ddpd, self.phi_switch

    def get_limits(self):
        bp1 = np.array(self.bp1[self.sector:self.sector+self.nr_segs]).T
        bp2 = np.array(self.bp2[self.sector:self.sector+self.nr_segs]).T
        br1 = np.array(self.br1[self.sector:self.sector+self.nr_segs]).T
        br2 = np.array(self.br2[self.sector:self.sector+self.nr_segs]).T
        return self.asymm_lower, self.asymm_upper, bp1, bp2, br1, br2

    def get_bound_params(self):
        s = np.array(self.s[self.sector:self.sector+self.nr_segs])
        e_p_min = np.array(self.e_p_min[self.sector:self.sector+self.nr_segs])
        e_r_min = np.array(self.e_r_min[self.sector:self.sector+self.nr_segs])
        e_p_max = np.array(self.e_p_max[self.sector:self.sector+self.nr_segs])
        e_r_max = np.array(self.e_r_max[self.sector:self.sector+self.nr_segs])
        return e_p_min, e_r_min, e_p_max, e_r_max, s

    def get_next_limits(self):
        pll = np.zeros(4)
        plu = np.zeros(4)
        idx = np.min((self.sector+self.nr_segs, len(self.p_lower)-1))
        pll[:2] = self.p_lower[idx]
        pll[2:] = self.r_lower[idx]
        plu[:2] = self.p_upper[idx]
        plu[2:] = self.r_upper[idx]
        return pll, plu

    def gram_schmidt_u(self, v, b, jac):
        """ Do one Gram Schmidt orthogonolization step. """
        return b - self.gram_schmidt_w(v, b, jac)

    def gram_schmidt_w(self, v, b, jac):
        """ Project the vector b onto v. The jacobian matrix is used for
        orientation projections """
        return jac @ ((v.T @ b) * v)
