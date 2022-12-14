import numpy as np
from scipy.integrate import odeint

from src.r0 import R0Generator


class RostHungaryModel:
    def __init__(self, init_values, contact_matrix, parameters, to_solve=True):
        self.s_0 = init_values["s"]
        self.l1_0 = init_values["l1"]
        self.l2_0 = init_values["l2"]
        self.ip_0 = init_values["ip"]
        self.ia1_0 = init_values["ia1"]
        self.ia2_0 = init_values["ia2"]
        self.ia3_0 = init_values["ia3"]
        self.is1_0 = init_values["is1"]
        self.is2_0 = init_values["is2"]
        self.is3_0 = init_values["is3"]
        self.ih_0 = init_values["ih"]
        self.ic_0 = init_values["ic"]
        self.icr_0 = init_values["icr"]
        self.r_0 = init_values["r"]
        self.d_0 = init_values["d"]
        self.c_0 = init_values["c"]

        self.contact_matrix = contact_matrix
        self.parameters = parameters
        self.r0 = self.get_r0_value()
        self.time_max = 400
        self.time_vector = np.linspace(0, self.time_max, self.time_max)

        self.solution = None
        if to_solve:
            self.solution = odeint(self._get_model, self._get_initial_values(), self.time_vector)

    def _get_model(self, xs, _):
        # Set parameters
        theta = np.array(self.parameters["p"])
        beta = self.parameters["beta"]
        asymp = self.parameters["inf_a"]
        zeta = np.array(self.parameters["xi"])
        h = np.array(self.parameters["h"])
        eta = np.array(self.parameters["mu"])
        susc = np.array(self.parameters["susc"])
        phi_l = self.parameters["alpha_l"]
        phi_p = self.parameters["alpha_p"]
        delta_a = self.parameters["gamma_a"]
        delta_s = self.parameters["gamma_s"]
        delta_h = self.parameters["gamma_h"]
        delta_c = self.parameters["gamma_c"]
        delta_cr = self.parameters["gamma_cr"]

        # Set initial conditions
        s, l1, l2, ip, ia1, ia2, ia3, is1, is2, is3, ih, ic, icr, r, d, c = xs.reshape(-1, 6)
        # [[16 elem ], [], [], ...]

        pop_per_age = self.s_0

        model_eq = [
            -susc * (s / pop_per_age) * beta * np.array((ip + asymp * (ia1 + ia2 + ia3) + (is1 + is2 + is3))).dot(
                self.contact_matrix),  # S'(t)
            susc * (s / pop_per_age) * beta * np.array((ip + asymp * (ia1 + ia2 + ia3) + (is1 + is2 + is3))).dot(
                self.contact_matrix) - 2 * phi_l * l1,  # L1'(t)
            2 * phi_l * l1 - 2 * phi_l * l2,  # L2'(t)
            2 * phi_l * l2 - phi_p * ip,  # Ip'(t)
            theta * phi_p * ip - 3 * delta_a * ia1,  # Ia1'(t)
            3 * delta_a * ia1 - 3 * delta_a * ia2,  # Ia2'(t)
            3 * delta_a * ia2 - 3 * delta_a * ia3,  # Ia3'(t)
            (1 - theta) * phi_p * ip - 3 * delta_s * is1,  # Is1'(t)
            3 * delta_s * is1 - 3 * delta_s * is2,  # Is2'(t)
            3 * delta_s * is2 - 3 * delta_s * is3,  # Is3'(t)
            h * (1 - zeta) * 3 * delta_s * is3 - delta_h * ih,  # Ih'(t)
            h * zeta * 3 * delta_s * is3 - delta_c * ic,  # Ic'(t)
            (1 - eta) * delta_c * ic - delta_cr * icr,  # Icr'(t)
            3 * delta_a * ia3 + (1 - h) * 3 * delta_s * is3 + delta_h * ih + delta_cr * icr,  # R'(t)
            eta * delta_c * ic,  # D'(t)
            susc * (s / pop_per_age) * beta * np.array((ip + asymp * (ia1 + ia2 + ia3) + (is1 + is2 + is3))).dot(
                self.contact_matrix)  # C'(t)
        ]
        v = np.array(model_eq).flatten()
        return v

    def _get_initial_values(self):
        init_values = [self.s_0, self.l1_0, self.l2_0, self.ip_0, self.ia1_0, self.ia2_0, self.ia3_0, self.is1_0,
                       self.is2_0, self.is3_0, self.ih_0, self.ic_0, self.icr_0, self.r_0, self.d_0, self.c_0]
        return np.array(init_values).flatten()

    def get_r0_value(self):
        r0generator = R0Generator(param=self.parameters)
        return r0generator.get_eig_val(self.contact_matrix) * self.parameters["beta"]
