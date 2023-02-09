import numpy as np

from src.dataloader import DataLoader
from src.simulation import Simulation


class Contacts:
    def __init__(self, susc: float = 1.0, base_r0: float = 3.68):
        self.data = DataLoader()
        self.country_names = list(self.data.age_data.keys())
        self.upper_tri_indexes = np.triu_indices(6)

        self.setting_contacts = dict()
        self.contacts = np.array([])
        self.susc = susc
        self.base_r0 = base_r0
        self.age_group = np.array([])
        self.contact_matrix = dict()
        self.data_cm_d2pca_col = []
        self.data_cm_d2pca_r = []

        self.data_clustering = []
        self.data_cm_d2pca_column = []
        self.data_cm_d2pca_row = []
        self.data_cm_pca = []

        self.indicator_data = []
        self.data_cm_1dpca = []

        self.get_contacts()

    def get_contacts(self):
        age = [(0, 0), (1, 2), (3, 3), (4, 4), (5, 12), (13, 15)]

        age_vector = self.data.age_data["Kenya"]["age"].flatten()

        # procedure to aggregate params to the desired 6 * 6 age group from 16 * 16
        p, x, m, h = (np.zeros(len(age)), np.zeros(len(age)), np.zeros(len(age)), np.zeros(len(age)))
        for i in range(len(age)):
            ps = self.data.model_parameters_data
            age_i = age_vector[age[i][0]]
            p[i] = np.sum(age_i * ps['p'][age[i][0]]) / np.sum(age_i)
            x[i] = np.sum(age_i * ps['xi'][age[i][0]]) / np.sum(age_i)
            m[i] = np.sum(age_i * ps['mu'][age[i][0]]) / np.sum(age_i)
            h[i] = np.sum(age_i * ps['h'][age[i][0]]) / np.sum(age_i)
        self.data.model_parameters_data.update({"p": p, "mu": m, "xi": x, "h": h})

        susceptibility = np.array([1.0] * 6)
        susceptibility[:3] = self.susc

        for country in self.country_names:
            # contact for all settings
            all_contact = self.data.contact_data[country]["HOME"] + self.data.contact_data[country]["SCHOOL"] + \
                      self.data.contact_data[country]["WORK"] + self.data.contact_data[country]["OTHER"]
            # contact for each setting and all settings
            h_contact, s_contact, w_contact, o_contact, f_contact = [self.data.contact_data[country]["HOME"],
                                                                     self.data.contact_data[country]["SCHOOL"],
                                                                     self.data.contact_data[country]["WORK"],
                                                                     self.data.contact_data[country]["OTHER"],
                                                                     all_contact
                                                                     ]

            age_vector = self.data.age_data[country]["age"].reshape((-1, 1))
            # create the age group
            age_group = np.array([np.sum(age_vector[age[i][0]:(age[i][1] + 1)]) for i in range(len(age))])
            self.data.age_data[country]["age"] = age_group
            self.age_group = np.array([age_group]).reshape((-1, 1))

            # aggregation method to transform age contact matrix for each setting from 16 * 16 to 6 * 6
            # entries for age groups in 6 * 6 that correspond to that in 16 * 16 remain unchanged

            h, s, w, o, a = (np.zeros(shape=(6, 6)), np.zeros(shape=(6, 6)), np.zeros(shape=(6, 6)),
                             np.zeros(shape=(6, 6)), np.zeros(shape=(6, 6)))
            home_contact, school_contact, work_contact, other_contact, all_contact, = [h_contact, s_contact,
                                                                                       w_contact, o_contact,
                                                                                       f_contact] * age_vector
            for i in range(len(age)):
                for j in range(len(age)):
                    h[i, j], s[i, j], w[i, j], o[i, j], a[i, j] = [np.sum(home_contact[age[i][0]:(age[i][1] + 1),
                                                                          age[j][0]:(age[j][1] + 1)]),
                                                                   np.sum(school_contact[age[i][0]:(age[i][1] + 1),
                                                                          age[j][0]:(age[j][1] + 1)]),
                                                                   np.sum(work_contact[age[i][0]:(age[i][1] + 1),
                                                                          age[j][0]:(age[j][1] + 1)]),
                                                                   np.sum(other_contact[age[i][0]:(age[i][1] + 1),
                                                                          age[j][0]:(age[j][1] + 1)]),
                                                                   np.sum(all_contact[age[i][0]:(age[i][1] + 1),
                                                                          age[j][0]:(age[j][1] + 1)])]
            home, school, work, other, full = [h, s, w, o, a] / age_group

            simulation = Simulation(data=self.data, base_r0=self.base_r0, contact_matrix=full,
                                    age_vector=self.age_group, susceptibility=susceptibility,
                                    country=country)
            # Create dictionary with the needed data
            self.setting_contacts.update(
                {country: {"beta": simulation.beta,
                           "age_vector": age_group,
                           "contact_full": full,
                           "contact_home": home,
                           "contact_school": school,
                           "contact_work": work,
                           "contact_other": other}
                 })
            # Create separated data structure for (2D)^2 PCA
            self.data_cm_d2pca_col.append(simulation.beta * full)
            self.data_cm_d2pca_r.append(simulation.beta * full)

            # create data for the indicators
            self.indicator_data = self.data.indicators_data

            # Create separated data structure for 1D PCA
            self.data_cm_1dpca.append(simulation.beta * full[self.upper_tri_indexes])
            self.data_clustering = np.array(self.data_cm_1dpca)

            # Final shape of the np.nd-arrays: (192, 6)
            self.data_cm_d2pca_column = np.vstack(self.data_cm_d2pca_col)
            self.data_cm_d2pca_row = np.vstack(self.data_cm_d2pca_r)
