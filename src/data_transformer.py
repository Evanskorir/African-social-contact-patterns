import numpy as np

from src.dataloader import DataLoader
from src.simulation import Simulation


class Contacts:
    def __init__(self, susc: float = 1.0, base_r0: float = 3.68):
        self.data = DataLoader()
        self.country_names = list(self.data.age_data.keys())
        self.upper_tri_indexes = np.triu_indices(6)

        self.full_contacts = dict()
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

        # params aggregation
        p, x, m, h = [np.zeros(len(age)), np.zeros(len(age)), np.zeros(len(age)), np.zeros(len(age))]

        for i in range(len(age)):
            age_vector = [self.data.age_data[country]["age"].reshape((-1, 1)) for country in self.country_names]

            p[i] = np.sum(age_vector[(age[i][0])] * self.data.model_parameters_data['p'][(age[i][0])]) / \
                   np.sum(age_vector[(age[i][0])])
            x[i] = np.sum(age_vector[(age[i][0])] * self.data.model_parameters_data['xi'][(age[i][0])]) / \
                   np.sum(age_vector[(age[i][0])])
            m[i] = np.sum(age_vector[age[i][0]] * self.data.model_parameters_data['mu'][age[i][0] + 1]) / \
                   np.sum(age_vector[age[i][0]])
            h[i] = np.sum(age_vector[age[i][0]] * self.data.model_parameters_data['h'][age[i][0] + 1]) / \
                   np.sum(age_vector[age[i][0]])
        self.data.model_parameters_data.update(
            {"p": p, "mu": m, "xi": x, "h": h
             })

        for country in self.country_names:
            contact = self.data.contact_data[country]["HOME"] + self.data.contact_data[country]["SCHOOL"] + \
                      self.data.contact_data[country]["WORK"] + self.data.contact_data[country]["OTHER"]

            age_vector = self.data.age_data[country]["age"].reshape((-1, 1))
            # create the age group
            age_group = np.array([age_vector[0][0], np.sum(age_vector[1:3]), age_vector[3][0],
                                  age_vector[4][0], np.sum(age_vector[5:13]), np.sum(age_vector[13:16])])
            self.data.age_data[country]["age"] = age_group
            self.age_group = np.array([age_group]).reshape((-1, 1))

            t_contact = contact * age_vector
            c = np.zeros(shape=(6, 6))
            for i in range(len(age)):
                for j in range(len(age)):
                    c[i, j] = np.sum(t_contact[age[i][0]:(age[i][1] + 1), age[j][0]:(age[j][1] + 1)])
            matrix = c / self.age_group

            susceptibility = np.array([1.0] * 6)
            susceptibility[:3] = self.susc
            #
            simulation = Simulation(data=self.data, base_r0=self.base_r0, contact_matrix=matrix,
                                    age_vector=age_group, susceptibility=susceptibility,
                                    country=country)
            # Create dictionary with the needed data
            self.full_contacts.update(
                {country: {"beta": simulation.beta,
                           "age_vector": self.age_group,
                           "contact_full": matrix
                           }
                 })

            # Create separated data structure for (2D)^2 PCA
            self.data_cm_d2pca_col.append(
                simulation.beta * matrix)
            self.data_cm_d2pca_r.append(
                simulation.beta * matrix
            )

            # create data for the indicators
            self.indicator_data = self.data.indicators_data

            # Create separated data structure for 1D PCA
            self.data_cm_1dpca.append(
                simulation.beta * matrix[self.upper_tri_indexes])
            self.data_clustering = np.array(self.data_cm_1dpca)

            # Final shape of the np.nd-arrays: (192, 6)
            self.data_cm_d2pca_column = np.vstack(self.data_cm_d2pca_col)
            self.data_cm_d2pca_row = np.vstack(self.data_cm_d2pca_r)


































