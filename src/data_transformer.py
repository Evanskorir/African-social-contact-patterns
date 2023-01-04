import numpy as np
from dataloader import DataLoader

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
        for country in self.country_names:
            contacts = [self.data.contact_data[country]["HOME"] +
                        self.data.contact_data[country]["SCHOOL"] +
                        self.data.contact_data[country]["WORK"] +
                        self.data.contact_data[country]["OTHER"]]

            for contact in contacts:
                age_vector = self.data.age_data[country]["age"].reshape((-1, 1))
                age_1 = age_vector[0]
                age_2 = age_vector[1] + age_vector[2]
                age_3 = age_vector[3]
                age_4 = age_vector[4]
                age_5 = age_vector[5] + age_vector[6] + age_vector[7] + age_vector[8] + age_vector[9] +\
                    age_vector[10] + age_vector[11] + age_vector[12]
                age_6 = age_vector[13] + age_vector[14] + age_vector[15]

                t_contacts = contact * age_vector

                c = np.zeros(shape=(6, 6))
                c[0, 0] = contact[0, 0]
                c[1, 0] = (t_contacts[1, 0] + t_contacts[2, 0]) / age_1
                c[2, 0] = contact[3, 0]
                c[3, 0] = contact[4, 0]
                c[4, 0] = (t_contacts[5, 0] + t_contacts[6, 0] + t_contacts[7, 0] + t_contacts[8, 0] +
                           t_contacts[9, 0] + t_contacts[10, 0] + t_contacts[11, 0] + t_contacts[12, 0]) / age_1
                c[5, 0] = (t_contacts[13, 0] + t_contacts[14, 0] + t_contacts[15, 0]) / age_1
                c[0, 1] = (t_contacts[0, 1] + t_contacts[0, 2]) / age_2
                c[1, 1] = (t_contacts[1, 1] + t_contacts[1, 2] + t_contacts[2, 1] + t_contacts[2, 2]) / age_2
                c[2, 1] = (t_contacts[3, 1] + t_contacts[3, 2]) / age_2
                c[3, 1] = (t_contacts[4, 1] + t_contacts[4, 2]) / age_2
                c[4, 1] = (t_contacts[5, 1] + t_contacts[5, 2] + t_contacts[6, 1] + t_contacts[6, 2] +
                           t_contacts[7, 1] + t_contacts[7, 2] + t_contacts[8, 1] + t_contacts[8, 2] +
                           t_contacts[9, 1] + t_contacts[9, 2] + t_contacts[10, 1] + t_contacts[10, 2] +
                           t_contacts[11, 1] + t_contacts[11, 2] + t_contacts[12, 1] + t_contacts[12, 2]) / age_2
                c[5, 1] = (t_contacts[13, 1] + t_contacts[13, 2] + t_contacts[14, 1] + t_contacts[14, 2] +
                           t_contacts[15, 1] + t_contacts[15, 2]) / age_2
                c[0, 2] = contact[0, 3]
                c[1, 2] = (t_contacts[1, 3] + t_contacts[2, 3]) / age_vector[3]
                c[2, 2] = contact[3, 3]
                c[3, 2] = contact[4, 3]
                c[4, 2] = (t_contacts[5, 3] + t_contacts[6, 3] + t_contacts[7, 3] + t_contacts[8, 3] +
                           t_contacts[9, 3] + t_contacts[10, 3] + t_contacts[11, 3] +
                           t_contacts[12, 3]) / age_vector[3]
                c[5, 2] = (t_contacts[13, 3] + t_contacts[14, 3] + t_contacts[15, 3]) / age_vector[3]
                c[0, 3] = contact[0, 4]
                c[1, 3] = (t_contacts[1, 4] + t_contacts[2, 4]) / age_vector[4]
                c[2, 3] = contact[3, 4]
                c[3, 3] = contact[4, 4]
                c[4, 3] = (t_contacts[5, 4] + t_contacts[6, 4] + t_contacts[7, 4] + t_contacts[8, 4] +
                           t_contacts[9, 4] + t_contacts[10, 4] + t_contacts[11, 4] +
                           t_contacts[12, 4]) / age_vector[4]
                c[5, 3] = (t_contacts[13, 4] + t_contacts[14, 4] + t_contacts[15, 4]) / age_vector[4]
                c[0, 4] = (t_contacts[0, 5] + t_contacts[0, 6] + t_contacts[0, 7] + t_contacts[0, 8] +
                           t_contacts[0, 9] + t_contacts[0, 10] + t_contacts[0, 11] + t_contacts[0, 12]) / age_5
                c[1, 4] = (t_contacts[1, 5] + t_contacts[1, 6] + t_contacts[1, 7] + t_contacts[1, 8] +
                           t_contacts[1, 9] + t_contacts[1, 10] + t_contacts[1, 11] + t_contacts[1, 12] +
                           t_contacts[2, 5] + t_contacts[2, 6] + t_contacts[2, 7] + t_contacts[2, 8] +
                           t_contacts[2, 9] + t_contacts[2, 10] + t_contacts[2, 11] + t_contacts[2, 12]) / age_5
                c[2, 4] = (t_contacts[3, 5] + t_contacts[3, 6] + t_contacts[3, 7] + t_contacts[3, 8] +
                           t_contacts[3, 9] + t_contacts[3, 10] + t_contacts[3, 11] + t_contacts[3, 12]) / age_5

                c[3, 4] = (t_contacts[4, 5] + t_contacts[4, 6] + t_contacts[4, 7] + t_contacts[4, 8] +
                           t_contacts[4, 9] + t_contacts[4, 10] + t_contacts[4, 11] + t_contacts[4, 12]) / age_5
                c[4, 4] = (t_contacts[5, 5] + t_contacts[5, 6] + t_contacts[5, 7] + t_contacts[5, 8] +
                           t_contacts[5, 9] + t_contacts[5, 10] + t_contacts[5, 11] + t_contacts[5, 12] +
                           t_contacts[6, 5] + t_contacts[6, 6] + t_contacts[6, 7] + t_contacts[6, 8] +
                           t_contacts[6, 9] + t_contacts[6, 10] + t_contacts[6, 11] + t_contacts[6, 12] +
                           t_contacts[7, 5] + t_contacts[7, 6] + t_contacts[7, 7] + t_contacts[7, 8] +
                           t_contacts[7, 9] + t_contacts[7, 10] + t_contacts[7, 11] + t_contacts[7, 12] +
                           t_contacts[8, 5] + t_contacts[8, 6] + t_contacts[8, 7] + t_contacts[8, 8] +
                           t_contacts[8, 9] + t_contacts[8, 10] + t_contacts[8, 11] + t_contacts[8, 12] +
                           t_contacts[9, 5] + t_contacts[9, 6] + t_contacts[9, 7] + t_contacts[9, 8] +
                           t_contacts[9, 9] + t_contacts[9, 10] + t_contacts[9, 11] + t_contacts[9, 12] +
                           t_contacts[10, 5] + t_contacts[10, 6] + t_contacts[10, 7] + t_contacts[10, 8] +
                           t_contacts[10, 9] + t_contacts[10, 10] + t_contacts[10, 11] + t_contacts[10, 12] +
                           t_contacts[11, 5] + t_contacts[11, 6] + t_contacts[11, 7] + t_contacts[11, 8] +
                           t_contacts[11, 9] + t_contacts[11, 10] + t_contacts[11, 11] + t_contacts[11, 12] +
                           t_contacts[12, 5] + t_contacts[12, 6] + t_contacts[12, 7] + t_contacts[12, 8] +
                           t_contacts[12, 9] + t_contacts[12, 10] + t_contacts[12, 11] + t_contacts[12, 12]) / age_5
                c[5, 4] = (t_contacts[13, 5] + t_contacts[13, 6] + t_contacts[13, 7] + t_contacts[13, 8] +
                           t_contacts[13, 9] + t_contacts[13, 10] + t_contacts[13, 11] + t_contacts[13, 12] +
                           t_contacts[14, 5] + t_contacts[14, 6] + t_contacts[14, 7] + t_contacts[14, 8] +
                           t_contacts[14, 9] + t_contacts[14, 10] + t_contacts[14, 11] + t_contacts[14, 12] +
                           t_contacts[15, 5] + t_contacts[15, 6] + t_contacts[15, 7] + t_contacts[15, 8] +
                           t_contacts[15, 9] + t_contacts[15, 10] + t_contacts[15, 11] + t_contacts[15, 12]) / age_5
                c[0, 5] = (t_contacts[0, 13] + t_contacts[0, 14] + t_contacts[0, 15]) / age_6
                c[1, 5] = (t_contacts[1, 13] + t_contacts[1, 14] + t_contacts[1, 15] + t_contacts[2, 13] +
                           t_contacts[2, 14] + t_contacts[2, 15]) / age_6
                c[2, 5] = (t_contacts[3, 13] + t_contacts[3, 14] + t_contacts[3, 15]) / age_6
                c[3, 5] = (t_contacts[4, 13] + t_contacts[4, 14] + t_contacts[4, 15]) / age_6
                c[4, 5] = (t_contacts[5, 13] + t_contacts[5, 14] + t_contacts[5, 15] + t_contacts[6, 13] +
                           t_contacts[6, 14] + t_contacts[6, 15] + t_contacts[7, 13] + t_contacts[7, 14] +
                           t_contacts[7, 15] + t_contacts[8, 13] + t_contacts[8, 14] + t_contacts[8, 15] +
                           t_contacts[9, 13] + t_contacts[9, 14] + t_contacts[9, 15] + t_contacts[10, 13] +
                           t_contacts[10, 14] + t_contacts[10, 15] + t_contacts[11, 13] + t_contacts[11, 14] +
                           t_contacts[11, 15] + t_contacts[12, 13] + t_contacts[12, 14] +
                           t_contacts[12, 15]) / age_6
                c[5, 5] = (t_contacts[13, 13] + t_contacts[13, 14] + t_contacts[13, 15] + t_contacts[14, 13] +
                           t_contacts[14, 14] + t_contacts[14, 15] + t_contacts[15, 13] +
                           t_contacts[15, 14] + t_contacts[15, 15]) / age_6

                # create the age group
                age_group = np.array([age_1, age_2, age_3, age_4, age_5, age_6])
                self.age_group = age_group.reshape((-1, 1))

                self.data.global_unit_set["age"] = self.age_group

                susceptibility = np.array([1.0] * 6)
                susceptibility[:4] = self.susc

                simulation = Simulation(data=self.data, base_r0=self.base_r0, contact_matrix=c,
                                        age_vector=age_group, susceptibility=susceptibility)
                # Create dictionary with the needed data
                self.full_contacts.update(
                    {country: {"beta": simulation.beta,
                               "age_vector": self.age_group,
                               "contact_full": c
                               }
                     })

                # Create separated data structure for (2D)^2 PCA
                self.data_cm_d2pca_col.append(
                    simulation.beta * c)
                self.data_cm_d2pca_r.append(
                    simulation.beta * c
                )

                # create data for the indicators
                self.indicator_data = self.data.indicators_data

                # Create separated data structure for 1D PCA
                self.data_cm_1dpca.append(
                    simulation.beta * c[self.upper_tri_indexes])
            self.data_clustering = np.array(self.data_cm_1dpca)

            # Final shape of the np.nd-arrays: (192, 6)
            self.data_cm_d2pca_column = np.vstack(self.data_cm_d2pca_col)
            self.data_cm_d2pca_row = np.vstack(self.data_cm_d2pca_r)

