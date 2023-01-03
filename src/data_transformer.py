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
            contacts = [self.data.contact_data[country]["OTHER"] +
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

                # first column
                # age group 0-4, 0-4
                col_0_0 = contact[0, 0]
                # age group 5-14, 0-4
                col_1_0 = (contact[1, 0] * age_vector[0]
                           + contact[2, 0] * age_vector[0]) / age_1
                # age group 15-19, 0-4
                col_2_0 = contact[3, 0]

                # age group 20-24, 0-4
                col_3_0 = contact[4, 0]
                # age group 25-64, 0-4
                col_4_0 = (contact[5, 0] * age_vector[0] + contact[6, 0] * age_vector[0] +
                           contact[7, 0] * age_vector[0] + contact[8, 0] * age_vector[0] +
                           contact[9, 0] * age_vector[0] + contact[10, 0] * age_vector[0] +
                           contact[11, 0] * age_vector[0] + contact[12, 0] * age_vector[0]) / age_1

                # age group 65+, 0-4
                col_5_0 = ((contact[13:14, 0:1] * age_vector[0] +
                            (contact[14:15, 0:1] * age_vector[0]) + contact[15:16, 0:1] * age_vector[0])) / age_1

                # second column
                # age group 0-4, 5-14
                col_0_1 = (contact[0, 1] * age_vector[1] + contact[0, 2] * age_vector[2]) / age_2
                # age group 5-14, 5-14
                col_1_1 = ((contact[1, 1] * age_vector[1] + contact[1, 2] * age_vector[2]) +
                           (contact[2, 1] * age_vector[1] + contact[2, 2] * age_vector[2])) / age_2
                # age group 15-19, 5-14
                col_2_1 = (contact[3, 1] * age_vector[1] + contact[3, 2] * age_vector[2]) / age_2
                # age group 20-24, 5-14
                col_3_1 = (contact[4, 1] * age_vector[1] + contact[4, 2] * age_vector[2]) / age_2
                # age group 25-64, 5-14
                col_4_1 = ((contact[5, 1] * age_vector[1] + contact[5, 2] * age_vector[2]) +
                           (contact[6, 1] * age_vector[1] + contact[6, 2] * age_vector[2]) +
                           (contact[7, 1] * age_vector[1] + contact[7, 2] * age_vector[2]) +
                           (contact[8, 1] * age_vector[1] + contact[8, 2] * age_vector[2]) +
                           (contact[9, 1] * age_vector[1] + contact[9, 2] * age_vector[2]) +
                           (contact[10, 1] * age_vector[1] + contact[10, 2] * age_vector[2]) +
                           (contact[11, 1] * age_vector[1] + contact[11, 2] * age_vector[2]) +
                           (contact[12, 1] * age_vector[1] + contact[12, 2] * age_vector[2])) / age_2
                # age group 65+, 5-14
                col_5_1 = ((contact[13, 1] * age_vector[1] + contact[13, 2] * age_vector[2]) +
                           (contact[14, 1] * age_vector[1] + contact[14, 2] * age_vector[2]) +
                           (contact[15, 1] * age_vector[1] + contact[15, 2] * age_vector[2])) / age_2
                # third column
                # age group 0-4, 15-19
                col_0_2 = contact[0, 3]
                # age group 5-14, 15-19
                col_1_2 = (contact[1, 3] * age_vector[3] + contact[2, 3] * age_vector[3]) / age_vector[3]
                # age group 15-19, 15-19
                col_2_2 = contact[3, 3]
                # age group 20-24, 15-19
                col_3_2 = contact[4, 3]
                col_4_2 = (contact[5, 3] * age_vector[3] + contact[6, 3] * age_vector[3] +
                           contact[7, 3] * age_vector[3] + contact[8, 3] * age_vector[3] +
                           contact[9, 3] * age_vector[3] + contact[10, 3] * age_vector[3] +
                           contact[11, 3] * age_vector[3] + contact[12, 3] * age_vector[3]) / age_vector[3]

                col_5_2 = (contact[13, 3] * age_vector[3] + contact[14, 3] * age_vector[3] +
                           contact[15, 3] * age_vector[3]) / age_vector[3]

                # 4th column
                # age group 0-4, 20-24
                col_0_3 = contact[0, 4]
                # age group 5-14, 20-24
                col_1_3 = (contact[1, 4] * age_vector[4] +
                           contact[2, 4] * age_vector[4]) / age_vector[4]
                # age group 15-19, 20-24
                col_2_3 = contact[3, 4]
                # age group 20-24, 15-19
                col_3_3 = contact[4, 4]
                col_4_3 = (contact[5, 4] * age_vector[4] + contact[6, 4] * age_vector[4] +
                           contact[7, 4] * age_vector[4] + contact[8, 4] * age_vector[4] +
                           contact[9, 4] * age_vector[4] + contact[10, 4] * age_vector[4] +
                           contact[11, 4] * age_vector[4] + contact[12, 4] * age_vector[4]) / age_vector[4]

                col_5_3 = (contact[13, 4] * age_vector[4] +
                           contact[14, 4] * age_vector[4] +
                           contact[15, 4] * age_vector[4]) / age_vector[4]

                # 5th column
                col_0_4 = ((contact[0, 5] * age_vector[5] + contact[0, 6] * age_vector[6] +
                            contact[0, 7] * age_vector[7] + contact[0, 8] * age_vector[8] +
                            contact[0, 9] * age_vector[9] + contact[0, 10] * age_vector[10] +
                            contact[0, 11] * age_vector[11] + contact[0, 12] * age_vector[12])) / age_5

                col_1_4 = (contact[1, 5] * age_vector[5] + contact[1, 6] * age_vector[6] +
                           contact[1, 7] * age_vector[7] + contact[1, 8] * age_vector[8] +
                           contact[1, 9] * age_vector[9] + contact[1, 10] * age_vector[10] +
                           contact[1, 11] * age_vector[11] + contact[1, 12] * age_vector[12] +
                           contact[2, 5] * age_vector[5] + contact[2, 6] * age_vector[6] +
                           contact[2, 7] * age_vector[7] + contact[2, 8] * age_vector[8] +
                           contact[2, 9] * age_vector[9] + contact[2, 10] * age_vector[10] +
                           contact[2, 11] * age_vector[11] + contact[2, 12] * age_vector[12]) / age_5

                col_2_4 = (contact[3, 5] * age_vector[5] + contact[3, 6] * age_vector[6] +
                           contact[3, 7] * age_vector[7] + contact[3, 8] * age_vector[8] +
                           contact[3, 9] * age_vector[9] + contact[3, 10] * age_vector[10] +
                           contact[3, 11] * age_vector[11] + contact[3, 12] * age_vector[12]) / age_5

                col_3_4 = (contact[4, 5] * age_vector[5] + contact[4, 6] * age_vector[6] +
                           contact[4, 7] * age_vector[7] + contact[4, 8] * age_vector[8] +
                           contact[4, 9] * age_vector[9] + contact[4, 10] * age_vector[10] +
                           contact[4, 11] * age_vector[11] + contact[4, 12] * age_vector[12]) / age_5

                col_4_4 = ((contact[5, 5] + contact[6, 5] + contact[7, 5] + contact[8, 5] +
                            contact[9, 5] + contact[10, 5] + contact[11, 5] + contact[12, 5]) * age_vector[5] +
                           (contact[5, 6] + contact[6, 6] + contact[7, 6] + contact[8, 6] + contact[9, 6] +
                            contact[10, 6] +
                            contact[11, 6] + contact[12, 6]) * age_vector[6] +
                           (contact[5, 7] + contact[6, 7] + contact[7, 7] + contact[8, 7] + contact[9, 7] +
                            contact[10, 7] + contact[11, 7] + contact[12, 7]) * age_vector[7] +
                           (contact[5, 8] + contact[6, 8] + contact[7, 8] + contact[8, 8] + contact[9, 8] +
                            contact[10, 8] + contact[11, 8] + contact[12, 8]) * age_vector[8] +
                           (contact[5, 9] + contact[6, 9] + contact[7, 9] + contact[8, 9] + contact[9, 9] +
                            contact[10, 9] + contact[11, 9] + contact[12, 9]) * age_vector[9] +
                           (contact[5, 10] + contact[6, 10] + contact[7, 10] + contact[8, 10] + contact[9, 10] +
                            contact[10, 10] + contact[11, 10] + contact[12, 10]) * age_vector[10] +
                           (contact[5, 11] + contact[6, 11] + contact[7, 11] +
                            contact[8, 11] + contact[9, 11] + contact[10, 11] + contact[11, 11] +
                            contact[12, 11]) * age_vector[11] +
                           (contact[5, 12] + contact[6, 12] + contact[7, 12] + contact[8, 12] + contact[9, 12] +
                            contact[10, 12] + contact[11, 12] + contact[12, 12]) * age_vector[12]) / age_5

                col_5_4 = ((contact[5, 13] + contact[6, 13] + contact[7, 13] + contact[8, 13] + contact[9, 13] +
                            contact[10, 13] + contact[11, 13] + contact[12, 13]) * age_vector[13] +
                           (contact[5, 14] + contact[6, 14] + contact[7, 14] + contact[8, 14] + contact[9, 14] +
                            contact[10, 14] + contact[11, 14] + contact[12, 14]) * age_vector[14] +
                           (contact[5, 15] + contact[6, 15] + contact[7, 15] + contact[8, 15] + contact[9, 15] +
                            contact[10, 15] + contact[11, 15] + contact[12, 15]) * age_vector[15]) / age_5

                # 6th column
                col_0_5 = (contact[0, 13] * age_vector[13] + contact[0, 14] * age_vector[14] +
                           contact[0, 15] * age_vector[15]) / age_6
                col_1_5 = ((contact[1, 13] + contact[2, 13]) * age_vector[13] +
                           (contact[1, 14] + contact[2, 14]) * age_vector[14] +
                           (contact[1, 15] + contact[2, 15]) * age_vector[15]) / age_6

                col_2_5 = (contact[3, 13] * age_vector[13] + contact[3, 14] * age_vector[14] +
                           contact[3, 15] * age_vector[15]) / age_6

                col_3_5 = (contact[4, 13] * age_vector[13] + contact[4, 14] * age_vector[14] +
                           contact[4, 15] * age_vector[15]) / age_6

                col_4_5 = ((contact[5, 13] + contact[6, 13] + contact[7, 13] + contact[8, 13] + contact[9, 13] +
                            contact[10, 13] + contact[11, 13] + contact[12, 13]) * age_vector[13] +
                           (contact[5, 14] + contact[6, 14] + contact[7, 14] + contact[8, 14] + contact[9, 14] +
                            contact[10, 14] + contact[11, 14] + contact[12, 14]) * age_vector[14] +
                           (contact[5, 15] + contact[6, 15] + contact[7, 15] + contact[8, 15] + contact[9, 15] +
                            contact[10, 15] + contact[11, 15] + contact[12, 15]) * age_vector[15]) / age_6

                col_5_5 = ((contact[13, 13] + contact[14, 13] + contact[15, 13]) * age_vector[13] +
                           (contact[13, 14] + contact[14, 14] + contact[15, 14]) * age_vector[14] +
                           (contact[13, 15] + contact[14, 15] + contact[15, 15]) * age_vector[15]) / age_6

                cont = np.zeros(shape=(6, 6))
                cont[0, 0] = col_0_0
                cont[1, 0] = col_1_0
                cont[2, 0] = col_2_0
                cont[3, 0] = col_3_0
                cont[4, 0] = col_4_0
                cont[5, 0] = col_5_0
                cont[0, 1] = col_0_1
                cont[1, 1] = col_1_1
                cont[2, 1] = col_2_1
                cont[3, 1] = col_3_1
                cont[4, 1] = col_4_1
                cont[5, 1] = col_5_1
                cont[0, 2] = col_0_2
                cont[1, 2] = col_1_2
                cont[2, 2] = col_2_2
                cont[3, 2] = col_3_2
                cont[4, 2] = col_4_2
                cont[5, 2] = col_5_2
                cont[0, 3] = col_0_3
                cont[1, 3] = col_1_3
                cont[2, 3] = col_2_3
                cont[3, 3] = col_3_3
                cont[4, 3] = col_4_3
                cont[5, 3] = col_5_3
                cont[0, 4] = col_0_4
                cont[1, 4] = col_1_4
                cont[2, 4] = col_2_4
                cont[3, 4] = col_3_4
                cont[4, 4] = col_4_4
                cont[5, 4] = col_5_4
                cont[0, 5] = col_0_5
                cont[1, 5] = col_1_5
                cont[2, 5] = col_2_5
                cont[3, 5] = col_3_5
                cont[4, 5] = col_4_5
                cont[5, 5] = col_5_5

                # create the age group
                age_group = np.array([age_1, age_2, age_3, age_4, age_5, age_6])
                self.age_group = age_group.reshape((-1, 1))

                self.data.global_unit_set["age"] = self.age_group

                susceptibility = np.array([1.0] * 6)
                susceptibility[:4] = self.susc

                simulation = Simulation(data=self.data, base_r0=self.base_r0, contact_matrix=cont,
                                        age_vector=age_group, susceptibility=susceptibility)
                # Create dictionary with the needed data
                self.full_contacts.update(
                    {country: {"beta": simulation.beta,
                               "age_vector": self.age_group,
                               "contact_full": cont
                               }
                     })

                # Create separated data structure for (2D)^2 PCA
                self.data_cm_d2pca_col.append(
                    simulation.beta * cont)
                self.data_cm_d2pca_r.append(
                    simulation.beta * cont.T
                )

                # create data for the indicators
                self.indicator_data = self.data.indicators_data

                # Create separated data structure for 1D PCA
                self.data_cm_1dpca.append(
                    simulation.beta * cont[self.upper_tri_indexes])
            self.data_clustering = np.array(self.data_cm_1dpca)

            # Final shape of the np.nd-arrays: (192, 6)
            self.data_cm_d2pca_column = np.vstack(self.data_cm_d2pca_col)
            self.data_cm_d2pca_row = np.vstack(self.data_cm_d2pca_r)
