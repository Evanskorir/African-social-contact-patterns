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

                # create the age group
                age_group = np.array([age_1, age_2, age_3, age_4, age_5, age_6])
                self.age_group = age_group.reshape((-1, 1))
                self.data.global_unit_set["age"] = self.age_group

                # create a variable t_contact
                t_contact = contact * age_vector

                # create a matrix c and fill it
                # the matrix is symmetrical
                c = np.zeros(shape=(6, 6))
                c[0, 0] = contact[0, 0]
                c[1, 0] = np.sum(t_contact[1:3, 0])
                c[2, 0] = contact[3, 0]
                c[3, 0] = contact[4, 0]
                c[4, 0] = np.sum(t_contact[5:13, 0])
                c[5, 0] = np.sum(t_contact[13:16, 0])
                c[0, 1] = c[1, 0]
                c[1, 1] = np.sum(t_contact[1:3, 1:3])
                c[2, 1] = np.sum(t_contact[3, 1:3])
                c[3, 1] = np.sum(t_contact[4, 1:3])
                c[4, 1] = np.sum(t_contact[5:13, 1:3])
                c[5, 1] = np.sum(t_contact[13:16, 1:3])
                c[0, 2] = c[2, 0]
                c[1, 2] = c[2, 1]
                c[2, 2] = contact[3, 3]
                c[3, 2] = contact[4, 3]
                c[4, 2] = np.sum(t_contact[5:13, 3])
                c[5, 2] = np.sum(t_contact[13:16, 3])
                c[0, 3] = c[3, 0]
                c[1, 3] = c[3, 1]
                c[2, 3] = c[3, 2]
                c[3, 3] = contact[4, 4]
                c[4, 3] = np.sum(t_contact[5:13, 4])
                c[5, 3] = np.sum(t_contact[13:16, 4])
                c[0, 4] = c[4, 0]
                c[1, 4] = c[4, 1]
                c[2, 4] = c[4, 2]
                c[3, 4] = c[4, 3]
                c[4, 4] = np.sum(t_contact[5:13, 5:13])
                c[5, 4] = np.sum(t_contact[5:13, 13:16])
                c[0, 5] = c[5, 0]
                c[1, 5] = c[5, 1]
                c[2, 5] = c[5, 2]
                c[3, 5] = c[5, 3]
                c[4, 5] = c[5, 4]
                c[5, 5] = np.sum(t_contact[13:16, 13:16])

                matrix = c/self.age_group

                # retain the unaffected contacts
                matrix[0, 0] = contact[0, 0]
                matrix[2, 0] = contact[3, 0]
                matrix[3, 0] = contact[4, 0]
                matrix[0, 2] = contact[0, 3]
                matrix[2, 2] = contact[3, 3]
                matrix[3, 2] = contact[4, 3]
                matrix[0, 3] = contact[0, 4]
                matrix[2, 3] = contact[3, 4]
                matrix[3, 3] = contact[4, 4]

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



