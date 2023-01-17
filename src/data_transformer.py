import numpy as np
import pandas as pd

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

        self.get_model_params()
        self.get_contacts()

    def get_model_params(self):
        # load the data using pandas for easy manipulation of all the countries
        age_pop = pd.read_excel("../data/Pop.xls")

        # drop the first column since it contains the name of the countries
        age_pop_2 = age_pop.iloc[:, 1:]

        # get the age groups population sum for the all countries
        pop_1 = np.sum(age_pop_2.iloc[:, 1:3])
        pop_2 = np.sum(age_pop_2.iloc[:, 5:13])
        pop_3 = np.sum(age_pop_2.iloc[:, 13:16])

        # get the Probability of asymptomatic course for the age groups
        p_1 = np.sum(np.sum(age_pop_2.iloc[:, 1:3]) * self.data.model_parameters_data['p'][1:3]) / np.sum(pop_1)
        p_2 = np.sum(np.sum(age_pop_2.iloc[:, 5:13]) * self.data.model_parameters_data['p'][5:13]) / np.sum(pop_2)
        p_3 = np.sum(np.sum(age_pop_2.iloc[:, 13:16]) * self.data.model_parameters_data['p'][13:16]) / np.sum(pop_3)

        # get the Probability of intensive care (given hospitalization) for the age groups
        x_1 = np.sum(np.sum(age_pop_2.iloc[:, 1:3]) * self.data.model_parameters_data['xi'][1:3]) / np.sum(pop_1)
        x_2 = np.sum(np.sum(age_pop_2.iloc[:, 5:13]) * self.data.model_parameters_data['xi'][5:13]) / np.sum(pop_2)
        x_3 = np.sum(np.sum(age_pop_2.iloc[:, 13:16]) * self.data.model_parameters_data['xi'][13:16]) / np.sum(pop_3)

        # get the Probability of fatal outcome
        m_1 = np.sum(np.sum(age_pop_2.iloc[:, 1:3]) * self.data.model_parameters_data['mu'][1:3]) / np.sum(pop_1)
        m_2 = np.sum(np.sum(age_pop_2.iloc[:, 5:13]) * self.data.model_parameters_data['mu'][5:13]) / np.sum(pop_2)
        m_3 = np.sum(np.sum(age_pop_2.iloc[:, 13:16]) * self.data.model_parameters_data['mu'][13:16]) / np.sum(pop_3)

        # get the "Probability of hospitalization (or intensive care)"
        h_1 = np.sum(np.sum(age_pop_2.iloc[:, 1:3]) * self.data.model_parameters_data['h'][1:3]) / np.sum(pop_1)
        h_2 = np.sum(np.sum(age_pop_2.iloc[:, 5:13]) * self.data.model_parameters_data['h'][5:13]) / np.sum(pop_2)
        h_3 = np.sum(np.sum(age_pop_2.iloc[:, 13:16]) * self.data.model_parameters_data['h'][13:16]) / np.sum(pop_3)

        # update the parameters
        self.data.model_parameters_data.update(
            {'p': [self.data.model_parameters_data['p'][0], p_1,
                   self.data.model_parameters_data['p'][3],
                   self.data.model_parameters_data['p'][4], p_2, p_3],
             'xi': [self.data.model_parameters_data['xi'][0], x_1,
                    self.data.model_parameters_data['xi'][3],
                    self.data.model_parameters_data['xi'][4], x_2, x_3],
             'h': [self.data.model_parameters_data['h'][0], h_1,
                   self.data.model_parameters_data['h'][3],
                   self.data.model_parameters_data['h'][4], h_2, h_3],
             'mu': [self.data.model_parameters_data['mu'][0], m_1,
                    self.data.model_parameters_data['mu'][3],
                    self.data.model_parameters_data['mu'][4], m_2, m_3]
             })
        # convert the Probabilities to a numpy array in their desired format
        self.data.model_parameters_data['p'] = np.array(self.data.model_parameters_data['p'])
        self.data.model_parameters_data['xi'] = np.array(self.data.model_parameters_data['xi'])
        self.data.model_parameters_data['mu'] = np.array(self.data.model_parameters_data['mu'])
        self.data.model_parameters_data['h'] = np.array(self.data.model_parameters_data['h'])
        self.data.model_parameters_data.update(self.data.model_parameters_data)

    def get_contacts(self):
        for country in self.country_names:
            contacts = [self.data.contact_data[country]["HOME"] +
                        self.data.contact_data[country]["SCHOOL"] +
                        self.data.contact_data[country]["WORK"] +
                        self.data.contact_data[country]["OTHER"]]
            for contact in contacts:
                age_vector = self.data.age_data[country]["age"].reshape((-1, 1))
                age_1 = age_vector[0][0]
                age_2 = np.sum(age_vector[1:3])
                age_3 = age_vector[3][0]
                age_4 = age_vector[4][0]
                age_5 = np.sum(age_vector[5:13])
                age_6 = np.sum(age_vector[13:16])

                # create the age group
                age_group = np.array([age_1, age_2, age_3, age_4, age_5, age_6])
                self.age_group = age_group.reshape((-1, 1))
                self.data.age_data[country]["age"] = age_group

                # create a variable t_contact
                t_contact = contact * age_vector

                # create a matrix c and fill it
                # the matrix is symmetrical
                c = np.zeros(shape=(6, 6))
                c[0, 0] = t_contact[0, 0]
                c[1, 0] = np.sum(t_contact[1:3, 0])
                c[2, 0] = t_contact[3, 0]
                c[3, 0] = t_contact[4, 0]
                c[4, 0] = np.sum(t_contact[5:13, 0])
                c[5, 0] = np.sum(t_contact[13:16, 0])
                c[0, 1] = np.copy(c[1, 0])
                c[0, 2] = np.copy(c[2, 0])
                c[0, 3] = np.copy(c[3, 0])
                c[0, 4] = np.copy(c[4, 0])
                c[0, 5] = np.copy(c[5, 0])

                c[1, 1] = np.sum(t_contact[1:3, 1:3])
                c[2, 1] = np.sum(t_contact[3, 1:3])
                c[3, 1] = np.sum(t_contact[4, 1:3])
                c[4, 1] = np.sum(t_contact[5:13, 1:3])
                c[5, 1] = np.sum(t_contact[13:16, 1:3])
                c[1, 2] = np.copy(c[2, 1])
                c[1, 3] = np.copy(c[3, 1])
                c[1, 4] = np.copy(c[4, 1])
                c[1, 5] = np.copy(c[5, 1])

                c[2, 2] = t_contact[3, 3]
                c[3, 2] = t_contact[4, 3]
                c[4, 2] = np.sum(t_contact[5:13, 3])
                c[5, 2] = np.sum(t_contact[13:16, 3])
                c[2, 3] = np.copy(c[3, 2])
                c[2, 4] = np.copy(c[4, 2])
                c[2, 5] = np.copy(c[5, 2])

                c[3, 3] = t_contact[4, 4]
                c[4, 3] = np.sum(t_contact[5:13, 4])
                c[5, 3] = np.sum(t_contact[13:16, 4])
                c[3, 4] = np.copy(c[4, 3])
                c[3, 5] = np.copy(c[5, 3])

                c[4, 4] = np.sum(t_contact[5:13, 5:13])
                c[5, 4] = np.sum(t_contact[13:16, 5:13])
                c[4, 5] = np.copy(c[5, 4])

                c[5, 5] = np.sum(t_contact[13:16, 13:16])

                matrix = c / self.age_group

                susceptibility = np.array([1.0] * 6)
                susceptibility[:4] = self.susc

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
