import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing

from src.contact_matrix_generator import ContactMatrixGenerator


class D2PCA:
    """
        This is (2D)^2 PCA class that applies both row-row and column-column directions to perform dimension reduction.
        input: 39 countries each 16 * 16 matrix concatenated row wise and column wise
        output: 39 countries each 2 * 2 matrix, and 39 * 4 (2 * 2 flatten matrix)
    """
    def __init__(self, c_mtx_gen: ContactMatrixGenerator, country_names: list,
                 to_print: bool = False):
        self.country_names = country_names
        self.c_mtx_gen = c_mtx_gen
        self.to_print = to_print

        self.data_contact_matrix = c_mtx_gen.data_cm_d2pca_column
        self.contact_matrix_transposed = c_mtx_gen.data_cm_d2pca_row

        self.data_split = []
        self.proj_matrix_2 = []
        self.proj_matrix_1 = []
        self.pca_reduced = []

    @staticmethod
    def preprocess_data(data):
        # center the data
        centered_data = data - np.mean(data, axis=0)
        # normalize data
        data_scaled = preprocessing.scale(centered_data)
        return data_scaled

    def column_pca(self, col_dim: int = 2):
        data_scaled = self.preprocess_data(data=self.data_contact_matrix)
        pca_1 = PCA(n_components=col_dim)
        pca_1.fit(data_scaled)

        if self.to_print:
            print("Explained variance ratios col:", pca_1.explained_variance_ratio_,
                  "->", sum(pca_1.explained_variance_ratio_), "Eigenvectors col:",
                  pca_1.components_,  # (col_dim, 6)
                  "Singular values col:", pca_1.singular_values_)  # col_dim leading eigenvalues

        # Projection matrix for row direction matrix
        proj_matrix_1 = pca_1.components_.T  # 6 * col_dim projection matrix 1

        return proj_matrix_1

    def row_pca(self, row_dim: int = 2):
        data_scaled_2 = self.preprocess_data(data=self.contact_matrix_transposed)

        pca_2 = PCA(n_components=row_dim)
        pca_2.fit(data_scaled_2)

        # print("Explained variance ratios row:", pca_2.explained_variance_ratio_,
        #      "->", sum(pca_2.explained_variance_ratio_), "Eigenvectors row:",
        #      pca_2.components_,  # (row_dim, 6)
        #     "Singular values row:", pca_2.singular_values_)  # row_dim leading eigenvalues
        # print("PC 2", pc2)

        # Projection matrix for column direction matrix
        proj_matrix_2 = pca_2.components_.T  # 6 * row_dim projection matrix 2
        return proj_matrix_2

    def run(self):
        # Now split concatenated original data into 39 sub-arrays of equal size i.e. 32 countries.
        data_scaled = self.preprocess_data(data=self.data_contact_matrix)
        split = np.array_split(data_scaled, 32)
        self.data_split = np.array(split)
        # Get projection matrix for column direction
        self.proj_matrix_1 = self.column_pca()
        # Get projection matrix for row direction
        self.proj_matrix_2 = self.row_pca()

        # Now apply (2D)^2 PCA simultaneously using projection matrix 1 and 2
        matrix = self.proj_matrix_1.T @ self.data_split @ self.proj_matrix_2

        # Now reshape the matrix to get desired 39 * 4
        self.pca_reduced = matrix.reshape((32, 4))
