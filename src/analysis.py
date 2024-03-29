import numpy as np
from sklearn.decomposition import PCA

from src.contact_matrix_generator import ContactMatrixGenerator
from src.d2pca import D2PCA
from src.hierarchical import Hierarchical
from src.indicators import Indicators


class Analysis:
    def __init__(self, c_mtx_gen: ContactMatrixGenerator, img_prefix, threshold,
                 n_components: int = 4,
                 dim_red: str = None, distance: str = "euclidean",
                 to_print: bool = False):
        self.c_mtx_gen = c_mtx_gen
        self.dim_red = dim_red
        self.img_prefix = img_prefix
        self.threshold = threshold
        self.distance = distance
        self.n_components = n_components
        self.to_print = to_print
        if dim_red is not None:
            self.apply_pca()
            self.contacts_apply_indicators()

        self.hierarchical = Hierarchical(c_mtx_gen=self.c_mtx_gen,
                                         country_names=self.c_mtx_gen.country_names,
                                         img_prefix=self.img_prefix,
                                         dist=self.distance)

    def apply_pca(self):
        if self.dim_red == "PCA":
            pca = PCA(n_components=self.n_components)
            pca.fit(self.c_mtx_gen.data_clustering)
            data_pcaa = pca.transform(self.c_mtx_gen.data_clustering)  # dim 4
            ind = Indicators(c_mtx_gen=self.c_mtx_gen, country_names=self.c_mtx_gen.country_names)

            # integrate indicators
            data_pca = np.append(data_pcaa, ind.pca_data, axis=1)  # dim 8
            if self.to_print:
                print("Explained variance ratios pca:",
                      pca.explained_variance_ratio_,
                      "->", sum(pca.explained_variance_ratio_))
        elif self.dim_red == "2DPCA":
            data_dpca = D2PCA(country_names=self.c_mtx_gen.country_names, c_mtx_gen=self.c_mtx_gen)
            data_dpca.run()
            data_pca = data_dpca.pca_reduced  # dim 4
        else:
            raise Exception("Provide a type for dimensionality reduction.")
        self.c_mtx_gen.data_clustering = data_pca

    def contacts_apply_indicators(self):
        """
        This part incorporates socioeconomic indicators from the indicators file.
        2DPCA approach is used.
        """
        if self.dim_red == "2DPCA":
            data_2 = D2PCA(country_names=self.c_mtx_gen.country_names, c_mtx_gen=self.c_mtx_gen)
            data_2.run()
            data_p = data_2.pca_reduced  # dim (32, 4)
            # incorporate socioeconomic indicators to the contact matrix using the same approach
            ind = Indicators(c_mtx_gen=self.c_mtx_gen, country_names=self.c_mtx_gen.country_names)
            ind.pca_apply()
            data_pca2 = np.append(data_p, ind.pca_data, axis=1)  # dim (32, 8)
        else:
            raise Exception("The indicators could not be incorporated.")
        self.c_mtx_gen.data_clustering = data_pca2
