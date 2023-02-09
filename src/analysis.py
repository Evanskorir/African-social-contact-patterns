import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.data_transformer import Contacts
from src.d2pca import D2PCA
from src.hierarchical import Hierarchical


class Analysis:
    def __init__(self, pca_data, data_tr: Contacts, img_prefix, threshold,
                 n_components: int = 4,
                 dim_red: str = None, distance: str = "euclidean"):
        self.data_tr = data_tr
        self.pca_data = pca_data
        self.dim_red = dim_red
        self.img_prefix = img_prefix
        self.threshold = threshold
        self.distance = distance
        self.n_components = n_components
        if dim_red is not None:
            self.apply_pca()
        self.hierarchical = Hierarchical(data_transformer=self.data_tr,
                                         country_names=self.data_tr.country_names,
                                         img_prefix=self.img_prefix,
                                         dist=self.distance)

    def apply_pca(self):
        if self.dim_red == "PCA":
            pca = PCA(n_components=self.n_components)
            pca.fit(self.data_tr.data_clustering)
            data_pcaa = pca.transform(self.data_tr.data_clustering)  # dim 4

            # integrate indicators
            data_pca = np.append(data_pcaa, self.pca_data, axis=1)  # dim 8
            print("Explained variance ratios pca:",
                  pca.explained_variance_ratio_,
                  "->", sum(pca.explained_variance_ratio_))
        elif self.dim_red == "2DPCA":
            data_dpca = D2PCA(country_names=self.data_tr.country_names, data_tr=self.data_tr)
            data_dpca.apply_dpca()
            data_pcaa = data_dpca.pca_reduced  # dim 4

            # integrate indicators to the contact matrices
            data_pca = np.append(data_pcaa, self.pca_data, axis=1)  # dim 8
        else:
            raise Exception("Provide a type for dimensionality reduction.")
        self.data_tr.data_clustering = data_pca


def kenya_contacts(data_tr):
    age_group = ["0-4", "5-14", "15-19", "20-24", "25-64", "65+"]
    plt.figure(figsize=(13, 12))
    full = plt.imshow(data_tr.full_contacts['Kenya']['contact_full'],
                      cmap='jet', alpha=.9, interpolation="nearest", vmin=0, vmax=16)

    ticks = np.arange(0, 6)
    cbar = plt.colorbar(full, shrink=0.87)
    tick_font_size = 40
    cbar.ax.tick_params(labelsize=tick_font_size)
    plt.xticks(ticks=ticks, labels=age_group, rotation=90, fontsize=32)
    plt.yticks(ticks=ticks, labels=age_group, rotation=0, fontsize=32)
    plt.gca().invert_yaxis()
    # plt.xlabel("Age", fontsize=28)
    # plt.ylabel("Age", fontsize=28)
    # plt.title("Other", fontsize=40)
    # plt.show()
    # here you create plots for contacts at home, school, work, other, and all
    # plt.savefig("../plots/" + "All.pdf")
    # by manipulating contact in data_transformer.py
