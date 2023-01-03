import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from data_transformer import Contacts
from d2pca import D2PCA
from hierarchical import Hierarchical
from Indicators import Indicators


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

    def run(self):
        hierarchical = Hierarchical(data_transformer=self.data_tr,
                                    country_names=self.data_tr.country_names,
                                    img_prefix=self.img_prefix,
                                    dist=self.distance)
        hierarchical.run(threshold=self.threshold)
        hierarchical.plot_distances()

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
            data_pcaa = data_dpca.pca_reduced     # dim 4

            # integrate indicators to the contact matrices
            data_pca = np.append(data_pcaa, self.pca_data, axis=1)   # dim 8
        else:
            raise Exception("Provide a type for dimensionality reduction.")
        self.data_tr.data_clustering = data_pca


def kenya_contacts(data_tr):
    # full contact
    age_group = ["0-4", "5-14", "15-19", "20-24", "25-64", "65+"]

    plt.figure(figsize=(10, 8))
    full = plt.imshow(data_tr.full_contacts['Kenya']['contact_full'].T,
                      cmap='jet', alpha=.9, interpolation="nearest")
    ticks = np.arange(0, 6)
    cbar = plt.colorbar(full)
    tick_font_size = 20
    cbar.ax.tick_params(labelsize=tick_font_size)

    plt.xticks(ticks=ticks, labels=age_group, rotation=45)
    plt.yticks(ticks=ticks, labels=age_group, rotation=0)
    # plt.title("All", fontsize=15)
    plt.gca().invert_yaxis()
    plt.xlabel("Age")
    plt.ylabel("Age")
    plt.savefig("../plots/" + "All.pdf")


def country_contacts(data_tr):
    for country in ["Ethiopia", "Algeria", "Burkina Faso"]:
        age_group = ["0-4", "5-14", "15-19", "20-24", "25-64", "65+"]
        matrix_to_plot = data_tr.full_contacts[country]["contact_full"] * data_tr.full_contacts[country]["beta"]
        img = plt.imshow(matrix_to_plot.T,
                         cmap='jet', vmin=0, vmax=0.9,
                         alpha=.9, interpolation="nearest")
        ticks = np.arange(0, 6)

        plt.yticks(ticks=ticks, labels=age_group, rotation=0)
        plt.xticks(ticks=ticks, labels=age_group, rotation=45)
        plt.gca().invert_yaxis()
        if country == "Burkina Faso":
            cbar = plt.colorbar(img)
            tick_font_size = 25
            cbar.ax.tick_params(labelsize=tick_font_size)
        plt.savefig("../plots/" + country + ".pdf")


def main():
    do_clustering_pca = False
    do_clustering_dpca = False

    # Create data for clustering
    susc = 1.0
    base_r0 = 3.68
    data_tr = Contacts(susc=susc, base_r0=base_r0)

    # execute class indicators
    ind = Indicators(data_tr=data_tr, country_names=data_tr.country_names)
    ind.pca_apply()
    ind.corr_pcs()
    ind.dendogram_pca()
    ind.project_2d()
    ind.plot_countries()

    kenya_contacts(data_tr=data_tr)
    country_contacts(data_tr=data_tr)

    # do analysis for original data
    Analysis(data_tr=data_tr, pca_data=ind.pca_data,
             img_prefix="original", threshold=0.55).run()

    # Do analysis of the pca
    if do_clustering_pca:
        n_components = 4
        # do analysis for reduced data
        Analysis(data_tr=data_tr, dim_red="PCA", n_components=4, pca_data=ind.pca_data,
                 img_prefix="pca_" + str(n_components), threshold=11).run()

    # do analysis of 2dpca
    if do_clustering_dpca:
        Analysis(data_tr=data_tr, pca_data=ind.pca_data, dim_red="2DPCA",
                 img_prefix="dpca_", threshold=11).run()


if __name__ == "__main__":
    main()
