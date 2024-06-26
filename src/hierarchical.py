from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from src.contact_matrix_generator import ContactMatrixGenerator


class Hierarchical:
    def __init__(self, c_mtx_gen: ContactMatrixGenerator, country_names: np.ndarray,
                 img_prefix: str,
                 dist: str = "euclidean",
                 to_export_plot: bool = False):
        self.c_mtx_gen = c_mtx_gen
        self.country_names = country_names
        self.img_prefix = img_prefix
        self.to_export_plot = to_export_plot

        if dist == "euclidean":
            self.get_distance_matrix = self.get_euclidean_distance
        elif dist == "manhattan":
            self.get_distance_matrix = self.get_manhattan_distance

    def get_manhattan_distance(self):
        """
        Calculates Manhattan distance of a 39 * 136 matrix and returns 39*39 distance matrix
        :return matrix: square distance matrix with zero diagonals
        """
        manhattan_distance = manhattan_distances(self.c_mtx_gen.data_clustering)  # get pairwise manhattan distance
        # convert the data into dataframe
        # replace the indexes of the distance with the country names
        # rename the columns and rows of the distance with country names and return a matrix distance
        dt = pd.DataFrame(manhattan_distance,
                          index=self.country_names, columns=self.country_names)
        return dt, manhattan_distance

    def get_euclidean_distance(self) -> np.array:
        """
        Calculates euclidean distance of a 39 * 136 matrix and returns 39*39 distance matrix
        :return matrix: square distance matrix with zero diagonals
        """
        # convert the data into dataframe
        euc_distance = euclidean_distances(self.c_mtx_gen.data_clustering)
        dt = pd.DataFrame(euc_distance,
                          index=self.country_names, columns=self.country_names)  # rename rows and columns
        return dt, euc_distance

    def plot_distances(self):
        distance, _ = self.get_distance_matrix()
        self.country_names = self.c_mtx_gen.country_names
        plt.figure(figsize=(44, 34))
        plt.xticks(ticks=np.arange(len(self.country_names)),
                   labels=self.country_names,
                   rotation=90, fontsize=39)
        plt.yticks(ticks=np.arange(len(self.country_names)),
                   labels=self.country_names,
                   rotation=0, fontsize=39)
        plt.title("Measure of closeness  between countries before reordering",
                  fontsize=42, fontweight="bold")
        az = plt.imshow(distance, cmap="jet",
                        interpolation="nearest",
                        vmin=0)
        cbar = plt.colorbar(az)
        tick_font_size = 110
        cbar.ax.tick_params(labelsize=tick_font_size)
        if self.to_export_plot:
            plt.savefig("../plots/" + self.img_prefix + "_" + "distances.pdf")
        else:
            plt.show()

    def calculate_ordered_distance_matrix(self, threshold, verbose: bool = True):
        dt, distance = self.get_distance_matrix()
        # Return a copy of the distance collapsed into one dimension.
        distances = distance[np.triu_indices(np.shape(distance)[0], k=1)].flatten()
        #  Perform hierarchical clustering using complete method.
        res = sch.linkage(distances, method="complete")
        #  flattens the dendrogram, obtaining as a result an assignation of the original data points to single clusters.
        order = sch.fcluster(res, threshold, criterion='distance')
        if verbose:
            for x in np.unique(order):
                print("cluster " + str(x) + ":", dt.columns[order == x])
        # Perform an indirect sort along the along first axis
        columns = [dt.columns.tolist()[i] for i in list((np.argsort(order)))]
        # Place columns(sorted countries) in the both axes
        dt = dt.reindex(columns, axis='index')
        dt = dt.reindex(columns, axis='columns')
        return columns, dt, res

    def plot_ordered_distance_matrix(self, columns, dt):
        plt.figure(figsize=(45, 35), dpi=300)
        az = plt.imshow(dt, cmap='jet',
                        alpha=.9, interpolation="nearest")
        plt.xticks(ticks=np.arange(len(columns)),
                   labels=columns,
                   rotation=90, fontsize=43)
        plt.yticks(ticks=np.arange(len(columns)),
                   labels=columns,
                   rotation=0, fontsize=43)
        cbar = plt.colorbar(az)
        tick_font_size = 115
        cbar.ax.tick_params(labelsize=tick_font_size)

        plt.savefig("./plots/" + self.img_prefix + "_" + "ordered_distance_1.pdf")
        plt.show()

    def plot_dendrogram(self, res):
        fig, axes = plt.subplots(1, 1, figsize=(35, 25), dpi=150)
        sch.dendrogram(res,
                       leaf_rotation=90,
                       leaf_font_size=25,
                       labels=self.country_names,
                       orientation="top",
                       show_leaf_counts=True,
                       distance_sort=True)
        axes.tick_params(axis='both', which='major', labelsize=26)
        plt.title('Cluster Analysis without threshold', fontsize=50, fontweight="bold")
        plt.ylabel('Distance between Clusters', fontsize=45)
        plt.tight_layout()
        if self.to_export_plot:
            plt.savefig("../plots/" + self.img_prefix + "_" + "ordered_distance_2.pdf")
        else:
            plt.show()

    def plot_dendrogram_with_threshold(self, res, threshold):
        fig, axes = plt.subplots(1, 1, figsize=(15, 12))
        colors = ['blue', 'green', 'red']
        default_color = 'black'
        sch.set_link_color_palette(colors)

        dendrogram = sch.dendrogram(res,
                       color_threshold=threshold,  # sets the color of the links above the color_threshold
                       leaf_rotation=90,
                       leaf_font_size=24,  # the size based on the number of nodes in the dendrogram.
                       show_leaf_counts=True,
                       labels=self.country_names,
                       above_threshold_color='black',
                       ax=axes,
                       orientation="top",
                       get_leaves=True,
                       distance_sort=True)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=25, fontweight="bold")
        plt.ylabel('Distance between Clusters', fontsize=20, fontweight="bold")
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=15)
        axes.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.axhline(y=200, color='gray', linestyle='--', linewidth=1)
        plt.text(-20, 200, 'Threshold = 200', fontsize=10, color='gray')

        if threshold > 3:
            plt.plot([], [], color=default_color, label='Single Country',
                     linewidth=5, alpha=0.8)

        for i, color in enumerate(colors):

            plt.plot([], [], color=color, label=f'Cluster {i + 1}', linewidth=5,
                     alpha=0.8)
        # plt.legend(loc='upper right', fontsize=15)

        dendrogram_color = 'lightgray'
        plt.axhspan(0, 200, facecolor=dendrogram_color, alpha=0.2)
        plt.tight_layout()
        if self.to_export_plot:
            plt.savefig("../plots/" + self.img_prefix + "_" + "ordered_distance_3.pdf")
        else:
            plt.show()
