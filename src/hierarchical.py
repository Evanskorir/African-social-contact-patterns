from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from src.data_transformer import Contacts


class Hierarchical:
    def __init__(self, data_transformer: Contacts, country_names: np.ndarray, img_prefix: str,
                 dist: str = "euclidean"):
        self.data_tr = data_transformer
        self.country_names = country_names
        self.img_prefix = img_prefix
        if dist == "euclidean":
            self.get_distance_matrix = self.get_euclidean_distance
        elif dist == "manhattan":
            self.get_distance_matrix = self.get_manhattan_distance

        # os.makedirs("../plots", exist_ok=True)

    def get_manhattan_distance(self):
        """
        Calculates Manhattan distance of a 39 * 136 matrix and returns 39*39 distance matrix
        :return matrix: square distance matrix with zero diagonals
        """
        manhattan_distance = manhattan_distances(self.data_tr.data_clustering)  # get pairwise manhattan distance
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
        euc_distance = euclidean_distances(self.data_tr.data_clustering)
        dt = pd.DataFrame(euc_distance,
                          index=self.country_names, columns=self.country_names)  # rename rows and columns
        return dt, euc_distance

    def plot_distances(self):
        distance, _ = self.get_distance_matrix()
        self.country_names = self.data_tr.country_names
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
        # plt.savefig("../plots/" + self.img_prefix + "_" + "distances.pdf")
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

    @staticmethod
    def plot_ordered_distance_matrix(columns, dt):
        plt.figure(figsize=(45, 35), dpi=300)
        az = plt.imshow(dt, cmap='jet',
                        alpha=.9, interpolation="nearest")
        plt.xticks(ticks=np.arange(len(columns)),
                   labels=columns,
                   rotation=90, fontsize=40)
        plt.yticks(ticks=np.arange(len(columns)),
                   labels=columns,
                   rotation=0, fontsize=40)
        cbar = plt.colorbar(az)
        tick_font_size = 115
        cbar.ax.tick_params(labelsize=tick_font_size)
        # plt.savefig("../plots/" + self.img_prefix + "_" + "ordered_distance_1.pdf")
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
        # plt.savefig("../plots/" + self.img_prefix + "_" + "ordered_distance_2.pdf")
        plt.show()

    def plot_dendrogram_with_threshold(self, res, threshold):
        fig, axes = plt.subplots(1, 1, figsize=(30, 25), dpi=300)
        sch.dendrogram(res,
                       color_threshold=threshold,  # sets the color of the links above the color_threshold
                       leaf_rotation=90,
                       leaf_font_size=24,  # the size based on the number of nodes in the dendrogram.
                       show_leaf_counts=True,
                       labels=self.country_names,
                       above_threshold_color='blue',
                       ax=axes,
                       orientation="top",
                       get_leaves=True,
                       distance_sort=True)
        plt.title('Cluster Analysis', fontsize=49, fontweight="bold")
        plt.ylabel('Distance between Clusters', fontsize=30)
        plt.tight_layout()
        axes.tick_params(axis='both', which='major', labelsize=26)
        # plt.savefig("../plots/" + self.img_prefix + "_" + "ordered_distance_3.pdf")
        plt.show()
