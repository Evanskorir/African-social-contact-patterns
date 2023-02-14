import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.contact_matrix_generator import ContactMatrixGenerator


class Indicators:
    pca_data = []

    def __init__(self, c_mtx_gen: ContactMatrixGenerator, country_names: list):
        self.country_names = country_names
        self.c_mtx_gen = c_mtx_gen
        self.indicator_data = c_mtx_gen.indicator_data
        self.pca_data = np.array([])
        self.pca2 = PCA(n_components=4, svd_solver='randomized', random_state=50)

    def pca_apply(self):
        # Standardization technique for scaling
        scaler = StandardScaler()
        country_data_scaled = scaler.fit_transform(self.c_mtx_gen.indicator_data)
        pca = PCA(svd_solver='randomized', random_state=50)
        pca.fit(country_data_scaled)
        self.pca_data = self.pca2.fit_transform(country_data_scaled)

        # Variance Ratio bar plot for each PCA components.
        plt.figure(figsize=(10, 8))
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
        plt.xlabel("PCA Components", fontweight='bold', fontsize=18)
        plt.ylabel("Variance Ratio", fontweight='bold', fontsize=18)
        plt.savefig("../plots/" + "variance.pdf")

        # Scree plot to visualize the Cumulative variance against the Number of components
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.vlines(x=3, ymax=1, ymin=0, colors="r", linestyles="--")
        plt.hlines(y=0.61, xmax=15, xmin=0, colors="g", linestyles="--")
        plt.xlabel('Number of PCA components')
        plt.ylabel('Cumulative Explained Variance')
        plt.savefig("../plots/" + "exp variance.pdf")

        # Let's check the variance ratios
        print("\n cumulative variance explained by indicators:", np.cumsum(self.pca2.explained_variance_ratio_))
        print("\n explained variance explained by indicator:", self.pca2.explained_variance_ratio_)

    def corr_data(self):
        # Let's check the correlation coefficients to see which variables are highly correlated
        plt.figure(figsize=(18, 18))
        country = self.c_mtx_gen.indicator_data.iloc[:, 1:]
        sns.heatmap(country.corr(), cmap="rainbow")
        plt.savefig("../plots/" + "corr.pdf")

    def dendogram_pca(self):
        # Hierarchical clustering based on only the indicators
        fig, axes = plt.subplots(1, 1, figsize=(40, 25), dpi=150)
        sch.dendrogram(sch.linkage(self.pca_data, method="complete"), color_threshold=8, get_leaves=True,
                       leaf_rotation=90, leaf_font_size=32, show_leaf_counts=True, orientation="top",
                       distance_sort=True, labels=self.c_mtx_gen.country_names)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=44, fontweight="bold")
        plt.ylabel('Distance between Clusters', fontsize=42, fontweight="bold")
        axes.tick_params(axis='both', which='major', labelsize=26)
        plt.savefig("../plots/" + "Dendrogram.pdf")

    def corr_pcs(self):
        plt.figure(figsize=(14, 12))
        _ = plt.gca()
        ax = plt.imshow(self.pca2.components_, cmap='jet',
                        alpha=.9, interpolation="nearest")
        feature_names = list(pd.DataFrame(self.c_mtx_gen.indicator_data).columns)
        plt.colorbar(ax, orientation='horizontal', ticks=[self.pca2.components_.min(), 0,
                                                          self.pca2.components_.max()], pad=0.6)
        plt.xticks(ticks=np.arange(len(pd.DataFrame(self.c_mtx_gen.indicator_data).columns)), labels=feature_names,
                   rotation=90, fontsize=20)
        plt.yticks(ticks=np.arange(0, 4),
                   labels=['PC1', 'PC2', 'PC3', 'PC4'], rotation=0, fontsize=20)

        plt.savefig("../plots/" + "components.pdf")

    def plot_countries(self):
        # Params
        n_samples = 32  # number of countries

        # Generate
        np.random.seed(42)
        names = self.country_names
        labels = [np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                                    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                                    'Z', '1', '2', '3', '4', '5', '6']) for _ in range(n_samples)]

        # Label to color dict (manual)
        label_color_dict = {'A': 'red', 'B': 'peru', 'C': 'blue', 'D': 'magenta',
                            'E': 'orange', 'F': 'yellow', 'G': 'black', 'H': "plum", 'I': 'olive',
                            'J': 'green', 'K': 'Plum', 'L': 'tomato', 'M': 'lime',
                            'N': 'YellowGreen', 'O': 'LemonChiffon', 'P': 'DarkGoldenrod',
                            'Q': 'Maroon', 'R': 'pink', 'S': 'ForestGreen', 'T': 'Sienna', 'U': 'chocolate',
                            'V': 'brown', 'W': 'DeepPink', 'X': 'DarkOrchid', 'Y': 'Violet', 'Z': 'MediumBlue',
                            '1': 'Aquamarine', '2': 'BlueViolet', '3': 'purple', '4': 'navy', '5': 'aqua',
                            '6': 'teal'}

        # Color vector creation
        col = [label_color_dict[label] for label in labels]

        # Create the scatter plot
        plt.figure(figsize=(14, 14))
        plt.scatter(self.pca_data[:, 0], self.pca_data[:, 1],
                    c=col, alpha=0.5)

        # Add the labels
        for name in self.country_names:
            # Get the index of the name
            i = names.index(name)

            # Add the text label
            labelpad = 0.03  # Adjust this based on your dataset
            plt.text(self.pca_data[i, 0] + labelpad, self.pca_data[i, 1] + labelpad, name, fontsize=20)

            # Mark the labeled observations with a star marker
            plt.scatter(self.pca_data[i, 0], self.pca_data[i, 1],
                        c=col[i], vmin=min(col), vmax=max(col), marker='*', s=120)

        # Add the axis labels
        plt.xlabel('First Dim (34.4%)', fontsize=20)
        plt.ylabel('Second Dim (10.6%)', fontsize=20)
        plt.savefig("../plots/" + "countries projection.pdf")
