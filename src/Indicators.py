import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.data_transformer import Contacts


class Indicators:
    pca_data = []

    def __init__(self, data_tr: Contacts, country_names: list):
        self.country_names = country_names
        self.data_tr = data_tr
        self.indicator_data = data_tr.indicator_data
        self.pca_data = np.array([])
        self.pca2 = []

    def corr_data(self):
        # Let's check the correlation coefficients to see which variables are highly correlated
        plt.figure(figsize=(18, 18))
        country = self.data_tr.indicator_data.iloc[:, 1:]
        sns.heatmap(country.corr(), cmap="rainbow")
        plt.savefig("../plots/" + "corr.pdf")

    def pca_apply(self):
        # Standardization technique for scaling
        scaler = StandardScaler()
        country_data_scaled = scaler.fit_transform(self.data_tr.indicator_data)
        pca = PCA(svd_solver='randomized', random_state=50)
        pca.fit(country_data_scaled)

        # Variance Ratio bar plot for each PCA components.
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
        plt.xlabel("PCA Components", fontweight='bold')
        plt.ylabel("Variance Ratio", fontweight='bold')
        plt.savefig("../plots/" + "variance.pdf")

        # Scree plot to visualize the Cumulative variance against the Number of components
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.vlines(x=3, ymax=1, ymin=0, colors="r", linestyles="--")
        plt.hlines(y=0.61, xmax=15, xmin=0, colors="g", linestyles="--")
        plt.xlabel('Number of PCA components')
        plt.ylabel('Cumulative Explained Variance')
        plt.savefig("../plots/" + "exp variance.pdf")

        pca2 = PCA(n_components=4, svd_solver='randomized', random_state=50)
        pca_data = pca2.fit_transform(country_data_scaled)

        # data projected to 4 dim
        self.pca_data = pca_data
        self.pca2 = pca2
        # Let's check the variance ratios
        print("\n cumulative variance explained by indicators:", np.cumsum(pca2.explained_variance_ratio_))
        print("\n explained variance explained by indicator:", pca2.explained_variance_ratio_)

    def dendogram_pca(self):
        # Hierarchical clustering based on only the indicators
        fig, axes = plt.subplots(1, 1, figsize=(40, 25), dpi=150)
        sch.dendrogram(sch.linkage(self.pca_data, method="complete"), color_threshold=8, get_leaves=True,
                       leaf_rotation=90, leaf_font_size=32, show_leaf_counts=True, orientation="top",
                       distance_sort=True, labels=self.data_tr.country_names)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=44, fontweight="bold")
        plt.ylabel('Distance between Clusters', fontsize=42, fontweight="bold")
        axes.tick_params(axis='both', which='major', labelsize=26)
        plt.savefig("../plots/" + "Dendrogram.pdf")

    def corr_pcs(self):
        a = pd.DataFrame(self.data_tr.indicator_data)
        b = a.columns
        print(b)
        plt.figure(figsize=(14, 12))
        ax = plt.gca()
        ax = plt.imshow(self.pca2.components_, cmap='jet',
                        alpha=.9, interpolation="nearest")
        feature_names = list(b)
        plt.colorbar(ax, orientation='horizontal', ticks=[self.pca2.components_.min(), 0,
                                                          self.pca2.components_.max()], pad=0.6)
        plt.xticks(ticks=np.arange(len(b)), labels=feature_names,
                   rotation=90, fontsize=20)
        plt.yticks(ticks=np.arange(0, 4),
                   labels=['PC1', 'PC2', 'PC3', 'PC4'], rotation=0, fontsize=20)

        plt.savefig("../plots/" + "components.pdf")
    #
    # def project_2d(self):
    #     # put feature values into dataframe
    #     stand = StandardScaler()
    #     scaled = stand.fit_transform(self.data_tr.indicator_data)
    #     pca3 = PCA(n_components=2, svd_solver='randomized', random_state=50)
    #     pca_dataa = pca3.fit_transform(scaled)
    #     components = pd.DataFrame(pca3.components_.T, index=pd.DataFrame(self.data_tr.indicator_data).columns,
    #                               columns=['PCA1', 'PCA2'])
    #
    #     # plot size
    #     plt.figure(figsize=(16, 12))
    #
    #     # main scatter-plot
    #     plt.scatter(pca_dataa[:, 0], pca_dataa[:, 1], cmap='jet', edgecolors='blue',
    #                 alpha=0.7, s=50)
    #     plt.xlabel('First Dim (34.4%)')
    #     plt.ylabel('Second Dim (10.6%)')
    #     plt.ylim(10, -10)
    #     plt.xlim(10, -10)
    #
    #     # individual feature values
    #     ax2 = plt.twinx().twiny()
    #     ax2.set_ylim(-0.4, 0.4)
    #     ax2.set_xlim(-0.4, 0.4)
    #
    #     # reference lines
    #     ax2.hlines(0, -0.4, 0.4, linestyles='dotted', colors='black')
    #     ax2.vlines(0, -0.4, 0.4, linestyles='dotted', colors='black')
    #
    #     # offset for labels
    #     offset = 1.05
    #
    #     # arrow & text
    #     for a, i in enumerate(components.index):
    #         ax2.arrow(0, 0, components['PCA1'][a], -components['PCA2'][a],
    #                   alpha=0.5, facecolor='grey', head_width=0.008)
    #         ax2.annotate(i, (components['PCA1'][a] * offset, -components['PCA2'][a] * offset), color='black')
    #     plt.savefig("../plots/" + "2D projection.pdf")

    def plot_countries(self):
        # Params
        n_samples = 32  # number of countries
        # m_features = 28  # number of economic indicators
        # selected_names = self.country_names

        # Generate
        np.random.seed(42)
        names = self.country_names
        labels = [np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                                    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                                    'Z', '1', '2', '3', '4', '5', '6']) for _ in range(n_samples)]
        # features = np.random.random((n_samples, m_features))

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
        plt.figure(figsize=(14, 12))
        plt.scatter(self.pca_data[:, 0], self.pca_data[:, 1],
                    c=col, alpha=0.5)

        # Add the labels
        for name in self.country_names:
            # Get the index of the name
            i = names.index(name)

            # Add the text label
            labelpad = 0.03  # Adjust this based on your dataset
            plt.text(self.pca_data[i, 0] + labelpad, self.pca_data[i, 1] + labelpad, name, fontsize=9)

            # Mark the labeled observations with a star marker
            plt.scatter(self.pca_data[i, 0], self.pca_data[i, 1],
                        c=col[i], vmin=min(col), vmax=max(col), marker='*', s=100)

        # Add the axis labels
        plt.xlabel('First Dim (34.4%)')
        plt.ylabel('Second Dim (10.6%)')

        plt.savefig("../plots/" + "countries projection.pdf")
