import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.contact_matrix_generator import ContactMatrixGenerator


class Indicators:
    pca_data = []

    def __init__(self, c_mtx_gen: ContactMatrixGenerator, country_names: list,
                 to_print: bool = False, to_export_plot: bool = False):
        self.country_names = country_names
        self.c_mtx_gen = c_mtx_gen
        self.indicator_data = c_mtx_gen.indicator_data
        self.to_print = to_print
        self.to_export_plot = to_export_plot
        self.pca_data = np.array([])
        self.pca2 = PCA(n_components=4, svd_solver='randomized', random_state=50)

    def pca_apply(self):
        # Standardization technique for scaling
        scaler = StandardScaler()
        country_data_scaled = scaler.fit_transform(self.c_mtx_gen.indicator_data)
        pca = PCA(svd_solver='randomized', random_state=50)
        pca.fit(country_data_scaled)
        self.pca_data = self.pca2.fit_transform(country_data_scaled)

    def var_plot_ratio(self):
        if not os.path.exists("../plots"):
            os.makedirs("../plots")
        # Variance Ratio bar plot for each PCA components.
        scaler = StandardScaler()
        country_data_scaled = scaler.fit_transform(self.c_mtx_gen.indicator_data)
        pca = PCA(svd_solver='randomized', random_state=50)
        pca.fit(country_data_scaled)
        self.pca_data = self.pca2.fit_transform(country_data_scaled)

        # Determine colors based on the percentage of variance explained
        colors = ['gray' if ratio < 0.07 else 'lightblue' for
                  ratio in pca.explained_variance_ratio_]
        plt.figure(figsize=(12, 8))
        bar_plot = plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
                           pca.explained_variance_ratio_, color=colors, edgecolor='black')

        # Annotate percentages of variance explained for the first 4 components
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            if i < 4:
                plt.text(i + 1, ratio + 0.005, f'{ratio * 100:.1f}%',
                         ha='center', va='bottom', fontsize=10, fontweight='bold',
                         color='black')
            else:
                plt.text(i + 1, ratio + 0.005, f'{ratio * 100:.1f}%',
                         ha='center', va='bottom', fontsize=7, fontweight='bold',
                         color='black')

        # Set labels and title
        plt.xlabel("Principal Components", fontweight='bold', fontsize=12)
        plt.ylabel("Variance Ratio", fontweight='bold', fontsize=12)
        plt.title("Variance Ratio Explained by Principal Components", fontweight='bold',
                  fontsize=18)

        # Set x-axis ticks labels
        plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1),
                   [f'PC{i}' for i in range(1, len(pca.explained_variance_ratio_) + 1)],
                   fontsize=8, fontweight='bold', rotation=90)

        # Adjust tick font sizes
        plt.yticks(fontsize=10, fontweight='bold')

        # Add gridlines
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        gray_legend = Patch(color='gray',
                            label=f'Explained variance by remaining PCs: '
                                  f'{100 - cumulative_variance[3] * 100:.1f}%')
        lightblue_legend = Patch(color='lightblue',
                                 label=f'Explained variance by 4 PCs: '
                                       f'{cumulative_variance[3] * 100:.1f}%')
        plt.legend(handles=[lightblue_legend, gray_legend], loc='upper right',
                   fontsize=10,  bbox_to_anchor=(1.0, 0.95))

        plt.tight_layout()
        if self.to_export_plot:
            plt.savefig("../plots/" + "variance.pdf")
        else:
            plt.show()

        # Let's check the variance ratios
        if self.to_print:
            print("\n cumulative variance explained by indicators:",
                  np.cumsum(self.pca2.explained_variance_ratio_))
            print("\n explained variance explained by indicator:",
                  self.pca2.explained_variance_ratio_)

    def corr_data(self):
        # Let's check the correlation coefficients to see which variables are highly correlated
        plt.figure(figsize=(18, 18))
        country = self.c_mtx_gen.indicator_data.iloc[:, 1:]
        sns.heatmap(country.corr(), cmap="rainbow")
        if self.to_export_plot:
            plt.savefig("../plots/" + "corr.pdf")

    def dendrogram_pca(self):
        fig, axes = plt.subplots(1, 1, figsize=(15, 12))
        colors = ['blue', 'green', 'red']
        labels = self.c_mtx_gen.country_names
        sch.set_link_color_palette(colors)

        dendrogram = sch.dendrogram(sch.linkage(self.pca_data, method="complete"),
                                    color_threshold=8,
                                    get_leaves=True, leaf_rotation=90, leaf_font_size=20,
                                    show_leaf_counts=True, orientation="top", distance_sort=True,
                                    labels=labels, above_threshold_color='black',
                                    ax=axes)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=25, fontweight="bold")
        plt.ylabel('Distance between Clusters', fontsize=20, fontweight="bold")
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=15)
        axes.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.axhline(y=200, color='gray', linestyle='--', linewidth=1)
        plt.text(-20, 200, 'Threshold = 200', fontsize=10, color='gray')

        for i, color in enumerate(colors):
            if i == 2:
                label = 'Single Country'
                color = 'black'
            else:
                label = f'Cluster {i + 1}'
            plt.plot([], [], color=color, label=label,
                     linewidth=5, alpha=0.8)
        plt.plot([], [], color="red", label='Cluster 3',
                 linewidth=5, alpha=0.8)
        # plt.legend(loc='upper right', fontsize=15)
        dendrogram_color = 'lightgray'
        plt.axhspan(0, 200, facecolor=dendrogram_color, alpha=0.2)
        plt.tight_layout()
        if self.to_export_plot:
            plt.savefig("../plots/" + "Dendrogram.pdf")
        else:
            plt.show()

    def corr_pcs(self):
        scaler = StandardScaler()
        country_data_scaled = scaler.fit_transform(self.c_mtx_gen.indicator_data)
        pca = PCA(svd_solver='randomized', random_state=50)
        pca.fit(country_data_scaled)
        self.pca_data = self.pca2.fit_transform(country_data_scaled)
        plt.figure(figsize=(20, 10))
        ax = plt.gca()
        im = ax.imshow(self.pca2.components_, cmap='jet',
                        alpha=.9, interpolation="nearest")
        feature_names = list(pd.DataFrame(self.c_mtx_gen.indicator_data).columns)
        cbar = plt.colorbar(im, orientation='horizontal', pad=0.45, shrink=0.5)
        cbar.ax.tick_params(labelsize=20)
        plt.xticks(ticks=np.arange(len(pd.DataFrame(self.c_mtx_gen.indicator_data).columns)),
                   labels=feature_names,
                   rotation=90, fontsize=20)
        plt.yticks(ticks=np.arange(0, 4),
                   labels=['PC1', 'PC2', 'PC3', 'PC4'], rotation=0, fontsize=18)

        # Add annotations
        for i in range(len(self.pca2.components_)):
            for j in range(len(self.pca2.components_[i])):
                plt.text(j, i, f'{self.pca2.components_[i, j]:.2f}', ha='center',
                         va='center', color='black',
                         fontsize=12)

        plt.tight_layout()
        if self.to_export_plot:
            plt.savefig("../plots/" + "components.pdf")
        else:
            plt.show()

    def plot_countries(self):
        # Define the clusters with individual countries
        clusters = {
            'Cluster 1': ['Ethiopia'],
            'Cluster 2': ['Lesotho'],
            'Cluster 3': ['Mauritania'],
            'Cluster 4': ['Liberia'],
            'Cluster 5': ['Nigeria'],
            'Cluster 6': ['Senegal'],
            'Cluster 7': ['Seychelles'],
            'Cluster 8': ['Sao Tome'],
            'Cluster 9': ['Tanzania'],
            'Cluster 10': ['Namibia'],
            'Cluster 11': ['Sierra Leone'],
            'Cluster 12': ['Egypt'],
            'Cluster 13': ['Cameroon'],
            'Cluster 14': ['Congo'],
            'Cluster 15': ['Morocco'],
            'Cluster 16': ['Algeria'],
            'Cluster 17': ['Cape Verde'],
            'Cluster 18': ['Kenya'],
            'Cluster 19': ['Zambia'],
            'Cluster 20': ['Mauritius'],
            'Cluster 21': ['Ghana'],
            'Cluster 22': ['Tunisia'],
            'Cluster 23': ['Uganda'],
            'Cluster 24': ['Botswana'],
            'Cluster 25': ['Mozambique'],
            'Cluster 26': ['South Africa'],
            'Cluster 27': ['Benin'],
            'Cluster 28': ['Rwanda'],
            'Cluster 29': ['Zimbabwe'],
            'Cluster 30': ['Niger'],
            'Cluster 31': ['Burkina Faso'],
            'Cluster 32': ['Guinea']
        }

        # Define colors for each cluster
        cluster_colors = {
            'Cluster 1': '#1f77b4',
            'Cluster 2': '#ff7f0e',
            'Cluster 3': '#2ca02c',
            'Cluster 4': '#d62728',
            'Cluster 5': '#9467bd',
            'Cluster 6': '#8c564b',
            'Cluster 7': '#e377c2',
            'Cluster 8': '#7f7f7f',
            'Cluster 9': '#bcbd22',
            'Cluster 10': '#17becf',
            'Cluster 11': '#aec7e8',
            'Cluster 12': '#ffbb78',
            'Cluster 13': '#98df8a',
            'Cluster 14': '#ff9896',
            'Cluster 15': '#c5b0d5',
            'Cluster 16': '#c49c94',
            'Cluster 17': '#f7b6d2',
            'Cluster 18': '#c7c7c7',
            'Cluster 19': '#dbdb8d',
            'Cluster 20': '#9edae5',
            'Cluster 21': '#c7c7c7',
            'Cluster 22': '#c7c7c7',
            'Cluster 23': 'purple',
            'Cluster 24': 'orange',
            'Cluster 25': 'pink',
            'Cluster 26': 'salmon',
            'Cluster 27': 'indigo',
            'Cluster 28': 'grey',
            'Cluster 29': 'cyan',
            'Cluster 30': 'skyblue',
            'Cluster 31': 'blue',
            'Cluster 32': 'black'
        }

        # Create the scatter plot
        plt.figure(figsize=(12, 10))  # Adjust figure size
        for cluster, countries in clusters.items():
            cluster_color = cluster_colors[cluster]
            for country in countries:
                # Get the index of the country
                i = self.country_names.index(country)

                # Adjust marker sizes and shapes
                marker = 'o'
                size = 150

                # Plot the point and add the text label
                plt.scatter(self.pca_data[i, 0], self.pca_data[i, 1],
                            c=cluster_color, alpha=0.7, marker=marker, s=size,
                            edgecolor='black', linewidth=0.5)
                plt.text(self.pca_data[i, 0], self.pca_data[i, 1], country,
                         fontsize=12, ha='center', va='center')

        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.5)

        # Add the axis labels
        plt.xlabel('First Dimension (34.4%)', fontsize=14, fontweight='bold')
        plt.ylabel('Second Dimension (10.6%)', fontsize=14, fontweight='bold')

        # Improve layout
        plt.tight_layout()

        # Save or display the plot
        if self.to_export_plot:
            plt.savefig("../plots/countries_projection.pdf", dpi=300)
        else:
            plt.show()