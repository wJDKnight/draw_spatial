import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit, 
                           QPushButton, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt

class AnalysisWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Analysis")
        self.setGeometry(200, 200, 800, 800)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Create table for cell type proportions
        self.table = QTableWidget()
        layout.addWidget(QLabel("Cell Type Proportions:"))
        layout.addWidget(self.table)

        # Create gene input section
        layout.addWidget(QLabel("Enter genes of interest (comma-separated):"))
        self.gene_input = QLineEdit()
        layout.addWidget(self.gene_input)

        # Create button to generate heatmap
        self.generate_btn = QPushButton("Generate Heatmap")
        self.generate_btn.clicked.connect(self.generate_heatmap)
        layout.addWidget(self.generate_btn)

        # Create matplotlib figure for heatmap
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Update the cell type proportions table
        self.update_proportions_table()

    def update_proportions_table(self):
        if not self.main_window.data is None and not self.main_window.cell_type_column is None:
            # Get all unique clusters and cell types
            all_clusters = set()
            cell_type_proportions = {}

            # Add selected points as a special cluster
            selected_data = self.main_window.data.iloc[list(self.main_window.selected_points)]
            if not selected_data.empty:
                all_clusters.add("Selected")
                cell_type_proportions["Selected"] = selected_data[self.main_window.cell_type_column].value_counts()

            # Add annotated clusters
            for cluster, indices in self.main_window.annotations.items():
                all_clusters.add(cluster)
                cluster_data = self.main_window.data.iloc[list(indices)]
                cell_type_proportions[cluster] = cluster_data[self.main_window.cell_type_column].value_counts()

            if all_clusters:
                # Get all unique cell types
                all_cell_types = set()
                for proportions in cell_type_proportions.values():
                    all_cell_types.update(proportions.index)

                # Create and populate the table
                self.table.setRowCount(len(all_clusters))
                self.table.setColumnCount(len(all_cell_types))
                self.table.setHorizontalHeaderLabels(list(all_cell_types))
                self.table.setVerticalHeaderLabels(list(all_clusters))

                # Fill in the proportions
                for i, cluster in enumerate(all_clusters):
                    cluster_props = cell_type_proportions[cluster]
                    total_cells = cluster_props.sum()
                    for j, cell_type in enumerate(all_cell_types):
                        proportion = cluster_props.get(cell_type, 0) / total_cells if total_cells > 0 else 0
                        percentage = f"{proportion:.1%}"
                        self.table.setItem(i, j, QTableWidgetItem(percentage))

                self.table.resizeColumnsToContents()

    def generate_heatmap(self):
        gene_list = [gene.strip() for gene in self.gene_input.text().split(",") if gene.strip()]
        if not gene_list:
            return

        # Clear the previous plot
        self.ax.clear()

        # Get data for selected and annotated cells
        clusters_data = {}
        
        # Add selected points as a special cluster
        if self.main_window.selected_points:
            selected_data = self.main_window.data.iloc[list(self.main_window.selected_points)]
            if not selected_data.empty:
                clusters_data["Selected"] = selected_data

        # Add annotated clusters
        for cluster, indices in self.main_window.annotations.items():
            cluster_data = self.main_window.data.iloc[list(indices)]
            clusters_data[cluster] = cluster_data

        if clusters_data:
            # Calculate mean expression for each gene in each cluster
            mean_expressions = []
            cluster_names = []
            
            for cluster, data in clusters_data.items():
                # Filter genes that exist in the data
                valid_genes = [gene for gene in gene_list if gene in data.columns]
                if valid_genes:
                    mean_expr = data[valid_genes].mean()
                    mean_expressions.append(mean_expr)
                    cluster_names.append(cluster)

            if mean_expressions:
                # Create DataFrame for heatmap
                mean_proportions = pd.DataFrame(mean_expressions, index=cluster_names)
                
                # Create heatmap
                # norm = TwoSlopeNorm(vmin=0, 
                #                   vcenter=0.5*mean_proportions.max().max(), 
                #                   vmax=mean_proportions.max().max())
                sns.heatmap(mean_proportions, cmap='viridis', 
                          annot=True, fmt='.2f', cbar=True, ax=self.ax)
                
                self.ax.set_title('Mean Expression Levels')
                self.ax.set_xlabel('Genes')
                self.ax.set_ylabel('Clusters')

        # Update the canvas
        self.canvas.draw() 