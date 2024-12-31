import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
import google.generativeai as genai
from .utils import run_gemini, load_config
from .prompt import zeroshot_celltype_geneorder_grouped
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit, 
                           QPushButton, QTableWidget, QTableWidgetItem,
                           QScrollArea, QMessageBox, QSizePolicy, QTextBrowser)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import markdown2

class AnalysisWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Analysis")
        self.setGeometry(200, 200, 800, 800)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        
        # Create a scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setFrameShape(QScrollArea.NoFrame)  # Remove frame
        
        # Create a widget for the scroll area
        self.scroll_widget = QWidget()
        self.scroll_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.scroll.setWidget(self.scroll_widget)
        
        main_layout.addWidget(self.scroll)
        
        self.setup_ui()

    def setup_ui(self):
        # Create layout for scroll widget with proper margins
        layout = QVBoxLayout(self.scroll_widget)
        layout.setSpacing(10)  # Add spacing between widgets
        layout.setContentsMargins(10, 10, 10, 10)  # Add margins inside scroll area

        # Create table for cell type proportions
        self.table = QTableWidget()
        self.table.setMinimumHeight(200)
        self.table.setMinimumWidth(400)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setDefaultSectionSize(100)
        self.table.verticalHeader().setDefaultSectionSize(30)
        
        proportion_label = QLabel("Cell Type Proportions:")
        proportion_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(proportion_label)
        layout.addWidget(self.table)

        # Gene input section
        gene_label = QLabel("Enter genes of interest (comma-separated):")
        gene_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(gene_label)
        
        self.gene_input = QLineEdit()
        self.gene_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.gene_input)

        # Heatmap section
        self.generate_btn = QPushButton("Generate Heatmap")
        self.generate_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.generate_btn.clicked.connect(self.generate_heatmap)
        layout.addWidget(self.generate_btn)

        # Create matplotlib figure for heatmap
        self.figure, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.canvas)

        # AI Analysis Section
        ai_label = QLabel("AI Analysis")
        ai_label.setAlignment(Qt.AlignCenter)
        ai_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(ai_label)
        
        # Domain text input
        domain_label = QLabel("Enter domain text (comma-separated):")
        domain_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(domain_label)
        
        self.domain_text_input = QLineEdit()
        self.domain_text_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.domain_text_input)
        
        # Analysis buttons
        self.preview_prompt_btn = QPushButton("Preview Prompt")
        self.preview_prompt_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.preview_prompt_btn.clicked.connect(self.preview_prompt)
        layout.addWidget(self.preview_prompt_btn)
        
        self.run_gemini_btn = QPushButton("Run AI Analysis")
        self.run_gemini_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.run_gemini_btn.clicked.connect(self.run_gemini_analysis)
        layout.addWidget(self.run_gemini_btn)
        
        # Results section
        results_label = QLabel("AI Analysis Results:")
        results_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(results_label)
        
        # Replace QTextEdit with QTextBrowser for markdown rendering
        self.results_text = QTextBrowser()
        self.results_text.setMinimumHeight(400)
        self.results_text.setMinimumWidth(400)
        self.results_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.results_text.setOpenExternalLinks(True)
        
        # Set stylesheet for better markdown rendering
        self.results_text.setStyleSheet("""
            QTextBrowser {
                background-color: white;
                color: black;
                font-family: Arial, sans-serif;
                font-size: 12px;
                padding: 10px;
            }
            QTextBrowser h1 {
                font-size: 18px;
                color: #2c3e50;
                margin-bottom: 15px;
            }
            QTextBrowser h2 {
                font-size: 16px;
                color: #34495e;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            QTextBrowser hr {
                border: 1px solid #ecf0f1;
                margin: 15px 0;
            }
        """)
        
        layout.addWidget(self.results_text)

        # Add stretch at the end to push everything up
        layout.addStretch()
        
        # Update the cell type proportions table
        self.update_proportions_table()

    def prepare_dataframes(self):
        if not self.main_window.data is None and not self.main_window.cell_type_column is None:
            # Prepare cell type proportions dataframe
            cell_type_proportions = {}
            
            # Add selected points
            selected_data = self.main_window.data.iloc[list(self.main_window.selected_points)]
            if not selected_data.empty:
                cell_type_proportions["Selected"] = selected_data[self.main_window.cell_type_column].value_counts()

            # Add annotated clusters
            for cluster, indices in self.main_window.annotations.items():
                cluster_data = self.main_window.data.iloc[list(indices)]
                cell_type_proportions[cluster] = cluster_data[self.main_window.cell_type_column].value_counts()

            # Create neighbor_celltype_df
            neighbor_celltype_df = pd.DataFrame(cell_type_proportions).T

            # Create neighbor_gene_df from mean_proportions
            gene_list = [gene.strip() for gene in self.gene_input.text().split(",") if gene.strip()]
            
            # If no genes provided, return only the celltype df
            if not gene_list:
                return neighbor_celltype_df, None

            # Process gene expressions if genes are provided
            clusters_data = {}
            if self.main_window.selected_points:
                clusters_data["Selected"] = selected_data
            for cluster, indices in self.main_window.annotations.items():
                clusters_data[cluster] = self.main_window.data.iloc[list(indices)]

            mean_expressions = []
            cluster_names = []
            for cluster, data in clusters_data.items():
                valid_genes = [gene for gene in gene_list if gene in data.columns]
                if valid_genes:
                    mean_expr = data[valid_genes].mean()
                    mean_expressions.append(mean_expr)
                    cluster_names.append(cluster)

            # Create gene expression dataframe if we have valid data
            if mean_expressions and cluster_names:
                neighbor_gene_df = pd.DataFrame(mean_expressions, index=cluster_names)
                return neighbor_celltype_df, neighbor_gene_df
            
            return neighbor_celltype_df, None
            
        return None, None

    def preview_prompt(self):
        neighbor_celltype_df, neighbor_gene_df = self.prepare_dataframes()
        # if neighbor_celltype_df is None or neighbor_gene_df is None:
        #     QMessageBox.warning(self, "Error", "Please ensure data and genes are properly loaded.")
        #     return
        
        try:
            config = load_config("model_config/config_zeroshot_cosmx.yaml")
            domain_text = self.domain_text_input.text()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            return
        if domain_text:
            domain_mapping = {i: domain_text.split(",")[i] for i in range(len(domain_text.split(",")))}
            config.domain_mapping = domain_mapping
            
        prompt = zeroshot_celltype_geneorder_grouped(neighbor_celltype_df, neighbor_gene_df, rows=[0], config=config)
        QMessageBox.information(self, "Prompt Preview", prompt)

    def run_gemini_analysis(self):
        reply = QMessageBox.question(self, "Confirm", "Do you want to run Gemini analysis?",
                                   QMessageBox.No | QMessageBox.Yes)
        if reply == QMessageBox.No:
            return
            
        neighbor_celltype_df, neighbor_gene_df = self.prepare_dataframes()
            
        try:
            genai.configure(api_key=os.environ["API_KEY"])
            model = genai.GenerativeModel("gemini-1.5-pro")
            gen_config = genai.types.GenerationConfig(temperature=1.0, max_output_tokens=4000)
            
            config = load_config("./model_config/config_zeroshot_cosmx.yaml")
            domain_text = self.domain_text_input.text()
            if domain_text:
                domain_mapping = {i: domain_text.split(",")[i] for i in range(len(domain_text.split(",")))}
                config.domain_mapping = domain_mapping
            
            gemini_results_df, store_responses = run_gemini(model, gen_config,
                                                          df=neighbor_celltype_df, config=config,
                                                          prompt_func=zeroshot_celltype_geneorder_grouped, 
                                                          n_rows=1,
                                                          df_extra=neighbor_gene_df,
                                                          column_name="zeroshot_gemini")
            
            # Format results as markdown text
            markdown_text = "# AI Analysis Results\n\n"
            for i, cluster in enumerate(neighbor_celltype_df.index):
                markdown_text += f"## Cluster {cluster}\n\n"
                markdown_text += f"{store_responses[i]}\n\n"
                markdown_text += "---\n\n"  # Add separator between clusters
            
            # Convert markdown to HTML and display
            html_content = markdown2.markdown(markdown_text, extras=["fenced-code-blocks", "tables"])
            self.results_text.setHtml(html_content)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

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
                #                   vcenter=0.2*mean_proportions.max().max(), 
                #                   vmax=mean_proportions.max().max())
                sns.heatmap(mean_proportions, cmap='viridis', 
                          annot=True, fmt='.2f', cbar=True, ax=self.ax)
                
                self.ax.set_title('Mean Expression Levels')
                self.ax.set_xlabel('Genes')
                self.ax.set_ylabel('Clusters')

        # Update the canvas
        self.canvas.draw() 