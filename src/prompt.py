from .utils import sig_figs, extract_output_microenvironments
import json
import os
import time
import pandas as pd
from .utils import get_farthest_cell_types


def zeroshot_celltype_geneorder_grouped(df_cell, df_gene, rows, config):
    # "cell" is a group of cells

    if config.domain_mapping is not None:
        text_domain = f"The group of cells can only belong to one of the {len(config.domain_mapping)} microenvironments: {', '.join(config.domain_mapping.values())}."
    else:
        text_domain = ""
    
    if df_cell is not None:
        df_cell = df_cell.copy()
        text_opening = "You will be provided with a list of cell types in the neighborhood of a group of cells."
        text_cell = "The neighbor cell-types are ordered from most frequent to least frequent."
    else:
        text_cell = ""

    if df_gene is not None:
        df_gene = df_gene.copy()
        text_opening = "You will be provided with a list of marker genes expressed in the neighborhood of a group of cells."
        text_gene = "The marker genes are ordered from highest expression to lowest expression."
    else:
        text_gene = ""

    if df_cell is not None and df_gene is not None:
        text_opening = "You will be provided with a list of cell types in the neighborhood of a group of cells and marker genes expressed in the neighborhood."

    if df_cell is None and df_gene is None:
        raise ValueError("No cell types or marker genes provided")

    strings_list = []
    for i in rows:
        if df_cell is not None:
            cell_row = df_cell.iloc[i]
            sorted_cells = cell_row.sort_values(ascending=False)[:config.top_n]
            cell_frequency = []
            # Iterate over sorted cell types
            for cell_type, count in sorted_cells.items():
                if count <= config.minimal_f:
                    continue
                count = round(count, 3)
                if config.use_full_name:
                    cell_name = config.cell_names_mapping[cell_type]
                else:
                    cell_name = cell_type
                if config.with_numbers:
                    cell_frequency.append(f"{cell_name}: {count}")
                else:
                    cell_frequency.append(f"{cell_name}")
            cell_frequency = ", ".join(cell_frequency)
        
        if df_gene is not None:
            gene_row = df_gene.iloc[i]
            sorted_genes = gene_row.sort_values(ascending=False)[:config.top_n]

            gene_frequency = []
            for gene, count in sorted_genes.items():
                if count <= config.minimal_gene_threshold:
                    continue
                count = round(count, 3)
                if config.with_numbers:
                    gene_frequency.append(f"{gene}: {count}")
                else:
                    gene_frequency.append(f"{gene}")
            gene_frequency = ", ".join(gene_frequency)

        if df_cell is not None and df_gene is not None:
            strings_list.append(f"Neighbor cell-types: {{{cell_frequency}}}, Marker genes: {{{gene_frequency}}}")
        elif df_cell is not None:
            strings_list.append(f"Neighbor cell-types: {{{cell_frequency}}}")
        elif df_gene is not None:
            strings_list.append(f"Marker genes: {{{gene_frequency}}}")
        else:
            raise ValueError("No cell types or marker genes provided")

    multi_rows_strings = ";\n ".join(strings_list)

    result = f"""{text_opening}{config.region1} \
{text_cell} \
{text_gene} \
Your task is to identify the microenvironments the group of cells belong to. \
Remember, microenvironments are not cell types. \
{text_domain} \
Below are the list of cell-types and/or marker genes in the cell's neighborhood:\n{multi_rows_strings}\n"""
    
    return result




