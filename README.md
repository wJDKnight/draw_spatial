# Spatial Transcriptomics Cell Annotation Tool

A GUI tool for manual annotation of cell types in spatial transcriptomics data.

## Setup Instructions

1. Create conda environment:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate spatial_annotation
```

## Input Format
The input CSV file should contain the following columns:
- x: X coordinates of cells
- y: Y coordinates of cells
- cell_type: Initial cell type classifications

## Features
- Interactive scatter plot visualization
- Color-coded cell types
- Single-point selection
- Lasso selection for multiple cells
- Export annotated cell types 