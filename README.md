# Spatial Transcriptomics Cell Annotation Tool

An interactive GUI tool for manual annotation of cell types in spatial transcriptomics data. This tool enables researchers to efficiently visualize, explore, and annotate cells in spatial transcriptomics datasets through an intuitive and user-friendly interface.

## Features

- **Interactive Visualization**
  - Scatter plot visualization of spatial data
  - Color-coded cell types
  - Adjustable point size and transparency
  - Zoom and pan functionality
  - RGB visualization mode for continuous variables
  - Show/hide annotated cells
  - Dynamic point size adjustment
  - Customizable transparency settings

- **Selection Tools**
  - Single-point selection (W)
  - Lasso selection (Q)
  - Brush selection (E)
  - Eraser tool (R)
  - Adjustable brush size (+/-)
  - Undo/Redo functionality (D/F)

- **Annotation Management**
  - Add new cell type annotations
  - Remove annotations
  - Undo/redo support for selections and annotations
  - Export annotated data to CSV
  - Auto-complete for annotation names
  - Annotation history tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/draw_spatial.git
cd draw_spatial
```

2. Create conda environment:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate spatial_annotation
```

## Configuration

1. Configure the LLM API key:

  Add Gemini "API_KEY" to the environment variable.

2. Configure the settings when running LLM for niche analysis:

  Change the settings in the "model_config/config_zeroshot.yaml" file.

## Usage

1. Launch the application:
```bash
python main.py
```

2. Load your data using the "Load Data" button
3. Select columns for X/Y coordinates and cell types
4. Use the selection tools to select cells:
   - Press W for single-point selection
   - Press Q for lasso selection
   - Press E for brush selection
   - Press R for eraser tool
   - Use +/- to adjust brush size
   - Use D/F for undo/redo

5. Enter a new annotation name and click confirm to annotate selected cells
6. Save your annotations using the "Save All Annotations" button

## Input Data Format

The input data should be in CSV format with the following requirements:

### Required Columns:
- Spatial coordinates (X, Y): Numeric columns representing spatial positions
- Cell type classifications: Initial cell type labels (if available)

### Optional Columns:
- Gene expression data: For RGB visualization
- Additional metadata: Any other cell-specific information

Example format:
```csv
X,Y,cell_type,gene1,gene2,metadata
100,200,T-cell,0.5,0.8,sample1
150,250,B-cell,0.3,0.6,sample1
...
```

## Keyboard Shortcuts

- Q: Lasso selection mode
- W: Single-point selection mode
- E: Brush selection mode
- R: Eraser mode
- D: Undo last selection
- F: Redo last selection
- +/-: Adjust brush size (for brush and eraser modes)

## Output

The tool exports a CSV file containing:
- All original columns
- A new 'new_annotation' column with the manual annotations

## Requirements

See the "environment.yml" file for the required packages.

## Platform Compatibility

- Tested on macOS (uses 'cocoa' backend)
- Should work on other platforms with appropriate Qt backend configuration

## License

[Add your license information here]