
# The meaning of this app

Manually annotating just a few cells can significantly improve the performance of large language models (LLMs) in identifying spatial niches. While LLMs possess some foundational biological knowledge, it may not be adequate for specialized research topics. This is because the background knowledge may not encompass all the information within the data, and conflicts can arise between real data and pre-existing knowledge. In such cases, relying solely on the pre-trained knowledge of an LLM may be insufficient. By training with human-annotated cells, LLMs can learn spatial niche patterns and subsequently predict the niches of unannotated cells.

Conversely, LLMs can also provide valuable insights for human scientists, enhancing their understanding of cellular spatial context. LLMs excel at complex information retrieval, such as integrating cell type and marker gene lists from their vast knowledge base, offering human scientists relevant information for reference.

Therefore, this application is designed to bridge this gap. It enables human scientists to annotate cells and utilize those annotations to train LLMs for spatial niche identification. Concurrently, it leverages LLMs to assist scientists in comprehending the spatial context of cells.

# Introduction

The application is a Spatial Cell Annotation Tool built with PyQt, designed for visualizing and annotating spatial data. Here's a breakdown of its functionalities: 

**1. User Interface Elements:**
- **Menu Bar:** Provides access to various functionalities through menus like "File", "Edit", "View", "Tools", and "Analysis".
- **Control Panel:** Contains buttons, sliders, and dropdown menus for controlling various aspects of the application, such as data loading, visualization settings, and annotation tools.
- **Matplotlib Canvas:** Displays the scatter plot of the data.


**2. Data Loading and Handling:**
- **Load Data:** Allows the user to load data from a file (CSV) using the "Load Data" button or Ctrl+O.
- **Column Selection:** Enables the user to specify which columns in the loaded data correspond to the X coordinates, Y coordinates, and cell type information using dropdown menus ("X Column", "Y Column", "Cell Type Column").

**3. Data Visualization:**
- **Scatter Plot:** Visualizes the data points as a scatter plot using Matplotlib.
- **Point Size Control:** Allows adjusting the size of the data points on the plot using a slider.
- **Transparency Control:** Enables adjusting the transparency (alpha) of the data points using a slider.
- **Color by Continuous Variables:** Provides the ability to color the data points based on the values in specified continuous variable columns (Red, Green, Blue channels). Users can select these columns from dropdown menus with autocompletion.
- **Refresh View:** Updates the plot based on changes to data, column selections, or visualization settings using the "Refresh View" button or F5.

**4. Cell Annotation:**
- **Selection Modes:** Offers different modes for selecting cells:
    - **Lasso:** Allows freehand selection of points.
    - **Single:** Enables selecting individual points.
    - **Brush:** Provides a brush tool to select points within a certain radius.
    - **Eraser:** Functions as a brush to deselect points.
- **Brush Size Control:** When in "Brush" or "Eraser" mode, a slider appears to control the size of the brush.
- **Annotating Selection:** After selecting cells, users can enter a name for the annotation in the "New Annotation Name" field and confirm the annotation using the confirm button.
- **Visual Feedback:** Selected cells are visually distinguished on the plot.
- **Annotation Display:** Shows a list of current annotations and the number of cells in each annotation.
- **Toggle Annotated Cells:** Allows showing or hiding the annotated cells on the plot using the "Hide Annotated Cells" button or Ctrl+T.

**5. Annotation Management:**
- **Clear Current Selection:** Deselects any currently selected cells using the "Clear Current Selection" button.
- **Remove Annotation from Selection:** Removes the annotation from the currently selected cells.
- **Undo/Redo Selection:** Allows undoing and redoing the last selection action using the "Undo Last Selection" and "Redo Last Selection" buttons or the 'D' and 'F' keys respectively.
- **Undo/Redo Annotation:** Enables undoing and redoing the last annotation action using the "Undo Last Annotation" and "Redo Last Annotation" buttons or Ctrl+Z and Ctrl+Y respectively.
- **Saving Annotations:** Saves all annotations to a file using the "Save All Annotations" button or Ctrl+S.
- **Saving/Loading Annotation State:** Allows saving and loading the current annotation state (annotations and selections) using the "Save Annotation State" and "Load Annotation State" buttons or Ctrl+Shift+S and Ctrl+Shift+O respectively.

**6. Analysis:**
- **Open Analysis Window:** Opens a separate window to perform analysis on the annotated data using the "Analysis" button or Ctrl+A.
- **Cell Type Proportion Analysis:** Displays a table showing the proportions of different cell types within the entire dataset, the current selection, and each annotation.
- **Gene Expression Heatmap:** Generates a heatmap visualizing the mean expression levels of user-specified genes across different annotations or the current selection.
- **AI-Powered Analysis:** Integrates with Google's Gemini AI model to provide insights into the biological context of the selected cells or annotations based on user-provided domain knowledge. Users can preview the prompt sent to the AI and view the AI's analysis results in a formatted manner.

**7. Shortcuts:**
- The application implements various keyboard shortcuts for quicker access to common functions.

In summary, this application provides a comprehensive set of tools for visualizing, annotating, and performing basic and AI-powered analysis on spatial cell data. It combines data loading, interactive plotting, multiple selection methods, annotation management, and advanced analytical capabilities within a user-friendly graphical interface.

