# --- START OF FILE data_operations.py ---
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QFileDialog

class DataOperations:
    def __init__(self, main_window):
        self.main_window = main_window

    def detect_coordinate_columns(self, df):
        """
        Detect likely X and Y coordinate columns based on column names first, then numeric data.
        Returns tuple of (x_col, y_col) or (None, None) if not found.
        """
        # Look for common coordinate column names
        x_keywords = ['x', 'col']
        y_keywords = ['y', 'row']

        x_candidates = []
        y_candidates = []

        # First pass: look for exact matches (case insensitive)
        for col in df.columns:
            col_lower = col.lower()
            # Check if column name exactly matches x or y
            if col_lower in x_keywords:
                x_candidates.append(col)
            elif col_lower in y_keywords:
                y_candidates.append(col)

        # Second pass: look for columns containing the keywords
        if not (x_candidates and y_candidates):
            for col in df.columns:
                col_lower = col.lower()
                # Check if column name contains coordinate keywords
                if any(k in col_lower for k in x_keywords):
                    x_candidates.append(col)
                if any(k in col_lower for k in y_keywords):
                    y_candidates.append(col)

        # If we found candidates by name, use the first ones
        if x_candidates and y_candidates:
            return x_candidates[0], y_candidates[0]

        # Fallback: find numeric columns with highest variance
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            variances = df[numeric_cols].var()
            coord_cols = variances.nlargest(2).index.tolist()
            return coord_cols[0], coord_cols[1]

        return None, None

    def detect_cell_type_column(self, df):
        """
        Detect likely cell type column based on column name first, then number of unique values.
        Returns column name or None if not found.
        """
        type_keywords = ['type', 'class', 'cluster']

        # First look for columns with type-related names
        type_candidates = []
        for col in df.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in type_keywords):
                # Check if the column has a reasonable number of unique values
                unique_vals = df[col].nunique()
                if 1 < unique_vals < 25:
                    return col
                type_candidates.append(col)

        # If no suitable named columns found, look for any column with appropriate number of unique values
        for col in df.columns:
            if col not in type_candidates:  # Skip already checked columns
                unique_vals = df[col].nunique()
                if 1 < unique_vals < 25:
                    return col

        return None

    def load_data(self):
        # print("Starting data load...")
        file_name, _ = QFileDialog.getOpenFileName(self.main_window, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            # print(f"Loading file: {file_name}")

            # Load data efficiently
            # print("Reading CSV...")
            self.main_window.data = pd.read_csv(file_name)
            # print(f"Loaded {len(self.main_window.data)} rows")

            # print("Clearing existing data structures...")
            self.main_window.annotations = {}
            self.main_window.selected_points.clear()

            # print("Updating column selectors...")
            self.main_window.x_combo.blockSignals(True)
            self.main_window.y_combo.blockSignals(True)
            self.main_window.cell_type_combo.blockSignals(True)

            self.main_window.x_combo.clear()
            self.main_window.y_combo.clear()
            self.main_window.cell_type_combo.clear()

            all_columns = self.main_window.data.columns
            # print(f"Found {len(all_columns)} columns")
            self.main_window.x_combo.addItems(all_columns)
            self.main_window.y_combo.addItems(all_columns)
            self.main_window.cell_type_combo.addItems(all_columns)

            # print("Auto-detecting columns...")
            x_col, y_col = self.detect_coordinate_columns(self.main_window.data)
            cell_type_col = self.detect_cell_type_column(self.main_window.data)
            # print(f"Detected columns - X: {x_col}, Y: {y_col}, Cell Type: {cell_type_col}")

            # print("Setting detected columns...")
            if x_col and y_col:
                self.main_window.x_combo.setCurrentText(x_col)
                self.main_window.y_combo.setCurrentText(y_col)

            if cell_type_col:
                self.main_window.cell_type_combo.setCurrentText(cell_type_col)

            self.main_window.x_combo.blockSignals(False)
            self.main_window.y_combo.blockSignals(False)
            self.main_window.cell_type_combo.blockSignals(False)

            # print("Clearing plot...")
            self.main_window.plot_ops.ax.clear()
            self.main_window.plot_ops.canvas.draw()

            # print("Checking for automatic plot update...")
            if x_col and y_col and cell_type_col:
                # print("Updating plot with detected columns...")
                self.main_window.plot_ops.update_plot()

            # print("Updating continuous variable selectors...")
            numeric_columns = self.main_window.data.select_dtypes(include=[np.number]).columns
            # print(f"Found {len(numeric_columns)} numeric columns")
            for combo in [self.main_window.red_combo, self.main_window.green_combo, self.main_window.blue_combo]:
                combo.clear()
                combo.addItem("None")
                combo.addItems(numeric_columns)

            # print("Data loading complete!")

    def save_annotations(self):
        if self.main_window.data is not None and self.main_window.annotations:
            # Create a copy of the data
            output_data = self.main_window.data.copy()
            # Add new annotation column
            output_data['new_annotation'] = 'NA'

            # Update annotations
            for new_type, indices in self.main_window.annotations.items():
                output_data.loc[list(indices), 'new_annotation'] = new_type

            # Save to file
            file_name, _ = QFileDialog.getSaveFileName(self.main_window, "Save Annotations", "", "CSV Files (*.csv)")
            if file_name:
                output_data.to_csv(file_name, index=False)

    def save_annotation_state(self):
        """Save current annotation state to a CSV file"""
        if self.main_window.data is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Save Annotation State",
            "",
            "CSV Files (*.csv)"
        )

        if file_name:
            try:
                # Create a DataFrame with cell indices and their annotations
                state_data = []
                for annotation_type, indices in self.main_window.annotations.items():
                    for idx in indices:
                        state_data.append({
                            'cell_index': idx,
                            'annotation': annotation_type
                        })

                if state_data:
                    state_df = pd.DataFrame(state_data)
                    state_df.to_csv(file_name, index=False)
                    # print(f"Annotation state saved to {file_name}")
                else:
                    print("No annotations to save")

            except Exception as e:
                print(f"Error saving annotation state: {e}")

    def load_annotation_state(self):
        """Load annotation state from a CSV file"""
        if self.main_window.data is None:
            return

        file_name, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Load Annotation State",
            "",
            "CSV Files (*.csv)"
        )

        if file_name:
            try:
                # Load the state file
                state_df = pd.read_csv(file_name)

                # Clear current annotations
                self.main_window.annotations = {}

                # Rebuild annotations dictionary
                for _, row in state_df.iterrows():
                    annotation_type = row['annotation']
                    cell_index = int(row['cell_index'])

                    # Initialize the set if this is a new annotation type
                    if annotation_type not in self.main_window.annotations:
                        self.main_window.annotations[annotation_type] = set()

                    # Add the cell index to the appropriate set
                    self.main_window.annotations[annotation_type].add(cell_index)

                # Update the display
                self.main_window.plot_ops.plot_data()
                self.main_window.update_annotation_display()
                # print(f"Annotation state loaded from {file_name}")

            except Exception as e:
                print(f"Error loading annotation state: {e}")

# --- END OF FILE data_operations.py ---