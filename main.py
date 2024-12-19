import sys
import os
os.environ['QT_QPA_PLATFORM'] = 'cocoa'  # Add this for macOS
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend before importing pyplot

import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                           QComboBox, QLineEdit, QSlider)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

class CellAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spatial Cell Annotation Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize data structures
        self.data = None
        self.coords = None  # NumPy array for coordinates
        self.selected_points = set()
        self.drawing_path = []
        self.is_selecting = False
        self.annotations = {}
        self.point_size = 10
        self.point_alpha = 0.3  # Default transparency value
        self.x_column = None
        self.y_column = None
        self.cell_type_column = None
        self.scatter_artists = {}  # Store scatter plot artists
        self.background = None  # Store background for blitting
        self.is_panning = False
        self.pan_start = None
        self.selection_history = []  # Track selection history
        self.redo_history = []  # Track redo history
        self.annotation_history = []  # Track annotation history
        self.annotation_redo_history = []  # Track annotation redo history
        self.temp_selection = None  # Store selection state at start of brush stroke
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # Create matplotlib figure with adjusted size for legend
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        # Add space for the legend on the right
        self.figure.subplots_adjust(right=0.85)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=4)
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        layout.addWidget(control_panel, stretch=1)
        
        # Add buttons
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_data)
        control_layout.addWidget(self.load_btn)
        
        # Point size control
        size_layout = QHBoxLayout()
        self.size_label = QLabel("Point Size:")
        size_layout.addWidget(self.size_label)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(100)
        self.size_slider.setValue(self.point_size)
        self.size_slider.valueChanged.connect(self.update_point_size)
        size_layout.addWidget(self.size_slider)
        self.size_value_label = QLabel(str(self.point_size))
        size_layout.addWidget(self.size_value_label)
        control_layout.addLayout(size_layout)
        
        # Transparency control
        alpha_layout = QHBoxLayout()
        self.alpha_label = QLabel("Transparency:")
        alpha_layout.addWidget(self.alpha_label)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(int(self.point_alpha * 100))
        self.alpha_slider.valueChanged.connect(self.update_point_alpha)
        alpha_layout.addWidget(self.alpha_slider)
        self.alpha_value_label = QLabel(f"{int(self.point_alpha * 100)}%")
        alpha_layout.addWidget(self.alpha_value_label)
        control_layout.addLayout(alpha_layout)
        
        # Selection mode
        self.selection_label = QLabel("Selection Mode:")
        control_layout.addWidget(self.selection_label)
        
        self.selection_combo = QComboBox()
        self.selection_combo.addItems([
            "Lasso (Q)", 
            "Single (W)", 
            "Brush (E)", 
            "Eraser (R)"
        ])
        control_layout.addWidget(self.selection_combo)
        
        # Add brush size control (initially hidden)
        self.brush_size_layout = QHBoxLayout()
        self.brush_size_label = QLabel("Brush Size:")
        self.brush_size_layout.addWidget(self.brush_size_label)
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setMinimum(1)  # 1% of view range
        self.brush_size_slider.setMaximum(50)  # 50% of view range
        self.brush_size_slider.setValue(5)  # Default to 5% of view range
        self.brush_size_layout.addWidget(self.brush_size_slider)
        self.brush_size_value_label = QLabel("5%")
        self.brush_size_layout.addWidget(self.brush_size_value_label)
        control_layout.addLayout(self.brush_size_layout)
        
        # Connect brush size slider
        self.brush_size_slider.valueChanged.connect(self.update_brush_size)
        
        # Hide brush size controls initially
        self.brush_size_label.setVisible(False)
        self.brush_size_slider.setVisible(False)
        self.brush_size_value_label.setVisible(False)
        
        # Connect selection mode change
        self.selection_combo.currentTextChanged.connect(self.on_selection_mode_changed)
        
        # New annotation name input
        self.new_type_label = QLabel("New Annotation Name:")
        control_layout.addWidget(self.new_type_label)
        self.new_type_input = QLineEdit()
        control_layout.addWidget(self.new_type_input)
        
        # Confirm selection button
        self.confirm_btn = QPushButton("Confirm Selection")
        self.confirm_btn.clicked.connect(self.confirm_selection)
        control_layout.addWidget(self.confirm_btn)
        
        # Clear current selection button
        self.clear_btn = QPushButton("Clear Current Selection")
        self.clear_btn.clicked.connect(self.clear_selection)
        control_layout.addWidget(self.clear_btn)
        
        # Add "Undo Last Selection" button after the Clear Selection button
        self.undo_selection_btn = QPushButton("Undo Last Selection (D)")
        self.undo_selection_btn.clicked.connect(self.undo_last_selection)
        control_layout.addWidget(self.undo_selection_btn)
        
        # Add "Redo Last Selection" button after the Undo button
        self.redo_selection_btn = QPushButton("Redo Last Selection (F)")
        self.redo_selection_btn.clicked.connect(self.redo_last_selection)
        control_layout.addWidget(self.redo_selection_btn)
        
        # Remove annotation button (moved here)
        self.remove_annotation_btn = QPushButton("Remove Annotation from Selection")
        self.remove_annotation_btn.clicked.connect(self.remove_annotation)
        control_layout.addWidget(self.remove_annotation_btn)
        
        # Add annotation undo/redo buttons
        self.undo_annotation_btn = QPushButton("Undo Last Annotation")
        self.undo_annotation_btn.clicked.connect(self.undo_last_annotation)
        control_layout.addWidget(self.undo_annotation_btn)
        
        self.redo_annotation_btn = QPushButton("Redo Last Annotation")
        self.redo_annotation_btn.clicked.connect(self.redo_last_annotation)
        control_layout.addWidget(self.redo_annotation_btn)
        
        # Save all annotations button
        self.save_btn = QPushButton("Save All Annotations")
        self.save_btn.clicked.connect(self.save_annotations)
        control_layout.addWidget(self.save_btn)
        
        # Add spacer
        control_layout.addStretch()
        
        # Current annotations display
        self.annotation_label = QLabel("Current Annotations:")
        control_layout.addWidget(self.annotation_label)
        self.annotation_display = QLabel("")
        self.annotation_display.setWordWrap(True)
        control_layout.addWidget(self.annotation_display)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Add column selectors after load button
        coord_layout = QVBoxLayout()
        
        # Add cell type column selector
        cell_type_layout = QHBoxLayout()
        cell_type_layout.addWidget(QLabel("Cell Type Column:"))
        self.cell_type_combo = QComboBox()
        cell_type_layout.addWidget(self.cell_type_combo)
        coord_layout.addLayout(cell_type_layout)
        
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X Column:"))
        self.x_combo = QComboBox()
        x_layout.addWidget(self.x_combo)
        coord_layout.addLayout(x_layout)
        
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y Column:"))
        self.y_combo = QComboBox()
        y_layout.addWidget(self.y_combo)
        coord_layout.addLayout(y_layout)
        
        # Add Refresh View button right after Y Column
        self.refresh_view_btn = QPushButton("Refresh View")
        self.refresh_view_btn.clicked.connect(self.refresh_view)
        coord_layout.addWidget(self.refresh_view_btn)
        
        control_layout.insertLayout(1, coord_layout)  # Insert after load button
        
        # After setting up the canvas, add key press event connection
        self.canvas.setFocusPolicy(Qt.StrongFocus)  # Make canvas focusable
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Add tooltips to show shortcuts
        self.selection_combo.setToolTip("Shortcuts: Q (Lasso), W (Single), E (Brush), R (Eraser)")
        
        # Update brush size control tooltip
        self.brush_size_label.setToolTip("Use + and - keys to adjust size")
        self.brush_size_slider.setToolTip("Use + and - keys to adjust size")
        self.brush_size_value_label.setToolTip("Use + and - keys to adjust size")
        
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
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            # Load data efficiently
            self.data = pd.read_csv(file_name)
            self.annotations = {}
            self.selected_points.clear()
            
            # Update all column selectors
            self.x_combo.clear()
            self.y_combo.clear()
            self.cell_type_combo.clear()
            
            all_columns = self.data.columns
            self.x_combo.addItems(all_columns)
            self.y_combo.addItems(all_columns)
            self.cell_type_combo.addItems(all_columns)
            
            # Auto-detect columns
            x_col, y_col = self.detect_coordinate_columns(self.data)
            cell_type_col = self.detect_cell_type_column(self.data)
            
            # Set detected columns in combo boxes
            if x_col and y_col:
                self.x_combo.setCurrentText(x_col)
                self.y_combo.setCurrentText(y_col)
            
            if cell_type_col:
                self.cell_type_combo.setCurrentText(cell_type_col)
            
            # Disconnect previous connections if they exist
            try:
                self.x_combo.currentTextChanged.disconnect()
                self.y_combo.currentTextChanged.disconnect()
                self.cell_type_combo.currentTextChanged.disconnect()
            except:
                pass
            
            # Connect column change events
            self.x_combo.currentTextChanged.connect(self.check_and_update_plot)
            self.y_combo.currentTextChanged.connect(self.check_and_update_plot)
            self.cell_type_combo.currentTextChanged.connect(self.check_and_update_plot)
            
            # Clear the plot
            self.ax.clear()
            self.canvas.draw()
            
            # Automatically update plot if all columns were detected
            if x_col and y_col and cell_type_col:
                self.update_plot()
    
    def check_and_update_plot(self):
        # Only update plot if all three columns are selected and different from initial state
        if (self.x_combo.currentText() and 
            self.y_combo.currentText() and 
            self.cell_type_combo.currentText() and
            self.x_combo.currentIndex() != 0 and
            self.y_combo.currentIndex() != 0 and
            self.cell_type_combo.currentIndex() != 0):
            self.update_plot()
    
    def update_plot(self):
        if not all([self.x_combo.currentText(), self.y_combo.currentText(), self.cell_type_combo.currentText()]):
            return
            
        self.x_column = self.x_combo.currentText()
        self.y_column = self.y_combo.currentText()
        self.cell_type_column = self.cell_type_combo.currentText()
        
        # Store coordinates in NumPy array for faster access
        self.coords = np.column_stack((
            self.data[self.x_column].values,
            self.data[self.y_column].values
        ))
        
        self.plot_data()
    
    def plot_data(self):
        if self.data is None or not all([self.x_column, self.y_column, self.cell_type_column]):
            return
            
        # Store current view limits before clearing
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        had_previous_view = hasattr(self, 'original_xlim')
            
        self.ax.clear()
        self.scatter_artists.clear()
        
        # Use numpy operations for faster processing
        unique_types = np.unique(self.data[self.cell_type_column].values)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
        color_dict = dict(zip(unique_types, colors))
        
        annotated_indices = np.array(list(set().union(*self.annotations.values()))) if self.annotations else np.array([])
        
        # Vectorized operations for plotting
        for cell_type in unique_types:
            mask = (self.data[self.cell_type_column].values == cell_type)
            if len(annotated_indices):
                mask &= ~np.isin(np.arange(len(self.data)), annotated_indices)
            
            if np.any(mask):
                scatter = self.ax.scatter(
                    self.coords[mask, 0],
                    self.coords[mask, 1],
                    c=[color_dict[cell_type]],
                    label=f"{cell_type}",
                    alpha=self.point_alpha,  # Use the transparency value
                    s=self.point_size
                )
                self.scatter_artists[cell_type] = scatter
        
        # Plot annotations with full opacity
        for new_type, indices in self.annotations.items():
            if indices:
                idx_array = np.array(list(indices))
                scatter = self.ax.scatter(
                    self.coords[idx_array, 0],
                    self.coords[idx_array, 1],
                    label=f"New: {new_type}",
                    alpha=1.0,  # Keep annotations fully visible
                    s=self.point_size
                )
                self.scatter_artists[new_type] = scatter
        
        # Plot selected points
        if self.selected_points:
            selected_array = np.array(list(self.selected_points))
            scatter = self.ax.scatter(
                self.coords[selected_array, 0],
                self.coords[selected_array, 1],
                c='red',
                s=self.point_size*1.5,
                alpha=0.5,  # Keep selection semi-transparent
                label='Selected'
            )
            self.scatter_artists['selected'] = scatter
        
        # Create legend with fixed marker size
        legend = self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        for handle in legend.legend_handles:
            handle._sizes = [30]  # Fix the size of legend markers to 10
        
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_aspect('equal', adjustable='box')
        
        # Store original view limits only on first plot
        if not had_previous_view:
            self.original_xlim = self.ax.get_xlim()
            self.original_ylim = self.ax.get_ylim()
        else:
            # Restore the previous view limits
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
        
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
    
    def confirm_selection(self):
        if not self.selected_points:
            return
            
        new_type = self.new_type_input.text().strip()
        if not new_type:
            return
        
        # Store current state for undo
        previous_annotations = {k: v.copy() for k, v in self.annotations.items()}
        self.annotation_history.append(previous_annotations)
        self.annotation_redo_history.clear()  # Clear redo history on new annotation
            
        # Add selected points to annotations
        if new_type not in self.annotations:
            self.annotations[new_type] = set()
        self.annotations[new_type].update(self.selected_points)
        
        # Clear current selection and selection history
        self.selected_points.clear()
        self.selection_history.clear()
        self.redo_history.clear()
        self.new_type_input.clear()
        
        # Update display
        self.plot_data()
        self.update_annotation_display()
    
    def update_annotation_display(self):
        text = []
        for cell_type, indices in self.annotations.items():
            text.append(f"{cell_type}: {len(indices)} cells")
        self.annotation_display.setText("\n".join(text))
    
    def on_mouse_press(self, event):
        if event.button == 1 and event.inaxes == self.ax:  # Left click
            self.is_selecting = True
            if self.selection_combo.currentText() == "Lasso (Q)":
                self.drawing_path = [(event.xdata, event.ydata)]
            elif self.selection_combo.currentText() in ["Brush (E)", "Eraser (R)"]:
                # Store initial state when starting a brush stroke
                self.temp_selection = self.selected_points.copy()
                # Add/remove initial points under brush
                brush_points = self.get_points_in_brush(event.xdata, event.ydata)
                if brush_points:
                    if self.selection_combo.currentText() == "Eraser (R)":
                        self.selected_points.difference_update(brush_points)
                    else:
                        self.selected_points.update(brush_points)
                    self.plot_data()
        elif event.button == 3 and event.inaxes == self.ax:  # Right click
            self.is_panning = True
            self.pan_start = (event.xdata, event.ydata)
            self.canvas.setCursor(Qt.ClosedHandCursor)
    
    def on_mouse_move(self, event):
        if event.inaxes != self.ax:
            return
            
        if self.is_selecting:
            if self.selection_combo.currentText() == "Lasso (Q)":
                self.drawing_path.append((event.xdata, event.ydata))
                if self.background is not None:
                    self.canvas.restore_region(self.background)
                    path = Path(self.drawing_path)
                    patch = patches.PathPatch(path, fill=False, color='red')
                    self.ax.add_patch(patch)
                    self.ax.draw_artist(patch)
                    self.canvas.blit(self.ax.bbox)
                    patch.remove()
            elif self.selection_combo.currentText() in ["Brush (E)", "Eraser (R)"]:
                # Add or remove points under brush without adding to history
                brush_points = self.get_points_in_brush(event.xdata, event.ydata)
                if brush_points:
                    if self.selection_combo.currentText() == "Eraser (R)":
                        self.selected_points.difference_update(brush_points)
                    else:
                        self.selected_points.update(brush_points)
                    self.plot_data()
                # Draw brush preview
                self.draw_brush_preview(event.xdata, event.ydata)
        elif self.is_panning and event.xdata and event.ydata:
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        elif self.selection_combo.currentText() in ["Brush (E)", "Eraser (R)"] and event.inaxes:
            # Show brush preview while moving
            self.draw_brush_preview(event.xdata, event.ydata)
    
    def on_mouse_release(self, event):
        if event.button == 1:  # Left click release
            self.is_selecting = False
            
            selection_mode = self.selection_combo.currentText()
            
            if selection_mode == "Single (W)":
                previous_selection = self.selected_points.copy()
                distances = np.sqrt(np.sum((self.coords - [event.xdata, event.ydata])**2, axis=1))
                closest_point = np.argmin(distances)
                self.selected_points.add(closest_point)
                
                # Add to history if selection changed
                if previous_selection != self.selected_points:
                    self.selection_history.append(previous_selection)
                    self.redo_history.clear()
                    
            elif selection_mode == "Lasso (Q)" and self.drawing_path:
                previous_selection = self.selected_points.copy()
                path = Path(self.drawing_path)
                selected = path.contains_points(self.coords)
                self.selected_points.update(np.where(selected)[0])
                
                # Add to history if selection changed
                if previous_selection != self.selected_points:
                    self.selection_history.append(previous_selection)
                    self.redo_history.clear()
                    
            elif selection_mode in ["Brush (E)", "Eraser (R)"]:
                # Add to history if the brush stroke changed the selection
                if self.temp_selection != self.selected_points:
                    self.selection_history.append(self.temp_selection)
                    self.redo_history.clear()
                self.temp_selection = None
            
            self.plot_data()
            self.drawing_path = []
        elif event.button == 3:  # Right click release
            self.is_panning = False
            self.canvas.setCursor(Qt.ArrowCursor)
    
    def on_scroll(self, event):
        if not event.inaxes:
            return
            
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        
        # Get the zoom factor
        base_scale = 1.1
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1
            
        # Calculate new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        
        self.ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * rely])
        self.canvas.draw()
        # Update background after view change
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
    
    def clear_selection(self):
        # Update to clear all history when clearing selection
        self.selection_history.clear()
        self.redo_history.clear()
        self.selected_points.clear()
        self.plot_data()
    
    def save_annotations(self):
        if self.data is not None and self.annotations:
            # Create a copy of the data
            output_data = self.data.copy()
            # Add new annotation column
            output_data['new_annotation'] = 'NA'
            
            # Update annotations
            for new_type, indices in self.annotations.items():
                output_data.loc[list(indices), 'new_annotation'] = new_type
            
            # Save to file
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", "CSV Files (*.csv)")
            if file_name:
                output_data.to_csv(file_name, index=False)
    
    def update_point_size(self, value):
        self.point_size = value
        self.size_value_label.setText(str(value))
        self.plot_data()
        
    def update_point_alpha(self, value):
        """Update the transparency of the points"""
        self.point_alpha = value / 100.0
        self.alpha_value_label.setText(f"{value}%")
        self.plot_data()
    
    def remove_annotation(self):
        if not self.selected_points:
            return
        
        # Store current state for undo
        previous_annotations = {k: v.copy() for k, v in self.annotations.items()}
        self.annotation_history.append(previous_annotations)
        self.annotation_redo_history.clear()  # Clear redo history on new change
            
        # Remove selected points from all annotation sets
        for annotation_set in self.annotations.values():
            annotation_set.difference_update(self.selected_points)
            
        # Remove empty annotations
        self.annotations = {k: v for k, v in self.annotations.items() if v}
        
        # Clear current selection
        self.selected_points.clear()
        
        # Update display
        self.plot_data()
        self.update_annotation_display()
    
    def refresh_view(self):
        """Reset the view to show all current data points"""
        if self.coords is not None and len(self.coords) > 0:
            # Add 5% padding to the limits
            padding = 0.05
            
            x_min, x_max = self.coords[:, 0].min(), self.coords[:, 0].max()
            y_min, y_max = self.coords[:, 1].min(), self.coords[:, 1].max()
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            self.ax.set_xlim([x_min - x_range * padding, 
                             x_max + x_range * padding])
            self.ax.set_ylim([y_min - y_range * padding, 
                             y_max + y_range * padding])
            
            self.canvas.draw()
            # Update background after view change
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
    
    def undo_last_selection(self):
        """Undo the last selection action"""
        # Only proceed if canvas has focus and there's history to undo
        if self.canvas.hasFocus() and self.selection_history:
            current_selection = self.selected_points.copy()
            self.selected_points = self.selection_history.pop()
            self.redo_history.append(current_selection)
            self.plot_data()

    def redo_last_selection(self):
        """Redo the last undone selection action"""
        # Only proceed if canvas has focus and there's history to redo
        if self.canvas.hasFocus() and self.redo_history:
            current_selection = self.selected_points.copy()
            self.selected_points = self.redo_history.pop()
            self.selection_history.append(current_selection)
            self.plot_data()

    def undo_last_annotation(self):
        """Undo the last annotation action"""
        if self.annotation_history:
            current_annotations = {k: v.copy() for k, v in self.annotations.items()}
            self.annotation_redo_history.append(current_annotations)
            self.annotations = self.annotation_history.pop()
            self.plot_data()
            self.update_annotation_display()

    def redo_last_annotation(self):
        """Redo the last undone annotation action"""
        if self.annotation_redo_history:
            current_annotations = {k: v.copy() for k, v in self.annotations.items()}
            self.annotation_history.append(current_annotations)
            self.annotations = self.annotation_redo_history.pop()
            self.plot_data()
            self.update_annotation_display()

    def update_brush_size(self, value):
        """Update the brush size value"""
        self.brush_size_value_label.setText(str(value))

    def on_selection_mode_changed(self, mode):
        """Handle selection mode changes"""
        # Show/hide brush size controls based on mode
        is_brush_mode = mode in ["Brush (E)", "Eraser (R)"]
        self.brush_size_label.setVisible(is_brush_mode)
        self.brush_size_slider.setVisible(is_brush_mode)
        self.brush_size_value_label.setVisible(is_brush_mode)

    def get_points_in_brush(self, x, y):
        """Get points within brush radius"""
        if self.coords is None:
            return set()
        
        # Get the current view limits to scale the brush size appropriately
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Scale brush size relative to the view extent (e.g. 1% of the view range)
        view_range = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
        scaled_brush_size = (self.brush_size_slider.value() / 100.0) * view_range
        
        # Calculate distances and return points within brush radius
        distances = np.sqrt(np.sum((self.coords - [x, y])**2, axis=1))
        return set(np.where(distances <= scaled_brush_size)[0])

    def draw_brush_preview(self, x, y):
        """Draw brush preview circle"""
        if self.background is not None and x is not None and y is not None:
            self.canvas.restore_region(self.background)
            
            # Scale brush size same as in get_points_in_brush
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            view_range = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
            scaled_brush_size = (self.brush_size_slider.value() / 100.0) * view_range
            
            # Use red for brush mode, blue for eraser mode
            color = 'blue' if self.selection_combo.currentText() == "Eraser (R)" else 'red'
            circle = plt.Circle((x, y), scaled_brush_size, fill=False, color=color)
            self.ax.add_patch(circle)
            self.ax.draw_artist(circle)
            self.canvas.blit(self.ax.bbox)
            circle.remove()

    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if not self.canvas.hasFocus():
            return
        
        if event.key == 'd':  # Undo last selection
            self.undo_last_selection()
        elif event.key == 'f':  # Redo last selection
            self.redo_last_selection()
        # Add selection mode shortcuts
        elif event.key == 'q':  # Lasso selection
            self.selection_combo.setCurrentText("Lasso (Q)")
        elif event.key == 'w':  # Single selection
            self.selection_combo.setCurrentText("Single (W)")
        elif event.key == 'e':  # Brush selection
            self.selection_combo.setCurrentText("Brush (E)")
        elif event.key == 'r':  # Eraser
            self.selection_combo.setCurrentText("Eraser (R)")
        # Add brush size shortcuts
        elif event.key in ['+', '=']:  # Increase brush size (both + and = keys work)
            if self.selection_combo.currentText() in ["Brush (E)", "Eraser (R)"]:
                current_size = self.brush_size_slider.value()
                new_size = min(current_size + 1, self.brush_size_slider.maximum())
                self.brush_size_slider.setValue(new_size)
        elif event.key == '-':  # Decrease brush size
            if self.selection_combo.currentText() in ["Brush (E)", "Eraser (R)"]:
                current_size = self.brush_size_slider.value()
                new_size = max(current_size - 1, self.brush_size_slider.minimum())
                self.brush_size_slider.setValue(new_size)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CellAnnotationTool()
    window.show()
    sys.exit(app.exec_()) 