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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.path import Path
import matplotlib.patches as patches

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
        self.point_size = 50
        self.x_column = None
        self.y_column = None
        self.cell_type_column = None
        self.scatter_artists = {}  # Store scatter plot artists
        self.background = None  # Store background for blitting
        
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
        self.size_slider.setMinimum(10)
        self.size_slider.setMaximum(200)
        self.size_slider.setValue(self.point_size)
        self.size_slider.valueChanged.connect(self.update_point_size)
        size_layout.addWidget(self.size_slider)
        self.size_value_label = QLabel(str(self.point_size))
        size_layout.addWidget(self.size_value_label)
        control_layout.addLayout(size_layout)
        
        # Selection mode
        self.selection_label = QLabel("Selection Mode:")
        control_layout.addWidget(self.selection_label)
        self.selection_combo = QComboBox()
        self.selection_combo.addItems([ "Lasso", "Single"])
        control_layout.addWidget(self.selection_combo)
        
        # New cell type input
        self.new_type_label = QLabel("New Cell Type Name:")
        control_layout.addWidget(self.new_type_label)
        self.new_type_input = QLineEdit()
        control_layout.addWidget(self.new_type_input)
        
        # Confirm selection button
        self.confirm_btn = QPushButton("Confirm Selection")
        self.confirm_btn.clicked.connect(self.confirm_selection)
        control_layout.addWidget(self.confirm_btn)
        
        # Remove annotation button
        self.remove_annotation_btn = QPushButton("Remove Annotation from Selection")
        self.remove_annotation_btn.clicked.connect(self.remove_annotation)
        control_layout.addWidget(self.remove_annotation_btn)
        
        # Clear current selection button
        self.clear_btn = QPushButton("Clear Current Selection")
        self.clear_btn.clicked.connect(self.clear_selection)
        control_layout.addWidget(self.clear_btn)
        
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
        
        control_layout.insertLayout(1, coord_layout)  # Insert after load button
        
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
                    label=f"Original: {cell_type}",
                    alpha=0.3,
                    s=self.point_size
                )
                self.scatter_artists[cell_type] = scatter
        
        # Plot annotations
        for new_type, indices in self.annotations.items():
            if indices:
                idx_array = np.array(list(indices))
                scatter = self.ax.scatter(
                    self.coords[idx_array, 0],
                    self.coords[idx_array, 1],
                    label=f"New: {new_type}",
                    alpha=1.0,
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
                alpha=0.5,
                label='Selected'
            )
            self.scatter_artists['selected'] = scatter
        
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_aspect('equal', adjustable='box')
        
        self.canvas.draw()
        # Store background for blitting
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
    
    def confirm_selection(self):
        if not self.selected_points:
            return
            
        new_type = self.new_type_input.text().strip()
        if not new_type:
            return
            
        # Add selected points to annotations
        if new_type not in self.annotations:
            self.annotations[new_type] = set()
        self.annotations[new_type].update(self.selected_points)
        
        # Clear current selection
        self.selected_points.clear()
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
        if event.button == 1 and event.inaxes == self.ax:
            self.is_selecting = True
            self.drawing_path = [(event.xdata, event.ydata)]
            
    def on_mouse_move(self, event):
        if self.is_selecting and event.inaxes == self.ax:
            if self.selection_combo.currentText() == "Lasso":
                self.drawing_path.append((event.xdata, event.ydata))
                # Efficient redraw using blitting
                if self.background is not None:
                    self.canvas.restore_region(self.background)
                    path = Path(self.drawing_path)
                    patch = patches.PathPatch(path, fill=False, color='red')
                    self.ax.add_patch(patch)
                    self.ax.draw_artist(patch)
                    self.canvas.blit(self.ax.bbox)
    
    def on_mouse_release(self, event):
        if event.button == 1 and event.inaxes == self.ax:
            self.is_selecting = False
            if self.selection_combo.currentText() == "Single":
                # Vectorized distance calculation
                distances = np.sqrt(np.sum((self.coords - [event.xdata, event.ydata])**2, axis=1))
                closest_point = np.argmin(distances)
                self.selected_points.add(closest_point)
            else:
                # Efficient lasso selection
                path = Path(self.drawing_path)
                selected = path.contains_points(self.coords)
                self.selected_points.update(np.where(selected)[0])
            
            self.plot_data()
            self.drawing_path = []
    
    def clear_selection(self):
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
        
    def remove_annotation(self):
        if not self.selected_points:
            return
            
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CellAnnotationTool()
    window.show()
    sys.exit(app.exec_()) 