# --- START OF FILE selection_operations.py ---
from PyQt5.QtCore import Qt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt

class SelectionOperations:
    def __init__(self, main_window):
        self.main_window = main_window

    def confirm_selection(self):
        if not self.main_window.selected_points:
            return

        new_type = self.main_window.new_type_input.text().strip()
        if not new_type:
            return

        # Store current state for undo
        previous_annotations = {k: v.copy() for k, v in self.main_window.annotations.items()}
        self.main_window.annotation_history.append(previous_annotations)
        self.main_window.annotation_redo_history.clear()  # Clear redo history on new annotation

        # Add selected points to annotations
        if new_type not in self.main_window.annotations:
            self.main_window.annotations[new_type] = set()
        self.main_window.annotations[new_type].update(self.main_window.selected_points)

        # Clear current selection and selection history
        self.main_window.selected_points.clear()
        self.main_window.selection_history.clear()
        self.main_window.redo_history.clear()
        self.main_window.new_type_input.clear()

        # Update display
        self.main_window.plot_ops.plot_data()
        self.main_window.update_annotation_display()

    def on_mouse_press(self, event):
        if event.button == 1 and event.inaxes == self.main_window.plot_ops.ax:  # Left click
            self.main_window.is_selecting = True
            if self.main_window.selection_combo.currentText() == "Lasso (Q)":
                self.main_window.drawing_path = [(event.xdata, event.ydata)]
            elif self.main_window.selection_combo.currentText() in ["Brush (E)", "Eraser (R)"]:
                # Store initial state when starting a brush stroke
                self.main_window.temp_selection = self.main_window.selected_points.copy()
                # Add/remove initial points under brush (respecting visibility)
                brush_points = self.get_points_in_brush(event.xdata, event.ydata)
                if brush_points:
                    if self.main_window.selection_combo.currentText() == "Eraser (R)":
                        self.main_window.selected_points.difference_update(brush_points)
                    else:
                        self.main_window.selected_points.update(brush_points)
                    self.main_window.plot_ops.plot_data()
            elif self.main_window.selection_combo.currentText() == "Single (W)":
                # For single selection, only select from visible points
                distances = np.sqrt(np.sum((self.main_window.coords - [event.xdata, event.ydata])**2, axis=1))
                if not self.main_window.show_annotated:
                    # Mask out annotated points
                    annotated_points = set().union(*self.main_window.annotations.values()) if self.main_window.annotations else set()
                    distances[list(annotated_points)] = np.inf
                closest_point = np.argmin(distances)
                if distances[closest_point] != np.inf:  # Only select if a valid point was found
                    self.main_window.selected_points.add(closest_point)
                    self.main_window.plot_ops.plot_data()
        elif event.button == 3 and event.inaxes == self.main_window.plot_ops.ax:  # Right click
            self.main_window.is_panning = True
            self.main_window.pan_start = (event.xdata, event.ydata)
            self.main_window.plot_ops.canvas.setCursor(Qt.ClosedHandCursor)

    def on_mouse_move(self, event):
        if event.inaxes != self.main_window.plot_ops.ax:
            return

        if self.main_window.is_selecting:
            if self.main_window.selection_combo.currentText() == "Lasso (Q)":
                self.main_window.drawing_path.append((event.xdata, event.ydata))
                if self.main_window.background is not None:
                    self.main_window.plot_ops.canvas.restore_region(self.main_window.background)
                    path = Path(self.main_window.drawing_path)
                    patch = patches.PathPatch(path, fill=False, color='red')
                    self.main_window.plot_ops.ax.add_patch(patch)
                    self.main_window.plot_ops.ax.draw_artist(patch)
                    self.main_window.plot_ops.canvas.blit(self.main_window.plot_ops.ax.bbox)
                    patch.remove()
            elif self.main_window.selection_combo.currentText() in ["Brush (E)", "Eraser (R)"]:
                # Add or remove points under brush without adding to history
                brush_points = self.get_points_in_brush(event.xdata, event.ydata)
                if brush_points:
                    if self.main_window.selection_combo.currentText() == "Eraser (R)":
                        self.main_window.selected_points.difference_update(brush_points)
                    else:
                        self.main_window.selected_points.update(brush_points)
                    self.main_window.plot_ops.plot_data()
                # Draw brush preview
                self.draw_brush_preview(event.xdata, event.ydata)
        elif self.main_window.is_panning and event.xdata and event.ydata:
            dx = event.xdata - self.main_window.pan_start[0]
            dy = event.ydata - self.main_window.pan_start[1]

            xlim = self.main_window.plot_ops.ax.get_xlim()
            ylim = self.main_window.plot_ops.ax.get_ylim()

            self.main_window.plot_ops.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.main_window.plot_ops.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.main_window.plot_ops.canvas.draw()
            self.main_window.background = self.main_window.plot_ops.canvas.copy_from_bbox(self.main_window.plot_ops.ax.bbox)
        elif self.main_window.selection_combo.currentText() in ["Brush (E)", "Eraser (R)"] and event.inaxes:
            # Show brush preview while moving
            self.draw_brush_preview(event.xdata, event.ydata)

    def on_mouse_release(self, event):
        if event.button == 1:  # Left click release
            self.main_window.is_selecting = False

            selection_mode = self.main_window.selection_combo.currentText()

            if selection_mode == "Single (W)":
                # Always store the previous state before making a new selection
                previous_selection = self.main_window.selected_points.copy()
                distances = np.sqrt(np.sum((self.main_window.coords - [event.xdata, event.ydata])**2, axis=1))

                # Filter out annotated points if they're hidden
                if not self.main_window.show_annotated:
                    annotated_points = set().union(*self.main_window.annotations.values()) if self.main_window.annotations else set()
                    distances[list(annotated_points)] = np.inf

                closest_point = np.argmin(distances)
                if distances[closest_point] != np.inf:  # Only select if a valid point was found
                    self.main_window.selected_points.add(closest_point)
                    # Always add to history for Single selection mode
                    self.main_window.selection_history.append(previous_selection)
                    self.main_window.redo_history.clear()

            elif selection_mode == "Lasso (Q)" and self.main_window.drawing_path:
                previous_selection = self.main_window.selected_points.copy()
                path = Path(self.main_window.drawing_path)
                selected = path.contains_points(self.main_window.coords)

                # Filter out annotated points if they're hidden
                if not self.main_window.show_annotated:
                    annotated_points = set().union(*self.main_window.annotations.values()) if self.main_window.annotations else set()
                    selected_indices = set(np.where(selected)[0]) - annotated_points
                else:
                    selected_indices = set(np.where(selected)[0])

                self.main_window.selected_points.update(selected_indices)

                # Add to history if selection changed
                if previous_selection != self.main_window.selected_points:
                    self.main_window.selection_history.append(previous_selection)
                    self.main_window.redo_history.clear()

            elif selection_mode in ["Brush (E)", "Eraser (R)"]:
                # Add to history if the brush stroke changed the selection
                if self.main_window.temp_selection != self.main_window.selected_points:
                    self.main_window.selection_history.append(self.main_window.temp_selection)
                    self.main_window.redo_history.clear()
                self.main_window.temp_selection = None

            self.main_window.plot_ops.plot_data()
            self.main_window.drawing_path = []
        elif event.button == 3:  # Right click release
            self.main_window.is_panning = False
            self.main_window.plot_ops.canvas.setCursor(Qt.ArrowCursor)

    def clear_selection(self):
        # Update to clear all history when clearing selection
        self.main_window.selection_history.clear()
        self.main_window.redo_history.clear()
        self.main_window.selected_points.clear()
        self.main_window.plot_ops.plot_data()

    def remove_annotation(self):
        if not self.main_window.selected_points:
            return

        # Store current state for undo
        previous_annotations = {k: v.copy() for k, v in self.main_window.annotations.items()}
        self.main_window.annotation_history.append(previous_annotations)
        self.main_window.annotation_redo_history.clear()  # Clear redo history on new change

        # Remove selected points from all annotation sets
        for annotation_set in self.main_window.annotations.values():
            annotation_set.difference_update(self.main_window.selected_points)

        # Remove empty annotations
        self.main_window.annotations = {k: v for k, v in self.main_window.annotations.items() if v}

        # Clear current selection
        self.main_window.selected_points.clear()

        # Update display
        self.main_window.plot_ops.plot_data()
        self.main_window.update_annotation_display()

    def undo_last_selection(self):
        """Undo the last selection action"""
        if self.main_window.selection_history:
            current_selection = self.main_window.selected_points.copy()
            self.main_window.selected_points = self.main_window.selection_history.pop()
            self.main_window.redo_history.append(current_selection)
            self.main_window.plot_ops.plot_data()

    def redo_last_selection(self):
        """Redo the last undone selection action"""
        if self.main_window.redo_history:
            current_selection = self.main_window.selected_points.copy()
            self.main_window.selected_points = self.main_window.redo_history.pop()
            self.main_window.selection_history.append(current_selection)
            self.main_window.plot_ops.plot_data()

    def undo_last_annotation(self):
        """Undo the last annotation action"""
        if self.main_window.annotation_history:
            current_annotations = {k: v.copy() for k, v in self.main_window.annotations.items()}
            self.main_window.annotation_redo_history.append(current_annotations)
            self.main_window.annotations = self.main_window.annotation_history.pop()
            self.main_window.plot_ops.plot_data()
            self.main_window.update_annotation_display()

    def redo_last_annotation(self):
        """Redo the last undone annotation action"""
        if self.main_window.annotation_redo_history:
            current_annotations = {k: v.copy() for k, v in self.main_window.annotations.items()}
            self.main_window.annotation_history.append(current_annotations)
            self.main_window.annotations = self.main_window.annotation_redo_history.pop()
            self.main_window.plot_ops.plot_data()
            self.main_window.update_annotation_display()

    def update_brush_size(self, value):
        """Update the brush size value"""
        self.main_window.brush_size_value_label.setText(str(value))

    def on_selection_mode_changed(self, mode):
        """Handle selection mode changes"""
        # Show/hide brush size controls based on mode
        is_brush_mode = mode in ["Brush (E)", "Eraser (R)"]
        self.main_window.brush_size_label.setVisible(is_brush_mode)
        self.main_window.brush_size_slider.setVisible(is_brush_mode)
        self.main_window.brush_size_value_label.setVisible(is_brush_mode)

    def get_points_in_brush(self, x, y):
        """Get points within brush radius"""
        if self.main_window.coords is None:
            return set()

        # Get the current view limits to scale the brush size appropriately
        xlim = self.main_window.plot_ops.ax.get_xlim()
        ylim = self.main_window.plot_ops.ax.get_ylim()

        # Scale brush size relative to the view extent
        view_range = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
        scaled_brush_size = (self.main_window.brush_size_slider.value() / 100.0) * view_range

        # Calculate distances
        distances = np.sqrt(np.sum((self.main_window.coords - [x, y])**2, axis=1))
        points_in_range = set(np.where(distances <= scaled_brush_size)[0])

        # If annotated cells are hidden, remove them from selectable points
        if not self.main_window.show_annotated:
            annotated_points = set().union(*self.main_window.annotations.values()) if self.main_window.annotations else set()
            points_in_range = points_in_range - annotated_points

        return points_in_range

    def draw_brush_preview(self, x, y):
        """Draw brush preview circle"""
        if self.main_window.background is not None and x is not None and y is not None:
            self.main_window.plot_ops.canvas.restore_region(self.main_window.background)

            # Scale brush size same as in get_points_in_brush
            xlim = self.main_window.plot_ops.ax.get_xlim()
            ylim = self.main_window.plot_ops.ax.get_ylim()
            view_range = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
            scaled_brush_size = (self.main_window.brush_size_slider.value() / 100.0) * view_range

            # Use red for brush mode, blue for eraser mode
            color = 'blue' if self.main_window.selection_combo.currentText() == "Eraser (R)" else 'red'
            circle = plt.Circle((x, y), scaled_brush_size, fill=False, color=color)
            self.main_window.plot_ops.ax.add_patch(circle)
            self.main_window.plot_ops.ax.draw_artist(circle)
            self.main_window.plot_ops.canvas.blit(self.main_window.plot_ops.ax.bbox)
            circle.remove()

# --- END OF FILE selection_operations.py ---