# --- START OF FILE plotting_operations.py ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.preprocessing import MinMaxScaler
from PyQt5.QtWidgets import QMessageBox

class PlottingOperations:
    def __init__(self, main_window):
        self.main_window = main_window
        self.figure = None
        self.ax = None
        self.canvas = None
        self.original_xlim = None
        self.original_ylim = None
        self.FigureCanvas = FigureCanvas

    def check_and_update_plot(self):
        """Only update plot if coordinates are selected"""
        if (self.main_window.x_combo.currentText() and
            self.main_window.y_combo.currentText()):
            self.update_plot()

    def update_plot(self):
        # print("Starting plot update...")
        if not all([self.main_window.x_combo.currentText(), self.main_window.y_combo.currentText(), self.main_window.cell_type_combo.currentText()]):
            print("Missing required columns, skipping plot update")
            return

        self.main_window.x_column = self.main_window.x_combo.currentText()
        self.main_window.y_column = self.main_window.y_combo.currentText()
        self.main_window.cell_type_column = self.main_window.cell_type_combo.currentText()

        # Check number of unique colors before plotting
        unique_types = np.unique(self.main_window.data[self.main_window.cell_type_column].values)
        if len(unique_types) > 100:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                f"The selected column '{self.main_window.cell_type_column}' has {len(unique_types)} unique values, which is too many to plot effectively. Please select a different column or filter your data."
            )
            return

        # print("Storing coordinates...")
        self.main_window.coords = np.column_stack((
            self.main_window.data[self.main_window.x_column].values,
            self.main_window.data[self.main_window.y_column].values
        ))

        # print("Calling plot_data...")
        self.plot_data()
        # print("Plot update complete!")

    def plot_data(self):
        # print("Starting plot_data...")
        if self.main_window.data is None or not all([self.main_window.x_column, self.main_window.y_column]):
            print("Missing required data, skipping plot")
            return

        try:
            # print("Setting up plot...")
            # Store current view limits before clearing
            try:
                current_xlim = self.ax.get_xlim()
                current_ylim = self.ax.get_ylim()
                had_previous_view = True
                # print("Stored previous view limits")
            except:
                had_previous_view = False
                print("No previous view limits")

            # print("Clearing plot...")
            self.ax.clear()
            self.main_window.scatter_artists.clear()

            # print("Checking RGB colors...")
            rgb_colors = self.get_rgb_colors()

            if rgb_colors is not None:
                # print("Plotting with RGB colors...")
                # Set dark grey background for RGB mode
                self.ax.set_facecolor('#333333')
                self.figure.set_facecolor('#333333')

                # Plot all points with RGB colors
                non_annotated = set(range(len(self.main_window.data))) - set().union(*self.main_window.annotations.values())
                if non_annotated:
                    non_annotated = list(non_annotated)
                    scatter = self.ax.scatter(
                        self.main_window.coords[non_annotated, 0],
                        self.main_window.coords[non_annotated, 1],
                        c=rgb_colors[non_annotated],
                        alpha=self.main_window.point_alpha,
                        s=self.main_window.point_size
                    )
                    self.main_window.scatter_artists['rgb'] = scatter

                # Use white for axis labels and ticks
                self.ax.tick_params(colors='white')
                for spine in self.ax.spines.values():
                    spine.set_color('white')
            else:
                # print("Plotting with categorical colors...")
                # Reset to default white background for categorical coloring
                self.ax.set_facecolor('white')
                self.figure.set_facecolor('white')

                # Original coloring by cell type
                unique_types = np.unique(self.main_window.data[self.main_window.cell_type_column].values)
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
                color_dict = dict(zip(unique_types, colors))

                annotated_indices = np.array(list(set().union(*self.main_window.annotations.values()))) if self.main_window.annotations else np.array([])

                # Get selected cell types
                selected_types = self.main_window.get_selected_cell_types()

                for cell_type in unique_types:
                    # Skip if not in selected types
                    if selected_types is not None and str(cell_type) not in selected_types:
                        continue

                    mask = (self.main_window.data[self.main_window.cell_type_column].values == cell_type)
                    if len(annotated_indices):
                        mask &= ~np.isin(np.arange(len(self.main_window.data)), annotated_indices)

                    if np.any(mask):
                        scatter = self.ax.scatter(
                            self.main_window.coords[mask, 0],
                            self.main_window.coords[mask, 1],
                            c=[color_dict[cell_type]],
                            label=f"{cell_type}",
                            alpha=self.main_window.point_alpha,
                            s=self.main_window.point_size
                        )
                        self.main_window.scatter_artists[cell_type] = scatter

                # Reset tick colors to default
                self.ax.tick_params(colors='black')
                for spine in self.ax.spines.values():
                    spine.set_color('black')

            # Modify the annotation plotting section to respect visibility toggle
            if self.main_window.show_annotated:
                # Plot annotations as before
                for new_type, indices in self.main_window.annotations.items():
                    if indices:
                        idx_array = np.array(list(indices))
                        scatter = self.ax.scatter(
                            self.main_window.coords[idx_array, 0],
                            self.main_window.coords[idx_array, 1],
                            label=f"New: {new_type}",
                            alpha=1.0,
                            s=self.main_window.point_size
                        )
                        self.main_window.scatter_artists[new_type] = scatter

            # Plot selected points
            if self.main_window.selected_points:
                selected_array = np.array(list(self.main_window.selected_points))
                scatter = self.ax.scatter(
                    self.main_window.coords[selected_array, 0],
                    self.main_window.coords[selected_array, 1],
                    c='red',
                    s=self.main_window.point_size*1.5,
                    alpha=0.5,
                    label='Selected'
                )
                self.main_window.scatter_artists['selected'] = scatter

            # Create legend only if there are labeled artists and not in RGB mode
            if rgb_colors is None:  # Only show legend for categorical coloring
                # Get all artists that have labels
                labeled_artists = [artist for artist in self.main_window.scatter_artists.values() 
                                 if artist.get_label() and not artist.get_label().startswith('_')]
                if labeled_artists:  # Only create legend if there are labeled artists
                    legend = self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                    for handle in legend.legend_handles:
                        handle._sizes = [30]

            self.ax.set_aspect('equal', adjustable='datalim')

            # Store or restore view limits
            if not had_previous_view:
                # Let matplotlib set the initial view
                self.ax.relim()
                self.ax.autoscale_view()
                self.original_xlim = self.ax.get_xlim()
                self.original_ylim = self.ax.get_ylim()
            else:
                # Restore previous view
                self.ax.set_xlim(current_xlim)
                self.ax.set_ylim(current_ylim)

            # print("Drawing plot...")
            self.canvas.draw()
            self.main_window.background = self.canvas.copy_from_bbox(self.ax.bbox)
            # print("Plot_data complete!")

        except Exception as e:
            print(f"Error in plot_data: {e}")

    def update_point_size(self, value):
        self.main_window.point_size = value
        self.main_window.size_value_label.setText(str(value))
        self.plot_data()

    def update_point_alpha(self, value):
        """Update the transparency of the points"""
        self.main_window.point_alpha = value / 100.0
        self.main_window.alpha_value_label.setText(f"{value}%")
        self.plot_data()

    def refresh_view(self):
        """Reset the view to show all current data points"""
        if self.main_window.coords is not None and len(self.main_window.coords) > 0:
            # Add 5% padding to the limits
            padding = 0.05

            x_min, x_max = self.main_window.coords[:, 0].min(), self.main_window.coords[:, 0].max()
            y_min, y_max = self.main_window.coords[:, 1].min(), self.main_window.coords[:, 1].max()

            x_range = x_max - x_min
            y_range = y_max - y_min

            self.ax.set_xlim([x_min - x_range * padding,
                             x_max + x_range * padding])
            self.ax.set_ylim([y_min - y_range * padding,
                             y_max + y_range * padding])

            self.canvas.draw()
            # Update background after view change
            self.main_window.background = self.canvas.copy_from_bbox(self.ax.bbox)

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
        self.main_window.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def get_rgb_colors(self):
        """Calculate RGB colors based on selected continuous variables"""
        if self.main_window.data is None:
            return None

        # Get selected columns
        rgb_cols = []
        for combo in [self.main_window.red_combo, self.main_window.green_combo, self.main_window.blue_combo]:
            col = combo.currentText()
            if col != "None":
                rgb_cols.append(col)

        if not rgb_cols:
            return None

        # Create color array
        colors = np.zeros((len(self.main_window.data), 3))
        scaler = MinMaxScaler()

        # Fill in selected channels
        for i, col in enumerate([self.main_window.red_combo.currentText(),
                               self.main_window.green_combo.currentText(),
                               self.main_window.blue_combo.currentText()]):
            if col != "None" and col != "":
                colors[:, i] = np.round(scaler.fit_transform(self.main_window.data[col].values.reshape(-1, 1)),6).ravel()
            else:
                colors[:, i] = 0

        return colors

    def clear_rgb_variables(self):
        """Reset all RGB variable selections to None"""
        self.main_window.red_combo.setCurrentText("None")
        self.main_window.green_combo.setCurrentText("None")
        self.main_window.blue_combo.setCurrentText("None")
        self.check_and_update_plot()

    def toggle_annotated_visibility(self):
        """Toggle visibility of annotated cells"""
        self.main_window.show_annotated = not self.main_window.show_annotated
        self.main_window.hide_annotated_btn.setText(
            "Show Annotated Cells" if not self.main_window.show_annotated else "Hide Annotated Cells"
        )
        self.plot_data()

# --- END OF FILE plotting_operations.py ---