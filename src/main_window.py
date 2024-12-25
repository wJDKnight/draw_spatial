# --- START OF FILE main_window.py ---
import sys
import os
os.environ['QT_QPA_PLATFORM'] = 'cocoa'  # Add this for macOS
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend before importing pyplot

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QFileDialog, QLabel,
                           QComboBox, QLineEdit, QSlider, QCompleter, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QKeySequence, QIcon

from .data_operations import DataOperations
from .plotting_operations import PlottingOperations
from .selection_operations import SelectionOperations
from .shortcuts_handler import ShortcutsHandler

class CellAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spatial Cell Annotation Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize data structures (shared state)
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
        self.show_annotated = True  # Add a class variable to track visibility state

        # Initialize helper classes
        self.data_ops = DataOperations(self)
        self.plot_ops = PlottingOperations(self)
        self.select_ops = SelectionOperations(self)
        self.shortcuts = ShortcutsHandler(self)

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Create matplotlib figure with adjusted size for legend
        self.plot_ops.figure, self.plot_ops.ax = matplotlib.pyplot.subplots(figsize=(10, 8))
        # Add space for the legend on the right
        self.plot_ops.figure.subplots_adjust(right=0.8)
        self.plot_ops.canvas = self.plot_ops.FigureCanvas(self.plot_ops.figure)
        layout.addWidget(self.plot_ops.canvas, stretch=4)

        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # Set fixed width and vertical resize for control panel
        control_panel.setFixedWidth(250)
        control_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # Add buttons
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.data_ops.load_data)
        control_layout.addWidget(self.load_btn)

        # Point size control
        size_layout = QHBoxLayout()
        self.size_label = QLabel("Point Size:")
        size_layout.addWidget(self.size_label)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(100)
        self.size_slider.setValue(self.point_size)
        self.size_slider.valueChanged.connect(self.plot_ops.update_point_size)
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
        self.alpha_slider.valueChanged.connect(self.plot_ops.update_point_alpha)
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
        self.brush_size_slider.valueChanged.connect(self.select_ops.update_brush_size)

        # Hide brush size controls initially
        self.brush_size_label.setVisible(False)
        self.brush_size_slider.setVisible(False)
        self.brush_size_value_label.setVisible(False)

        # Connect selection mode change
        self.selection_combo.currentTextChanged.connect(self.select_ops.on_selection_mode_changed)

        # New annotation name input with icon button layout
        annotation_layout = QVBoxLayout()  # Change to vertical layout
        self.new_type_label = QLabel("New Annotation Name:")
        annotation_layout.addWidget(self.new_type_label)

        # Create horizontal layout for input and confirm button
        input_layout = QHBoxLayout()
        self.new_type_input = QLineEdit()
        input_layout.addWidget(self.new_type_input)

        # Add confirm button to same row as input
        self.confirm_btn = QPushButton()
        self.confirm_btn.setIcon(QIcon("icons/confirm.png"))
        self.confirm_btn.setToolTip("Confirm Selection")
        # Get the height of the input field and set button size to 1.2x that height
        input_height = self.new_type_input.sizeHint().height()
        button_size = int(input_height * 1.2)
        self.confirm_btn.setFixedSize(button_size, button_size)
        # Set the icon size to match the button size
        self.confirm_btn.setIconSize(QSize(button_size, button_size))
        self.confirm_btn.clicked.connect(self.select_ops.confirm_selection)
        input_layout.addWidget(self.confirm_btn)

        # Add the input layout to the main annotation layout
        annotation_layout.addLayout(input_layout)
        control_layout.addLayout(annotation_layout)

        # Clear current selection button
        self.clear_btn = QPushButton("Clear Current Selection")
        self.clear_btn.clicked.connect(self.select_ops.clear_selection)
        control_layout.addWidget(self.clear_btn)

        # Add "Undo Last Selection" button after the Clear Selection button
        self.undo_selection_btn = QPushButton("Undo Last Selection (D)")
        self.undo_selection_btn.clicked.connect(self.select_ops.undo_last_selection)
        control_layout.addWidget(self.undo_selection_btn)

        # Add "Redo Last Selection" button after the Undo button
        self.redo_selection_btn = QPushButton("Redo Last Selection (F)")
        self.redo_selection_btn.clicked.connect(self.select_ops.redo_last_selection)
        control_layout.addWidget(self.redo_selection_btn)

        # Remove annotation button (moved here)
        self.remove_annotation_btn = QPushButton("Remove Annotation from Selection")
        self.remove_annotation_btn.clicked.connect(self.select_ops.remove_annotation)
        control_layout.addWidget(self.remove_annotation_btn)

        # Add annotation undo/redo buttons
        self.undo_annotation_btn = QPushButton("Undo Last Annotation")
        self.undo_annotation_btn.clicked.connect(self.select_ops.undo_last_annotation)
        control_layout.addWidget(self.undo_annotation_btn)

        self.redo_annotation_btn = QPushButton("Redo Last Annotation")
        self.redo_annotation_btn.clicked.connect(self.select_ops.redo_last_annotation)
        control_layout.addWidget(self.redo_annotation_btn)

        # Add "Hide Annotated Cells" button after the save button
        self.hide_annotated_btn = QPushButton("Hide Annotated Cells")
        self.hide_annotated_btn.setCheckable(True)  # Make it toggleable
        self.hide_annotated_btn.clicked.connect(self.plot_ops.toggle_annotated_visibility)
        control_layout.addWidget(self.hide_annotated_btn)

        # Add save/load annotation state buttons after the save annotations button
        self.save_state_btn = QPushButton("Save Annotation State")
        self.save_state_btn.clicked.connect(self.data_ops.save_annotation_state)
        control_layout.addWidget(self.save_state_btn)

        self.load_state_btn = QPushButton("Load Annotation State")
        self.load_state_btn.clicked.connect(self.data_ops.load_annotation_state)
        control_layout.addWidget(self.load_state_btn)

        # Save all annotations button
        self.save_btn = QPushButton("Save All Annotations")
        self.save_btn.clicked.connect(self.data_ops.save_annotations)
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
        self.plot_ops.canvas.mpl_connect('button_press_event', self.select_ops.on_mouse_press)
        self.plot_ops.canvas.mpl_connect('button_release_event', self.select_ops.on_mouse_release)
        self.plot_ops.canvas.mpl_connect('motion_notify_event', self.select_ops.on_mouse_move)
        self.plot_ops.canvas.mpl_connect('scroll_event', self.plot_ops.on_scroll)

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
        self.refresh_view_btn.clicked.connect(self.plot_ops.refresh_view)
        coord_layout.addWidget(self.refresh_view_btn)

        control_layout.insertLayout(1, coord_layout)  # Insert after load button

        # After setting up the canvas, add key press event connection
        self.plot_ops.canvas.setFocusPolicy(Qt.StrongFocus)  # Make canvas focusable
        self.plot_ops.canvas.mpl_connect('key_press_event', self.shortcuts.on_key_press)

        # Add tooltips to show shortcuts
        self.selection_combo.setToolTip("Shortcuts: Q (Lasso), W (Single), E (Brush), R (Eraser)")

        # Update brush size control tooltip
        self.brush_size_label.setToolTip("Use + and - keys to adjust size")
        self.brush_size_slider.setToolTip("Use + and - keys to adjust size")
        self.brush_size_value_label.setToolTip("Use + and - keys to adjust size")

        # After the existing column selectors, add continuous variable selectors
        continuous_layout = QVBoxLayout()

        # Create horizontal layout for label and clear button
        continuous_header = QHBoxLayout()
        continuous_header.addWidget(QLabel("Color by Continuous Variables:"))

        # Add clear button with icon
        clear_rgb_btn = QPushButton()
        clear_rgb_btn.setIcon(QIcon("icons/clear.png"))
        clear_rgb_btn.setToolTip("Clear RGB Variables")
        # Set button size similar to confirm button
        button_size = int(self.new_type_input.sizeHint().height() * 1.2)
        clear_rgb_btn.setFixedSize(button_size, button_size)
        clear_rgb_btn.setIconSize(QSize(button_size, button_size))
        clear_rgb_btn.clicked.connect(self.plot_ops.clear_rgb_variables)
        continuous_header.addWidget(clear_rgb_btn)

        continuous_layout.addLayout(continuous_header)

        # Function to create combo box with autocomplete
        def create_channel_combo(label, color):
            layout = QHBoxLayout()
            layout.addWidget(QLabel(f"{label}:"))
            combo = QComboBox()
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.NoInsert)  # Don't automatically insert text

            # Remove automatic completion
            combo.completer().setCompletionMode(QCompleter.PopupCompletion)
            combo.completer().setCompletionMode(QCompleter.InlineCompletion)
            combo.completer().popup() # Hide completion popup

            combo.addItem("None")
            combo.setCurrentText("None")

            # Style the combo box with a color indicator
            combo.setStyleSheet(f"""
                QComboBox {{
                    padding-left: 5px;
                    border: 1px solid {color};
                }}
                QComboBox:editable {{
                    background-color: white;
                }}
            """)

            layout.addWidget(combo)
            return layout, combo

        # Create RGB channel selectors
        blue_layout, self.blue_combo = create_channel_combo("Blue", "blue")
        green_layout, self.green_combo = create_channel_combo("Green", "green")
        red_layout, self.red_combo = create_channel_combo("Red", "red")

        continuous_layout.addLayout(blue_layout)
        continuous_layout.addLayout(green_layout)
        continuous_layout.addLayout(red_layout)

        control_layout.insertLayout(2, continuous_layout)

        # Create a scroll area and set its widget to the control panel
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow the widget to resize with the scroll area
        scroll_area.setWidget(control_panel)

        # Set fixed width and vertical resize for scroll area
        scroll_area.setFixedWidth(270)
        scroll_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        layout.addWidget(scroll_area, stretch=1)

        # Connect column selector changes after UI is set up
        self.x_combo.currentTextChanged.connect(self.plot_ops.check_and_update_plot)
        self.y_combo.currentTextChanged.connect(self.plot_ops.check_and_update_plot)
        self.cell_type_combo.currentTextChanged.connect(self.plot_ops.check_and_update_plot)
        self.red_combo.currentTextChanged.connect(self.plot_ops.check_and_update_plot)
        self.green_combo.currentTextChanged.connect(self.plot_ops.check_and_update_plot)
        self.blue_combo.currentTextChanged.connect(self.plot_ops.check_and_update_plot)

    def update_annotation_display(self):
        text = []
        for cell_type, indices in self.annotations.items():
            text.append(f"{cell_type}: {len(indices)} cells")
        self.annotation_display.setText("\n".join(text))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CellAnnotationTool()
    window.show()
    sys.exit(app.exec_())
# --- END OF FILE main_window.py ---