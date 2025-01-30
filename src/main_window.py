# --- START OF FILE main_window.py ---
import sys
import os
os.environ['QT_QPA_PLATFORM'] = 'cocoa'  # Add this for macOS
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend before importing pyplot

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QFileDialog, QLabel,
                           QComboBox, QLineEdit, QSlider, QCompleter, QScrollArea, QSizePolicy,
                           QMenuBar, QMenu, QAction, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QKeySequence, QIcon

from .data_operations import DataOperations
from .plotting_operations import PlottingOperations
from .selection_operations import SelectionOperations
from .shortcuts_handler import ShortcutsHandler
from .analysis_window import AnalysisWindow

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
        self.analysis_window = None  # Store reference to analysis window

        # Initialize helper classes
        self.data_ops = DataOperations(self)
        self.plot_ops = PlottingOperations(self)
        self.select_ops = SelectionOperations(self)
        self.shortcuts = ShortcutsHandler(self)

        # Create menu bar
        self.create_menu_bar()


        # Setup UI
        self.setup_ui()

    def create_menu_bar(self):
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu('&File')
        
        load_data_action = QAction('&Load Data', self)
        load_data_action.setShortcut('Ctrl+O')
        load_data_action.triggered.connect(self.data_ops.load_data)
        file_menu.addAction(load_data_action)

        save_annotations_action = QAction('&Save All Annotations', self)
        save_annotations_action.setShortcut('Ctrl+S')
        save_annotations_action.triggered.connect(self.data_ops.save_annotations)
        file_menu.addAction(save_annotations_action)

        save_state_action = QAction('Save Annotation &State', self)
        save_state_action.setShortcut('Ctrl+Shift+S')
        save_state_action.triggered.connect(self.data_ops.save_annotation_state)
        file_menu.addAction(save_state_action)

        load_state_action = QAction('&Load Annotation State', self)
        load_state_action.setShortcut('Ctrl+Shift+O')
        load_state_action.triggered.connect(self.data_ops.load_annotation_state)
        file_menu.addAction(load_state_action)

        # Edit Menu
        edit_menu = menubar.addMenu('&Edit')

        undo_selection_action = QAction('&Undo Last Selection', self)
        undo_selection_action.setShortcut('D')
        undo_selection_action.triggered.connect(self.select_ops.undo_last_selection)
        edit_menu.addAction(undo_selection_action)

        redo_selection_action = QAction('&Redo Last Selection', self)
        redo_selection_action.setShortcut('F')
        redo_selection_action.triggered.connect(self.select_ops.redo_last_selection)
        edit_menu.addAction(redo_selection_action)

        edit_menu.addSeparator()

        undo_annotation_action = QAction('Undo Last &Annotation', self)
        undo_annotation_action.setShortcut('Ctrl+Z')
        undo_annotation_action.triggered.connect(self.select_ops.undo_last_annotation)
        edit_menu.addAction(undo_annotation_action)

        redo_annotation_action = QAction('Redo Last A&nnotation', self)
        redo_annotation_action.setShortcut('Ctrl+Y')
        redo_annotation_action.triggered.connect(self.select_ops.redo_last_annotation)
        edit_menu.addAction(redo_annotation_action)

        edit_menu.addSeparator()

        clear_selection_action = QAction('&Clear Current Selection', self)
        # clear_selection_action.setShortcut('Escape')
        clear_selection_action.triggered.connect(self.select_ops.clear_selection)
        edit_menu.addAction(clear_selection_action)

        remove_annotation_action = QAction('&Remove Annotation from Selection', self)
        # remove_annotation_action.setShortcut('Delete')
        remove_annotation_action.triggered.connect(self.select_ops.remove_annotation)
        edit_menu.addAction(remove_annotation_action)

        # View Menu
        view_menu = menubar.addMenu('&View')

        refresh_view_action = QAction('&Refresh View', self)
        refresh_view_action.setShortcut('F5')
        refresh_view_action.triggered.connect(self.plot_ops.refresh_view)
        view_menu.addAction(refresh_view_action)

        toggle_annotated_action = QAction('&Toggle Annotated Cells', self)
        toggle_annotated_action.setShortcut('Ctrl+T')
        toggle_annotated_action.setCheckable(True)
        toggle_annotated_action.triggered.connect(self.plot_ops.toggle_annotated_visibility)
        view_menu.addAction(toggle_annotated_action)

        clear_rgb_action = QAction('Clear &RGB Variables', self)
        clear_rgb_action.setShortcut('Ctrl+R')
        clear_rgb_action.triggered.connect(self.plot_ops.clear_rgb_variables)
        view_menu.addAction(clear_rgb_action)

        # Tools Menu
        tools_menu = menubar.addMenu('&Tools')

        # Selection Mode submenu
        selection_menu = QMenu('&Selection Mode', self)
        tools_menu.addMenu(selection_menu)

        lasso_action = QAction('&Lasso', self)
        lasso_action.setShortcut('Q')
        lasso_action.triggered.connect(lambda: self.selection_combo.setCurrentText("Lasso (Q)"))
        selection_menu.addAction(lasso_action)

        single_action = QAction('&Single', self)
        single_action.setShortcut('W')
        single_action.triggered.connect(lambda: self.selection_combo.setCurrentText("Single (W)"))
        selection_menu.addAction(single_action)

        brush_action = QAction('&Brush', self)
        brush_action.setShortcut('E')
        brush_action.triggered.connect(lambda: self.selection_combo.setCurrentText("Brush (E)"))
        selection_menu.addAction(brush_action)

        eraser_action = QAction('&Eraser', self)
        eraser_action.setShortcut('R')
        eraser_action.triggered.connect(lambda: self.selection_combo.setCurrentText("Eraser (R)"))
        selection_menu.addAction(eraser_action)

        # Analysis Menu
        analysis_menu = menubar.addMenu('&Analysis')
        
        open_analysis_action = QAction('&Open Analysis Window', self)
        open_analysis_action.setShortcut('Ctrl+A')
        open_analysis_action.triggered.connect(self.show_analysis_window)
        analysis_menu.addAction(open_analysis_action)

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

        # Add analysis button after load button
        self.analysis_btn = QPushButton("Analysis")
        self.analysis_btn.clicked.connect(self.show_analysis_window)
        control_layout.addWidget(self.analysis_btn)

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
        self.cell_type_combo.currentTextChanged.connect(self.update_cell_type_filter)
        cell_type_layout.addWidget(self.cell_type_combo)
        coord_layout.addLayout(cell_type_layout)

        # Add cell type filter dropdown
        cell_type_filter_layout = QHBoxLayout()
        cell_type_filter_layout.addWidget(QLabel("Filter Cell Types:"))
        self.cell_type_filter_list = QListWidget()
        self.cell_type_filter_list.setMaximumHeight(100)  # Limit height to show ~5 items
        self.cell_type_filter_list.itemChanged.connect(self.plot_ops.check_and_update_plot)
        cell_type_filter_layout.addWidget(self.cell_type_filter_list)
        coord_layout.addLayout(cell_type_filter_layout)

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

    def update_cell_type_filter(self):
        """Update cell type filter list when cell type column changes"""
        # Disconnect existing connections if any
        try:
            self.cell_type_filter_list.itemChanged.disconnect()
        except:
            pass
            
        self.cell_type_filter_list.clear()
        
        if self.data is not None and self.cell_type_combo.currentText():
            # Add "Select All" item
            select_all_item = QListWidgetItem("Select All")
            select_all_item.setFlags(select_all_item.flags() | Qt.ItemIsUserCheckable)
            select_all_item.setCheckState(Qt.Checked)
            self.cell_type_filter_list.addItem(select_all_item)
            
            # Add cell type items
            cell_type_col = self.cell_type_combo.currentText()
            unique_types = sorted(self.data[cell_type_col].unique())
            for cell_type in unique_types:
                item = QListWidgetItem(str(cell_type))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.cell_type_filter_list.addItem(item)
            
            # Connect select all functionality after adding all items
            self.cell_type_filter_list.itemChanged.connect(self.handle_select_all)
        
        self.plot_ops.check_and_update_plot()
        
    def handle_select_all(self, item):
        """Handle the Select All checkbox state change"""
        # Temporarily disconnect the itemChanged signal to prevent multiple updates
        self.cell_type_filter_list.itemChanged.disconnect()
        
        try:
            if item.text() == "Select All":
                state = item.checkState()
                # Update all items at once
                for i in range(1, self.cell_type_filter_list.count()):
                    self.cell_type_filter_list.item(i).setCheckState(state)
            else:
                # Update Select All state based on other items
                all_checked = True
                for i in range(1, self.cell_type_filter_list.count()):
                    if self.cell_type_filter_list.item(i).checkState() == Qt.Unchecked:
                        all_checked = False
                        break
                # Update Select All without triggering itemChanged
                self.cell_type_filter_list.item(0).setCheckState(Qt.Checked if all_checked else Qt.Unchecked)
        finally:
            # Reconnect the signal and update plot once
            self.cell_type_filter_list.itemChanged.connect(self.handle_select_all)
            self.plot_ops.check_and_update_plot()

    def get_selected_cell_types(self):
        """Get list of selected cell types from the filter list"""
        selected_types = []
        for i in range(1, self.cell_type_filter_list.count()):  # Skip Select All item
            item = self.cell_type_filter_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_types.append(item.text())
        return selected_types if selected_types else None  # Return None if no types selected

    def show_analysis_window(self):
        if self.analysis_window is None:
            self.analysis_window = AnalysisWindow(self)
        self.analysis_window.show()
        self.analysis_window.update_proportions_table()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CellAnnotationTool()
    window.show()
    sys.exit(app.exec_())
# --- END OF FILE main_window.py ---