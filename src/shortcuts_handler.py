# --- START OF FILE shortcuts_handler.py ---
class ShortcutsHandler:
    def __init__(self, main_window):
        self.main_window = main_window

    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if not self.main_window.plot_ops.canvas.hasFocus():
            return

        if event.key == 'd':  # Undo last selection
            self.main_window.select_ops.undo_last_selection()
        elif event.key == 'f':  # Redo last selection
            self.main_window.select_ops.redo_last_selection()
        # Add selection mode shortcuts
        elif event.key == 'q':  # Lasso selection
            self.main_window.selection_combo.setCurrentText("Lasso (Q)")
        elif event.key == 'w':  # Single selection
            self.main_window.selection_combo.setCurrentText("Single (W)")
        elif event.key == 'e':  # Brush selection
            self.main_window.selection_combo.setCurrentText("Brush (E)")
        elif event.key == 'r':  # Eraser
            self.main_window.selection_combo.setCurrentText("Eraser (R)")
        # Add brush size shortcuts
        elif event.key in ['+', '=']:  # Increase brush size (both + and = keys work)
            if self.main_window.selection_combo.currentText() in ["Brush (E)", "Eraser (R)"]:
                current_size = self.main_window.brush_size_slider.value()
                new_size = min(current_size + 1, self.main_window.brush_size_slider.maximum())
                self.main_window.brush_size_slider.setValue(new_size)
        elif event.key == '-':  # Decrease brush size
            if self.main_window.selection_combo.currentText() in ["Brush (E)", "Eraser (R)"]:
                current_size = self.main_window.brush_size_slider.value()
                new_size = max(current_size - 1, self.main_window.brush_size_slider.minimum())
                self.main_window.brush_size_slider.setValue(new_size)

# --- END OF FILE shortcuts_handler.py ---