import sys
from src.main_window import CellAnnotationTool
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    window = CellAnnotationTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 