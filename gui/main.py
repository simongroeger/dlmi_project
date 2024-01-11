from PyQt6.QtWidgets import *
import sys


# 
# create GUI 
#
from classification_gui import *

def create_gui():
    """Create classification GUI"""

    gui_path = "gui/classification_gui.ui"
    # images_path = "Sourcecode/pytorch-image-models/images/all"
    # metadata_path = "Sourcecode/gui.py/metadata.csv"
    images_path = "pytorch-image-models/images/all"
    metadata_path = "pytorch-image-models/images/metadata.csv"
    application = QApplication(sys.argv)
    root = ClassificationGUI(gui_path, images_path, metadata_path)
    application.exec()


if __name__ == '__main__':
    create_gui()