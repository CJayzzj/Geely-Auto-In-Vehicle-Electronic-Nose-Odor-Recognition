# -*- coding: utf-8 -*-
"""Tool1 source entry: alcohol versus smoke."""

import os
import sys

from PyQt5 import QtCore, QtWidgets

from gui.main_window import MainWindow
from task_config import get_task_config


def main():
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(get_task_config("alcohol_vs_smoke"))
    window.show()
    if os.environ.get("JL_GUI_SMOKE_TEST") == "1":
        QtCore.QTimer.singleShot(200, app.quit)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
