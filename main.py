import sys
from PyQt5.QtWidgets import QApplication

import gui

app = QApplication(sys.argv)


main_window = gui.Ui_MainWindow()
main_window.setupUi(main_window)

main_window.show()

sys.exit(app.exec_())