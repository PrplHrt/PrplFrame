import sys
import PyQt6
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidget, QTableWidgetItem

import pandas as pd

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a QTableWidget to display the dataset head
        self.table = QTableWidget()

        # Set the window title
        self.setWindowTitle("CSV Dataset Head")

        # Set the central widget to the QTableWidget
        self.setCentralWidget(self.table)

        # Create a menu bar
        self.menuBar = self.menuBar()

        # Create a File menu
        self.fileMenu = self.menuBar.addMenu("File")

        # Create an Open action
        self.openAction = self.fileMenu.addAction("Open")
        self.openAction.triggered.connect(self.openFile)

        # Show the window
        self.show()

    def openFile(self):
        # Get the path of the selected file
        filePath, _ = QFileDialog.getOpenFileName(self, "Select CSV file", filter="CSV files (*.csv)")

        # If a file was selected, read it using pandas and display the head
        if filePath:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(filePath)

            # Display the head of the DataFrame in the QTableWidget
            self.table.setRowCount(10)
            self.table.setColumnCount(len(df.columns))
            self.table.setHorizontalHeaderLabels(df.columns)
            for i in range(10):
                for j in range(len(df.columns)):
                    self.table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))
            self.table.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create a MainWindow instance
    window = MainWindow()

    # Start the main loop
    app.exec()