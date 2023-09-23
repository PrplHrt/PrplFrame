import sys
import PyQt6
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QTableWidget, 
                             QTableWidgetItem, QMessageBox, QDialog, QVBoxLayout, 
                             QLabel, QLineEdit, QPushButton)
from evaluation import utils
from models import regression
from output import render
from models import classification
import os

import pandas as pd

class MainWindow(QMainWindow):
    class DatasetInfoDialog(QDialog):
        def __init__(self, dataset_info):
            super().__init__()

            self.dataset_info = dataset_info

            # Create a layout for the dialog
            layout = QVBoxLayout()

            # Create a label for the dataset name input
            datasetNameLabel = QLabel("Dataset Name:")
            layout.addWidget(datasetNameLabel)

            # Create a line edit for the dataset name input
            self.datasetNameLineEdit = QLineEdit()
            layout.addWidget(self.datasetNameLineEdit)

            # Create a label for the dataset type input
            datasetTypeLabel = QLabel("Type (regression/classification):")
            layout.addWidget(datasetTypeLabel)

            # Create a line edit for the dataset type input
            self.datasetTypeLineEdit = QLineEdit()
            layout.addWidget(self.datasetTypeLineEdit)

            # Create a label for the target column input
            targetColumnLabel = QLabel("Target column:")
            layout.addWidget(targetColumnLabel)

            # Create a line edit for the target column input
            self.targetColumnLineEdit = QLineEdit()
            layout.addWidget(self.targetColumnLineEdit)

            # Create a label for the test split input
            testSplitLabel = QLabel("Test split (between 0 and 1):")
            layout.addWidget(testSplitLabel)

            # Create a line edit for the test split input
            self.testSplitLineEdit = QLineEdit()
            layout.addWidget(self.testSplitLineEdit)

            # Create a label for the dataset source input
            datasetSourceLabel = QLabel("Source of dataset:")
            layout.addWidget(datasetSourceLabel)

            # Create a line edit for the dataset source input
            self.datasetSourceLineEdit = QLineEdit()
            layout.addWidget(self.datasetSourceLineEdit)

            # Create a button to submit the input
            submitButton = QPushButton("Submit")
            submitButton.clicked.connect(self.submit)
            layout.addWidget(submitButton)

            # Set the layout for the dialog
            self.setLayout(layout)

        def submit(self):
            # Get the user input
            self.dataset_info["name"] = self.datasetNameLineEdit.text()
            self.dataset_info["type"] = self.datasetTypeLineEdit.text()
            self.dataset_info["target"] = self.targetColumnLineEdit.text().split(',') if ',' in self.targetColumnLineEdit.text() else [self.targetColumnLineEdit.text()]
            self.dataset_info["split"] = float(self.testSplitLineEdit.text())
            self.dataset_info["source"] = self.datasetSourceLineEdit.text()
            # Close the dialog
            self.close()

    def __init__(self):
        super().__init__()

        # Create a QTableWidget to display the dataset head
        self.table = QTableWidget()

        # Set the window title
        self.setWindowTitle("BAIN")

        # Set the central widget to the QTableWidget
        self.setCentralWidget(self.table)

        # Create a menu bar
        self.menuBar = self.menuBar()

        # Create a File menu
        self.fileMenu = self.menuBar.addMenu("File")
        # Create an Open action
        self.openAction = self.fileMenu.addAction("Open")
        self.openAction.triggered.connect(self.openFile)

        # Create a Train and Test menu action
        self.testMenu = self.menuBar.addAction("Train and Test")
        self.testMenu.triggered.connect(self.trainAndTest)
        # Disabled until a dataset is loaded in
        self.testMenu.setEnabled(False)

        # Create a parametric study menu action
        self.paraMenu = self.menuBar.addAction("Parametric")
        self.paraMenu.triggered.connect(self.parametric)
        # Disabled until train and test is done
        self.paraMenu.setEnabled(False)

        # Create dictionary that will store dataset info
        self.dataset_info = {}

        # Show the window
        self.show()

    def openFile(self):
        self.testMenu.setEnabled(False)
        self.paraMenu.setEnabled(False)
        # Get the path of the selected file
        filePath, _ = QFileDialog.getOpenFileName(self, "Select CSV file", filter="CSV files (*.csv)")

        # If a file was selected, read it using pandas and display the head
        if filePath:
            # Read the CSV file into a pandas DataFrame
            self.df = pd.read_csv(filePath)

            # Display the head of the DataFrame in the QTableWidget
            self.table.setRowCount(10)
            self.table.setColumnCount(len(self.df.columns))
            self.table.setHorizontalHeaderLabels(self.df.columns)
            for i in range(10):
                for j in range(len(self.df.columns)):
                    self.table.setItem(i, j, QTableWidgetItem(str(self.df.iloc[i, j])))
            self.table.show()

            # Pop-up for collecting dataset info
            self.getDatasetInfo()

            self.dataset_info['path'] = filePath
            self.dataset_info['size'] = len(self.df)

            self.testMenu.setEnabled(True)

    def getDatasetInfo(self):
        dialog = MainWindow.DatasetInfoDialog(self.dataset_info)
        dialog.exec()
    
    def trainAndTest(self):
        # Loading in the train and test set
        data = utils.load_Xy(self.df, self.dataset_info['target'], self.dataset_info['split'])

        # Using a temporary variable to hold the best performing model
        # we'll define the best performing model by the highest f1 score
        self.top_model = None
        metric = None
        self.scores=[]

        if self.dataset_info['type'].lower() == 'classification':
            # Define which models to use
            models = [  classification.LogisticRegression(),
                        classification.SVC(),
                        classification.DecisionTreeClassifier(),
                        classification.GaussianProcessClassifier(),
                        classification.RandomForestClassifier(),
                        classification.DecisionTreeClassifier(),
                        classification.MLPClassifier()
                    ]
            for model in models:
                score = utils.classification_train_and_test(model, *data)
                # Check for best f1 score
                if (not metric) or (score['f1'] > metric):
                    self.top_model = model
                    metric = score['f1']
                self.scores.append(score)
        elif self.dataset_info['type'].lower() == 'regression':
            models = [  regression.GaussianProcessRegressor(),
                        regression.KNeighborsRegressor(),
                        regression.Ridge(),
                        regression.LinearRegression(),
                        regression.MLPRegressor(),
                        regression.PolynomialRegression(),
                        regression.SVR(),
                        regression.DecisionTreeRegressor(),
                        regression.Lasso(),
                        regression.RandomForestRegressor()
                    ]
            for model in models:
                score = utils.regression_train_and_test(model, *data)
                # Check for best R2 score
                if (not metric) or (score['r2'] > metric):
                    self.top_model = model
                    metric = score['r2']
                self.scores.append(score)
        else:
            msgBox = QMessageBox()
            msgBox.setText(f"The dataset type '{self.dataset_info['type']}' is not supported. Use either 'classification' or 'regression'.")
            msgBox.exec()
            return
        
        msgBox = QMessageBox()
        msgBox.setText(f"Best performing model:  {type(self.top_model).__name__} w/ score: {metric}")
        msgBox.exec()
        self.paraMenu.setEnabled(True)

    def parametric(self):
        print("Parametric")

        

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create a MainWindow instance
    window = MainWindow()

    # Start the main loop
    app.exec()