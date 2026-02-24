from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QFileDialog, QRadioButton,
    QGroupBox
)


import sys

class DetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Text Detector")
        self.resize(700, 500)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # -------- Input Mode --------
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)


        self.text_box = QTextEdit()
        self.text_box.setPlaceholderText("Paste or type text here...")
        input_layout.addWidget(self.text_box)


        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)

        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.file_label)

        input_layout.addLayout(file_layout)

        layout.addWidget(input_group)

        # -------- Analyze Button --------
        self.analyze_button = QPushButton("Analyze")
        layout.addWidget(self.analyze_button)
        self.analyze_button.clicked.connect(self.analyse_essay)

        # -------- Results --------
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.result_label = QLabel("Prediction: —")
        self.confidence_label = QLabel("Confidence: —")

        results_layout.addWidget(self.result_label)
        results_layout.addWidget(self.confidence_label)

        layout.addWidget(results_group)

        # -------- Status --------
        self.statusBar().showMessage("Ready")

    def analyse_essay(self):
        text = self.text_box.toPlainText()
        print(text)
        from essay_analyzer import predict_essay
        results = predict_essay(text)
        print(results)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select document", "", "Word Documents (*.docx)"
        )
        if path:
            self.file_label.setText(path)
            


app = QApplication(sys.argv)
window = DetectorApp()
window.show()
sys.exit(app.exec())


