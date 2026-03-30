from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QFileDialog, QRadioButton,
    QGroupBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QIcon
from docx import Document
from PyPDF2 import PdfReader
import os
from essayThread import EssayWorker 
os.environ["TORCH_DISABLE_DYNAMO"] = "1"
os.environ["TORCH_COMPILE_MODE"] = "OFF"

from gauge import GaugeWidget

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
        self.gauge = GaugeWidget()
        

        results_layout.addWidget(self.result_label)
        results_layout.addWidget(self.confidence_label)
        results_layout.addWidget(self.gauge, alignment=Qt.AlignRight)

        layout.addWidget(results_group)

        # -------- Status --------
        self.statusBar().showMessage("Ready")

    def analyse_essay(self):
        # Prevent multiple threads
        if hasattr(self, "worker") and self.worker is not None and self.worker.isRunning():
            return
    
        text = self.text_box.toPlainText().strip()
        if not text:
            return
    
        # Disable UI (important)
        self.analyze_button.setEnabled(False)
    
        self.worker = EssayWorker(text)
    
        self.worker.progress.connect(self.update_status)
        self.worker.finished.connect(self.analysis_complete)
    
        # Clean up thread properly
        self.worker.finished.connect(self.cleanup_worker)
    
        self.worker.start()

    def cleanup_worker(self):
        if self.worker:
            self.worker.deleteLater()
            self.worker = None



    def update_status(self, msg):
        self.statusBar().showMessage(msg)

    def analysis_complete(self, result):

        conf = result["meta_results"][0]
        label = ""
        if(conf < 0.30):
            label = "unlikely to be AI generated"
        elif(conf < 0.70):
            label = "likely contains AI generated elements"
        elif(conf > 0.70):
            label = "likely to be ai generated"
        

        sentence_probs = result["sentence_results"]

        self.result_label.setText(f"Prediction: {label}")
        self.confidence_label.setText(f"Confidence: {conf:.2%}")

        self.gauge.setValue(int(conf * 100))
        

        self.highlight_sentences(sentence_probs)
        self.statusBar().showMessage("Done")
        self.analyze_button.setEnabled(True)

    def highlight_sentences(self, sentence_probs):
        doc = self.text_box.document()

        # clear previous formatting
        cursor = QTextCursor(doc)
        cursor.select(QTextCursor.Document)
        cursor.setCharFormat(QTextCharFormat())

        fmt = QTextCharFormat()
        fmt.setForeground(QColor("black"))

        for sentence in sentence_probs:
            s = sentence["sentence"]
            prob = sentence["prob"]
            if prob > 0.40:
                if prob < 0.50:
                    fmt.setBackground(QColor("#fbff00"))
                elif prob < 0.60:
                    fmt.setBackground(QColor("#ff9100"))
                elif prob > 0.70:
                    fmt.setBackground(QColor("#ff4040"))
                else:
                    break

                find_cursor = QTextCursor(doc)
                while True:
                    find_cursor = doc.find(s, find_cursor)
                    if find_cursor.isNull():
                        break
                    find_cursor.mergeCharFormat(fmt)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
                self,
                "Select document",
                "",
                "Documents (*.docx *.pdf *.txt)"
        )
        if path:
            self.file_label.setText(path)
            self.addText(path)

    def addText(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext == ".docx":
            text = self.load_docx(path)
        elif ext == ".pdf":
            text = self.load_pdf(path)
        elif ext == ".txt":
            text = self.load_txt(path)
        else:
            text = ""
        self.text_box.setPlainText(text)

    def load_docx(self, path):
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)


    def load_pdf(self, path):
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)


    def load_txt(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def resource_path(relative_path):
    """Get absolute path for PyInstaller or dev environment."""
    if hasattr(sys, "_MEIPASS"):  # PyInstaller temp folder
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(__file__), relative_path)
      
        

app = QApplication(sys.argv)
icon_path = resource_path("logo.ico")
app.setWindowIcon(QIcon(icon_path))
window = DetectorApp()
window.show()
sys.exit(app.exec())


