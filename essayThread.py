from PySide6.QtCore import QThread, Signal
from essay_analyzer import predict_essay

class EssayWorker(QThread):
    finished = Signal(dict)
    progress = Signal(str)

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        
        
        self.progress.emit("Analyzing please wait...")
        result = predict_essay(self.text)

        self.finished.emit(result)