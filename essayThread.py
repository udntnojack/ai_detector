from PySide6.QtCore import QThread, Signal


class EssayWorker(QThread):
    finished = Signal(dict)
    progress = Signal(str)

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        self.progress.emit("loading LLM model please wait...")
        from essay_analyzer import predict_essay   
        
        self.progress.emit("Analyzing please wait...")
        result = predict_essay(self.text, progress_callback=self.progress.emit)

        self.finished.emit(result)