from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QFileDialog, QMessageBox
)

from AudioEngine import render_offline, save_wav


class DownloadDialog(QDialog):
    """Süre seçip WAV olarak indirmeyi sağlayan küçük pencere.

    build_voices_fn: () -> {isim: MelodyEngine}  şeklinde, TAZE (canlı
    akışa dokunmayan) MelodyEngine kopyaları üreten bir fonksiyon.
    Bu, indirme işleminin çalan sesi asla bozmamasını garanti eder."""

    def __init__(self, title, build_voices_fn, sample_rate=44100, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setStyleSheet(
            "QDialog { background-color: #05070a; color: #d8ffe0; }"
            "QLabel { color: #9dffb0; }"
            "QDoubleSpinBox { background-color: #0d1210; color: #39ff14; "
            "border: 1px solid #1f3d1f; padding: 4px; border-radius: 4px; }"
            "QPushButton { background-color: #10241a; color: #39ff14; "
            "border: 1px solid #39ff14; padding: 6px 12px; border-radius: 6px; "
            "font-weight: bold; }"
            "QPushButton:hover { background-color: #1d4d2a; }"
        )

        self.build_voices_fn = build_voices_fn
        self.sample_rate = sample_rate

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(title))

        duration_row = QHBoxLayout()
        duration_row.addWidget(QLabel("Süre (saniye):"))
        self.duration_input = QDoubleSpinBox()
        self.duration_input.setRange(1, 600)
        self.duration_input.setValue(60)
        self.duration_input.setSingleStep(5)
        duration_row.addWidget(self.duration_input)
        layout.addLayout(duration_row)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        button_row = QHBoxLayout()
        self.download_button = QPushButton("⬇ İndir")
        self.download_button.clicked.connect(self._on_download)
        button_row.addWidget(self.download_button)

        cancel_button = QPushButton("İptal")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)

        layout.addLayout(button_row)

    def _on_download(self):
        duration = self.duration_input.value()

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Ses dosyasını kaydet", "melodi.wav", "WAV Dosyası (*.wav)"
        )
        if not filepath:
            return

        self.status_label.setText("Render ediliyor, lütfen bekleyin...")
        self.repaint()

        try:
            voices = self.build_voices_fn()
            if not voices:
                QMessageBox.warning(self, "Hata", "İndirilecek denklem bulunamadı.")
                return

            pcm16 = render_offline(voices, duration, self.sample_rate)
            save_wav(filepath, pcm16, self.sample_rate)

            self.status_label.setText("Tamamlandı.")
            QMessageBox.information(self, "Tamamlandı", f"Kaydedildi:\n{filepath}")
            self.accept()
        except Exception as error:
            QMessageBox.critical(self, "Hata", f"Render sırasında hata oluştu:\n{error}")
