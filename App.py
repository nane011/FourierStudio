import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QScrollArea, QFrame, QDoubleSpinBox
)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor

from EquationEngine import EquationEngine
from NoteMapper import NoteMapper
from MelodyEngine import MelodyEngine
from NeonPlot import (
    WaveformPlot, SpectrumPlot, VectorPlot,
    MultiWaveformPlot, MultiSpectrumPlot, MultiVectorPlot,
)
from DownloadDialog import DownloadDialog


INSTRUMENTS = ["gitar", "ronroco", "ukulele", "flut", "kalimba", "santur"]
SCALES = list(NoteMapper.SCALES.keys())
NEON_COLORS = ["#39ff14", "#00e5ff", "#ff2fd0", "#ffe600", "#ff6a00", "#a5ff00"]

DEFAULT_EQUATION = (
    "sin(t)*(exp(cos(t)) - 2*cos(4*t) - sin(t/12)**5), "
    "cos(t)*(exp(cos(t)) - 2*cos(4*t) - sin(t/12)**5)"
)

DARK_STYLESHEET = """
QMainWindow, QWidget { background-color: #05070a; color: #d8ffe0; font-family: 'Consolas'; }
QLineEdit, QComboBox, QDoubleSpinBox {
    background-color: #0d1210; color: #39ff14; border: 1px solid #1f3d1f;
    padding: 4px; border-radius: 4px;
}
QComboBox QAbstractItemView { background-color: #0d1210; color: #39ff14; selection-background-color: #1f3d1f; }
QPushButton {
    background-color: #10241a; color: #39ff14; border: 1px solid #39ff14;
    padding: 6px 12px; border-radius: 6px; font-weight: bold;
}
QPushButton:hover { background-color: #1d4d2a; }
QLabel { color: #9dffb0; }
QScrollArea { border: none; }
QFrame#row { border-radius: 8px; }
"""

# GUI zamanlayıcısı: hafif işler (kayan osiloskop) her tikte, ağır
# işler (FFT spektrum, denklem eğrisi yeniden hesaplama) her N tikte
# bir yapılır. Bu, GIL'in ses callback'ini geciktirmesini önlemeye
# yardımcı olur.
FAST_TICK_MS = 60
HEAVY_TICK_EVERY = 4


class EquationRow(QFrame):
    """Tek bir denklem satırı = tek bir 'ses' (voice): rol, denklem,
    enstrüman, gam, baz frekans, nota süresi, ses seviyesi + kendi
    ton kayıt istasyonu / spektrum / vektör matrisi grafikleri."""

    def __init__(self, row_id, index, synth, sample_rate=44100, parent=None):
        super().__init__(parent)
        self.setObjectName("row")
        self.row_id = row_id                # AudioEngine/dict anahtarı olarak kullanılan BENZERSİZ id
        self.sample_rate = sample_rate
        self.synth = synth
        self.color_hex = NEON_COLORS[index % len(NEON_COLORS)]
        self.color = QColor(self.color_hex)
        self.is_playing = False

        self._build_ui()
        self._apply_border(active=False)

        self.equation_engine = EquationEngine(
            self.equation_input.text(), role=self.role_input.text(), color=self.color_hex
        )
        self.note_mapper = NoteMapper(
            base_frequency=self.freq_input.value(), scale=self.scale_combo.currentText()
        )
        self.melody_engine = MelodyEngine(
            self.equation_engine, self.note_mapper, self.synth,
            instrument=self.instrument_combo.currentText(),
            note_duration=self.duration_input.value(),
            gain=self.gain_input.value(),
            sample_rate=self.sample_rate,
        )

    # ------------------------------------------------------------

    def _build_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        self.role_input = QLineEdit("Ana Ses")
        self.role_input.setStyleSheet(
            f"color: {self.color_hex}; font-weight: bold; border: none; "
            "background: transparent; font-size: 13px;"
        )
        layout.addWidget(self.role_input, 0, 0, 1, 2)

        self.badge_label = QLabel("")
        self.badge_label.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self.badge_label, 0, 2, 1, 2)

        self.download_button = QPushButton("⬇")
        self.download_button.setFixedWidth(30)
        self.download_button.setToolTip("Bu sesi WAV olarak indir")
        layout.addWidget(self.download_button, 0, 4)

        self.remove_button = QPushButton("✕")
        self.remove_button.setFixedWidth(30)
        layout.addWidget(self.remove_button, 0, 5)

        layout.addWidget(QLabel("Denklem:"), 1, 0)
        self.equation_input = QLineEdit(DEFAULT_EQUATION)
        layout.addWidget(self.equation_input, 1, 1, 1, 5)

        layout.addWidget(QLabel("Enstrüman:"), 2, 0)
        self.instrument_combo = QComboBox()
        self.instrument_combo.addItems(INSTRUMENTS)
        layout.addWidget(self.instrument_combo, 2, 1)

        layout.addWidget(QLabel("Gam:"), 2, 2)
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(SCALES)
        layout.addWidget(self.scale_combo, 2, 3)

        layout.addWidget(QLabel("Baz Hz:"), 2, 4)
        self.freq_input = QDoubleSpinBox()
        self.freq_input.setRange(50, 2000)
        self.freq_input.setValue(220)
        layout.addWidget(self.freq_input, 2, 5)

        layout.addWidget(QLabel("Nota süresi (sn):"), 3, 0)
        self.duration_input = QDoubleSpinBox()
        self.duration_input.setRange(0.05, 2.0)
        self.duration_input.setSingleStep(0.05)
        self.duration_input.setValue(0.35)
        layout.addWidget(self.duration_input, 3, 1)

        layout.addWidget(QLabel("Ses seviyesi:"), 3, 2)
        self.gain_input = QDoubleSpinBox()
        self.gain_input.setRange(0.0, 2.0)
        self.gain_input.setSingleStep(0.05)
        self.gain_input.setValue(1.0)
        layout.addWidget(self.gain_input, 3, 3)

        self.play_button = QPushButton("▶ Tekli Çal")
        layout.addWidget(self.play_button, 3, 4)

        self.stop_button = QPushButton("⏹ Durdur")
        layout.addWidget(self.stop_button, 3, 5)

        self.waveform_plot = WaveformPlot(color=self.color_hex, sample_rate=self.sample_rate)
        self.spectrum_plot = SpectrumPlot(color=self.color_hex)
        self.vector_plot = VectorPlot(color=self.color_hex)

        graphs = QHBoxLayout()
        graphs.addWidget(self._titled(self.waveform_plot, "TON KAYIT İSTASYONU"))
        graphs.addWidget(self._titled(self.spectrum_plot, "SPEKTRUM"))
        graphs.addWidget(self._titled(self.vector_plot, "VEKTÖR MATRİSİ"))

        layout.addLayout(graphs, 4, 0, 1, 6)

    @staticmethod
    def _titled(widget, title):
        box = QFrame()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        label = QLabel(title)
        label.setStyleSheet("font-size: 9px; color: #5f9d6f; letter-spacing: 1px;")
        v.addWidget(label)
        v.addWidget(widget)
        return box

    # ------------------------------------------------------------

    def _apply_border(self, active):
        width = 2 if active else 1
        opacity = "" if active else "88"  # pasifken hafif soluk kenarlık
        self.setStyleSheet(
            f"QFrame#row {{ border: {width}px solid {self.color_hex}{opacity}; }}"
        )

    def set_playing(self, is_playing):
        self.is_playing = is_playing
        self._apply_border(active=is_playing)
        self.badge_label.setText("🔊 ÇALIYOR" if is_playing else "")
        if not is_playing:
            self.clear_graphs()

    def clear_graphs(self):
        self.waveform_plot.reset_ring()
        self.spectrum_plot.clear()
        self.vector_plot.clear()

    def apply_settings(self):
        """Arayüzdeki güncel değerleri motora aktarır (her Çal/Birleştir'de)."""
        self.equation_engine.set_equation(self.equation_input.text())
        self.equation_engine.role = self.role_input.text()
        self.note_mapper.base_frequency = self.freq_input.value()
        self.note_mapper.set_scale(self.scale_combo.currentText())
        self.melody_engine.instrument = self.instrument_combo.currentText()
        self.melody_engine.note_duration = self.duration_input.value()
        self.melody_engine.gain = self.gain_input.value()

    def make_fresh_melody_engine(self):
        """İndirme (offline render) için canlı akışa dokunmayan, aynı
        ayarlara sahip TAZE bir MelodyEngine kopyası üretir."""
        self.apply_settings()
        return MelodyEngine(
            self.equation_engine, self.note_mapper, self.synth,
            instrument=self.melody_engine.instrument,
            note_duration=self.melody_engine.note_duration,
            gain=self.melody_engine.gain,
            sample_rate=self.sample_rate,
        )

    def update_waveform(self, wave_chunk):
        self.waveform_plot.push_audio(wave_chunk)

    def update_heavy_graphs(self, wave_chunk):
        """Spektrum + vektör grafiklerini günceller. Denklem eğrisini
        (xs, ys) SADECE BURADA bir kez hesaplar ve döndürür; MainWindow
        bu sonucu COMBINED panel için tekrar (aynı denklemi ikinci kez
        eval etmeden) kullanır."""
        self.spectrum_plot.update_from_audio(wave_chunk)

        elapsed = self.melody_engine._next_note_sample / self.sample_rate
        window = np.linspace(max(0.0, elapsed - 2.0), elapsed + 0.01, 100)
        xs, ys = self.equation_engine.evaluate_array(window)
        self.vector_plot.update_from_xy(xs, ys)
        return xs, ys


class MainWindow(QMainWindow):

    def __init__(self, synth, audio_engine):
        super().__init__()
        self.setWindowTitle("Sinyal Laboratuvarı — Matematiksel Enstrüman Sentezleyici")
        self.resize(1320, 900)
        self.setStyleSheet(DARK_STYLESHEET)

        self.sample_rate = 44100
        self.synth = synth
        self.audio_engine = audio_engine
        self.audio_engine.on_chunk = self._on_chunk

        self.rows = []
        self._row_counter = 0
        self._last_combined = None
        self._last_per_voice = {}
        self._tick_counter = 0

        self._build_ui()
        self.add_row()

        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh_graphs)
        self.timer.start(FAST_TICK_MS)

    # ------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        header = QHBoxLayout()
        title = QLabel("SİNYAL LABORATUVARI")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #39ff14; letter-spacing: 2px;")
        header.addWidget(title)
        header.addStretch()

        self.add_button = QPushButton("+ Denklem Ekle")
        self.add_button.clicked.connect(self.add_row)
        header.addWidget(self.add_button)

        main_layout.addLayout(header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.rows_container = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_container)
        self.rows_layout.addStretch()
        self.scroll.setWidget(self.rows_container)
        main_layout.addWidget(self.scroll, stretch=3)

        # ---------------- COMBINED PANEL ----------------
        combined_frame = QFrame()
        combined_frame.setStyleSheet("QFrame { border: 1px solid #ffffff33; border-radius: 8px; }")
        combined_layout = QVBoxLayout(combined_frame)

        combined_header = QHBoxLayout()
        combined_label = QLabel("COMBINED")
        combined_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffffff; letter-spacing: 2px;")
        combined_header.addWidget(combined_label)
        combined_header.addStretch()

        self.combine_button = QPushButton("🎛 Birleştir ve Çal")
        self.combine_button.clicked.connect(self.play_combined)
        combined_header.addWidget(self.combine_button)

        self.stop_all_button = QPushButton("⏹ Tümünü Durdur")
        self.stop_all_button.clicked.connect(self.stop_all)
        combined_header.addWidget(self.stop_all_button)

        self.combined_download_button = QPushButton("⬇ Birleşimi İndir")
        self.combined_download_button.clicked.connect(self.download_combined)
        combined_header.addWidget(self.combined_download_button)

        combined_layout.addLayout(combined_header)

        self.legend_layout = QHBoxLayout()
        combined_layout.addLayout(self.legend_layout)

        self.combined_waveform = MultiWaveformPlot(sample_rate=self.sample_rate)
        self.combined_spectrum = MultiSpectrumPlot()
        self.combined_vector = MultiVectorPlot()

        combined_graphs = QHBoxLayout()
        combined_graphs.addWidget(self._titled(self.combined_waveform, "TON KAYIT İSTASYONU"))
        combined_graphs.addWidget(self._titled(self.combined_spectrum, "SPEKTRUM"))
        combined_graphs.addWidget(self._titled(self.combined_vector, "VEKTÖR MATRİSİ"))
        combined_layout.addLayout(combined_graphs)

        main_layout.addWidget(combined_frame, stretch=2)

    @staticmethod
    def _titled(widget, title):
        box = QFrame()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        label = QLabel(title)
        label.setStyleSheet("font-size: 9px; color: #cfcfcf; letter-spacing: 1px;")
        v.addWidget(label)
        v.addWidget(widget)
        return box

    # ------------------------------------------------------------

    def add_row(self):
        self._row_counter += 1
        row_id = f"ses_{self._row_counter}"
        index = len(self.rows)

        row = EquationRow(row_id, index, self.synth, sample_rate=self.sample_rate)
        row.remove_button.clicked.connect(lambda: self.remove_row(row))
        row.play_button.clicked.connect(lambda: self.play_single(row))
        row.stop_button.clicked.connect(self.stop_all)
        row.download_button.clicked.connect(lambda: self.download_single(row))

        self.rows.append(row)
        self.rows_layout.insertWidget(self.rows_layout.count() - 1, row)
        self._rebuild_legend()

    def remove_row(self, row):
        if row in self.rows:
            self.rows.remove(row)
            row.setParent(None)
            row.deleteLater()
            self._rebuild_legend()

    def _rebuild_legend(self):
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for row in self.rows:
            label = QLabel(f"● {row.role_input.text() or row.row_id}")
            label.setStyleSheet(f"color: {row.color_hex}; font-size: 11px; font-weight: bold;")
            self.legend_layout.addWidget(label)
        self.legend_layout.addStretch()

    # ------------------------------------------------------------

    def play_single(self, row):
        row.apply_settings()

        for r in self.rows:
            r.set_playing(r is row)

        self.audio_engine.set_voices({row.row_id: row.melody_engine})
        self.audio_engine.start()

    def play_combined(self):
        if not self.rows:
            return

        for row in self.rows:
            row.apply_settings()
            row.set_playing(True)

        voices = {row.row_id: row.melody_engine for row in self.rows}
        self.audio_engine.set_voices(voices)
        self.audio_engine.start()

    def stop_all(self):
        self.audio_engine.stop()
        for row in self.rows:
            row.set_playing(False)
        self.combined_waveform.reset_rings()

    # ------------------------------------------------------------

    def download_single(self, row):
        def build_voices():
            engine = row.make_fresh_melody_engine()
            return {row.row_id: engine}

        dialog = DownloadDialog(
            f"'{row.role_input.text() or row.row_id}' sesini indir",
            build_voices, sample_rate=self.sample_rate, parent=self
        )
        dialog.exec()

    def download_combined(self):
        def build_voices():
            return {row.row_id: row.make_fresh_melody_engine() for row in self.rows}

        dialog = DownloadDialog(
            "Birleştirilmiş tüm denklemleri indir",
            build_voices, sample_rate=self.sample_rate, parent=self
        )
        dialog.exec()

    # ------------------------------------------------------------

    def _on_chunk(self, combined_wave, per_voice):
        self._last_combined = combined_wave
        self._last_per_voice = per_voice

    def _refresh_graphs(self):
        if self._last_combined is None:
            return

        self._tick_counter += 1
        heavy = (self._tick_counter % HEAVY_TICK_EVERY == 0)

        colors = {row.row_id: row.color for row in self.rows}

        # Hızlı (her tik): kayan osiloskoplar
        for row in self.rows:
            wave = self._last_per_voice.get(row.row_id)
            if wave is not None:
                row.update_waveform(wave)

        self.combined_waveform.push(self._last_per_voice, colors)

        # Ağır (her N tikte bir): spektrum + vektör. Denklem eğrisi
        # (xs, ys) her satır için SADECE BİR KEZ hesaplanır
        # (update_heavy_graphs içinde) ve hem satırın kendi vektör
        # grafiğinde hem de COMBINED vektör grafiğinde tekrar
        # kullanılır — eskiden aynı eval() ikinci kez de çağrılıyordu.
        if heavy:
            xy_dict = {}
            for row in self.rows:
                wave = self._last_per_voice.get(row.row_id)
                if wave is not None:
                    xs, ys = row.update_heavy_graphs(wave)
                    xy_dict[row.row_id] = (xs, ys)

            self.combined_spectrum.push(self._last_per_voice, colors)
            self.combined_vector.push(xy_dict, colors)

    def closeEvent(self, event):
        self.audio_engine.hard_stop()
        super().closeEvent(event)


def run_app(synth, audio_engine):
    import sys
    app = QApplication(sys.argv)
    window = MainWindow(synth, audio_engine)
    window.show()
    sys.exit(app.exec())
