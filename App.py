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


INSTRUMENTS = ["gitar", "bowed_guitar", "ronroco", "ukulele", "bas_gitar", "kick_bass", "piyano", "ney", "kalimba", "flut"]
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

# Grafik zamanlayıcısı yaklaşık 60 FPS hedefler.
FAST_TICK_MS = 16
HEAVY_TICK_EVERY = 4

# Varsayılan kelebek eğrisi için standart parametrik aralık 0..12π.
# AudioEngine.playback_position_seconds bu aralığın çizim ilerlemesini sürer.
VECTOR_T_END = 12.0 * np.pi
VECTOR_POINT_COUNT = 1800


class EquationRow(QFrame):
    """Tek bir denklem satırı = tek bir ses (voice): rol, denklem,
    enstrüman, gam, baz frekans, nota süresi, ses seviyesi + kendi
    ton kayıt istasyonu / spektrum / vektör matrisi grafikleri."""

    def __init__(self, row_id, index, synth, sample_rate=44100, parent=None):
        super().__init__(parent)
        self.setObjectName("row")
        self.row_id = row_id
        self.sample_rate = sample_rate
        self.synth = synth
        self.color_hex = NEON_COLORS[index % len(NEON_COLORS)]
        self.color = QColor(self.color_hex)
        self.is_playing = False

        self.vector_t_end = VECTOR_T_END
        self.vector_point_count = VECTOR_POINT_COUNT
        self._vector_path_prepared = False
        self._vector_xs = None
        self._vector_ys = None

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

        self.waveform_plot = WaveformPlot(
            color=self.color_hex, sample_rate=self.sample_rate
        )
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
        opacity = "" if active else "88"
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
        self._vector_path_prepared = False
        self._vector_xs = None
        self._vector_ys = None

    def apply_settings(self):
        """Arayüzdeki güncel değerleri motora aktarır (her Çal/Birleştir'de)."""
        self.equation_engine.set_equation(self.equation_input.text())
        self.equation_engine.role = self.role_input.text()
        self.note_mapper.base_frequency = self.freq_input.value()
        self.note_mapper.set_scale(self.scale_combo.currentText())
        self.melody_engine.instrument = self.instrument_combo.currentText()
        self.melody_engine.note_duration = self.duration_input.value()
        self.melody_engine.gain = self.gain_input.value()
        self._vector_path_prepared = False

    def prepare_vector_path(self):
        """Denklemin sabit çizim yolunu çalma başlamadan bir kez hesaplar."""
        t_samples = np.linspace(
            0.0,
            self.vector_t_end,
            self.vector_point_count,
            dtype=np.float64
        )

        xs, ys = self.equation_engine.evaluate_array(t_samples)
        self._vector_xs = np.asarray(xs, dtype=np.float64)
        self._vector_ys = (
            None if ys is None else np.asarray(ys, dtype=np.float64)
        )

        self.vector_plot.update_from_xy(self._vector_xs, self._vector_ys)
        self.vector_plot.set_draw_progress(0.0)
        self._vector_path_prepared = True

        return self._vector_xs, self._vector_ys

    def update_spectrum(self, wave_chunk):
        """FFT spektrumu, vektör çiziminden bağımsız güncellenir."""
        self.spectrum_plot.update_from_audio(wave_chunk)

    def update_heavy_graphs(self, wave_chunk):
        """Eski çağrı noktalarıyla uyumluluk. Denklem yolu kaydırılmaz."""
        self.update_spectrum(wave_chunk)
        if not self._vector_path_prepared:
            return self.prepare_vector_path()
        return self._vector_xs, self._vector_ys

    def update_waveform(self, wave_chunk):
        self.waveform_plot.push_audio(wave_chunk)

    def make_fresh_melody_engine(self):
        """İndirme için canlı akışa dokunmayan, aynı ayarlara sahip kopya üretir."""
        self.apply_settings()
        return MelodyEngine(
            self.equation_engine, self.note_mapper, self.synth,
            instrument=self.melody_engine.instrument,
            note_duration=self.melody_engine.note_duration,
            gain=self.melody_engine.gain,
            sample_rate=self.sample_rate,
        )


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
        self._audio_chunk_version = 0
        # Tek tuple halinde atomik referans değişimi; ses callback'i lock beklemez.
        self._audio_snapshot = (None, {}, 0)
        self._last_visualized_chunk_version = -1
        self._tick_counter = 0
        self._combined_vector_duration = None
        self._accept_audio_chunks = False

        self._build_ui()
        self.add_row()

        self.timer = QTimer(self)
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

    def _prepare_vector_plots(self, rows):
        """Vektör yollarını bir kere üretir ve çizimi sıfırdan başlatır."""
        xy_dict = {}
        colors = {}

        self.combined_vector.clear()

        for row in rows:
            xs, ys = row.prepare_vector_path()
            xy_dict[row.row_id] = (xs, ys)
            colors[row.row_id] = row.color

        if xy_dict:
            self.combined_vector.push(xy_dict, colors)
            self.combined_vector.set_draw_progress(0.0)
            self._combined_vector_duration = max(
                row.vector_t_end for row in rows
            )
        else:
            self._combined_vector_duration = None

    # ------------------------------------------------------------

    def play_single(self, row):
        row.apply_settings()

        for other_row in self.rows:
            other_row.set_playing(other_row is row)

        self._prepare_vector_plots([row])
        self._accept_audio_chunks = False
        self.audio_engine.set_voices({row.row_id: row.melody_engine})
        self.audio_engine.start()
        self._accept_audio_chunks = True

    def play_combined(self):
        if not self.rows:
            return

        for row in self.rows:
            row.apply_settings()
            row.set_playing(True)

        self._prepare_vector_plots(self.rows)
        voices = {row.row_id: row.melody_engine for row in self.rows}
        self._accept_audio_chunks = False
        self.audio_engine.set_voices(voices)
        self.audio_engine.start()
        self._accept_audio_chunks = True

    def stop_all(self):
        self._accept_audio_chunks = False
        self.audio_engine.stop()
        for row in self.rows:
            row.set_playing(False)

        self.combined_waveform.reset_rings()
        self.combined_spectrum.clear()
        self.combined_vector.clear()
        self._combined_vector_duration = None
        self._last_combined = None
        self._last_per_voice = {}
        self._audio_snapshot = (None, {}, self._audio_chunk_version)
        self._last_visualized_chunk_version = self._audio_chunk_version

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
            return {
                row.row_id: row.make_fresh_melody_engine()
                for row in self.rows
            }

        dialog = DownloadDialog(
            "Birleştirilmiş tüm denklemleri indir",
            build_voices, sample_rate=self.sample_rate, parent=self
        )
        dialog.exec()

    # ------------------------------------------------------------

    def _on_chunk(self, combined_wave, per_voice):
        # Callback ses thread'inden gelir. QWidget'e dokunmadan yalnızca son bloğu aktar.
        if not self._accept_audio_chunks:
            return

        self._audio_chunk_version += 1
        # Python referans ataması atomiktir; ses callback'i GUI kilidinde beklemez.
        self._audio_snapshot = (
            combined_wave,
            per_voice,
            self._audio_chunk_version
        )
        self._last_combined = combined_wave
        self._last_per_voice = per_voice

    def _refresh_graphs(self):
        # Ses thread'inin tek hamlede yayınladığı son blok görüntüsünü al.
        combined_wave, per_voice_snapshot, chunk_version = self._audio_snapshot
        per_voice = dict(per_voice_snapshot)

        if combined_wave is None:
            return

        self._tick_counter += 1
        heavy = (self._tick_counter % HEAVY_TICK_EVERY == 0)
        colors = {row.row_id: row.color for row in self.rows}

        # Aynı ses bloğunu 60 FPS boyunca tekrar tekrar RingBuffer'a yazma.
        # Waveform'lar yalnızca yeni AudioEngine bloğu geldiğinde beslenir.
        new_audio_chunk = chunk_version != self._last_visualized_chunk_version
        if new_audio_chunk:
            for row in self.rows:
                wave = per_voice.get(row.row_id)
                if wave is not None:
                    row.update_waveform(wave)

            self.combined_waveform.push(per_voice, colors)
            self._last_visualized_chunk_version = chunk_version

        # Tek zaman kaynağı: AudioEngine'in DAC zamanından hesaplanan oynatma konumu.
        playback_time = self.audio_engine.playback_position_seconds

        # Kalemin ilerlemesi her GUI karesinde güncellenir; geometri sabit kalır.
        for row in self.rows:
            if row.is_playing and row._vector_path_prepared:
                progress = np.clip(
                    playback_time / max(row.vector_t_end, 1e-9),
                    0.0,
                    1.0
                )
                row.vector_plot.set_draw_progress(progress)

        if self._combined_vector_duration is not None:
            combined_progress = np.clip(
                playback_time / max(self._combined_vector_duration, 1e-9),
                0.0,
                1.0
            )
            self.combined_vector.set_draw_progress(combined_progress)

        # FFT yalnızca yeni bir blok geldiğinde ve ağır tikte çalışır.
        if heavy and new_audio_chunk:
            for row in self.rows:
                wave = per_voice.get(row.row_id)
                if wave is not None:
                    row.update_spectrum(wave)

            self.combined_spectrum.push(per_voice, colors)

    def closeEvent(self, event):
        self.audio_engine.hard_stop()
        super().closeEvent(event)


def run_app(synth, audio_engine):
    import sys
    app = QApplication(sys.argv)
    window = MainWindow(synth, audio_engine)
    window.show()
    sys.exit(app.exec())
