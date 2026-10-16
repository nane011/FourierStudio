import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QLinearGradient, QBrush
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPolygonF
from PyQt6.QtCore import QPointF
import time

class RingBuffer:
    """Gerçek dairesel tampon: O(chunk_len) yazma, O(max_points) okuma."""

    # Değişken isimleri slots içinde ve metotlarda tamamen eşitlendi
    __slots__ = ("length", "buf", "write_pos", "filled", "_cached_count", "_cached_idx_filled", "_cached_idx_unfilled")

    def __init__(self, length):
        self.length = max(1, int(length))
        self.buf = np.zeros(self.length, dtype=np.float32)
        self.write_pos = 0
        self.filled = False
        
        # Performans önbelleği başlangıç değerleri
        self._cached_count = -1
        self._cached_idx_filled = None
        self._cached_idx_unfilled = {}

    def write(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32)
        n = len(chunk)
        if n == 0:
            return

        if n >= self.length:
            self.buf[:] = chunk[-self.length:]
            self.write_pos = 0
            self.filled = True
            return

        end = self.write_pos + n
        if end <= self.length:
            self.buf[self.write_pos:end] = chunk
        else:
            first = self.length - self.write_pos
            self.buf[self.write_pos:] = chunk[:first]
            self.buf[:end - self.length] = chunk[first:]

        if end >= self.length:
            self.filled = True

        self.write_pos = end % self.length

    def read_downsampled(self, max_points):
        """Hafıza korumalı (cache) fancy-index yapısı ile işlemciyi hiç yormadan
        verileri hızlıca seyreltip grafik motoruna teslim eder."""
        
        if not self.filled:
            n = self.write_pos
            if n < 2:
                return np.zeros(2, dtype=np.float32)
            
            count = min(max_points, n)
            
            cache_key = (n, count)
            if cache_key not in self._cached_idx_unfilled:
                self._cached_idx_unfilled[cache_key] = np.linspace(0, n - 1, count).astype(np.int64)
            
            idx = self._cached_idx_unfilled[cache_key]
            return self.buf[idx]

        # Tampon tamamen dolduğunda tetiklenen ultra hızlı alan
        count = min(max_points, self.length)
        
        # Eğer max_points değişmediyse linspace'i baştan hesaplama, hafızadan çağır!
        if self._cached_count != count:
            self._cached_count = count
            self._cached_idx_filled = np.linspace(0, self.length - 1, count).astype(np.int64)

        # Modüler dairesel indeks kaydırması
        idx = (self.write_pos + self._cached_idx_filled) % self.length
        return self.buf[idx]

    def reset(self):
        self.buf[:] = 0
        self.write_pos = 0
        self.filled = False
        # Temizlik anında önbellek alanlarını da sıfırlıyoruz
        self._cached_count = -1
        self._cached_idx_unfilled = {}
        self._cached_idx_filled = None

class NeonLinePlot(QWidget):
    def __init__(self, color="#39ff14", parent=None):
        super().__init__(parent)
        self._series = {}   # {isim: (xs_array, ys_array, QColor, fade_bool)}
        self._default_color = QColor(color)
        self.setMinimumSize(160, 120)
        self.setStyleSheet("background-color: #000000; border-radius: 6px;")

    def set_color(self, color):
        self._default_color = QColor(color)

    def clear(self):
        self._series = {}
        self.update()

    def set_points(self, xs, ys, fade=True):
        """Tek seri (geriye dönük uyumluluk için). xs, ys: -1..1
        aralığında normalize edilmiş dizi/liste."""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        self._series = {"_default": (xs, ys, self._default_color, fade)}
        self.update()

    def set_series(self, series_dict):
        """series_dict: {isim: (xs_array, ys_array, color, fade_bool)}"""
        self._series = series_dict
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))

        w = self.width()
        h = self.height()

        # Önceki 6 px yerine 2 px: çizgi alanı biraz uzar.
        margin = 2

        available_w = max(1, w - 2 * margin)
        available_h = max(1, h - 2 * margin)
        square_size = min(available_w, available_h)

        offset_x = margin + (available_w - square_size) / 2
        offset_y = margin + (available_h - square_size) / 2

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing, True
        )

        for _, entry in self._series.items():
            xs, ys, color, fade = entry
            n = len(xs)

            if n < 2:
                continue

            px = offset_x + (xs * 0.5 + 0.5) * square_size
            py = offset_y + (0.5 - ys * 0.5) * square_size

            points = [
                QPointF(float(x), float(y))
                for x, y in zip(px, py)
            ]

            path = QPainterPath()
            path.addPolygon(QPolygonF(points))

            # Uçlara doğru saydamlaşan yatay gradient.
            if fade:
                start_color = QColor(color)
                end_color = QColor(color)
                start_color.setAlpha(0)
                end_color.setAlpha(0)

                gradient = QLinearGradient(
                    offset_x, 0,
                    offset_x + square_size, 0
                )
                gradient.setColorAt(0.00, start_color)
                gradient.setColorAt(0.08, color)
                gradient.setColorAt(0.92, color)
                gradient.setColorAt(1.00, end_color)

                brush = QBrush(gradient)
            else:
                brush = QBrush(color)

            # Neon dış parıltısı.
            glow_pen = QPen(brush, 6)
            glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            glow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

            painter.setOpacity(0.25)
            painter.setPen(glow_pen)
            painter.drawPath(path)

            # Keskin, ana çizgi.
            core_pen = QPen(brush, 1.8)
            core_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            core_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

            painter.setOpacity(1.0)
            painter.setPen(core_pen)
            painter.drawPath(path)

            painter.setOpacity(1.0)

# ------------------------------------------------------------------
# TEKLİ (tek renk) grafikler — her denklem satırının kendi paneli
# ------------------------------------------------------------------

class WaveformPlot(NeonLinePlot):
    """Ton kayıt istasyonu — EKG tarzı KAYAN osiloskop. Gerçek dairesel
    tampon (RingBuffer) kullanıyor; her push'ta sadece gelen küçük
    chunk yazılıyor, tüm 3 saniyelik geçmiş asla yeniden kopyalanmıyor."""

    def __init__(self, color="#39ff14", history_seconds=3.0, sample_rate=44100, parent=None):
        super().__init__(color=color, parent=parent)
        self.sample_rate = sample_rate
        self._ring = RingBuffer(int(history_seconds * sample_rate))
        self._last_draw_time = 0.0
        self._draw_interval = 1.0 / 30.0

    def push_audio(self, audio_chunk, max_points=350):
        chunk = np.asarray(audio_chunk, dtype=np.float32)
        if len(chunk) == 0:
            return

        # Ses verisini her zaman tamponla.
        self._ring.write(chunk)

        # Ekranı en fazla 30 FPS hızında güncelle.
        now = time.perf_counter()
        if now - self._last_draw_time < self._draw_interval:
            return

        self._last_draw_time = now

        display = self._ring.read_downsampled(max_points)
        n_display = len(display)

        if (
            not hasattr(self, "_cached_xs")
            or len(self._cached_xs) != n_display
        ):
            self._cached_xs = np.linspace(
                -1, 1, n_display, dtype=np.float64
            )

        ys = np.clip(display, -1, 1)
        self.set_points(self._cached_xs, ys)

    # eski API ile uyum
    def update_from_audio(self, audio_buffer):
        self.push_audio(audio_buffer)

    def reset_ring(self):
        self._ring.reset()
        self.clear()


class SpectrumPlot(NeonLinePlot):
    """Radyal simetrik spektrum. Açı tepeden (-90°) başlıyor ki şekil
    yana yatık değil, izleyiciye 'karşı duruşta' görünsün."""

    def update_from_audio(self, audio_buffer, n_bins=24):
        audio_buffer = np.asarray(audio_buffer, dtype=np.float64)
        n = len(audio_buffer)
        if n < 8:
            return

        spectrum = np.abs(np.fft.rfft(audio_buffer * np.hanning(n)))
        if len(spectrum) < 2:
            return

        idx = np.linspace(1, len(spectrum) - 1, n_bins).astype(int)
        mags = spectrum[idx]
        if mags.max() > 0:
            mags = mags / mags.max()

        angles = np.linspace(0, 2 * np.pi, n_bins, endpoint=True) - np.pi / 2
        radius = 0.15 + 0.85 * mags
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        self.set_points(xs, ys)


class VectorPlot(NeonLinePlot):
    """Parametrik (x, y) eğrisi için 'vektör matrisi' grafiği."""

    def update_from_xy(self, xs, ys=None):
        xs = np.asarray(xs, dtype=np.float64)
        if ys is None:
            ys = np.roll(xs, max(1, len(xs) // 20))
        else:
            ys = np.asarray(ys, dtype=np.float64)

        if len(xs) == 0:
            return

        max_r = max(1e-6, float(np.max(np.abs(xs))), float(np.max(np.abs(ys))))
        xs_n = xs / max_r
        ys_n = ys / max_r
        self.set_points(xs_n, ys_n)


# ------------------------------------------------------------------
# ÇOKLU (renkli overlay) grafikler — COMBINED panel için
# ------------------------------------------------------------------

class MultiWaveformPlot(NeonLinePlot):
    """Her ses (voice) için ayrı bir RingBuffer tutar, hepsini KENDİ
    RENGİNDE aynı koordinat sisteminde üst üste çizer."""

    def __init__(self, history_seconds=3.0, sample_rate=44100, parent=None):
        super().__init__(parent=parent)
        self.sample_rate = sample_rate
        self.history_len = int(history_seconds * sample_rate)
        self._rings = {}   # isim -> RingBuffer
        self._xs_cache = {}
        self._last_draw_time = 0.0
        self._draw_interval = 1.0 / 30.0

    def push(self, per_voice_waves, colors, max_points=300):
        # 1. Gelen bütün ses parçalarını kaydet.
        for name, chunk in per_voice_waves.items():
            ring = self._rings.get(name)
    
            if ring is None:
                ring = RingBuffer(self.history_len)
                self._rings[name] = ring
    
            ring.write(chunk)
    
        # Artık bulunmayan kanalları temizle.
        for stale in list(self._rings.keys()):
            if stale not in per_voice_waves:
                del self._rings[stale]
    
        # 2. Çizim sıklığını sınırla.
        now = time.perf_counter()
    
        if now - self._last_draw_time < self._draw_interval:
            return
    
        self._last_draw_time = now
    
        # 3. Yalnızca çizim zamanı geldiğinde koordinatları üret.
        series = {}
    
        for name in per_voice_waves:
            ring = self._rings[name]
            display = ring.read_downsampled(max_points)
            n_display = len(display)
    
            xs = self._xs_cache.get(n_display)
    
            if xs is None:
                xs = np.linspace(
                    -1, 1, n_display, dtype=np.float64
                )
                self._xs_cache[n_display] = xs
    
            ys = np.clip(display, -1, 1)
    
            series[name] = (
                xs,
                ys,
                colors.get(name, self._default_color),
                True
            )
    
        self.set_series(series)

    def reset_rings(self):
        """Tüm ses tamponlarını ve çizimleri sıfırlar."""
        self._rings.clear()
        self.clear()
        self._last_draw_time = 0.0

class MultiSpectrumPlot(NeonLinePlot):

    def push(self, per_voice_waves, colors, n_bins=22):
        series = {}
        for name, chunk in per_voice_waves.items():
            chunk = np.asarray(chunk, dtype=np.float64)
            n = len(chunk)
            if n < 8:
                continue
            spectrum = np.abs(np.fft.rfft(chunk * np.hanning(n)))
            if len(spectrum) < 2:
                continue
            idx = np.linspace(1, len(spectrum) - 1, n_bins).astype(int)
            mags = spectrum[idx]
            if mags.max() > 0:
                mags = mags / mags.max()
            angles = np.linspace(0, 2 * np.pi, n_bins, endpoint=True) - np.pi / 2
            radius = 0.15 + 0.85 * mags
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            series[name] = (xs, ys, colors.get(name, self._default_color), True)
        self.set_series(series)


class MultiVectorPlot(NeonLinePlot):

    def push(self, xy_dict, colors):
        series = {}
        for name, pair in xy_dict.items():
            xs, ys = pair
            xs = np.asarray(xs, dtype=np.float64)
            if ys is None:
                ys = np.roll(xs, max(1, len(xs) // 20))
            else:
                ys = np.asarray(ys, dtype=np.float64)
            if len(xs) == 0:
                continue
            max_r = max(1e-6, float(np.max(np.abs(xs))), float(np.max(np.abs(ys))))
            xs_n = xs / max_r
            ys_n = ys / max_r
            series[name] = (xs_n, ys_n, colors.get(name, self._default_color), True)
        self.set_series(series)
