import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QLinearGradient, QBrush
from PyQt6.QtCore import Qt


class RingBuffer:
    """Gerçek dairesel tampon: O(chunk_len) yazma, O(max_points) okuma.

    ÖNCEKİ SÜRÜMDEKİ SORUN: her push'ta `np.concatenate([ring[n:], chunk])`
    ile TÜM geçmiş (örn. 3 saniye @44100 = 132300 eleman) yeniden
    kopyalanıyordu — saniyede onlarca MB gereksiz memmove. Burada artık
    sadece gelen küçük chunk (örn. 2048 örnek) yazılıyor; okuma da
    sadece ekranda gösterilecek kadar (max_points) örnek çekiyor, tüm
    tarihçeyi hiç kopyalamadan (modüler indeksleme ile)."""

    __slots__ = ("length", "buf", "write_pos", "filled")

    def __init__(self, length):
        self.length = max(1, int(length))
        self.buf = np.zeros(self.length, dtype=np.float32)
        self.write_pos = 0
        self.filled = False

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
        """Tüm tarihçeyi hiç birleştirmeden (concatenate), doğrudan
        modüler (wrap-around) fancy-index ile örnekleyip döndürür."""
        if not self.filled:
            n = self.write_pos
            if n < 2:
                return np.zeros(2, dtype=np.float32)
            count = min(max_points, n)
            idx = np.linspace(0, n - 1, count).astype(np.int64)
            return self.buf[idx]

        count = min(max_points, self.length)
        idx = np.linspace(0, self.length - 1, count).astype(np.int64)
        idx = (self.write_pos + idx) % self.length
        return self.buf[idx]

    def reset(self):
        self.buf[:] = 0
        self.write_pos = 0
        self.filled = False


class NeonLinePlot(QWidget):
    """Siyah zemin üzerinde neon çizgili temel grafik widget'ı.

    PERFORMANS NOTU (bu sürümdeki değişiklik): Önceki sürüm her nokta
    için ayrı bir QPointF nesnesi oluşturup path.lineTo(QPointF)
    çağırıyordu — n nokta için n QPointF nesnesi + n Python fonksiyon
    çağrısı. Artık:
      1) Koordinat dönüşümü (normalize -1..1 -> piksel) TEK SEFERDE,
         numpy ile vektörize yapılıyor (Python döngüsü yok).
      2) QPointF hiç oluşturulmuyor; path.lineTo(x, y) float
         overload'u doğrudan kullanılıyor (nesne oluşturma maliyeti
         sıfır).
      3) QPainterPath.reserve(n) ile iç dizi baştan büyük ayrılıyor,
         lineTo çağrıları sırasında tekrar tekrar yeniden ayırma
         (realloc) olmuyor.
    Kalan tek Python döngüsü sadece path.lineTo çağrılarının kendisi;
    bu artık en ucuz kısım."""

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
        margin = 6
        span_x = max(1, w - 2 * margin)
        span_y = max(1, h - 2 * margin)

        for _, entry in self._series.items():
            xs, ys, color, fade = entry
            n = len(xs)
            if n < 2:
                continue

            # --- Vektörize koordinat dönüşümü (Python döngüsü yok) ---
            px = margin + (xs * 0.5 + 0.5) * span_x
            py = margin + (0.5 - ys * 0.5) * span_y
            px_list = px.tolist()
            py_list = py.tolist()

            path = QPainterPath()
            if hasattr(path, "reserve"):
                path.reserve(n)
            path.moveTo(px_list[0], py_list[0])
            for x, y in zip(px_list[1:], py_list[1:]):
                path.lineTo(x, y)

            if fade:
                grad = QLinearGradient(0, 0, w, 0)
                faint = QColor(color)
                faint.setAlpha(20)
                bright = QColor(color)
                bright.setAlpha(235)
                grad.setColorAt(0.0, faint)
                grad.setColorAt(1.0, bright)
                brush = QBrush(grad)
            else:
                brush = QBrush(color)

            # PERFORMANS: profil çıkardığımda gerçek maliyetin gradyan
            # DEĞİL, geniş (6px) çizginin anti-alias'lı birleşim
            # (join)/uç (cap) hesaplaması olduğunu gördüm — nokta
            # sayısıyla doğru orantılı büyüyor. Glow katmanı zaten
            # bulanık/yarı saydam olduğu için AA'sız çizilse de fark
            # edilmiyor; ~4-6x daha hızlı. İnce, parlak çekirdek
            # çizgide netlik için AA açık bırakılıyor (o da ucuz,
            # çünkü çizgi ince).
            glow_pen = QPen(brush, 6)
            glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            glow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.setOpacity(0.30)
            painter.setPen(glow_pen)
            painter.drawPath(path)

            core_pen = QPen(brush, 1.8)
            core_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            core_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setOpacity(1.0)
            painter.setPen(core_pen)
            painter.drawPath(path)


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

    def push_audio(self, audio_chunk, max_points=90):
        chunk = np.asarray(audio_chunk, dtype=np.float32)
        if len(chunk) == 0:
            return

        self._ring.write(chunk)
        display = self._ring.read_downsampled(max_points)

        xs = np.linspace(-1, 1, len(display))
        ys = np.clip(display, -1, 1)
        self.set_points(xs, ys)

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

    def push(self, per_voice_waves, colors, max_points=80):
        series = {}
        for name, chunk in per_voice_waves.items():
            ring = self._rings.get(name)
            if ring is None:
                ring = RingBuffer(self.history_len)
                self._rings[name] = ring
            ring.write(chunk)

            display = ring.read_downsampled(max_points)
            xs = np.linspace(-1, 1, len(display))
            ys = np.clip(display, -1, 1)
            series[name] = (xs, ys, colors.get(name, self._default_color), True)

        for stale in list(self._rings.keys()):
            if stale not in per_voice_waves:
                del self._rings[stale]

        self.set_series(series)

    def reset_rings(self):
        self._rings = {}
        self.clear()


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
