import numpy as np
import time
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QLinearGradient, QBrush,
    QPolygonF
)
from PyQt6.QtCore import Qt, QPointF


class RingBuffer:
    """Gerçek dairesel tampon: O(chunk_len) yazma, O(max_points) okuma."""

    __slots__ = (
        "length", "buf", "write_pos", "filled",
        "_cached_count", "_cached_idx_filled", "_cached_idx_unfilled"
    )

    def __init__(self, length):
        self.length = max(1, int(length))
        self.buf = np.zeros(self.length, dtype=np.float32)
        self.write_pos = 0
        self.filled = False
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
        if not self.filled:
            n = self.write_pos
            if n < 2:
                return np.zeros(2, dtype=np.float32)

            count = min(max(2, int(max_points)), n)
            cache_key = (n, count)
            if cache_key not in self._cached_idx_unfilled:
                self._cached_idx_unfilled[cache_key] = np.linspace(
                    0, n - 1, count
                ).astype(np.int64)

            return self.buf[self._cached_idx_unfilled[cache_key]]

        count = min(max(2, int(max_points)), self.length)
        if self._cached_count != count:
            self._cached_count = count
            self._cached_idx_filled = np.linspace(
                0, self.length - 1, count
            ).astype(np.int64)

        idx = (self.write_pos + self._cached_idx_filled) % self.length
        return self.buf[idx]

    def reset(self):
        self.buf[:] = 0
        self.write_pos = 0
        self.filled = False
        self._cached_count = -1
        self._cached_idx_unfilled = {}
        self._cached_idx_filled = None


class NeonLinePlot(QWidget):
    """Neon çizgi grafiği.

    Koordinatları sabit bir yol olarak saklar. set_draw_progress() yalnızca
    yolun ne kadarının görünür olduğunu değiştirir; eğriyi yeniden şekillendirmez.
    """

    def __init__(self, color="#39ff14", parent=None):
        super().__init__(parent)
        self._series = {}
        self._default_color = QColor(color)
        self.setMinimumSize(160, 120)
        self.setStyleSheet("background-color: #000000; border-radius: 6px;")
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        # None: bütün eğriyi göster. 0..1: sesle senkronize kalem ilerlemesi.
        self._draw_progress = None

        # Hareketli kalemin ekranda bıraktığı sabit uzunluktaki iz.
        # İstenirse bu iki değer doğrudan ayarlanabilir.
        self._trail_length_px = 190.0
        self._trail_fade_px = 30.0
        self._trail_segments = 22

        # Geometri sadece seri/ölçü değiştiğinde üretilir; her karede tekrar kurulmaz.
        self._geometry_cache = {}
        self._geometry_cache_size = None

    def set_color(self, color):
        self._default_color = QColor(color)
        self.update()

    def clear(self):
        self._series = {}
        self._geometry_cache.clear()
        self._geometry_cache_size = None
        self._draw_progress = None
        self.update()

    def set_draw_progress(self, progress):
        """Kalemin ilerlemesini 0..1 aralığında dışarıdan ayarla."""
        self._draw_progress = float(np.clip(progress, 0.0, 1.0))
        self.update()

    def set_points(self, xs, ys, fade=True):
        """Tek seri (geriye dönük uyumluluk için). xs, ys: -1..1."""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        count = min(len(xs), len(ys))

        if count < 2:
            self.clear()
            return

        self._series = {
            "_default": (
                xs[:count].copy(), ys[:count].copy(),
                self._default_color, fade
            )
        }
        self._geometry_cache.clear()
        self._geometry_cache_size = None
        self.update()

    def set_series(self, series_dict):
        """series_dict: {isim: (xs, ys, color, fade[, progress])}."""
        normalized = {}
        for name, entry in series_dict.items():
            if len(entry) == 5:
                xs, ys, color, fade, local_progress = entry
                local_progress = float(np.clip(local_progress, 0.0, 1.0))
            else:
                xs, ys, color, fade = entry
                local_progress = None

            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
            count = min(len(xs), len(ys))
            if count < 2:
                continue

            normalized[name] = (
                xs[:count].copy(), ys[:count].copy(), color, fade,
                local_progress
            )

        self._series = normalized
        self._geometry_cache.clear()
        self._geometry_cache_size = None
        self.update()

    def _build_geometry(self, entry, square_size, offset_x, offset_y):
        xs, ys, color, fade = entry[:4]
        n = min(len(xs), len(ys))
        if n < 2:
            return None

        px = offset_x + (xs[:n] * 0.5 + 0.5) * square_size
        py = offset_y + (0.5 - ys[:n] * 0.5) * square_size

        # QPointF nesneleri bir kere oluşturulur ve sonraki karelerde tekrar kullanılır.
        points = [QPointF(float(x), float(y)) for x, y in zip(px, py)]
        full_polyline = QPolygonF(points)

        dx = np.diff(px)
        dy = np.diff(py)
        segment_lengths = np.hypot(dx, dy)
        cumulative = np.empty(n, dtype=np.float64)
        cumulative[0] = 0.0
        cumulative[1:] = np.cumsum(segment_lengths)
        total_length = float(cumulative[-1])

        return {
            "points": points,
            "polyline": full_polyline,
            "cumulative": cumulative,
            "total_length": total_length,
            "color": color,
            "fade": fade,
        }

    def _get_geometry(self, name, entry, square_size, offset_x, offset_y):
        size_key = (
            self.width(), self.height(), square_size,
            float(offset_x), float(offset_y)
        )
        if self._geometry_cache_size != size_key:
            self._geometry_cache.clear()
            self._geometry_cache_size = size_key

        if name not in self._geometry_cache:
            self._geometry_cache[name] = self._build_geometry(
                entry, square_size, offset_x, offset_y
            )

        return self._geometry_cache[name]

    def _partial_polyline(self, geometry, progress):
        """Aynı sabit yolun yalnızca ilerleme oranına kadar olan bölümünü döndür."""
        if progress >= 0.9999:
            return geometry["polyline"]
        if progress <= 0.0:
            return None

        points = geometry["points"]
        cumulative = geometry["cumulative"]
        total_length = geometry["total_length"]
        if total_length <= 0.0:
            return None

        distance = total_length * progress
        last_full = int(np.searchsorted(cumulative, distance, side="right") - 1)
        last_full = max(0, min(last_full, len(points) - 1))

        # Önceden hesaplanan noktalardan yalnızca görünür prefix alınır.
        visible_points = points[:last_full + 1]

        if last_full < len(points) - 1:
            start_distance = float(cumulative[last_full])
            end_distance = float(cumulative[last_full + 1])
            segment_length = end_distance - start_distance

            if segment_length > 1e-12:
                ratio = float(np.clip(
                    (distance - start_distance) / segment_length,
                    0.0, 1.0
                ))
                p0 = points[last_full]
                p1 = points[last_full + 1]
                end_point = QPointF(
                    p0.x() + (p1.x() - p0.x()) * ratio,
                    p0.y() + (p1.y() - p0.y()) * ratio
                )
                visible_points = visible_points + [end_point]

        if len(visible_points) < 2:
            return None

        return QPolygonF(visible_points)

    def _point_at_distance(self, geometry, distance):
        """Sabit polylinenin yay uzunluğundaki bir konumu bul."""
        points = geometry["points"]
        cumulative = geometry["cumulative"]

        if distance <= 0.0:
            return points[0]
        if distance >= cumulative[-1]:
            return points[-1]

        index = int(np.searchsorted(cumulative, distance, side="right") - 1)
        index = max(0, min(index, len(points) - 2))

        d0 = float(cumulative[index])
        d1 = float(cumulative[index + 1])
        segment_length = d1 - d0
        if segment_length <= 1e-12:
            return points[index + 1]

        ratio = float(np.clip((distance - d0) / segment_length, 0.0, 1.0))
        p0 = points[index]
        p1 = points[index + 1]
        return QPointF(
            p0.x() + (p1.x() - p0.x()) * ratio,
            p0.y() + (p1.y() - p0.y()) * ratio
        )

    def _trail_polylines(self, geometry, progress):
        """Sabit uzunlukta iz döndür; iki uçta opaklık yumuşakça azalır.

        Her parça orijinal polyline üzerindeki gerçek segmentleri kullanır.
        İlerleme büyüdükçe çizilecek nokta sayısı büyümez; maliyet trail boyuyla sınırlıdır.
        """
        total_length = geometry["total_length"]
        if total_length <= 0.0 or progress <= 0.0:
            return []

        end_distance = total_length * float(np.clip(progress, 0.0, 1.0))
        start_distance = max(0.0, end_distance - self._trail_length_px)
        visible_length = end_distance - start_distance
        if visible_length < 1.0:
            return []

        fade_length = min(self._trail_fade_px, visible_length * 0.45)
        chunk_count = max(
            8,
            min(self._trail_segments, int(np.ceil(visible_length / 7.0)))
        )
        distances = np.linspace(start_distance, end_distance, chunk_count + 1)

        points = geometry["points"]
        cumulative = geometry["cumulative"]
        segments = []

        for i in range(chunk_count):
            d0 = float(distances[i])
            d1 = float(distances[i + 1])
            p0 = self._point_at_distance(geometry, d0)
            p1 = self._point_at_distance(geometry, d1)

            # Orijinal köşe noktalarını ara parçaya dahil ederek eğrinin şeklini koru.
            first_inner = int(np.searchsorted(cumulative, d0, side="right"))
            end_inner = int(np.searchsorted(cumulative, d1, side="left"))
            piece_points = [p0]
            if end_inner > first_inner:
                piece_points.extend(points[first_inner:end_inner])
            piece_points.append(p1)

            middle = (d0 + d1) * 0.5
            if fade_length > 0.0:
                fade_factor = min(
                    1.0,
                    (middle - start_distance) / fade_length,
                    (end_distance - middle) / fade_length
                )
            else:
                fade_factor = 1.0

            # Smoothstep: solma iki uçta daha yumuşak başlar.
            fade_factor = float(np.clip(fade_factor, 0.0, 1.0))
            alpha = fade_factor * fade_factor * (3.0 - 2.0 * fade_factor)

            if alpha > 0.003 and len(piece_points) >= 2:
                segments.append((QPolygonF(piece_points), alpha))

        return segments

    def set_trail_style(self, length_px=None, fade_px=None, segments=None):
        """İz uzunluğunu ve uçların solma mesafesini çalışma anında ayarla."""
        if length_px is not None:
            self._trail_length_px = max(10.0, float(length_px))
        if fade_px is not None:
            self._trail_fade_px = max(0.0, float(fade_px))
        if segments is not None:
            self._trail_segments = max(8, min(40, int(segments)))
        self.update()

    def _make_brush(self, color, fade, offset_x, square_size):
        if not fade:
            return QBrush(color)

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
        return QBrush(gradient)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))

        w = self.width()
        h = self.height()
        margin = 2
        available_w = max(1, w - 2 * margin)
        available_h = max(1, h - 2 * margin)
        square_size = min(available_w, available_h)
        offset_x = margin + (available_w - square_size) / 2
        offset_y = margin + (available_h - square_size) / 2

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        for name, entry in self._series.items():
            geometry = self._get_geometry(
                name, entry, square_size, offset_x, offset_y
            )
            if geometry is None:
                continue

            total_length = geometry["total_length"]
            if total_length < 1.0:
                continue

            # Dışarıdan gelen ses ilerlemesi tüm grafik için önceliklidir.
            # İlerleme verilmemişse 5 elemanlı eski seri biçimi desteklenir.
            local_progress = entry[4] if len(entry) >= 5 else None
            if self._draw_progress is not None:
                progress = self._draw_progress
            elif local_progress is not None:
                progress = local_progress
            else:
                progress = 1.0

            brush = self._make_brush(
                geometry["color"], geometry["fade"], offset_x, square_size
            )

            if self._draw_progress is not None or local_progress is not None:
                # Sesle senkronize kalem: yalnızca sabit uzunluklu hareketli iz çizilir.
                # Her karede tüm eğrinin prefix'i yeniden oluşturulmadığı için maliyet sabit kalır.
                trail_segments = self._trail_polylines(geometry, progress)

                for trail_polyline, alpha in trail_segments:
                    glow_pen = QPen(brush, 6)
                    glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    glow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                    painter.setOpacity(0.25 * alpha)
                    painter.setPen(glow_pen)
                    painter.drawPolyline(trail_polyline)

                    core_pen = QPen(brush, 1.8)
                    core_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    core_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                    painter.setOpacity(alpha)
                    painter.setPen(core_pen)
                    painter.drawPolyline(trail_polyline)
            else:
                # İlerlemesi dışarıdan kontrol edilmeyen grafikler olduğu gibi çizilir.
                glow_pen = QPen(brush, 6)
                glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                glow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setOpacity(0.25)
                painter.setPen(glow_pen)
                painter.drawPolyline(geometry["polyline"])

                core_pen = QPen(brush, 1.8)
                core_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                core_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setOpacity(1.0)
                painter.setPen(core_pen)
                painter.drawPolyline(geometry["polyline"])

        painter.setOpacity(1.0)

    def resizeEvent(self, event):
        self._geometry_cache.clear()
        self._geometry_cache_size = None
        super().resizeEvent(event)


# ------------------------------------------------------------------
# TEKLİ (tek renk) grafikler — her denklem satırının kendi paneli
# ------------------------------------------------------------------

class WaveformPlot(NeonLinePlot):
    """Kayan osiloskop için dairesel tampon tabanlı waveform grafiği."""

    def __init__(self, color="#39ff14", history_seconds=3.0, sample_rate=44100, parent=None):
        super().__init__(color=color, parent=parent)
        self.sample_rate = sample_rate
        self._ring = RingBuffer(int(history_seconds * sample_rate))
        self._last_draw_time = 0.0
        self._draw_interval = 1.0 / 60.0
        self._cached_xs = None

    def push_audio(self, audio_chunk, max_points=350):
        chunk = np.asarray(audio_chunk, dtype=np.float32)
        if len(chunk) == 0:
            return

        self._ring.write(chunk)
        now = time.perf_counter()
        if now - self._last_draw_time < self._draw_interval:
            return
        self._last_draw_time = now

        display = self._ring.read_downsampled(max_points)
        n_display = len(display)

        if self._cached_xs is None or len(self._cached_xs) != n_display:
            self._cached_xs = np.linspace(-1, 1, n_display, dtype=np.float64)

        ys = np.clip(display, -1, 1)
        self.set_points(self._cached_xs, ys, fade=True)

    def update_from_audio(self, audio_buffer):
        self.push_audio(audio_buffer)

    def reset_ring(self):
        self._ring.reset()
        self._cached_xs = None
        self.clear()


class SpectrumPlot(NeonLinePlot):
    """Radyal simetrik spektrum; açı tepeden (-90 derece) başlar."""

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
        self.set_points(xs, ys, fade=True)


class VectorPlot(NeonLinePlot):
    """Parametrik (x, y) eğrisi için vektör grafiği."""

    def update_from_xy(self, xs, ys=None):
        xs = np.asarray(xs, dtype=np.float64)
        if ys is None:
            ys = np.roll(xs, max(1, len(xs) // 20))
        else:
            ys = np.asarray(ys, dtype=np.float64)

        if len(xs) == 0 or len(ys) == 0:
            return

        max_r = max(1e-6, float(np.max(np.abs(xs))), float(np.max(np.abs(ys))))
        self.set_points(xs / max_r, ys / max_r, fade=True)


# ------------------------------------------------------------------
# ÇOKLU (renkli overlay) grafikler — COMBINED panel için
# ------------------------------------------------------------------

class MultiWaveformPlot(NeonLinePlot):
    """Her voice için ayrı RingBuffer tutar ve hepsini üst üste çizer."""

    def __init__(self, history_seconds=3.0, sample_rate=44100, parent=None):
        super().__init__(parent=parent)
        self.sample_rate = sample_rate
        self.history_len = int(history_seconds * sample_rate)
        self._rings = {}
        self._xs_cache = {}
        self._last_draw_time = 0.0
        self._draw_interval = 1.0 / 60.0

    def push(self, per_voice_waves, colors, max_points=300):
        for name, chunk in per_voice_waves.items():
            ring = self._rings.get(name)
            if ring is None:
                ring = RingBuffer(self.history_len)
                self._rings[name] = ring
            ring.write(chunk)

        for stale in list(self._rings.keys()):
            if stale not in per_voice_waves:
                del self._rings[stale]

        now = time.perf_counter()
        if now - self._last_draw_time < self._draw_interval:
            return
        self._last_draw_time = now

        series = {}
        for name in per_voice_waves:
            ring = self._rings[name]
            display = ring.read_downsampled(max_points)
            n_display = len(display)

            xs = self._xs_cache.get(n_display)
            if xs is None:
                xs = np.linspace(-1, 1, n_display, dtype=np.float64)
                self._xs_cache[n_display] = xs

            ys = np.clip(display, -1, 1)
            series[name] = (xs, ys, colors.get(name, self._default_color), True)

        self.set_series(series)

    def reset_rings(self):
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

            if len(xs) == 0 or len(ys) == 0:
                continue

            max_r = max(1e-6, float(np.max(np.abs(xs))), float(np.max(np.abs(ys))))
            series[name] = (
                xs / max_r,
                ys / max_r,
                colors.get(name, self._default_color),
                True
            )

        self.set_series(series)
