import atexit
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from scipy.io import wavfile


class AudioEngine:
    """Bir veya birden fazla MelodyEngine sesini gerçek zamanlı olarak
    üretir ve sounddevice üzerinden çalar.

    Grafik senkronizasyonu için üretilen örnek konumu ile ses kartının
    gerçek oynatma zamanını birbirinden ayrı takip eder.
    """

    # Mevcut ses üretim yapısını koruyoruz. Bu boyut, ses üretim yükünü
    # değiştirmeden DAC zaman damgasıyla hassas ilerleme tahmini sağlar.
    BLOCKSIZE = 4096
    BUFFER_COUNT = 16
    FADE_SPEED = 0.05

    def __init__(self, sample_rate=44100, on_chunk=None):
        self.sample_rate = sample_rate
        self.on_chunk = on_chunk

        # {isim: MelodyEngine}
        self._voices = {}

        self._stream = None
        self._sample_position = 0

        # Fade durumu.
        self._fade_state = "idle"
        self._fade_volume = 0.0

        # Hazır ses bloklarının tutulduğu kuyruk.
        self._audio_queue = queue.Queue(maxsize=self.BUFFER_COUNT)

        # Sesleri arka planda üreten thread.
        self._worker_thread = None
        self._worker_running = False

        # Aynı anda start/stop işlemlerinin çakışmasını önlemek için.
        self._lock = threading.Lock()

        # Her blok için (başlangıç örneği, kare sayısı, DAC zamanı, ses var mı)
        # tutulur. DAC zamanı, bloğun ses kartında oynatılmasının beklendiği andır.
        self._scheduled_segments = []
        self._last_stream_time = None
        self._last_callback_perf = None
        self._last_reported_sample = 0.0
        self._clock_lock = threading.Lock()
        self._underrun_count = 0

        atexit.register(self.hard_stop)

    @property
    def underrun_count(self):
        """Ses üretimi yetişemediği için sessizlik gönderilen callback sayısı."""
        return self._underrun_count

    def set_voices(self, voice_dict):
        """Çalınacak sesleri belirler."""
        with self._lock:
            self._voices = dict(voice_dict)

    def start(self):
        """Ses sistemini başlatır."""
        with self._lock:
            if not self._voices:
                return

        # Önce eski stream'i ve thread'i temizle.
        self.hard_stop()

        self._sample_position = 0
        self._fade_state = "in"
        self._fade_volume = 0.0
        self._reset_playback_clock()
        self._underrun_count = 0

        # Her MelodyEngine yeni oturumdan başlasın.
        with self._lock:
            voices = list(self._voices.values())

        for engine in voices:
            engine.reset()

        self._audio_queue = queue.Queue(maxsize=self.BUFFER_COUNT)
        self._worker_running = True

        # Ağır ses hesaplamasını ayrı thread'e taşıyoruz.
        self._worker_thread = threading.Thread(
            target=self._generate_audio,
            daemon=True
        )
        self._worker_thread.start()

        def callback(outdata, frames, time_info, status):
            """Hazır bloğu ses kartına verir ve DAC zaman çizelgesini kaydeder."""
            callback_perf = time.perf_counter()
            try:
                current_stream_time = float(time_info.currentTime)
                dac_time = float(time_info.outputBufferDacTime)
            except Exception:
                current_stream_time = 0.0
                dac_time = 0.0

            with self._clock_lock:
                self._last_stream_time = current_stream_time
                self._last_callback_perf = callback_perf

            try:
                block_start, combined, per_voice = self._audio_queue.get_nowait()

                if len(combined) != frames:
                    outdata.fill(0)
                    self._record_silence_segment(
                        frames, dac_time, current_stream_time
                    )
                    return

                outdata[:, 0] = combined
                outdata[:, 1] = combined

                self._record_segment(
                    block_start, frames, dac_time, has_audio=True
                )

                # Grafik sistemine hazır veriyi gönderiyoruz.
                if self.on_chunk is not None:
                    self.on_chunk(combined, per_voice)

            except queue.Empty:
                # Ses kuyruğu boşaldı: bu bir underrun sayılır.
                self._underrun_count += 1
                # Ses hazır değilse sessizlik gönder. Grafik zamanı, sessizlik
                # süresince bir sonraki ses bloğuna kadar bekletilir.
                outdata.fill(0)
                self._record_silence_segment(
                    frames, dac_time, current_stream_time
                )

            except Exception:
                outdata.fill(0)
                self._record_silence_segment(
                    frames, dac_time, current_stream_time
                )

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            dtype="float32",
            callback=callback,
            blocksize=self.BLOCKSIZE,
            latency="high"
        )
        self._stream.start()

    def _reset_playback_clock(self):
        with self._clock_lock:
            self._scheduled_segments = []
            self._last_stream_time = None
            self._last_callback_perf = None
            self._last_reported_sample = 0.0

    def _record_segment(self, start_sample, frames, dac_time, has_audio=True):
        if not np.isfinite(dac_time) or dac_time < 0.0:
            with self._clock_lock:
                stream_time = self._last_stream_time or 0.0
            dac_time = stream_time

        with self._clock_lock:
            self._scheduled_segments.append(
                (int(start_sample), int(frames), float(dac_time), bool(has_audio))
            )
            # 512 blok yaklaşık 47 saniyedir; bu, ses kartı gecikmesinden
            # çok uzundur ve eski kayıtların sınırsız büyümesini engeller.
            if len(self._scheduled_segments) > 512:
                del self._scheduled_segments[:256]

    def _position_at_stream_time(self, stream_time, segments=None):
        if segments is None:
            with self._clock_lock:
                segments = list(self._scheduled_segments)

        if not segments:
            return 0.0

        # En son başlamış olan ve stream_time'ı kapsayan bloğu bul.
        for start_sample, frames, dac_time, has_audio in reversed(segments):
            if stream_time < dac_time:
                continue

            elapsed = stream_time - dac_time
            block_duration = frames / self.sample_rate

            if elapsed < block_duration:
                if has_audio:
                    return start_sample + elapsed * self.sample_rate
                return float(start_sample)

            # Bu blok tamamen oynatılmıştır. Sessizlik içindeysek örnek
            # konumu ilerlemez; ses bloğundaysak bloğun sonuna ulaşır.
            if has_audio:
                return float(start_sample + frames)
            return float(start_sample)

        # İlk planlanmış bloğun DAC zamanı henüz gelmemiştir.
        return float(segments[0][0])

    @property
    def playback_position_seconds(self):
        """Ses kartının kendi PortAudio saatine göre oynatılan içeriğin konumu.

        DAC zaman damgaları ile aynı saat alanını kullanır; perf_counter ile
        ikinci bir saati tahmini olarak eşleştirmediği için uzun çalmada saat
        kayması riskini azaltır.
        """
        stream = self._stream
        if stream is None:
            with self._clock_lock:
                return max(0.0, self._last_reported_sample / self.sample_rate)

        try:
            # sounddevice Stream.time, callback time_info zaman damgalarıyla
            # aynı PortAudio saatini kullanır.
            stream_now = float(stream.time)
        except Exception:
            with self._clock_lock:
                last_stream_time = self._last_stream_time
                segments = list(self._scheduled_segments)
            if last_stream_time is None or not segments:
                return 0.0
            stream_now = last_stream_time

        with self._clock_lock:
            segments = list(self._scheduled_segments)

        if not segments:
            return 0.0

        sample_position = self._position_at_stream_time(stream_now, segments)

        with self._clock_lock:
            # Küçük saat jitter'ı çizimi geriye götürmesin.
            self._last_reported_sample = max(
                self._last_reported_sample,
                sample_position
            )
            sample_position = self._last_reported_sample

        return max(0.0, sample_position / self.sample_rate)

    def _record_silence_segment(self, frames, dac_time, current_stream_time):
        # Sessizlik bloğu başlamadan hemen önceki içerik konumunu hesapla.
        # DAC zamanı kullanıldığı için önceki ses bloğunun sonuna kadar ilerler.
        with self._clock_lock:
            segments = list(self._scheduled_segments)

        hold_sample = self._position_at_stream_time(dac_time, segments)
        self._record_segment(
            int(round(hold_sample)), frames, dac_time, has_audio=False
        )

    def _generate_audio(self):
        """MelodyEngine'lerden ses bloklarını arka planda üretir."""
        while self._worker_running:
            try:
                with self._lock:
                    voices = list(self._voices.items())

                if not voices:
                    self._worker_running = False
                    break

                block_start = self._sample_position
                combined = np.zeros(self.BLOCKSIZE, dtype=np.float32)
                per_voice = {}

                for name, engine in voices:
                    wave = engine.generate_chunk(block_start, self.BLOCKSIZE)
                    wave = np.asarray(wave, dtype=np.float32)

                    if len(wave) != self.BLOCKSIZE:
                        fixed_wave = np.zeros(self.BLOCKSIZE, dtype=np.float32)
                        copy_length = min(len(wave), self.BLOCKSIZE)
                        if copy_length > 0:
                            fixed_wave[:copy_length] = wave[:copy_length]
                        wave = fixed_wave

                    per_voice[name] = wave
                    combined += wave

                # Fade
                if self._fade_state == "in":
                    self._fade_volume = min(
                        1.0, self._fade_volume + self.FADE_SPEED
                    )
                    if self._fade_volume >= 1.0:
                        self._fade_state = "playing"
                elif self._fade_state == "out":
                    self._fade_volume = max(
                        0.0, self._fade_volume - self.FADE_SPEED
                    )

                # Master ses
                combined *= 1.1
                np.tanh(combined, out=combined)
                combined *= (0.35 * self._fade_volume)

                # Bloğu kuyruğa koy. Başlangıç örnek numarası bloğun yanında taşınır.
                while self._worker_running:
                    try:
                        self._audio_queue.put(
                            (block_start, combined, per_voice),
                            timeout=0.05
                        )
                        self._sample_position = block_start + self.BLOCKSIZE
                        break
                    except queue.Full:
                        continue

                time.sleep(0.002)

                if self._fade_state == "out" and self._fade_volume <= 0.01:
                    self._worker_running = False

            except Exception as error:
                print(f"Ses üretim hatası: {error}")
                silent = np.zeros(self.BLOCKSIZE, dtype=np.float32)
                silent_start = self._sample_position
                try:
                    self._audio_queue.put(
                        (silent_start, silent, {}),
                        timeout=0.05
                    )
                    self._sample_position = silent_start + self.BLOCKSIZE
                except Exception:
                    pass

    def stop(self):
        """Sesi fade-out ile durdurur."""
        if self._stream is not None:
            self._fade_state = "out"

    def hard_stop(self):
        """Ses stream'ini ve üretici thread'i tamamen durdurur."""
        self._worker_running = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if (
            self._worker_thread is not None
            and self._worker_thread.is_alive()
        ):
            self._worker_thread.join(timeout=0.5)

        self._worker_thread = None
        self._fade_state = "idle"
        self._fade_volume = 0.0

        try:
            while True:
                self._audio_queue.get_nowait()
        except queue.Empty:
            pass

        self._reset_playback_clock()


# ------------------------------------------------------------
# OFFLINE RENDER
# ------------------------------------------------------------

def render_offline(voices, duration_seconds, sample_rate=44100):
    """Canlı playback'e dokunmadan sesi baştan sona üretir."""
    for engine in voices.values():
        engine.reset()

    total_samples = max(1, int(duration_seconds * sample_rate))
    block = 4096
    output = np.zeros(total_samples, dtype=np.float64)
    position = 0

    while position < total_samples:
        frames = min(block, total_samples - position)
        combined = np.zeros(frames, dtype=np.float64)

        for engine in voices.values():
            wave = engine.generate_chunk(position, frames)
            combined += wave

        output[position:position + frames] = combined
        position += frames

    output = np.tanh(output * 1.1) * 0.6

    fade_len = min(int(0.5 * sample_rate), total_samples // 2)
    if fade_len > 0:
        ramp = np.linspace(0.0, 1.0, fade_len)
        output[:fade_len] *= ramp
        output[-fade_len:] *= ramp[::-1]

    output = np.clip(output, -1.0, 1.0)
    pcm16 = (output * 32767).astype(np.int16)
    return pcm16


def save_wav(filepath, pcm16, sample_rate=44100):
    """16-bit PCM verisini WAV olarak kaydeder."""
    wavfile.write(filepath, sample_rate, pcm16)
