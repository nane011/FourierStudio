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

    Ses üretimini ses kartının callback'inden ayırıyoruz. Böylece gitar,
    santur gibi daha ağır enstrümanların hesaplaması ses kartının
    zamanlamasını doğrudan bozmaz.
    """

    # Her callback'te 4096 örnek işlenir.
    # 44100 Hz'de yaklaşık 93 ms'lik ses bloğudur.
    BLOCKSIZE = 4096

    # Arka planda önceden hazırlanmış 12 blok tutulabilir.
    BUFFER_COUNT = 12

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
        self._audio_queue = queue.Queue(
            maxsize=self.BUFFER_COUNT
        )

        # Sesleri arka planda üreten thread.
        self._worker_thread = None
        self._worker_running = False

        # Aynı anda start/stop işlemlerinin çakışmasını önlemek için.
        self._lock = threading.Lock()

        atexit.register(self.hard_stop)

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

        # Her MelodyEngine yeni oturumdan başlasın.
        with self._lock:
            voices = list(self._voices.values())

        for engine in voices:
            engine.reset()

        # Yeni ses kuyruğu.
        self._audio_queue = queue.Queue(
            maxsize=self.BUFFER_COUNT
        )

        self._worker_running = True

        # Ağır ses hesaplamasını ayrı thread'e taşıyoruz.
        self._worker_thread = threading.Thread(
            target=self._generate_audio,
            daemon=True
        )
        self._worker_thread.start()

        def callback(outdata, frames, time_info, status):
            """Ses kartı yeni ses istediğinde çalışan hafif callback."""

            try:
                combined, per_voice = (
                    self._audio_queue.get_nowait()
                )

                if len(combined) != frames:
                    outdata.fill(0)
                    return

                # combined zaten _generate_audio içinde float32 olarak ayarlandı
                # Stereo çıkış.
                outdata[:, 0] = combined
                outdata[:, 1] = combined

                # Grafik sistemine hazır veriyi gönderiyoruz.
                if self.on_chunk is not None:
                    self.on_chunk(
                        combined,
                        per_voice
                    )

            except queue.Empty:
                # Henüz yeni blok hazır değilse ses kartını
                # bekletmek yerine sessizlik gönderiyoruz.
                outdata.fill(0)

            except Exception:
                # Callback içinde hata olursa stream'in çökmesini
                # engellemek için sessizlik gönder.
                outdata.fill(0)

        # Ses kartı stream'i.
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            dtype="float32",
            callback=callback,
            blocksize=self.BLOCKSIZE,
            latency="high"
        )

        self._stream.start()

    def _generate_audio(self):
        """MelodyEngine'lerden ses bloklarını arka planda üretir."""

        while self._worker_running:

            try:
                # O an kullanılacak voice listesinin kopyasını alıyoruz.
                with self._lock:
                    voices = list(
                        self._voices.items()
                    )

                if not voices:
                    self._worker_running = False
                    break

                # Birleştirilmiş ses (float32 olarak başlatılıyor).
                combined = np.zeros(
                    self.BLOCKSIZE,
                    dtype=np.float32
                )

                # Her voice ayrı ayrı üretilir.
                per_voice = {}

                for name, engine in voices:

                    wave = engine.generate_chunk(
                        self._sample_position,
                        self.BLOCKSIZE
                    )

                    # Tür dönüşümünü ve kopyalamayı kesinleştir
                    wave = np.asarray(
                        wave,
                        dtype=np.float32
                    )

                    # Güvenlik: beklenmeyen uzunlukta veri gelirse
                    if len(wave) != self.BLOCKSIZE:
                        fixed_wave = np.zeros(
                            self.BLOCKSIZE,
                            dtype=np.float32
                        )
                        copy_length = min(
                            len(wave),
                            self.BLOCKSIZE
                        )
                        if copy_length > 0:
                            fixed_wave[:copy_length] = (
                                wave[:copy_length]
                            )
                        wave = fixed_wave

                    per_voice[name] = wave

                    # Voice'ları mix ediyoruz.
                    combined += wave

                # --------------------------------------------------
                # FADE
                # --------------------------------------------------

                if self._fade_state == "in":
                    self._fade_volume = min(
                        1.0,
                        self._fade_volume + self.FADE_SPEED
                    )
                    if self._fade_volume >= 1.0:
                        self._fade_state = "playing"

                elif self._fade_state == "out":
                    self._fade_volume = max(
                        0.0,
                        self._fade_volume - self.FADE_SPEED
                    )

                # --------------------------------------------------
                # MASTER SES (BELLEK OPTİMİZASYONU)
                # --------------------------------------------------
                # KRİTİK DEĞİŞİKLİK: Sürekli yeni bellek bloğu açarak (Garbage Collector'ı yorarak)
                # kasmayı engellemek için, aynı array üzerinde (in-place) matematik işlemi yapıyoruz.
                
                combined *= 1.1
                np.tanh(combined, out=combined)
                combined *= (0.35 * self._fade_volume)

                # --------------------------------------------------
                # HAZIR BLOĞU KUYRUĞA KOY
                # --------------------------------------------------

                while self._worker_running:
                    try:
                        self._audio_queue.put(
                            (
                                combined,
                                per_voice
                            ),
                            timeout=0.05
                        )
                        break
                    except queue.Full:
                        continue

                self._sample_position += self.BLOCKSIZE

                # --------------------------------------------------
                # GIL YIELD (ARAYÜZÜN DONMASINI ENGELLEYEN MOLA)
                # --------------------------------------------------
                # Eğer birden fazla ses işleniyorsa bu thread çok yoğun çalışır ve PyQt'ye 
                # (arayüze) nefes aldırmaz. 2 milisaniyelik bu uyku, kilidi serbest bırakır 
                # ve arayüzün akıcı bir şekilde grafik çizmesine imkan tanır.
                time.sleep(0.002)

                # Fade-out bittiyse üretimi durdur.
                if (
                    self._fade_state == "out"
                    and self._fade_volume <= 0.01
                ):
                    self._worker_running = False

            except Exception as error:

                print(
                    f"Ses üretim hatası: {error}"
                )

                silent = np.zeros(
                    self.BLOCKSIZE,
                    dtype=np.float32
                )
                try:
                    self._audio_queue.put(
                        (silent, {}),
                        timeout=0.05
                    )
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
            self._worker_thread.join(
                timeout=0.5
            )

        self._worker_thread = None

        self._fade_state = "idle"
        self._fade_volume = 0.0

        try:
            while True:
                self._audio_queue.get_nowait()
        except queue.Empty:
            pass

# ------------------------------------------------------------
# OFFLINE RENDER
# ------------------------------------------------------------

def render_offline(
    voices,
    duration_seconds,
    sample_rate=44100
):
    """Canlı playback'e dokunmadan sesi baştan sona üretir."""

    for engine in voices.values():
        engine.reset()

    total_samples = max(
        1,
        int(
            duration_seconds
            * sample_rate
        )
    )

    block = 4096
    output = np.zeros(
        total_samples,
        dtype=np.float64
    )
    position = 0

    while position < total_samples:
        frames = min(
            block,
            total_samples - position
        )
        combined = np.zeros(
            frames,
            dtype=np.float64
        )
        for engine in voices.values():
            wave = engine.generate_chunk(
                position,
                frames
            )
            combined += wave

        output[
            position:
            position + frames
        ] = combined
        position += frames

    output = np.tanh(
        output * 1.1
    ) * 0.6

    fade_len = min(
        int(0.5 * sample_rate),
        total_samples // 2
    )
    if fade_len > 0:
        ramp = np.linspace(
            0.0,
            1.0,
            fade_len
        )
        output[:fade_len] *= ramp
        output[-fade_len:] *= (
            ramp[::-1]
        )

    output = np.clip(
        output,
        -1.0,
        1.0
    )
    pcm16 = (
        output * 32767
    ).astype(np.int16)

    return pcm16


def save_wav(
    filepath,
    pcm16,
    sample_rate=44100
):
    """16-bit PCM verisini WAV olarak kaydeder."""

    wavfile.write(
        filepath,
        sample_rate,
        pcm16
    )