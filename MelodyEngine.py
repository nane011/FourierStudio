import numpy as np
from NoteEvent import NoteEvent


class MelodyEngine:
    """EquationEngine -> NoteMapper -> NoteEvent -> InstrumentSynth
    zincirini yöneten, TEK BİR sesin (voice) zaman içinde nota
    üretmesini sağlayan sınıf.

    Çalışma mantığı: denklem, ses hızını (audio-rate) değiştirmek için
    DEĞİL, sadece "hangi nota, ne zaman çalınacak" sorusuna cevap
    vermek için örnekleniyor. Her not_duration saniyede bir denklem
    elapsed_time anında değerlendirilir (EquationEngine), sonuç bir
    notaya çevrilir (NoteMapper), NoteEvent olarak paketlenir ve
    InstrumentSynth.play_note() ile SIFIRDAN, tam uzunlukta bir ses
    bloğu render edilir. Bu blok, ait olduğu zaman diliminde mix'e
    eklenir; birden fazla notanın çınlama kuyruğu (ring tail) doğal
    olarak üst üste binebilir (polifoni).

    Birden fazla MelodyEngine (arka plan + bas + ana ses gibi) bir
    araya gelip AudioEngine tarafında karıştırılabilir (mix).
    """

    def __init__(self, equation_engine, note_mapper, synth, instrument,
                 note_duration=0.35, gain=1.0, sample_rate=44100):
        self.equation_engine = equation_engine
        self.note_mapper = note_mapper
        self.synth = synth
        self.instrument = instrument
        self.note_duration = note_duration
        self.gain = gain
        self.sample_rate = sample_rate

        self._voices = []            # [{"audio": np.ndarray, "start_sample": int}, ...]
        self._next_note_sample = 0
        self.last_note_info = None   # UI/grafik için: {"raw_value", "frequency", "velocity"}

    def reset(self):
        self._voices = []
        self._next_note_sample = 0
        self.last_note_info = None

    def _trigger_note(self, trigger_sample):
        elapsed_seconds = trigger_sample / self.sample_rate

        raw_value = self.equation_engine.evaluate(elapsed_seconds)
        frequency, velocity = self.note_mapper.map_value(raw_value)

        note = NoteEvent(
            instrument=self.instrument,
            frequency=frequency,
            duration=self.note_duration,
            velocity=velocity,
            start_time=elapsed_seconds,
        )

        audio = self.synth.play_note(
            note.instrument, note.frequency, note.duration, note.velocity
        )

        self._voices.append({"audio": audio, "start_sample": trigger_sample})
        self.last_note_info = {
            "raw_value": raw_value, "frequency": frequency, "velocity": velocity,
            "elapsed_seconds": elapsed_seconds,
        }

    def generate_chunk(self, start_sample, frames):
        """[start_sample, start_sample+frames) aralığındaki ses bloğunu
        üretir. Gerekirse yeni notaları bu aralık içinde tetikler."""

        end_sample = start_sample + frames
        note_length_samples = max(1, int(self.note_duration * self.sample_rate))

        while self._next_note_sample < end_sample:
            self._trigger_note(self._next_note_sample)
            self._next_note_sample += note_length_samples

        mix = np.zeros(frames, dtype=np.float64)
        still_alive = []

        for voice in self._voices:
            offset = voice["start_sample"] - start_sample
            audio = voice["audio"]

            src_start = max(0, -offset)
            dst_start = max(0, offset)
            length = min(len(audio) - src_start, frames - dst_start)

            if length > 0:
                mix[dst_start:dst_start + length] += audio[src_start:src_start + length]

            if voice["start_sample"] + len(audio) > start_sample:
                still_alive.append(voice)

        self._voices = still_alive

        return mix * self.gain
