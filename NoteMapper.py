import numpy as np


class NoteMapper:
    """Matematiksel değerleri müzikal notalara çeviren sınıf. Projenin
    'müzikal zekası' burada yaşıyor: ham float/2D değerler, kulağa hoş
    gelen, birbiriyle uyumlu bir gamdaki perdelere (pitch) kuantalanır.

    - Denklem tek boyutluysa (float): tanh ile sınırlanır, gam
      derecesine yuvarlanır.
    - Denklem 2D parametrikse (x, y): açısı (atan2) perdeye, yarıçapı
      (hypot) ise ses şiddetine (velocity/dinamik) eşlenir. Böylece
      eğri döndükçe melodi değişir, eğri büyüyüp küçüldükçe ses
      yükselip alçalır — Hugo'nun kelebek eğrisi gibi şekiller bu
      sayede gerçek bir melodiye dönüşür.
    """

    SCALES = {
        "pentatonik_minor": [0, 3, 5, 7, 10],
        "pentatonik_major": [0, 2, 4, 7, 9],
        "dogal_minor": [0, 2, 3, 5, 7, 8, 10],
        "major": [0, 2, 4, 5, 7, 9, 11],
    }

    def __init__(self, base_frequency=220.0, scale="pentatonik_minor", octave_range=2):
        self.base_frequency = base_frequency
        self.octave_range = octave_range
        self.set_scale(scale)

    def set_scale(self, scale_name):
        self.scale_name = scale_name
        self.semitones = self.SCALES.get(scale_name, self.SCALES["pentatonik_minor"])

    def _frequency_from_index(self, index):
        notes_per_octave = len(self.semitones)
        octave, degree = divmod(index, notes_per_octave)
        semitone = self.semitones[degree] + 12 * octave
        return self.base_frequency * (2.0 ** (semitone / 12.0))

    def map_value(self, raw_value):
        """raw_value: EquationEngine.evaluate() çıktısı — float ya da
        (x, y) tuple. Döner: (frequency, velocity)."""

        total_notes = max(1, len(self.semitones) * self.octave_range)

        if isinstance(raw_value, tuple):
            x, y = float(raw_value[0]), float(raw_value[1])
            angle = float(np.arctan2(y, x))          # -pi .. pi -> perde
            radius = float(np.hypot(x, y))             # -> ses şiddeti
            norm = angle / np.pi                        # -1 .. 1
            velocity = float(np.clip(radius / 3.0, 0.25, 1.0))
        else:
            norm = float(np.tanh(float(raw_value)))
            velocity = 0.75

        index = int(round(norm * total_notes / 2)) + total_notes // 2
        index = index % total_notes

        frequency = self._frequency_from_index(index)
        return frequency, velocity
