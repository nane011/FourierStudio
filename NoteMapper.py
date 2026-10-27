import numpy as np


class NoteMapper:
    """Matematiksel değerleri müzikal notalara eşler.

    1D denklem:
        Değer tanh ile sınırlanır ve nota perdesine eşlenir.

    2D parametrik denklem (x, y):
        DİKEY y koordinatı perdeyi belirler. Grafik yukarı çıktıkça perde
        yükselir, aşağı indikçe perde alçalır. Orijinden uzaklık (yarıçap)
        ses şiddetini belirler.

    Frekanslar seçilen gamın notalarına kuantalanır; dolayısıyla perde
    tamamen sürekli kaymak yerine müzikal basamaklarla değişir.
    """

    SCALES = {
        "pentatonik_minor": [0, 3, 5, 7, 10],
        "pentatonik_major": [0, 2, 4, 7, 9],
        "dogal_minor": [0, 2, 3, 5, 7, 8, 10],
        "major": [0, 2, 4, 5, 7, 9, 11],
        "kromatik": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "harmonik_minor": [0, 2, 3, 5, 7, 8, 11],
    }

    def __init__(self, base_frequency=220.0, scale="pentatonik_minor", octave_range=2):
        self.base_frequency = float(base_frequency)
        self.octave_range = max(1, int(octave_range))
        self.set_scale(scale)

    def set_scale(self, scale_name):
        self.scale_name = scale_name
        self.semitones = self.SCALES.get(
            scale_name, self.SCALES["pentatonik_minor"]
        )

    def _frequency_from_index(self, index):
        notes_per_octave = len(self.semitones)
        total_notes = max(1, notes_per_octave * self.octave_range)

        # Nota indeksini aralıkta tutuyoruz; üst notadan başlangıç notasına
        # dönen modulo davranışı, grafikte perde sıçraması oluşturabiliyordu.
        exact_index = int(np.clip(np.round(index), 0, total_notes - 1))
        octave, degree = divmod(exact_index, notes_per_octave)
        semitone = self.semitones[degree] + 12 * octave

        return self.base_frequency * (2.0 ** (semitone / 12.0))

    def map_value(self, raw_value):
        """Girdi: float veya (x, y). Çıktı: (frequency_hz, velocity)."""
        total_notes = max(1, len(self.semitones) * self.octave_range)

        if isinstance(raw_value, (tuple, list)) and len(raw_value) == 2:
            x, y = float(raw_value[0]), float(raw_value[1])

            # NaN/inf değerleri nota hesabını bozmasın.
            x = float(np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6))
            y = float(np.nan_to_num(y, nan=0.0, posinf=20.0, neginf=-20.0))

            # Temel düzeltme: perde açıdan değil, dikey y değerinden gelir.
            # tanh, farklı denklem ölçeklerinde aralığı (-1, 1) içine alır.
            norm = float(np.tanh(y))

            # Ses şiddeti yarıçapa bağlı kalır.
            radius = float(np.hypot(x, y))
            velocity = float(np.clip(radius / 3.0, 0.25, 1.0))
        else:
            value = float(np.nan_to_num(
                float(raw_value), nan=0.0, posinf=20.0, neginf=-20.0
            ))
            norm = float(np.tanh(value))
            velocity = 0.75

        # norm=-1 -> en alt nota, norm=+1 -> en üst nota.
        # total_notes-1 kullanımı, üst uçta modulo ile ilk notaya dönmeyi önler.
        index = ((norm + 1.0) * 0.5) * (total_notes - 1)
        frequency = self._frequency_from_index(index)
        return frequency, velocity
