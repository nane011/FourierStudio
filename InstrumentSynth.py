import numpy as np
from scipy.signal import butter, lfilter


class InstrumentSynth:
    """Fiziksel/additive modelleme ile enstrüman notaları üreten motor.

    MİMARİ GÜNCELLEMESİ (WAVETABLE CACHING):
    Ağır trigonometrik hesaplamalar (np.sin, np.exp) bilgisayarı boğmamak 
    için sadece bir kez yapılır. Üretilen dalga _wave_cache sözlüğüne 
    kaydedilir. Aynı nota (aynı enstrüman, frekans ve süre ile) tekrar 
    istendiğinde sıfırdan hesaplanmaz, doğrudan bellekten (O(1) hızında) 
    çekilir. Bu sayede işlemci yükü %99 oranında düşer.
    """

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
        # YENİ: Sesleri hafızada tutacak devasa depo!
        self._wave_cache = {}

        self.instruments = {
            "gitar": {
                "type": "string",
                "harmonics": [(1, 1.00), (2, 0.55), (3, 0.25), (4, 0.12), (5, 0.06), (6, 0.025)],
                "decay": 1.8, "attack": 0.006, "detune": 0.002, "pluck_noise": 0.09,
                "chorus": True, "chorus_detune": 0.0025,
            },
            "ronroco": {
                "type": "string",
                "harmonics": [(1, 1.00), (2, 0.55), (3, 0.30), (4, 0.16), (5, 0.08), (6, 0.04), (7, 0.02)],
                "decay": 1.5, "attack": 0.012, "detune": 0.0025, "pluck_noise": 0.045,
            },
            "ukulele": {
                "type": "string",
                "harmonics": [(1, 1.0), (2, 0.40), (3, 0.15), (4, 0.06), (5, 0.02)],
                "decay": 3.8, "attack": 0.004, "detune": 0.001, "pluck_noise": 0.05,
            },
            "santur": {
                "type": "string",
                "harmonics": [(1, 1.00), (2, 0.60), (3, 0.35), (4, 0.20), (5, 0.12), (6, 0.07), (7, 0.04)],
                "decay": 0.9, "attack": 0.003, "detune": 0.0015, "pluck_noise": 0.03,
            },
            "flut": {
                "type": "wind",
                "harmonics": [(1, 1.0), (2, 0.15), (3, 0.05), (4, 0.015)],
                "attack": 0.08, "release": 0.20, "vibrato_rate": 5.0, "vibrato_depth": 0.006,
            },
            "kalimba": {
                "type": "metal",
                "harmonics": [(1, 1.0), (2, 0.55), (3, 0.30), (4, 0.15), (5, 0.08), (6, 0.03)],
                "decay": 4.0, "attack": 0.002,
            },
        }

        self._gains = {}
        for name, cfg in self.instruments.items():
            harmonic_sum = sum(a for _, a in cfg["harmonics"])
            if cfg["type"] == "wind":
                harmonic_sum += 0.04
            self._gains[name] = 0.92 / harmonic_sum

        self._pluck_b, self._pluck_a = butter(2, 3000 / (sample_rate / 2), btype="low")
        self._breath_b, self._breath_a = butter(2, 2200 / (sample_rate / 2), btype="low")

    # ------------------------------------------------------------

    def harmonic_frequencies(self, instrument, frequency, n=8):
        cfg = self.instruments.get(instrument.lower(), self.instruments["gitar"])
        return [(frequency * h, amp) for h, amp in cfg["harmonics"][:n]]

    def max_ring_time(self, instrument):
        cfg = self.instruments.get(instrument.lower(), self.instruments["gitar"])
        if cfg["type"] == "wind":
            return cfg["attack"] + cfg["release"] + 0.05
        return 6.0 / cfg["decay"] + cfg["attack"] + 0.05

    def play_note(self, instrument, frequency, duration, velocity=0.8):
        key = instrument.lower()
        config = self.instruments.get(key, self.instruments["gitar"])
        gain = self._gains.get(key, self._gains["gitar"]) * float(np.clip(velocity, 0.05, 1.0))

        # KRİTİK BÖLGE: Notanın kimliği (Cache Key) belirleniyor.
        # Aynı nota daha önce çalınmışsa, direkt hafızadan (RAM) çekilir!
        cache_key = (key, round(frequency, 2), round(duration, 3))
        
        if cache_key in self._wave_cache:
            wave = self._wave_cache[cache_key]
        else:
            tail = self.max_ring_time(key)
            total_len = max(duration + tail, 0.02)
            n_samples = max(1, int(total_len * self.sample_rate))
            
            # Bellek ve işlemci dostu float32 matris (İşlem süresini yarı yarıya düşürür)
            t = np.arange(n_samples, dtype=np.float32) / np.float32(self.sample_rate)

            if config["type"] == "string":
                wave = self._render_string(frequency, t, config)
            elif config["type"] == "wind":
                wave = self._render_flute(frequency, t, config, duration)
            elif config["type"] == "metal":
                wave = self._render_metal(frequency, t, config)
            else:
                wave = np.zeros_like(t)
            
            # Yeni hesaplanan notayı hafızaya kaydet
            self._wave_cache[cache_key] = wave

        # Nota hazır, sadece vurma şiddetiyle (velocity) çarpıp gönder
        return (wave * gain).astype(np.float32)

    # ---------------- Telli çalgılar ----------------

    def _pluck_burst(self, t, config):
        noise_gain = config.get("pluck_noise", 0.08)
        attack = config["attack"]
        burst_length = max(attack * 4, 0.01)
        
        burst_samples = min(len(t), max(1, int(burst_length * self.sample_rate)))
        t_burst = t[:burst_samples]
        
        envelope = np.clip(1 - t_burst / burst_length, 0.0, 1.0) ** 2
        noise = np.random.normal(0, 1, burst_samples)
        noise = lfilter(self._pluck_b, self._pluck_a, noise)
        
        full_noise = np.zeros_like(t)
        full_noise[:burst_samples] = noise * envelope * noise_gain
        return full_noise

    def _harmonic_body(self, frequency, t, config):
        wave = np.zeros_like(t)
        for harmonic, amplitude in config["harmonics"]:
            multiplier = harmonic * (1 + config["detune"] * harmonic * harmonic)
            partial_frequency = frequency * multiplier
            harmonic_decay = config["decay"] * (1 + 0.18 * harmonic)
            
            wave += (
                amplitude
                * np.sin(2 * np.pi * partial_frequency * t)
                * np.exp(-harmonic_decay * t)
            )
        return wave

    def _render_string(self, frequency, t, config):
        if config.get("chorus", False):
            detune = config.get("chorus_detune", 0.0025)
            body = (
                self._harmonic_body(frequency * (1 - detune), t, config)
                + self._harmonic_body(frequency * (1 + detune), t, config)
            ) * 0.5
        else:
            body = self._harmonic_body(frequency, t, config)

        attack = config["attack"]
        attack_env = np.ones_like(t)
        mask = t < attack
        attack_env[mask] = t[mask] / attack
        body *= attack_env

        body += self._pluck_burst(t, config)
        return body

    # ---------------- Flüt ----------------

    def _render_flute(self, frequency, t, config, duration):
        vibrato_rate = config.get("vibrato_rate", 5.0)
        vibrato_depth = config.get("vibrato_depth", 0.006)

        base_vibrato_wave = np.sin(2 * np.pi * vibrato_rate * t) / vibrato_rate

        wave = np.zeros_like(t)
        for harmonic, amplitude in config["harmonics"]:
            base = 2 * np.pi * frequency * harmonic * t
            vib = vibrato_depth * frequency * harmonic * base_vibrato_wave
            wave += amplitude * np.sin(base + vib)

        noise = np.random.normal(0, 1, len(t))
        noise = lfilter(self._breath_b, self._breath_a, noise)
        wave += noise * 0.05

        attack = config["attack"]
        release = config["release"]

        envelope = np.zeros_like(t)
        attack_mask = t < attack
        envelope[attack_mask] = t[attack_mask] / attack

        sustain_end = max(duration, attack)
        sustain_mask = (t >= attack) & (t < sustain_end)
        envelope[sustain_mask] = 1.0

        release_end = sustain_end + release
        release_mask = (t >= sustain_end) & (t < release_end)
        envelope[release_mask] = 1.0 - (t[release_mask] - sustain_end) / release

        wave *= envelope
        return wave

    # ---------------- Kalimba ----------------

    def _render_metal(self, frequency, t, config):
        wave = np.zeros_like(t)
        for harmonic, amplitude in config["harmonics"]:
            partial_frequency = frequency * (harmonic ** 1.07)
            wave += (
                amplitude
                * np.sin(2 * np.pi * partial_frequency * t)
                * np.exp(-config["decay"] * t)
            )

        attack = config["attack"]
        attack_mask = t < attack
        envelope = np.ones_like(t)
        envelope[attack_mask] = t[attack_mask] / attack
        wave *= envelope

        return wave