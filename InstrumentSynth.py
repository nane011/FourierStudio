import numpy as np
from scipy.signal import butter, lfilter


class InstrumentSynth:
    """Fiziksel/additive modelleme ile enstrüman notaları üreten motor.
    """

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self._wave_cache = {}

        self.instruments = {
            "gitar": {
                "type": "string",
                "model": "plucked_string",
                "harmonics": [(1, 1.00)], # Karplus-Strong modelinde harmonikler fiziksel olarak kendiliğinden oluşur
                "decay": 0.8,        # Genel ses kuyruğu (sustain)
                "attack": 0.003,     
                "detune": 0.0012,
                "pluck_noise": 0.15, # Mızrap vuruş sesi düşürüldü (lastik hissini kırmak için)
                "string_damping": 0.0015, # SIFIRA NE KADAR YAKINSA SES O KADAR UZAR (Sustain)
                "body_gain": 5,
                "body_freq": 1.0,
                "body_width": 95.0,
            },
            "ronroco": {
                "type": "string",
                "model": "double_string",
                "harmonics": [(1, 1.00)],
                "decay": 0.25,        # Dağlardaki o uzun yankı (Sustain)
                "attack": 0.005,     
                "detune": 0.040,    # AND DAĞLARI TINISI: Çift tellerin kasti akortsuzluğu (Chorus efekti)
                "pluck_noise": 0.08,
                "string_damping": 0.0010, # Tel çok daha uzun titreşsin
                "body_gain": 0.25,
                "body_freq": 0.1,
                "body_width": 0.1,
            },
            "ukulele": {
                "type": "string",
                "model": "plucked_string",
                "harmonics": [(1, 1.00)],
                "decay": 0.5,        
                "attack": 0.004,
                "detune": 0.0010,
                "pluck_noise": 0.16,
                "string_damping": 0.10,
                "body_gain": 0.13,
                "body_freq": 330.0,
                "body_width": 120.0,
            },
            "bas_gitar": {
                "type": "string",
                "model": "plucked_string",
                "dark_excitation": True, # YENİ: Bas gitarın o çok tok/karanlık vuruşu için
                "harmonics": [(1, 1.00)],
                "decay": 1.2,
                "attack": 0.015,     # Daha etli bir parmak vuruşu
                "detune": 0.000,
                "pluck_noise": 0.01, # Tiz gürültü tamamen kısıldı
                "string_damping": 0.03, # Bas telinin uzun, boğuk çınlaması
                "body_gain": 0.18,
                "body_freq": 60.0,   # Sub-bass gövde frekansı
                "body_width": 55.0,
            },
            "piyano": {
                "type": "string",
                "model": "piano",
                "harmonics": [(1, 1.00), (2, 0.62), (3, 0.39), (4, 0.22), (5, 0.13), (6, 0.075), (7, 0.040), (8, 0.022)],
                "decay": 3.0,
                "attack": 0.005,
                "detune": 0.0007,
                "pluck_noise": 0.02,
                "inharmonicity": 0.0009,
            },
            "ney": {
                "type": "wind",
                "harmonics": [(1, 1.00), (2, 0.40), (3, 0.25), (4, 0.12), (5, 0.06)],
                "attack": 0.12,      
                "release": 0.25,     
                "vibrato_rate": 4.5, 
                "vibrato_depth": 0.012, 
            },
            "kalimba": {
                "type": "metal",
                "harmonics": [(1, 1.00), (3, 0.45), (6, 0.10)],
                "decay": 2.5, "attack": 0.001, "detune": 0.0005, "pluck_noise": 0.12,
            },
            "flut": {
                "type": "wind",
                "harmonics": [(1, 1.0), (2, 0.15), (3, 0.05), (4, 0.015)],
                "attack": 0.08, "release": 0.20, "vibrato_rate": 5.0, "vibrato_depth": 0.006,
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

        cache_key = (key, round(frequency, 2), round(duration, 3))
        
        if cache_key in self._wave_cache:
            wave = self._wave_cache[cache_key]
        else:
            tail = self.max_ring_time(key)
            total_len = max(duration + tail, 0.02)
            n_samples = max(1, int(total_len * self.sample_rate))
            
            t = np.arange(n_samples, dtype=np.float32) / np.float32(self.sample_rate)

            if config["type"] == "string":
                wave = self._render_string(frequency, t, config)
            elif config["type"] == "wind":
                wave = self._render_flute(frequency, t, config, duration)
            elif config["type"] == "metal":
                wave = self._render_metal(frequency, t, config)
            else:
                wave = np.zeros_like(t)
            
            self._wave_cache[cache_key] = wave

        return (wave * gain).astype(np.float32)

    # ---------------- Telli çalgılar ----------------

    def _pluck_burst(self, t, config):
        noise_gain = config.get("pluck_noise", 0.08)
        attack = config["attack"]
        burst_length = max(attack * 8, 0.020) # Sesin ince çıtlamasını önlemek için vuruş süresi uzatıldı
        
        burst_samples = min(len(t), max(1, int(burst_length * self.sample_rate)))
        t_burst = t[:burst_samples]
        
        # vurus sesini ayarlar
        envelope = np.clip(1 - t_burst / burst_length, 0.0, 1.0) ** 1.0
        noise = np.random.normal(0, 1, burst_samples)
        noise = lfilter(self._pluck_b, self._pluck_a, noise)
        
        full_noise = np.zeros_like(t)
        full_noise[:burst_samples] = noise * envelope * noise_gain
        return full_noise

    def _harmonic_body(self, frequency, t, config):
        wave = np.zeros_like(t)
        inharmonicity = config.get("inharmonicity", 0.0)
        for harmonic, amplitude in config["harmonics"]:
            partial_frequency = frequency * harmonic * np.sqrt(
                1 + inharmonicity * harmonic * harmonic
            )
            harmonic_decay = config["decay"] * (1 + 0.18 * harmonic)
            
            wave += (
                amplitude
                * np.sin(2 * np.pi * partial_frequency * t)
                * np.exp(-harmonic_decay * t)
            )
        return wave

    def _string_wave(self, frequency, t, config, seed_offset=0):
        # YENİ VEKTÖREL KARPLUS-STRONG: Donmaları önlemek için for döngüsü yerine lfilter kullanılır.
        n_samples = len(t)
        delay = max(2, int(self.sample_rate / max(frequency, 20.0)))
        
        # 1. Başlangıç Vuruşu (Sadece ilk delay süresi kadar)
        rng = np.random.default_rng(int(frequency * 13 + seed_offset * 997))
        excitation = np.zeros(n_samples, dtype=np.float32)
        noise_burst = rng.normal(0, 1, delay).astype(np.float32)

        # BAS GİTAR İÇİN KARANLIK (DARK) FİLTRE:
        if config.get("dark_excitation", False):
            # Tiz sesleri kesip tok bir vuruş elde etmek için Low-Pass
            b_low, a_low = butter(1, 400 / (self.sample_rate / 2), btype="low")
            noise_burst = lfilter(b_low, a_low, noise_burst)

        # Doğal tel vuruşu için hafif fade-out
        noise_burst *= np.linspace(1.0, 0.0, delay, dtype=np.float32)
        excitation[:delay] = noise_burst

        # 2. Işık Hızında Geribildirim (Feedback) Hesaplaması
        string_damping = config.get("string_damping", 0.05)
        g = 1.0 - string_damping # Kazanç (1'e ne kadar yakınsa o kadar uzun sustain)
        
        # IIR Filtre katsayıları (y[n] = x[n] + 0.5 * g * (y[n-delay] + y[n-delay-1]))
        b = np.array([1.0], dtype=np.float32)
        a = np.zeros(delay + 2, dtype=np.float32)
        a[0] = 1.0
        a[delay] = -0.5 * g
        a[delay + 1] = -0.5 * g

        # İşlemciyi donduran for döngüsü silindi, tek bir C-tabanlı matris operasyonu eklendi:
        wave = lfilter(b, a, excitation)

        # 3. Genel Sönümlenme
        envelope = np.exp(-config["decay"] * t)
        wave *= envelope
        
        return wave

    def _body_resonance(self, frequency, t, wave, config):
        body_freq = config.get("body_freq", frequency * 0.8)
        body_width = config.get("body_width", 100.0)
        body_gain = config.get("body_gain", 0.12)
        
        resonance = np.sin(2 * np.pi * body_freq * t) * np.exp(-5.5 * t)
        body_env = np.clip(1.0 - np.abs(frequency - body_freq) / max(body_width, 1.0), 0.0, 1.0)
        return wave + resonance * body_gain * body_env

    def _render_string(self, frequency, t, config):
        model = config.get("model", "plucked_string")
        
        if model == "piano":
            body = self._harmonic_body(frequency, t, config)
            attack = config["attack"]
            attack_env = np.ones_like(t)
            mask = t < attack
            attack_env[mask] = t[mask] / attack
            body *= attack_env
            return body + self._pluck_burst(t, config)
        
        if model == "double_string":
            detune = config.get("detune", 0.0018)
            string_a = self._string_wave(frequency, t, config, seed_offset=1)
            string_b = self._string_wave(frequency * (1.0 + detune), t, config, seed_offset=2)
            body = (string_a + string_b) * 0.50
        else:
            body = self._string_wave(frequency, t, config)
        
        attack = config["attack"]
        attack_env = np.ones_like(t)
        mask = t < attack
        attack_env[mask] = t[mask] / max(attack, 0.0001)
        body *= attack_env
        
        body = self._body_resonance(frequency, t, body, config)
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
