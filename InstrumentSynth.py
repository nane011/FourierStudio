import numpy as np
from scipy.signal import butter, lfilter
from numba import njit

@njit(cache=True)
def _karplus_feedback(excitation, delay, g):
    n = len(excitation)
    wave = np.zeros(n, dtype=np.float32)

    for i in range(n):
        value = excitation[i]

        if i >= delay:
            value += 0.5 * g * wave[i - delay]

        if i >= delay + 1:
            value += 0.5 * g * wave[i - delay - 1]

        wave[i] = value

    return wave

class InstrumentSynth:
    """Fiziksel/additive modelleme ile enstrüman notaları üreten motor.
    
    YENİ NESİL KARPLUS-STRONG: Python for döngüleri tamamen kaldırılarak 
    vektörel IIR filtre (lfilter) mimarisine geçildi. Telin çınlaması 
    işlemciyi dondurmadan O(1) maliyetle anında hesaplanır.
    """

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self._wave_cache = {}

        self.instruments = {
            "gitar": {
                "type": "string",
                "model": "plucked_string",
                "harmonics": [(1, 1.00)], 
                "decay": 0.15,
                "attack": 0.003,     
                "detune": 0.012,
                "pluck_noise": 0.30, 
                "string_damping": 0.0015, 
                "body_gain": 20,
                "body_freq": 0.0,
                "body_width": 300.0,
            },
            "ronroco": {
                "type": "string",
                "model": "double_string",

                "harmonics": [
                    (1, 10.00),  # Güçlü temel ses: dolgun ve sıcak ton
                    (2, 17.50),  # Güçlü oktav: belirgin ilk vuruş
                    (3, 2.20),   # Ronroco karakteri ve gövde
                    (4, 0.90),   # Kontrollü parlaklık
                    (5, 0.35),   # Hafif tel parıltısı
                    (6, 0.14),   # Az miktarda atak sertliği
                    (7, 0.05),   # Çok hafif metalik ayrıntı
                    (8, 0.015)   # Üst frekansları sınırlama
                ],

                "decay": 0.65,
                "attack": 0.025,
                "detune": 0.0020,
                "pluck_noise": 0.075,
                "string_damping": 0.0015,

                "body_gain": 1.00,
                "body_freq": 220.0,
                "body_width": 180.0,
            },
            "bowed_guitar": {
                "type": "string",
                "model": "bowed_string",

                "harmonics": [
                    (1, 5.00),    # Sıcak ve dolgun ana ses
                    (2, 0.85),    # Yumuşak oktav desteği
                    (3, 0.32),    # Hafif nefesli karakter
                    (4, 0.10),    # Kontrollü üst orta frekans
                    (5, 0.32),    # Hafif yay rezonansı
                    (6, 0.09),    # İnce yay dokusu
                    (7, 0.025)    # Çok hafif üst frekans
                ],

                "decay": 0.05,
                "attack": 0.40,
                "release": 1.2,

                "string_damping": 0.0001,

                "vibrato_rate": 3.4,
                "vibrato_depth": 0.0012,
            },
            "ukulele": {
                "type": "string",
                "model": "plucked_string",

                "harmonics": [
                    (1, 1.00),
                    (2, 0.42),
                    (3, 0.18),
                    (4, 0.065),
                    (5, 0.020)
                ],

                "harmonic_blend": 0.18,

                "decay": 0.65,
                "attack": 0.003,
                "detune": 0.0010,
                "pluck_noise": 0.10,
                "string_damping": 0.025,

                "body_gain": 0.13,
                "body_freq": 330.0,
                "body_width": 120.0,
            },
            "bas_gitar": {
                "type": "string",
                "model": "plucked_string",
                "dark_excitation": True, 
                "harmonics": [
                    (1, 1.00),   # Temel bas gövdesi
                    (2, 0.45),   # YENİ KATMAN: Yüksek notalara çıkıldığında sesin cılızlaşmasını önleyen, tınıyı koruyan oktav harmoniği
                    (3, 0.15)    # Sese o telli bas gitar hissiyatını (growl) veren hafif pürüz
                ],
                "decay": 1.4,    # Yüksek notaların biraz daha süzülerek duyulması için hafifçe uzatıldı
                "attack": 0.012,     
                "detune": 0.000,
                "pluck_noise": 0.01, 
                "string_damping": 0.02, # Yüksek notalarda telin enerjisini çok hızlı kaybetmesini engellemek için biraz düşürüldü
                "kick_burst": 15.0,     # Üst notalarda tokat sesine dönmemesi için biraz dengelendi
                "body_gain": 8.5,       # Filtre genişlediği için kazanç dengelendi
                "body_freq": 65.0,      # DÜZELTME: 20Hz'den 65Hz'e çekildi. Yüksek notaların da filtreden geçebilmesini sağlar
                "body_width": 140.0,    # DÜZELTME: 55'ten 140'a fırlatıldı. Filtre aralığı genişlediği için üst notalar artık boğulmayacak
            },

            "kick_bass": {
                "type": "kick_bass",
                "harmonics": [(1, 0.35)],
                "decay": 2.5,
                "attack": 0.01,
                "kick_burst": 1.5,
                "body_freq": 45.0,
                "kick_start_freq": 65.0,
                "body_gain": 1.0,
                "ring_time": 1.0,
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
                "harmonics": [(1, 1.00), (2, 0.55), (3, 0.75), (4, 0.20), (5, 0.45), (6, 0.12), (7, 0.08)],
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

            if cfg["type"] == "kick_bass":
                self._gains[name] = 5
                continue

            harmonic_sum = sum(a for _, a in cfg["harmonics"])

            if cfg["type"] == "wind":
                harmonic_sum += 0.04

            if cfg.get("model") == "double_string":
                # Harmonikler render aşamasında normalize edilecek.
                # Bu nedenle harmonik toplamına göre tekrar kısma yapma.
                self._gains[name] = 0.92
            else:
                self._gains[name] = 0.92 / max(harmonic_sum, 1e-8)

        # INIT İÇİNE BOWED GUITAR İÇİN ÖZEL ELEKTRO GİTAR AMFİ FİLTRESİ EKLENDİ (PERFORMANS İÇİN)
        self._pluck_b, self._pluck_a = butter(2, 3000 / (sample_rate / 2), btype="low")
        self._breath_b, self._breath_a = butter(2, 2200 / (sample_rate / 2), btype="low")
        self._amp_b, self._amp_a = butter(2, 3500 / (sample_rate / 2), btype="low")

        # Kick/bas davul filtresi
        kick_cutoff = min(180.0, self.sample_rate * 0.45)

        self._kick_b, self._kick_a = butter(
            2,
            kick_cutoff / (self.sample_rate / 2),
            btype="low"
        )

        self._kick_b = self._kick_b.astype(np.float32)
        self._kick_a = self._kick_a.astype(np.float32)
    # ------------------------------------------------------------

    def harmonic_frequencies(self, instrument, frequency, n=8):
        cfg = self.instruments.get(instrument.lower(), self.instruments["gitar"])
        return [(frequency * h, amp) for h, amp in cfg["harmonics"][:n]]

    def max_ring_time(self, instrument):
        cfg = self.instruments.get(instrument.lower(), self.instruments["gitar"])

        if cfg["type"] == "kick_bass":
            return (
                float(cfg.get("ring_time", 1.0))
                + cfg["attack"]
                + 0.05
            )
    
        # DÜZELTME: Eğer enstrüman üflemeliyse veya keman yaylı gitarsa, süre patlamasını engelle
        if cfg["type"] == "wind" or cfg.get("model") == "bowed_string":
            return cfg["attack"] + cfg.get("release", 0.25) + 0.05
            
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
                if config.get("model") == "bowed_string":
                    wave = self._render_bowed(frequency, t, config, duration)
                else:
                    wave = self._render_string(frequency, t, config)
            elif config["type"] == "kick_bass":
                wave = self._render_kick_bass(frequency, t, config)
            elif config["type"] == "wind":
                wave = self._render_flute(frequency, t, config, duration)
            elif config["type"] == "metal":
                wave = self._render_metal(frequency, t, config)
            else:
                wave = np.zeros_like(t)

            if len(self._wave_cache) >= 128:
                self._wave_cache.pop(next(iter(self._wave_cache)))

            self._wave_cache[cache_key] = wave

        return wave * np.float32(gain)

    #bowd gitar için
    def _render_bowed(self, frequency, t, config, duration):
        n_samples = len(t)

        vibrato_rate = config.get("vibrato_rate", 3.8)
        vibrato_depth = config.get("vibrato_depth", 0.004)

        base_vibrato_wave = np.sin(2 * np.pi * vibrato_rate * t) / vibrato_rate

        wave = np.zeros_like(t)
        for harmonic, amplitude in config["harmonics"]:
            base = 2 * np.pi * frequency * harmonic * t
            vib = vibrato_depth * frequency * harmonic * base_vibrato_wave
            wave += amplitude * np.sin(base + vib)

        # Gürültü üretiminde önceden türetilmiş hazır filtre katsayıları kullanılıyor
        noise = np.random.normal(0, 1, n_samples)
        noise = lfilter(self._breath_b, self._breath_a, noise)
        wave += noise * 0.04

        attack = config["attack"]
        release = config["release"]

        envelope = np.zeros_like(t)
        attack_mask = t < attack
        if attack > 0:
            envelope[attack_mask] = t[attack_mask] / attack

        sustain_end = max(duration, attack)
        sustain_mask = (t >= attack) & (t < sustain_end)
        envelope[sustain_mask] = 1.0

        release_end = sustain_end + release
        release_mask = (t >= sustain_end) & (t < release_end)
        if release > 0:
            envelope[release_mask] = 1.0 - (t[release_mask] - sustain_end) / release

        wave *= envelope

        # AMFİ REZONANSI
        wave = lfilter(self._amp_b, self._amp_a, wave)

        return wave


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
            cutoff_freq = 100 
            nyquist = self.sample_rate / 2
            b_low, a_low = butter(4, cutoff_freq / nyquist, btype="low")
            noise_burst = lfilter(b_low, a_low, noise_burst)

            # 2. KAYBOLAN SESİ DÜZELTME (Makyaj/Gain)
            bass_boost_gain = 7000.5 
            noise_burst = noise_burst * bass_boost_gain

        # Doğal tel vuruşu için hafif fade-out
        noise_burst *= np.linspace(1.0, 0.0, delay, dtype=np.float32)
        excitation[:delay] = noise_burst

        # 2. Işık Hızında Geribildirim (Feedback) Hesaplaması
        string_damping = config.get("string_damping", 0.05)
        g = 1.0 - string_damping # Kazanç (1'e ne kadar yakınsa o kadar uzun sustain)
        
        wave = _karplus_feedback(excitation, delay, g)

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

    def _render_double_string(self, frequency, t, config):

        detune = config.get("detune", 0.0020)
        decay = config.get("decay", 0.65)

        # 1. Doğal tel sesi: Karplus-Strong
        string_a = self._string_wave(
            frequency, t, config, seed_offset=1
        )
        string_b = self._string_wave(
            frequency * (1.0 + detune),
            t, config, seed_offset=2
        )

        string_body = 0.5 * (string_a + string_b)

        # 2. Harmonik katman
        harmonics = config["harmonics"]
        harmonics = harmonics[:6]

        total = max(
            sum(abs(amp) for _, amp in harmonics),
            1e-8
        )

        harmonic_body = np.zeros_like(t, dtype=np.float32)

        for harmonic, amplitude in harmonics:
            partial_decay = decay * (
                1.0 + 0.35 * (harmonic - 1)
            )

            harmonic_body += (
                (amplitude / total)
                * np.sin(2.0 * np.pi * frequency * harmonic * t)
                * np.exp(-partial_decay * t)
            ).astype(np.float32)

        # Doğal tel baskın, harmonikler tınıyı renklendiriyor.
        wave = 0.80 * string_body + 0.20 * harmonic_body

        # Gövde rezonansı ve tel çekme sesi
        body_config = dict(config)
        body_config["body_gain"] = config.get("body_gain", 0.25)

        wave = self._body_resonance(
            frequency, t, wave, body_config
        )
        wave += self._pluck_burst(t, config)

        return wave.astype(np.float32)

    def _render_string(self, frequency, t, config):
        model = config.get("model", "plucked_string")

        if model == "double_string":
            return self._render_double_string(
                frequency, t, config
            )

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

        blend = config.get("harmonic_blend", 0.0)

        if blend > 0.0:
            harmonics = config["harmonics"]
            total = max(
                sum(abs(amp) for _, amp in harmonics),
                1e-8
            )
        
            harmonic_body = np.zeros_like(t, dtype=np.float32)
        
            for harmonic, amplitude in harmonics:
                partial_decay = config["decay"] * (
                    1.0 + 0.35 * (harmonic - 1)
                )
        
                harmonic_body += (
                    (amplitude / total)
                    * np.sin(
                        2.0 * np.pi * frequency * harmonic * t
                    )
                    * np.exp(-partial_decay * t)
                ).astype(np.float32)
        
            body = (
                (1.0 - blend) * body
                + blend * harmonic_body
            )
        
        attack = config["attack"]
        attack_env = np.ones_like(t)
        mask = t < attack
        attack_env[mask] = t[mask] / max(attack, 0.0001)
        body *= attack_env
        
        body = self._body_resonance(frequency, t, body, config)
        body += self._pluck_burst(t, config)
        return body

    def _render_kick_bass(self, frequency, t, config):
        t = np.asarray(t, dtype=np.float32)

        # 1. Temel frekans
        harmonic = config["harmonics"][0]
        h_mult = float(harmonic[0])
        h_amp = float(harmonic[1])

        phase = 2 * np.pi * frequency * h_mult * t
        wave = h_amp * np.sin(phase)

        # 2. Kick gövdesi
        kick_gain = float(config.get("kick_burst", 2.5))
        burst_len = max(float(config["attack"]) * 12.0, 0.035)

        kick_envelope = np.exp(-t / burst_len).astype(np.float32)

        kick_freq = float(config.get("body_freq", 45.0))
        kick_start_freq = float(
            config.get("kick_start_freq", kick_freq * 2.1)
        )

        # Başlangıçta yüksek, ardından düşük frekansa inen kick.
        # Faz, anlık frekansın zamana göre integraliyle hesaplanır.
        kick_phase = 2 * np.pi * (
            kick_freq * t
            + (kick_start_freq - kick_freq)
            * burst_len
            * (1.0 - kick_envelope)
        )

        kick_wave = np.sin(kick_phase)
        wave += kick_wave * kick_envelope * kick_gain

        # 3. Genel sönümlenme
        decay = max(float(config.get("decay", 2.5)), 0.05)
        decay_env = np.exp(-decay * t).astype(np.float32)
        wave *= decay_env

        # 4. Gövde filtresi
        wave = lfilter(
            self._kick_b,
            self._kick_a,
            wave.astype(np.float32, copy=False)
        )

        wave = np.asarray(wave, dtype=np.float32)
        wave *= float(config.get("body_gain", 1.0))

        return wave
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
