from dataclasses import dataclass


@dataclass
class NoteEvent:
    """MelodyEngine tarafından üretilen, InstrumentSynth'e gönderilen
    tek bir nota olayı. EquationEngine -> NoteMapper zincirinin çıktısı
    budur; InstrumentSynth.play_note() bu bilgilerle sesi üretir."""

    instrument: str
    frequency: float
    duration: float
    velocity: float
    start_time: float = 0.0
