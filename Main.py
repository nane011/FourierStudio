# Main.py
from InstrumentSynth import InstrumentSynth
from AudioEngine import AudioEngine
from App import run_app


def main():
    synth = InstrumentSynth(sample_rate=44100)
    audio_engine = AudioEngine(sample_rate=44100)
    run_app(synth, audio_engine)


if __name__ == "__main__":
    main()
