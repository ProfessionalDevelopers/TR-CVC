import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
import sounddevice as sd
import curses
import threading
import random
import json
import os

IS_EXITING = False

FS = 44100  # sample rate
BPM = 120.0
BPMFRAME = (60 / BPM) / 4
SEQUENCE_FILE = "sequence.json"  # the file where we'll save and load the sequence
MASTER_LEVEL = 0.8  # master level
GRID = [
    "x" * 16 for _ in range(11)
]  # add extra rows for the bassline, mid tom and clap
CURSOR = [0, 0]
COMPLETE_SEQUENCE = np.zeros(16 * int(FS * BPMFRAME), dtype=np.float32)
SWING = 50
PLAYBACK_THREAD = None
CURRENT_KIT = "808"
BASSLINE_FILTER_FREQ = 880.0
SLIDE_AMT = 0.1


class Instrument:
    def __init__(self, label, sound, level):
        self.label = label
        self.sound = sound
        self.level = level


def generate_sound(freq, decay_factor, length, noise=False):
    x = np.arange(length)
    y = np.random.normal(0, 1, length) if noise else np.sin(2 * np.pi * freq * x / FS)
    decay = np.exp(-decay_factor * x)
    return (y * decay).astype(np.float32)


def generate_kick_sound(start_freq, end_freq, decay_factor, length):
    x = np.arange(length)
    y = np.sin(
        2
        * np.pi
        * start_freq
        * np.exp(np.log(end_freq / start_freq) * x / length)
        * x
        / FS
    )
    decay = np.exp(-decay_factor * x)
    return (y * decay).astype(np.float32)


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, data)


def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = lfilter(b, a, data)
    return y


def generate_acid_bassline(
    note_frequency, note_duration, slide_duration, resonance, distortion, fs=44100
):
    t = np.arange(note_duration * fs)
    bassline = signal.square(2 * np.pi * note_frequency * t / fs)  # using square wave

    if slide_duration > 0:
        slide_frequencies = np.linspace(
            note_frequency, note_frequency * 2, int(fs * slide_duration)
        )
        slide = signal.square(
            2 * np.pi * slide_frequencies * np.arange(len(slide_frequencies)) / fs
        )
        bassline[: len(slide)] = slide

    if resonance > 0:
        sos = signal.butter(10, note_frequency, "hp", fs=fs, output="sos")
        bassline = signal.sosfilt(sos, bassline)

    if distortion > 0:
        bassline = np.clip(bassline, -distortion, distortion)

    return bassline


def generate_piano_sound(freq, decay_factor, length):
    x = np.arange(length)
    y = np.sin(2 * np.pi * freq * x / FS)
    decay = np.exp(-decay_factor * x)
    return (y * decay).astype(np.float32)


def generate_piano_chord(root_freq, decay_factor, length):
    major_third_freq = root_freq * 2 ** (4 / 12)
    perfect_fifth_freq = root_freq * 2 ** (7 / 12)
    major_seventh_freq = root_freq * 2 ** (11 / 12)

    root_note = np.sin(2 * np.pi * root_freq * np.arange(length) / FS)
    major_third = np.sin(2 * np.pi * major_third_freq * np.arange(length) / FS)
    perfect_fifth = np.sin(2 * np.pi * perfect_fifth_freq * np.arange(length) / FS)
    major_seventh = np.sin(2 * np.pi * major_seventh_freq * np.arange(length) / FS)

    chord = root_note + major_third + perfect_fifth + major_seventh
    decay = np.exp(-decay_factor * np.arange(length))
    return (chord * decay).astype(np.float32)


KICK_808 = generate_kick_sound(65.0, 50.0, 0.0003, int(FS * 0.4))
SNARE_808 = generate_sound(180.0, 0.0015, int(FS * BPMFRAME))
SNARE_808 += generate_sound(0, 0.0015, int(FS * BPMFRAME), noise=True)
HIHAT_808 = bandpass_filter(
    generate_sound(0, 0.005, int(FS * 0.5), noise=True), 7000, 9000, FS
)
OPEN_HIHAT_808 = bandpass_filter(
    generate_sound(0, 0.001, int(FS * 1.4), noise=True), 7000, 9000, FS
)
COWBELL_808 = generate_sound(380.0, 0.002, int(FS * BPMFRAME))
HIGH_TOM_909 = generate_kick_sound(350.0, 150.0, 0.0005, int(FS * 0.2))
MID_TOM_909 = generate_kick_sound(300.0, 125.0, 0.0005, int(FS * 0.2))
LOW_TOM_909 = generate_kick_sound(200.0, 100.0, 0.0005, int(FS * 0.2))
CLAP_808 = generate_sound(0, 0.001, int(FS * BPMFRAME), noise=True)

KICK_909 = generate_kick_sound(150.0, 30.0, 0.0002, int(FS * 0.5))  # longer and boomy
SNARE_909 = generate_sound(220.0, 0.001, int(FS * BPMFRAME))
SNARE_909 += generate_sound(0, 0.001, int(FS * BPMFRAME), noise=True)
HIHAT_909 = bandpass_filter(
    generate_sound(0, 0.0025, int(FS * 0.5), noise=True), 7000, 9000, FS
)
OPEN_HIHAT_909 = bandpass_filter(
    generate_sound(100, 0.0005, int(FS * 1.4), noise=True), 6000, 9000, FS
)
COWBELL_909 = generate_sound(480.0, 0.001, int(FS * 0.075))
HIGH_TOM_808 = generate_kick_sound(350.0, 150.0, 0.0005, int(FS * 0.2))
MID_TOM_808 = generate_kick_sound(300.0, 125.0, 0.0005, int(FS * 0.2))
LOW_TOM_808 = generate_kick_sound(200.0, 100.0, 0.0005, int(FS * 0.2))
CLAP_909 = generate_sound(0, 0.001, int(FS * BPMFRAME), noise=True)

# PIANO_SOUND = generate_piano_sound(440.0, 0.001, int(FS * BPMFRAME))  # A4 note


INSTRUMENTS_808 = [
    Instrument("⦿ BD", KICK_808, 0.8),
    Instrument("◼ SD", SNARE_808, 1.0),
    Instrument("⚆ LT", LOW_TOM_808, 0.8),
    Instrument("⚇ MT", MID_TOM_808, 0.7),
    Instrument("⚈ HT", HIGH_TOM_808, 0.9),
    Instrument("॥ CP", CLAP_808, 0.6),
    Instrument("Ⓚ CB", COWBELL_808, 1.0),
    Instrument("⨂ HH", HIHAT_808, 1.0),
    Instrument("⨁ OH", OPEN_HIHAT_808, 1.0),
    Instrument("♩ BL", None, 0.2),
    Instrument("♪ PA", None, 0.8),
]

INSTRUMENTS_909 = [
    Instrument("⦿ BD", KICK_909, 0.8),
    Instrument("◼ SD", SNARE_909, 1.0),
    Instrument("⚆ LT", LOW_TOM_909, 0.8),
    Instrument("⚇ MT", MID_TOM_909, 0.7),
    Instrument("⚈ HT", HIGH_TOM_909, 0.9),
    Instrument("॥ CP", CLAP_909, 0.6),
    Instrument("Ⓚ CB", COWBELL_909, 1.0),
    Instrument("⨂ HH", HIHAT_909, 1.0),
    Instrument("⨁ OH", OPEN_HIHAT_909, 1.0),
    Instrument("♩ BL", None, 0.2),
    Instrument("♪ PA", None, 0.8),
]


stdscr = curses.initscr()
curses.noecho()
curses.cbreak()
stdscr.keypad(True)

INSTRUMENT_MUTE_STATUS = {i: False for i in range(len(INSTRUMENTS_808))}
ORIGINAL_LEVELS = {i: inst.level for i, inst in enumerate(INSTRUMENTS_808)}

if os.path.exists(SEQUENCE_FILE):
    with open(SEQUENCE_FILE) as f:
        state = json.load(f)
        GRID = state["grid"]
        SWING = state["swing"]
        CURRENT_KIT = state["current_kit"]
        BPM = state.get("bpm", 120.0)
        BASSLINE_FILTER_FREQ = state.get("bassline_freq", 880.0)
        SLIDE_AMT = state.get("slide_amt", 0.1)
        # Only update the mute status if it exists in the loaded state
        if "mute_status" in state:
            INSTRUMENT_MUTE_STATUS = {
                int(k): v for k, v in state["mute_status"].items()
            }

        for i, mute in INSTRUMENT_MUTE_STATUS.items():
            if mute:
                INSTRUMENTS_808[i].level = 0.0
                INSTRUMENTS_909[i].level = 0.0
            else:
                INSTRUMENTS_808[i].level = ORIGINAL_LEVELS[i]
                INSTRUMENTS_909[i].level = ORIGINAL_LEVELS[i]
    BPMFRAME = (60 / BPM) / 4


instruments = INSTRUMENTS_808 if CURRENT_KIT == "808" else INSTRUMENTS_909


def dump_sequence():
    with open(SEQUENCE_FILE, "w") as f:
        json.dump(
            {
                "grid": GRID,
                "swing": SWING,
                "current_kit": CURRENT_KIT,
                "bpm": BPM,
                "bassline_freq": BASSLINE_FILTER_FREQ,
                "mute_status": INSTRUMENT_MUTE_STATUS,  # Add the mute status to the saved state
                "slide_amt": SLIDE_AMT,
            },
            f,
        )


# Define frequencies for 'o', 'u', 'p' for the bassline and piano
bassline_freqs = [55, 110, 220]
piano_freqs = [262, 330, 440]  # frequencies for C4, E4, A4 notes


def update_sequence():
    dump_sequence()
    global COMPLETE_SEQUENCE, instruments
    instruments = INSTRUMENTS_808 if CURRENT_KIT == "808" else INSTRUMENTS_909

    sequences = []
    for j in range(len(instruments)):
        if INSTRUMENT_MUTE_STATUS.get(j, False):
            # If instrument is muted, skip this iteration
            continue

        instrument_sequence = np.zeros(16 * int(FS * BPMFRAME), dtype=np.float32)
        for i in range(16):
            # Start index is shifted forward by a certain amount for even steps
            swing_shift = ((FS * BPMFRAME) * (SWING - 50) / 100) if i % 2 == 1 else 0
            start_index = min(
                int(i * FS * BPMFRAME + swing_shift), instrument_sequence.size - 1
            )

            if GRID[j][i] != "x" and instruments[j].sound is not None:
                sound = instruments[j].sound * instruments[j].level
                end_index = min(start_index + sound.size, instrument_sequence.size)
                instrument_sequence[start_index:end_index] += sound[
                    : end_index - start_index
                ]

        sequences.append(instrument_sequence)

    # Handle the bassline and piano lines separately
    for j in [-2, -1]:  # the last two lines are the 'BL' and 'PA' lines
        bassline_sequence = np.zeros(16 * int(FS * BPMFRAME), dtype=np.float32)
        for i in range(16):
            # Start index is shifted forward by a certain amount for even steps
            swing_shift = ((FS * BPMFRAME) * (SWING - 50) / 100) if i % 2 == 1 else 0
            start_index = min(
                int(i * FS * BPMFRAME + swing_shift), bassline_sequence.size - 1
            )

            if GRID[j][i] in "oup":
                freqs = bassline_freqs if j == -2 else piano_freqs
                if j == -2:  # if it's the 'BL' line
                    sound = (
                        generate_acid_bassline(
                            freqs["oup".index(GRID[j][i])],
                            BPMFRAME,
                            BPMFRAME * SLIDE_AMT,
                            90,
                            0,
                        )
                        * instruments[j].level
                    )
                    sound = lowpass_filter(
                        sound, BASSLINE_FILTER_FREQ, FS
                    )  # Apply low-pass filter
                else:  # if it's the 'PA' line
                    sound = (
                        generate_piano_chord(
                            freqs["oup".index(GRID[j][i])], 0.001, int(FS * BPMFRAME)
                        )
                        * instruments[j].level
                    )
                end_index = min(start_index + sound.size, bassline_sequence.size)
                bassline_sequence[start_index:end_index] += sound[
                    : end_index - start_index
                ]
        sequences.append(bassline_sequence)

    COMPLETE_SEQUENCE = sum(sequences) * MASTER_LEVEL


def playback_function():
    global PLAYBACK_THREAD, instruments
    with sd.OutputStream(samplerate=FS, channels=1) as stream:
        while PLAYBACK_THREAD is not None:
            if IS_EXITING:
                stream.abort(True)
                return
            update_sequence()
            stream.write(COMPLETE_SEQUENCE)


try:
    while not IS_EXITING:
        for i, row in enumerate(GRID):
            row_str = " ".join(row[j: j + 4] for j in range(0, len(row), 4))
            stdscr.addstr(
                i, 0, f"{instruments[i].label} {instruments[i].level:.2f}: {row_str}"
            )

        stdscr.addstr(
            len(GRID) + 1, 0, "\n"
        )  # Add a blank line between the sequencer and the status
        stdscr.addstr(
            len(GRID) + 1,
            0,
            f'Move with (arrows), press (space) to toggle a step, \n(x) to clear the pattern, (z) to clear the row\n\n⇧/(-/=) BPM: {BPM}\n⇧/(5/6/0) Swing: {SWING}%\n(8/9): Selected Kit: {CURRENT_KIT}\n(s): Status: {"Playing" if PLAYBACK_THREAD else "Stopped"}\n(f/g): Bass Filter Freq: {BASSLINE_FILTER_FREQ}\n(o/p): Slide Amount: {SLIDE_AMT * 100}%\n(m): Mute/Unmute Track\n\n(q) to quit',
        )

        stdscr.move(
            CURSOR[0],
            CURSOR[1] // 4 * 5 + CURSOR[1] % 4 +
            len(instruments[CURSOR[0]].label) + 7,
        )
        stdscr.refresh()

        c = stdscr.getch()

        if c == curses.KEY_UP and CURSOR[0] > 0:
            CURSOR[0] -= 1
        elif c == curses.KEY_DOWN and CURSOR[0] < len(instruments) - 1:
            CURSOR[0] += 1
        elif c == curses.KEY_LEFT and CURSOR[1] > 0:
            CURSOR[1] -= 1
        elif c == curses.KEY_RIGHT and CURSOR[1] < 15:
            CURSOR[1] += 1
        elif c == ord(" "):
            if CURSOR[0] in [
                len(instruments) - 2,
                len(instruments) - 1,
            ]:  # if cursor is at the 'BL' or 'PA' line
                GRID[CURSOR[0]] = (
                    GRID[CURSOR[0]][: CURSOR[1]]
                    + {"x": "o", "o": "u", "u": "p",
                        "p": "x"}[GRID[CURSOR[0]][CURSOR[1]]]
                    + GRID[CURSOR[0]][CURSOR[1] + 1:]
                )
            else:
                GRID[CURSOR[0]] = (
                    GRID[CURSOR[0]][: CURSOR[1]]
                    + {"x": "o", "o": "x"}[GRID[CURSOR[0]][CURSOR[1]]]
                    + GRID[CURSOR[0]][CURSOR[1] + 1:]
                )
        elif c in (ord("8"), ord("9")):
            CURRENT_KIT = {ord("8"): "808", ord("9"): "909"}[c]
        elif c in (ord("8"), ord("9")):
            CURRENT_KIT = {ord("8"): "808", ord("9"): "909"}[c]
            instruments = INSTRUMENTS_808 if CURRENT_KIT == "808" else INSTRUMENTS_909
        elif c == ord("5"):
            SWING = SWING = max(SWING - 5, 0)
        elif c == ord("6"):
            SWING = min(SWING + 5, 100)
        elif c == ord("%"):
            SWING = SWING = max(SWING - 1, 0)
        elif c == ord("^"):
            SWING = min(SWING + 1, 100)
        elif c == ord("p"):
            SLIDE_AMT = min(SLIDE_AMT + 0.05, 1)  # swing can't go above 1
        elif c == ord("o"):
            SLIDE_AMT = max(SLIDE_AMT - 0.05, 0)  # swing can't go below 0
        elif c == ord("-"):
            BPM = max(BPM - 5, 1)  # BPM cannot go below 1
            BPMFRAME = (60 / BPM) / 4
        elif c == ord("="):
            BPM += 5
            BPMFRAME = (60 / BPM) / 4
        elif c == ord("_"):
            BPM = max(BPM - 1, 1)  # BPM cannot go below 1
            BPMFRAME = (60 / BPM) / 4
        elif c == ord("+"):
            BPM += 1
            BPMFRAME = (60 / BPM) / 4
        elif c == ord("g"):
            BASSLINE_FILTER_FREQ = min(BASSLINE_FILTER_FREQ * 2, 12800.0)
        elif c == ord("f"):
            BASSLINE_FILTER_FREQ /= 2
        elif c == ord("x"):
            GRID = ["x" * 16 for _ in range(len(instruments))]
        elif c == ord("z"):
            GRID[CURSOR[0]] = "x" * STEP_COUNT
        elif c == ord("m"):  # Mute/unmute the current track
            if instruments[CURSOR[0]].level == 0.0:
                INSTRUMENT_MUTE_STATUS[CURSOR[0]] = False  # Unmute the track
                instruments[CURSOR[0]].level = ORIGINAL_LEVELS[CURSOR[0]]
                INSTRUMENTS_808[CURSOR[0]].level = ORIGINAL_LEVELS[CURSOR[0]]
                INSTRUMENTS_909[CURSOR[0]].level = ORIGINAL_LEVELS[CURSOR[0]]
            else:
                INSTRUMENT_MUTE_STATUS[CURSOR[0]] = True  # Mute the track
                ORIGINAL_LEVELS[CURSOR[0]] = instruments[
                    CURSOR[0]
                ].level  # Save the current level
                instruments[CURSOR[0]].level = 0.0
                INSTRUMENTS_808[CURSOR[0]].level = 0.0
                INSTRUMENTS_909[CURSOR[0]].level = 0.0
        elif c == ord("s"):
            update_sequence()
            if PLAYBACK_THREAD is None:
                PLAYBACK_THREAD = threading.Thread(target=playback_function)
                PLAYBACK_THREAD.start()
            else:
                PLAYBACK_THREAD = None
        elif c == ord("q"):
            PLAYBACK_THREAD = None
            IS_EXITING = True
            break
except KeyboardInterrupt:
    print('Interrupted. Exiting.')
    IS_EXITING = True
    PLAYBACK_THREAD = None
    curses.endwin()

curses.nocbreak()
stdscr.keypad(False)
curses.echo()
curses.endwin()
