import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
import sounddevice as sd
import curses
import threading
import random
import json
import os

FS = 44100  # sample rate
SEQUENCE_FILE = 'sequence.json'  # the file where we'll save and load the sequence
LABELS = ['BD', 'SD', 'HH', 'OH', 'CB', 'HT', 'LT', 'BL']
# add a level for the bassline at the end
STEPS = 32
LEVELS = [0.8, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.2]
MASTER_LEVEL = 0.8  # master level
GRID = ['x'*STEPS for _ in range(8)]  # added an extra row for the bassline
CURSOR = [0, 0]
COMPLETE_SEQUENCE = np.zeros(STEPS * int(FS * 0.125), dtype=np.float32)
SWING = 0
PLAYBACK_THREAD = None
CURRENT_KIT = "808"


def generate_sound(freq, decay_factor, length, noise=False):
    x = np.arange(length)
    y = np.random.normal(0, 1, length) if noise else np.sin(
        2 * np.pi * freq * x / FS)
    decay = np.exp(-decay_factor * x)
    return (y * decay).astype(np.float32)


def generate_kick_sound(start_freq, end_freq, decay_factor, length):
    x = np.arange(length)
    y = np.sin(2 * np.pi * start_freq *
               np.exp(np.log(end_freq/start_freq)*x/length) * x / FS)
    decay = np.exp(-decay_factor * x)
    return (y * decay).astype(np.float32)


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


def generate_acid_bassline(note_frequency, note_duration, slide_duration, resonance, distortion, fs=44100):
    t = np.arange(note_duration * fs)
    bassline = signal.square(
        2 * np.pi * note_frequency * t / fs)  # using square wave

    if slide_duration > 0:
        slide_frequencies = np.linspace(
            note_frequency, note_frequency * 2, int(fs * slide_duration))
        slide = signal.square(2 * np.pi * slide_frequencies *
                              np.arange(len(slide_frequencies)) / fs)
        bassline[:len(slide)] = slide

    if resonance > 0:
        sos = signal.butter(10, note_frequency, 'hp', fs=fs, output='sos')
        bassline = signal.sosfilt(sos, bassline)

    if distortion > 0:
        bassline = np.clip(bassline, -distortion, distortion)

    return bassline


KICK_808 = generate_kick_sound(65.0, 50.0, 0.0003, int(FS * 0.4))
SNARE_808 = generate_sound(180.0, 0.0015, int(FS * 0.125))
SNARE_808 += generate_sound(0, 0.0015, int(FS * 0.125), noise=True)
HIHAT_808 = bandpass_filter(generate_sound(
    0, 0.005, int(FS * 0.5), noise=True), 7000, 9000, FS)
OPEN_HIHAT_808 = bandpass_filter(generate_sound(
    0, 0.001, int(FS * 1.4), noise=True), 7000, 9000, FS)
COWBELL_808 = generate_sound(380.0, 0.002, int(FS * 0.125))
HIGH_TOM_808 = generate_kick_sound(300.0, 150.0, 0.0005, int(FS * 0.2))
LOW_TOM_808 = generate_kick_sound(200.0, 75.0, 0.0005, int(FS * 0.2))

KICK_909 = generate_kick_sound(
    150.0, 30.0, 0.0002, int(FS * 0.5))  # longer and boomy
SNARE_909 = generate_sound(220.0, 0.001, int(FS * 0.125))
SNARE_909 += generate_sound(0, 0.001, int(FS * 0.125), noise=True)
HIHAT_909 = bandpass_filter(generate_sound(
    0, 0.0025, int(FS * 0.5), noise=True), 7000, 9000, FS)
OPEN_HIHAT_909 = bandpass_filter(generate_sound(
    100, 0.0005, int(FS * 1.4), noise=True), 6000, 9000, FS)
COWBELL_909 = generate_sound(480.0, 0.001, int(FS * 0.075))
HIGH_TOM_909 = generate_kick_sound(300.0, 150.0, 0.0005, int(FS * 0.2))
LOW_TOM_909 = generate_kick_sound(200.0, 75.0, 0.0005, int(FS * 0.2))

SOUNDS_808 = [KICK_808, SNARE_808, HIHAT_808,
              OPEN_HIHAT_808, COWBELL_808, HIGH_TOM_808, LOW_TOM_808]
SOUNDS_909 = [KICK_909, SNARE_909, HIHAT_909,
              OPEN_HIHAT_909, COWBELL_909, HIGH_TOM_909, LOW_TOM_909]

stdscr = curses.initscr()
curses.noecho()
curses.cbreak()
stdscr.keypad(True)

if os.path.exists(SEQUENCE_FILE):
    with open(SEQUENCE_FILE) as f:
        state = json.load(f)
        GRID = state['grid']
        SWING = state['swing']
        CURRENT_KIT = state['current_kit']


def dump_sequence():
    with open(SEQUENCE_FILE, 'w') as f:
        json.dump({'grid': GRID, 'swing': SWING,
                  'current_kit': CURRENT_KIT}, f)


def update_sequence():
    dump_sequence()
    global COMPLETE_SEQUENCE
    sequences = [np.zeros(STEPS * int(FS * 0.125), dtype=np.float32)
                 for _ in range(8)]
    sounds = SOUNDS_808 if CURRENT_KIT == '808' else SOUNDS_909

    for i in range(STEPS):
        start_index = int(i * FS * 0.125 * (1 + SWING)
                          ) if SWING and i % 2 else i * int(FS * 0.125)
        end_indices = [start_index + sound.size for sound in sounds]
        end_indices.append(start_index + int(FS * 0.125))

        for j in range(7):
            if GRID[j][i] != 'x':
                sound = sounds[j] * LEVELS[j]
                sequences[j][start_index:min(end_indices[j], sequences[j].size)] += sound[:min(
                    end_indices[j], sequences[j].size) - start_index]

        min_index = min(end_indices[7], sequences[7].size)
        if GRID[7][i] in 'oup':
            bassline_freqs = [55, 110, 220]  # Frequencies for 'o', 'u', 'p'
            sound = generate_acid_bassline(
                bassline_freqs['oup'.index(GRID[7][i])], 0.125, 0, 90, 0) * LEVELS[7]
            sequences[7][start_index:min_index] += sound[:min_index - start_index]

    COMPLETE_SEQUENCE = sum(sequences) * MASTER_LEVEL


def playback_function():
    global PLAYBACK_THREAD
    with sd.OutputStream(samplerate=FS, channels=1) as stream:
        while PLAYBACK_THREAD is not None:
            update_sequence()
            stream.write(COMPLETE_SEQUENCE)


while True:
    for i, row in enumerate(GRID):
        row_str = ' '.join(row[j:j+4] for j in range(0, len(row), 4))
        stdscr.addstr(i, 0, f'{LABELS[i]} {LEVELS[i]:.2f}: {row_str}')
    stdscr.addstr(
        8, 0, f'Selected Kit: {CURRENT_KIT}\nSwing: {SWING * 100:.0f}%\nStatus: {"Playing" if PLAYBACK_THREAD else "Stopped"}\nMaster level: {MASTER_LEVEL}')
    stdscr.move(CURSOR[0], CURSOR[1] // 4 * 5 + CURSOR[1] %
                4 + len(LABELS[CURSOR[0]]) + 7)
    stdscr.refresh()

    c = stdscr.getch()

    if c == curses.KEY_UP and CURSOR[0] > 0:
        CURSOR[0] -= 1
    elif c == curses.KEY_DOWN and CURSOR[0] < 7:
        CURSOR[0] += 1
    elif c == curses.KEY_LEFT and CURSOR[1] > 0:
        CURSOR[1] -= 1
    elif c == curses.KEY_RIGHT and CURSOR[1] < STEPS - 1:
        CURSOR[1] += 1
    elif c == ord(' '):
        GRID[CURSOR[0]] = GRID[CURSOR[0]][:CURSOR[1]] + {'x': 'o', 'o': 'u' if CURSOR[0] == 7 else 'x', 'u': 'p', 'p': 'x'}[
            GRID[CURSOR[0]][CURSOR[1]]] + GRID[CURSOR[0]][CURSOR[1]+1:]
    elif c in (ord('8'), ord('9')):
        CURRENT_KIT = {ord('8'): '808', ord('9'): '909'}[c]
    elif c == ord('r'):
        GRID[CURSOR[0]] = ''.join(random.choice(
            ['x', 'o'] + (['u', 'p'] if [CURSOR[0]] == 7 else [])) for _ in range(STEPS))
    elif c == ord('0'):
        SWING = 0
    elif c == ord('5'):
        SWING = 0.5
    elif c == ord('6'):
        SWING = 0.6
    elif c == ord('x'):
        GRID[CURSOR[0]] = 'x'*STEPS
    elif c == ord('s'):
        update_sequence()
        if PLAYBACK_THREAD is None:
            PLAYBACK_THREAD = threading.Thread(target=playback_function)
            PLAYBACK_THREAD.start()
        else:
            PLAYBACK_THREAD = None
    elif c == ord('q'):
        PLAYBACK_THREAD = None
        exit()

curses.nocbreak()
stdscr.keypad(False)
curses.echo()
curses.endwin()
