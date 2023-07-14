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
BPM = 120.0
BPMFRAME = (60/BPM)/4
SEQUENCE_FILE = 'sequence.json'  # the file where we'll save and load the sequence
MASTER_LEVEL = 0.8  # master level
GRID = ['x'*16 for _ in range(11)]  # add extra rows for the bassline, mid tom and clap
CURSOR = [0, 0]
COMPLETE_SEQUENCE = np.zeros(16 * int(FS * BPMFRAME), dtype=np.float32)
SWING = 0
PLAYBACK_THREAD = None
CURRENT_KIT = "808"
BASSLINE_FILTER_FREQS = [110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0, 14080.0]  # Low-pass filter frequencies for bassline
BASSLINE_FILTER_INDEX = 0  # Index for the current filter frequency
BASSLINE_FILTER_DIRECTION = 1  # 1 for increasing, -1 for decreasing



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
    y = np.sin(2 * np.pi * start_freq * np.exp(np.log(end_freq/start_freq)*x/length) * x / FS)
    decay = np.exp(-decay_factor * x)
    return (y * decay).astype(np.float32)

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def generate_acid_bassline(note_frequency, note_duration, slide_duration, resonance, distortion, fs=44100):
    t = np.arange(note_duration * fs)
    bassline = signal.square(2 * np.pi * note_frequency * t / fs)  # using square wave

    if slide_duration > 0:
        slide_frequencies = np.linspace(note_frequency, note_frequency * 2, int(fs * slide_duration))
        slide = signal.square(2 * np.pi * slide_frequencies * np.arange(len(slide_frequencies)) / fs)
        bassline[:len(slide)] = slide

    if resonance > 0:
        sos = signal.butter(10, note_frequency, 'hp', fs=fs, output='sos')
        bassline = signal.sosfilt(sos, bassline)

    if distortion > 0:
        bassline = np.clip(bassline, -distortion, distortion)

    return bassline

def generate_piano_sound(freq, decay_factor, length):
    x = np.arange(length)
    y = np.sin(2 * np.pi * freq * x / FS)
    decay = np.exp(-decay_factor * x)
    return (y * decay).astype(np.float32)

KICK_808 = generate_kick_sound(65.0, 50.0, 0.0003, int(FS * 0.4))
SNARE_808 = generate_sound(180.0, 0.0015, int(FS * BPMFRAME))
SNARE_808 += generate_sound(0, 0.0015, int(FS * BPMFRAME), noise=True)
HIHAT_808 = bandpass_filter(generate_sound(0, 0.005, int(FS * 0.5), noise=True), 7000, 9000, FS)
OPEN_HIHAT_808 = bandpass_filter(generate_sound(0, 0.001, int(FS * 1.4), noise=True), 7000, 9000, FS)
COWBELL_808 = generate_sound(380.0, 0.002, int(FS * BPMFRAME))
HIGH_TOM_808 = generate_kick_sound(300.0, 150.0, 0.0005, int(FS * 0.2))
LOW_TOM_808 = generate_kick_sound(200.0, 75.0, 0.0005, int(FS * 0.2))
MID_TOM_808 = generate_kick_sound(250.0, 100.0, 0.0005, int(FS * 0.2))  # example sound
CLAP_808 = generate_sound(0, 0.001, int(FS * BPMFRAME), noise=True)  # example sound

KICK_909 = generate_kick_sound(150.0, 30.0, 0.0002, int(FS * 0.5))  # longer and boomy
SNARE_909 = generate_sound(220.0, 0.001, int(FS * BPMFRAME))
SNARE_909 += generate_sound(0, 0.001, int(FS * BPMFRAME), noise=True)
HIHAT_909 = bandpass_filter(generate_sound(0, 0.0025, int(FS * 0.5), noise=True), 7000, 9000, FS)
OPEN_HIHAT_909 = bandpass_filter(generate_sound(100, 0.0005, int(FS * 1.4), noise=True), 6000, 9000, FS)
COWBELL_909 = generate_sound(480.0, 0.001, int(FS * 0.075))
HIGH_TOM_909 = generate_kick_sound(300.0, 150.0, 0.0005, int(FS * 0.2))
LOW_TOM_909 = generate_kick_sound(200.0, 75.0, 0.0005, int(FS * 0.2))
MID_TOM_909 = generate_kick_sound(250.0, 100.0, 0.0005, int(FS * 0.2))  # example sound
CLAP_909 = generate_sound(0, 0.001, int(FS * BPMFRAME), noise=True)  # example sound

PIANO_SOUND = generate_piano_sound(440.0, 0.001, int(FS * BPMFRAME))  # A4 note

INSTRUMENTS_808 = [Instrument('BD', KICK_808, 0.8), Instrument('SD', SNARE_808, 1.0), Instrument('HH', HIHAT_808, 1.0),
                   Instrument('OH', OPEN_HIHAT_808, 1.0), Instrument('CB', COWBELL_808, 1.0), Instrument('HT', HIGH_TOM_808, 0.9),
                   Instrument('LT', LOW_TOM_808, 0.8), Instrument('MT', MID_TOM_808, 0.7), Instrument('CP', CLAP_808, 0.6), Instrument('BL', None, 0.2), Instrument('PA', PIANO_SOUND, 0.8)]

INSTRUMENTS_909 = [Instrument('BD', KICK_909, 0.8), Instrument('SD', SNARE_909, 1.0), Instrument('HH', HIHAT_909, 1.0),
                   Instrument('OH', OPEN_HIHAT_909, 1.0), Instrument('CB', COWBELL_909, 1.0), Instrument('HT', HIGH_TOM_909, 0.9),
                   Instrument('LT', LOW_TOM_909, 0.8), Instrument('MT', MID_TOM_909, 0.7), Instrument('CP', CLAP_909, 0.6), Instrument('BL', None, 0.2), Instrument('PA', PIANO_SOUND, 0.8)]

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
        BASSLINE_FILTER_INDEX = state.get('bassline_freq_index', 0)  # Add this line


instruments = INSTRUMENTS_808 if CURRENT_KIT == '808' else INSTRUMENTS_909
ORIGINAL_LEVELS = {i: inst.level for i, inst in enumerate(instruments)}
BASSLINE_FILTER_CUTOFF = BASSLINE_FILTER_FREQS[BASSLINE_FILTER_INDEX]  # Initial filter cutoff frequency

def dump_sequence():
    with open(SEQUENCE_FILE, 'w') as f:
        json.dump({
            'grid': GRID, 
            'swing': SWING, 
            'current_kit': CURRENT_KIT, 
            'bassline_freq_index': BASSLINE_FILTER_INDEX  # Add this line
        }, f)

# Define frequencies for 'o', 'u', 'p' for the bassline and piano
bassline_freqs = [55, 110, 220]
piano_freqs = [262, 330, 392]  # frequencies for C4, E4, G4 notes

def update_sequence():
    dump_sequence()
    global COMPLETE_SEQUENCE, instruments
    sequences = [np.zeros(16 * int(FS * BPMFRAME), dtype=np.float32) for _ in range(len(INSTRUMENTS_808))]
    instruments = INSTRUMENTS_808 if CURRENT_KIT == '808' else INSTRUMENTS_909

    for i in range(16):
        start_index = int(i * FS * BPMFRAME * (1 + SWING)) if SWING and i % 2 else i * int(FS * BPMFRAME)
        end_indices = [start_index + inst.sound.size if inst.sound is not None else start_index for inst in instruments]
        end_indices.append(start_index + int(FS * BPMFRAME))

        for j in range(len(instruments)):
            if GRID[j][i] != 'x' and instruments[j].sound is not None:
                sound = instruments[j].sound * instruments[j].level
                sequences[j][start_index:min(end_indices[j], sequences[j].size)] += sound[:min(end_indices[j], sequences[j].size) - start_index]

        # Handle the bassline and piano lines separately
        for j in [-2, -1]:  # the last two lines are the 'BA' and 'PA' lines
            min_index = min(end_indices[j], sequences[j].size)
            if GRID[j][i] in 'oup':
                freqs = bassline_freqs if j == -2 else piano_freqs
                if j == -2:  # if it's the 'BL' line
                    sound = generate_acid_bassline(freqs['oup'.index(GRID[j][i])], BPMFRAME, 0, 90, 0) * instruments[j].level
                    sound = lowpass_filter(sound, BASSLINE_FILTER_FREQS[BASSLINE_FILTER_INDEX], FS)  # Apply low-pass filter
                else:  # if it's the 'PA' line
                    sound = generate_piano_sound(freqs['oup'.index(GRID[j][i])], 0.001, int(FS * BPMFRAME)) * instruments[j].level
                sequences[j][start_index:min_index] += sound[:min_index - start_index]
   



    COMPLETE_SEQUENCE = sum(sequences) * MASTER_LEVEL



def playback_function():
    global PLAYBACK_THREAD, instruments
    with sd.OutputStream(samplerate=FS, channels=1) as stream:
        while PLAYBACK_THREAD is not None:
            update_sequence()
            stream.write(COMPLETE_SEQUENCE)


while True:
    for i, row in enumerate(GRID):
        row_str = ' '.join(row[j:j+4] for j in range(0, len(row), 4))
        stdscr.addstr(i, 0, f'{instruments[i].label} {instruments[i].level:.2f}: {row_str}')
        
    stdscr.addstr(len(GRID)+1, 0, '\n')  # Add a blank line between the sequencer and the status
    stdscr.addstr(len(GRID)+2, 0, f'(8/9): Selected Kit: {CURRENT_KIT}\n(5/6/0): Swing: {SWING * 100:.0f}%\n(s): Status: {"Playing" if PLAYBACK_THREAD else "Stopped"}\n(f): Bassline Filter Cutoff: {BASSLINE_FILTER_CUTOFF}\n(m): Mute/Unmute Track\nMaster level: {MASTER_LEVEL}')
    
    stdscr.move(CURSOR[0], CURSOR[1] // 4 * 5 + CURSOR[1] % 4 + len(instruments[CURSOR[0]].label) + 7)
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
    elif c == ord(' '):
        if CURSOR[0] in [len(instruments) - 2, len(instruments) - 1]:  # if cursor is at the 'BL' or 'PA' line
            GRID[CURSOR[0]] = GRID[CURSOR[0]][:CURSOR[1]] + {'x': 'o', 'o': 'u', 'u': 'p', 'p': 'x'}[GRID[CURSOR[0]][CURSOR[1]]] + GRID[CURSOR[0]][CURSOR[1]+1:]
        else:
            GRID[CURSOR[0]] = GRID[CURSOR[0]][:CURSOR[1]] + {'x': 'o', 'o': 'x'}[GRID[CURSOR[0]][CURSOR[1]]] + GRID[CURSOR[0]][CURSOR[1]+1:]
    elif c in (ord('8'), ord('9')):
        CURRENT_KIT = {ord('8'): '808', ord('9'): '909'}[c]
    elif c in (ord('8'), ord('9')):
        CURRENT_KIT = {ord('8'): '808', ord('9'): '909'}[c]
        instruments = INSTRUMENTS_808 if CURRENT_KIT == '808' else INSTRUMENTS_909  # Add this line
    elif c == ord('0'):
        SWING = 0
    elif c == ord('5'):
        SWING = 0.5
    elif c == ord('6'):
        SWING = 0.6
    elif c == ord('f'):
        BASSLINE_FILTER_INDEX += BASSLINE_FILTER_DIRECTION
        if BASSLINE_FILTER_INDEX == len(BASSLINE_FILTER_FREQS) - 1 or BASSLINE_FILTER_INDEX == 0:
            BASSLINE_FILTER_DIRECTION *= -1
        BASSLINE_FILTER_CUTOFF = BASSLINE_FILTER_FREQS[BASSLINE_FILTER_INDEX]
    elif c == ord('x'):
        GRID = ['x'*16 for _ in range(len(instruments))]
    elif c == ord('m'):  # Mute/unmute the current track
        if instruments[CURSOR[0]].level == 0.0:
            instruments[CURSOR[0]].level = ORIGINAL_LEVELS[CURSOR[0]]  # Unmute the track
        else:
            ORIGINAL_LEVELS[CURSOR[0]] = instruments[CURSOR[0]].level  # Save the current level
            instruments[CURSOR[0]].level = 0.0  # Mute the track
    elif c == ord('s'):
        update_sequence()
        if PLAYBACK_THREAD is None:
            PLAYBACK_THREAD = threading.Thread(target=playback_function)
            PLAYBACK_THREAD.start()
        else:
            PLAYBACK_THREAD = None

curses.nocbreak()
stdscr.keypad(False)
curses.echo()
curses.endwin()