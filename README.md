# TR-CVC

A very basic drum machine and bassline synthesizer, based incredibly loosely on the Roland TR-303, TR-909 and TB-808 machines.

This was built entirely through interaction with OpenAI's ChatGPT, specifically using GPT-4's Code Intepretation module.

The goal was to build a drum machine with reatlime editing capabilities, combined with the ability to add additional sounds and sequencing modes. 

Video and sound examples:

https://github.com/ProfessionalDevelopers/TR-CVC/assets/1034155/59721b06-78c2-44bd-ae2a-881506157d4d

# Features:
- Simple 16-step sequencer: press space to toggle a step, and on bass and piano lines, repeat it to choose a different note.
- Bad approxmiations of classic roland 808 and 909 sounds: Bass Drum, Snare, Low- Mid- and High-Toms, Clap, Cowbell, Hihat and Open Hihat, all done in code, no samples.
- A simplistic bassline sequencer allowing octave jumps of a square wav
- Configurable highpass filter for the bass sequence
- A ...piano(?) sound. Another sound.
- Per-channel mute and un-mutes
- Change kits while playing
- Adjustable tempo: -/= for jumps of 5, hold shift for single beat increments 
- Configurable per-instrument mixer levels per
- Automatic saving and restoration of sequences

#To come:
- Adjustable BPM
- Swing
- FX?