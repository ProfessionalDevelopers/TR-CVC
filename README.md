# TR-CVC

A very basic drum machine and bassline synthesizer, based incredibly loosely on the Roland TR-303, TR-909 and TB-808 machines.

The goal was to build a drum machine with reatlime editing capabilities, combined with the ability to add additional sounds and sequencing modes.

Video and sound examples:

https://github.com/ProfessionalDevelopers/TR-CVC/assets/1034155/5152e76a-297c-4dca-b71a-1a5703ee3cfe

# Features

- Simple 16/32/64-step sequencer with double-up pattern expansion with velocity control
- Easily pre-fill with standard rhythms, or shift rhythms across a pattern
- Bad approxmiations of classic roland 808 and 909 sounds: Bass Drum, Snare, Low- Mid- and High-Toms, Clap, Cowbell, Hihat and Open Hihat, all done in code, no samples.
- Can also load a sample kit.
- Bassline instrument including filter and slide
- Piano stabs! 
- Cycle between kits while playing
- Configurable per-instrument mixer levels, and per-channel live mute and unmute
- Adjustable tempo and swing
- Automatic saving and restoration of sequences

# Installation and Development

Right now, the way to install it is the same as the way to develop it: get the code, install the dependencies and run it.

Note: you'll need a pretty big terminal window. Resize it before launching.

## Requirements

- Python > 3.11.0
- Pipenv (`pip install pipenv`)
- Turn your speakers on!

## Installation

1. Clone this repo: `git clone https://github.com/ProfessionalDevelopers/tr-cvc.git`
1. `cd tr-cvc`
1. `pipenv install`

## Usage

Start up with `pipenv run python main.py`. The command keys are explained on the screen.


## Linux Tips 
Follow the instructions as above. 
If you get an error like 

`OSError: PortAudio library not found`

1. this S/O link will help: https://stackoverflow.com/a/35593426
1. then you may need to `pip install librosa` depending on your setup

# TODO

- FX?
