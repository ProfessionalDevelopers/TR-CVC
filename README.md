# TR-CVC

A very basic drum machine and bassline synthesizer, based incredibly loosely on the Roland TR-303, TR-909 and TB-808 machines.

This was built entirely through interaction with OpenAI's ChatGPT, specifically using GPT-4's Code Intepreter plugin.

The goal was to build a drum machine with reatlime editing capabilities, combined with the ability to add additional sounds and sequencing modes.

Video and sound examples:

<https://github.com/ProfessionalDevelopers/TR-CVC/assets/1034155/1e1cb11a-2c46-49c6-a75b-2a59f77cc64f>

# Features

- Simple 16/32/64-step sequencer with double-up pattern expansion
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

# TODO

- FX?
