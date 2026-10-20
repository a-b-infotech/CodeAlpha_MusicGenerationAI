# CodeAlpha_MusicGenerationAI
### Task 3: Music Generation with Deep Learning (LSTM) — CodeAlpha Artificial Intelligence Internship

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![music21](https://img.shields.io/badge/Music-music21-purple.svg)](https://web.mit.edu/music21/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning music generation system that learns temporal structures, melodies, and chords from MIDI music files using a **Long Short-Term Memory (LSTM)** recurrent neural network and generates original, polyphonic musical compositions.

---

## 📌 Features

- **MIDI Data Ingestion**: Parses standard `.mid` and `.midi` files into monophonic note pitches and polyphonic chord representations using MIT's `music21` library.
- **Auto-Bootstrapping Dataset**: Includes a built-in synthetic multi-motif MIDI generator so the system trains and runs immediately without requiring manual dataset downloads.
- **Multi-Layer LSTM Neural Network**:
  - Embedding Layer: Maps discrete note tokens to continuous dense vector representations.
  - 2-Layer Stacked LSTM with Dropout: Captures long-range musical context and prevents overfitting.
  - Linear Output Projection: Predicts the next pitch distribution across the learned vocabulary.
- **Temperature-Controlled Sampling**: Tunes generation creativity (lower temperature for predictable harmony, higher temperature for experimental variations).
- **MIDI Serialization**: Converts generated note and chord sequences into standard MIDI streams (`.mid`) playable with any standard media player (VLC, Windows Media Player, GarageBand, etc.).

---

## 🛠️ Tech Stack

- **Deep Learning Framework**: [PyTorch](https://pytorch.org/)
- **Music & Audio Processing**: [music21](https://web.mit.edu/music21/)
- **Numerical Computation**: [NumPy](https://numpy.org/)
- **Language**: Python 3.9+

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/a-b-infotech/CodeAlpha_MusicGenerationAI.git
cd CodeAlpha_MusicGenerationAI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train and Generate Music
Run the automated end-to-end pipeline:
```bash
python train_and_generate.py --epochs 25 --notes 60 --temp 0.85 --output generated_music.mid
```

The script will:
1. Parse/create MIDI motif data.
2. Train the PyTorch LSTM network.
3. Save the trained model checkpoint to `music_lstm_model.pth`.
4. Generate 60 new musical notes and export them to `generated_music.mid`.

### 4. Custom MIDI Training Data
To train on your own favorite genre (classical, jazz, lo-fi, etc.):
1. Create or place your `.mid` files inside the `midi_dataset/` folder.
2. Re-run `python train_and_generate.py --midi_dir midi_dataset`.

---

## 📖 Project Structure
```
CodeAlpha_MusicGenerationAI/
│
├── model.py                 # PyTorch MusicLSTM neural network architecture
├── train_and_generate.py    # Training loop, inference, and MIDI synthesis
├── requirements.txt         # Project dependencies
└── README.md                # Documentation & instructions
```

---

## 👨‍💻 Author
- **Name**: Abhi B Sheth
- **Internship**: CodeAlpha Artificial Intelligence Internship
- **GitHub**: [@a-b-infotech](https://github.com/a-b-infotech)
