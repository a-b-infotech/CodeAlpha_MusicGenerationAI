import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from music21 import converter, instrument, note, chord, stream
from model import MusicLSTM

def create_sample_midi_data(midi_folder: str):
    """Generates melodic MIDI files if the dataset directory is empty."""
    os.makedirs(midi_folder, exist_ok=True)
    if any(f.endswith((".mid", ".midi")) for f in os.listdir(midi_folder)):
        return

    print(f"Creating sample classical MIDI motifs in '{midi_folder}'...")
    scales = [
        ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
        ["A3", "C4", "E4", "A4", "B4", "C5", "E5", "A5"],
        ["G3", "B3", "D4", "G4", "A4", "B4", "D5", "G5"],
        ["F3", "A3", "C4", "F4", "G4", "A4", "C5", "F5"]
    ]
    
    for i, scale in enumerate(scales):
        s = stream.Stream()
        s.append(instrument.Piano())
        for _ in range(8):
            for pitch in scale:
                n = note.Note(pitch)
                n.quarterLength = 0.5
                s.append(n)
            # Add harmonic chord
            ch = chord.Chord([scale[0], scale[2], scale[4]])
            ch.quarterLength = 1.0
            s.append(ch)
        midi_path = os.path.join(midi_folder, f"sample_motif_{i+1}.mid")
        s.write('midi', fp=midi_path)
    print(f"Sample MIDI motifs created successfully.")

def parse_midi_files(midi_folder: str):
    """Parses MIDI files into pitch and chord token sequences using music21."""
    notes = []
    midi_files = [f for f in os.listdir(midi_folder) if f.endswith(('.mid', '.midi'))]
    print(f"Parsing {len(midi_files)} MIDI files from '{midi_folder}'...")

    for file in midi_files:
        path = os.path.join(midi_folder, file)
        try:
            midi = converter.parse(path)
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Skipping {file} due to read error: {e}")

    if not notes:
        # Fallback progression
        notes = ["C4", "E4", "G4", "B4", "C5", "A4", "F4", "G4"] * 20
    return notes

def prepare_sequences(notes, seq_length=16):
    """Maps notes to integer tokens and constructs rolling window sequences."""
    pitchnames = sorted(list(set(notes)))
    vocab_size = len(pitchnames)
    note_to_int = {n: i for i, n in enumerate(pitchnames)}
    int_to_note = {i: n for i, n in enumerate(pitchnames)}

    inputs, targets = [], []
    for i in range(len(notes) - seq_length):
        inputs.append([note_to_int[n] for n in notes[i:i + seq_length]])
        targets.append(note_to_int[notes[i + seq_length]])

    X = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    return X, y, pitchnames, note_to_int, int_to_note

def train_model(model, X, y, epochs=25, lr=0.005, device="cpu"):
    """Trains the LSTM music generation model."""
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training for {epochs} epochs on {device}...")
    dataset_size = len(X)
    batch_size = min(32, dataset_size)

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(dataset_size)
        total_loss = 0.0
        batches = 0

        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X[indices].to(device), y[indices].to(device)

            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches if batches > 0 else 0
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f}")

    return model

def sample_with_temperature(logits, temperature=1.0):
    """Samples next token from logits with temperature control for creativity."""
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1).item()
    probs = torch.softmax(logits / temperature, dim=-1).cpu().detach().numpy().flatten()
    probs = np.nan_to_num(probs)
    if probs.sum() == 0:
        return int(np.random.choice(len(probs)))
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

def generate_notes(model, seed_seq, int_to_note, num_notes=50, temperature=0.9, device="cpu"):
    """Generates a novel sequence of musical notes using the trained LSTM."""
    model.eval()
    pattern = list(seed_seq)
    generated = []

    with torch.no_grad():
        for _ in range(num_notes):
            input_tensor = torch.tensor([pattern], dtype=torch.long).to(device)
            logits, _ = model(input_tensor)
            pred_idx = sample_with_temperature(logits[0], temperature=temperature)
            generated.append(int_to_note[pred_idx])
            pattern = pattern[1:] + [pred_idx]

    return generated

def save_midi(generated_notes, output_file="output_generated_music.mid"):
    """Converts pitch and chord sequence to MIDI format and saves to disk."""
    output_stream = []
    offset = 0.0

    for item in generated_notes:
        if ('.' in item) or item.isdigit():
            # Chord representation
            notes_in_chord = item.split('.')
            chord_notes = [note.Note(int(n)) for n in notes_in_chord]
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_stream.append(new_chord)
        else:
            # Single note
            new_note = note.Note(item)
            new_note.offset = offset
            output_stream.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_stream)
    midi_stream.write('midi', fp=output_file)
    print(f"Successfully exported generated music to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="CodeAlpha AI Music Generation with LSTM")
    parser.add_argument("--midi_dir", type=str, default="midi_dataset", help="Directory containing MIDI training files")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--notes", type=int, default=60, help="Number of notes to generate")
    parser.add_argument("--temp", type=float, default=0.85, help="Sampling temperature (creativity)")
    parser.add_argument("--output", type=str, default="generated_music.mid", help="Output MIDI file path")
    args = parser.parse_args()

    # Step 1: Ensure dataset exists
    create_sample_midi_data(args.midi_dir)

    # Step 2: Parse MIDI data
    notes = parse_midi_files(args.midi_dir)
    print(f"Extracted {len(notes)} total musical elements.")

    # Step 3: Build sequences & vocabulary
    X, y, pitchnames, note_to_int, int_to_note = prepare_sequences(notes, seq_length=16)
    vocab_size = len(pitchnames)
    print(f"Vocabulary size: {vocab_size} unique notes/chords.")

    # Step 4: Model setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicLSTM(vocab_size=vocab_size, embed_dim=128, hidden_dim=256, num_layers=2)

    # Step 5: Training
    train_model(model, X, y, epochs=args.epochs, lr=0.005, device=device)

    # Save model checkpoint
    torch.save(model.state_dict(), "music_lstm_model.pth")
    print("Saved model checkpoint to 'music_lstm_model.pth'.")

    # Step 6: Generation
    seed_idx = random.randint(0, len(X) - 1)
    seed_sequence = X[seed_idx].tolist()
    print("Generating new music sequence from seed...")
    generated_notes = generate_notes(model, seed_sequence, int_to_note, num_notes=args.notes, temperature=args.temp, device=device)

    # Step 7: Export to MIDI
    save_midi(generated_notes, output_file=args.output)
    print("Task 3 execution completed successfully!")

if __name__ == "__main__":
    main()
