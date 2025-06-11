"""This module generates notes for a MIDI file using a trained hybrid LSTM-Transformer model with variable note lengths."""

import glob
import pickle
import numpy as np
from music21 import instrument, note, stream, chord
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM, Embedding, MultiHeadAttention, LayerNormalization
from keras.optimizers import Adam
import tensorflow as tf
import os

def load_notes_and_vocab(notes_path="thesis_models/Hybrid/using_Transposed_Full51_dataset/Notes/full51_notes.pkl"):
    """Loads notes from notes.pkl and creates vocabulary mappings."""
    if not os.path.exists(notes_path):
        raise FileNotFoundError(f"Notes file not found at {notes_path}")
    with open(notes_path, 'rb') as filepath:
        notes = pickle.load(filepath)
    print(f"Loaded {len(notes)} notes from {notes_path}")
    
    note_names = sorted(set(notes))
    vocab_size = len(note_names)
    note_to_int = dict((note, number) for number, note in enumerate(note_names))
    int_to_note = dict((number, note) for number, note in enumerate(note_names))
    return notes, note_names, vocab_size, note_to_int, int_to_note

def prepare_sequences(notes, note_to_int, sequence_length=20, normalize=False, vocab_size=None):
    """Prepares input sequences for music generation."""
    network_input = []
    for i in range(len(notes) - sequence_length):
        input_sequence = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in input_sequence])
    if not network_input:
        raise ValueError(f"No sequences generated. Need at least {sequence_length + 1} notes, got {len(notes)}.")
    network_input = np.array(network_input)
    
    if normalize and vocab_size is not None:
        network_input = network_input / float(vocab_size)
        network_input = np.reshape(network_input, (network_input.shape[0], network_input.shape[1], 1))
    
    print(f"Generated {len(network_input)} sequences")
    return network_input

def create_hybrid_model(sequence_length, vocab_size):
    """Creates the hybrid LSTM-Transformer model structure."""
    d_model = 512  # Embedding dimension, matched to lstm_units
    num_heads = 8  # Number of attention heads
    dff = 512      # Feed-forward layer dimension
    lstm_units = 512  # LSTM units

    inputs = Input(shape=(sequence_length,))
    x = Embedding(input_dim=vocab_size, output_dim=d_model)(inputs)
    
    x = LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.3)(x)
    x = Dropout(0.3)(x)
    x = LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.3)(x)
    x = Dropout(0.3)(x)
    
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    attn_output = Dropout(0.1)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = Dense(dff, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(0.1)(ffn_output)
    x = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    
    x = Dense(256, activation='relu')(x[:, -1, :])
    x = Dropout(0.3)(x)
    outputs = Dense(vocab_size, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')
    return model

def generate_notes(model, network_input, int_to_note, sequence_length=20, generation_length=500, normalize=False, vocab_size=None, use_first_sequence=True):
    """Generates notes using the trained hybrid model."""
    # Choose starting pattern
    if use_first_sequence:
        pattern = network_input[0].copy()
    else:
        start = np.random.randint(0, len(network_input) - 1)
        pattern = network_input[start].copy()
    
    prediction_output = []
    
    for _ in range(generation_length):
        prediction_input = np.reshape(pattern, (1, sequence_length))
        if normalize and vocab_size is not None:
            prediction_input = prediction_input / float(vocab_size)
            prediction_input = np.reshape(prediction_input, (1, sequence_length, 1))
        
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = np.append(pattern[1:], index)
    
    return prediction_output

def create_midi(prediction_output, output_dir="thesis_generated_music/Hybrid", output_file="hybrid_full51_generated_music.mid"):
    """Converts generated note/chord sequence to MIDI file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    
    offset = 0
    output_notes = []
    
    for pattern in prediction_output:
        try:
            if '_' in pattern:
                pitch_duration, duration_class = pattern.rsplit('_', 1)
                duration = 0.5 if duration_class == 'short' else 1.0 if duration_class == 'medium' else 1.5
                
                if '.' in pitch_duration:
                    notes_in_chord = pitch_duration.split('.')
                    notes = []
                    for current_note in notes_in_chord:
                        new_note = note.Note(int(current_note))
                        new_note.duration.quarterLength = duration
                        new_note.storedInstrument = instrument.Piano()
                        notes.append(new_note)
                    new_chord = chord.Chord(notes)
                    new_chord.offset = offset
                    new_chord.duration.quarterLength = duration
                    output_notes.append(new_chord)
                else:
                    new_note = note.Note(pitch_duration)
                    new_note.offset = offset
                    new_note.duration.quarterLength = duration
                    new_note.storedInstrument = instrument.Piano()
                    output_notes.append(new_note)
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.duration.quarterLength = 0.5  # Default duration
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
                duration = 0.5
            
            offset += duration
        except Exception as e:
            print(f"Error processing {pattern}: {e}")
    
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_path)
    print(f"Saved MIDI file to {output_path}")

def generate():
    """Generates a piano MIDI file using the trained hybrid LSTM-Transformer model."""
    try:
        # Load notes and vocabulary
        notes, note_names, vocab_size, note_to_int, int_to_note = load_notes_and_vocab()
        
        # Prepare sequences
        sequence_length = 25  # Set to 25 if model was trained with that length
        normalize = False  # Set to True to normalize inputs like LSTM model
        network_input = prepare_sequences(notes, note_to_int, sequence_length, normalize, vocab_size)
        
        # Create model
        model = create_hybrid_model(sequence_length, vocab_size)
        
        # Load latest weights
        weights_path = "thesis_models/Hybrid/using_Transposed_Full51_dataset/Weights/"
        weight_files = glob.glob(weights_path + "weights_hybrid-epoch130-loss1.7931.keras")
        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {weights_path}")
        latest_weights = max(weight_files, key=os.path.getctime)
        model.load_weights(latest_weights)
        print(f"Loaded weights from {latest_weights}")
        
        # Generate notes
        generated = generate_notes(
            model,
            network_input,
            int_to_note,
            sequence_length,
            generation_length=500,
            normalize=normalize,
            vocab_size=vocab_size,
            use_first_sequence=True  # Matches LSTM/Transformer scripts
        )
        print("Generated music sequence (first 10):", generated[:10])
        
        # Save to MIDI
        create_midi(generated)
    except Exception as e:
        print(f"Generation failed: {e}")

if __name__ == '__main__':
    generate()