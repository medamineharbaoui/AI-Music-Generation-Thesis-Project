"""
This module generates notes for a MIDI file using a Transformer model trained with variable note lengths.
"""

# Import required libraries
import pickle  # For loading and saving data in binary format
import numpy as np  # Library for numerical calculations and multi-dimensional arrays
from music21 import instrument, note, stream, chord  # Library for working with musical data
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Flatten
from keras.layers import MultiHeadAttention, LayerNormalization, Add
from keras.optimizers import Adam

def generate():
    """ Generates a piano MIDI file using the trained Transformer model """
    # Load the notes used to train the model
    with open('thesis_models/Transformer/using_Transposed_Full51_dataset/Notes/full51_notes.pkl', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names and durations
    note_names = sorted(set(notes))
    vocab_size = len(note_names)  # Vocabulary size

    # Prepare input sequences for the model
    network_input, int_to_note = prepare_sequences(notes, note_names)

    # Create the Transformer model structure and load trained weights
    model = create_transformer_model(network_input, vocab_size)

    # Replace 'pesos_transformer-XX-XXXX.keras' with your actual weights filename
    model.load_weights('thesis_models/Transformer/using_Transposed_Full51_dataset/Weights/weights_transformer-full_51-epoch98-loss0.0646.keras')

    # Generate notes using the trained model
    prediction_output = generate_notes(model, network_input, int_to_note, vocab_size)

    # Create a MIDI file from the generated notes
    create_midi(prediction_output)

def prepare_sequences(notes, note_names):
    """ Prepares the sequences used by the Transformer model """
    # Map notes to integers and vice versa
    note_to_int = dict((note, number) for number, note in enumerate(note_names))
    int_to_note = dict((number, note) for number, note in enumerate(note_names))

    sequence_length = 25  # Define input sequence length
    network_input = []  # List to store input sequences

    # Create input sequences from the notes
    for i in range(len(notes) - sequence_length):
        input_sequence = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in input_sequence])

    network_input = np.array(network_input)

    return network_input, int_to_note  # Return input sequences and conversion dictionary

def create_transformer_model(network_input, vocab_size):
    """ Creates the Transformer model structure for prediction """
    import tensorflow as tf  # Import TensorFlow

    d_model = 256
    num_heads = 8
    dff = 512

    # Get input_length from network_input
    input_length = network_input.shape[1]
    input_length = int(input_length)  # Ensure it's an integer

    inputs = Input(shape=(input_length,))

    # Note embedding layer
    embedding = Embedding(input_dim=vocab_size, output_dim=d_model)(inputs)

    # Positional Encoding embedding layer
    positions = tf.range(start=0, limit=input_length, delta=1)
    positions = positions[tf.newaxis, :]  # Shape: (1, input_length)
    position_embedding_layer = Embedding(input_dim=input_length, output_dim=d_model)
    position_embeddings = position_embedding_layer(positions)

    # Add embedding and positional encoding
    x = embedding + position_embeddings  # Broadcasting over batch dimension

    # Transformer block
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = Dropout(0.1)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn_output = Dense(dff, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(0.1)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    # Output Layer
    out_flat = Flatten()(out2)
    outputs = Dense(vocab_size, activation='softmax')(out_flat)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy')

    return model

def generate_notes(model, network_input, int_to_note, vocab_size):
    """ Generates notes from the Transformer model based on a note sequence """
    # Use first input sequence as starting point
    pattern = network_input[0].tolist()  # Convert to list

    prediction_output = []  # List to store generated notes

    # Generate 500 notes
    for note_index in range(500):
        # Prepare model input
        prediction_input = np.array([pattern])

        # Predict next note
        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)  # Get index with highest probability
        result = int_to_note[index]  # Convert index to note
        prediction_output.append(result)  # Add note to output

        # Update pattern for next prediction
        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output  # Return generated notes

def create_midi(prediction_output):
    """ Converts prediction output to notes and creates a MIDI file """
    offset = 0  # Offset for each note
    output_notes = []  # List to store notes and chords

    # Create note and chord objects based on generated values
    for pattern in prediction_output:
        if '_' in pattern:
            # Split pattern into pitch/chord and duration class
            pitch_duration, duration_class = pattern.split('_')

            # Determine actual duration value based on duration class
            if duration_class == 'short':
                duration = 0.5  # Representative value for 'short'
            elif duration_class == 'medium':
                duration = 1.0  # Representative value for 'medium'
            else:  # 'long'
                duration = 1.5  # Representative value for 'long'

            if '.' in pitch_duration or pitch_duration.isdigit():
                # Pattern is a chord
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
                # Pattern is a note
                new_note = note.Note(pitch_duration)
                new_note.offset = offset
                new_note.duration.quarterLength = duration
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
        else:
            # Handle cases where duration is missing (optional)
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
            duration = 0.5  # Default duration

        # Increment offset by duration to prevent note overlap
        offset += duration

    # Create music stream with generated notes
    midi_stream = stream.Stream(output_notes)

    # Write stream to MIDI file
    midi_stream.write('midi', fp='thesis_generated_music/Transfomer/Transformer_full51_generated_music.mid')
    print("MIDI file successfully generated")

if __name__ == '__main__':
    generate()  # Call main function to generate MIDI file