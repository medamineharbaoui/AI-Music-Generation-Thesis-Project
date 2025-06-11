"""
This module prepares MIDI file data and feeds it to a Transformer model for training with variable note lengths.
"""

# Import required libraries
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Flatten
from keras.layers import MultiHeadAttention, LayerNormalization, Add
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

def train_transformer():
    """ Trains a Transformer model to generate music """
    notes = get_notes()  # Get all notes and chords from MIDI files

    vocab_size = len(set(notes))  # Vocabulary size

    network_input, network_output = prepare_sequences(notes, vocab_size)

    model = create_transformer(network_input, vocab_size)

    train_model(model, network_input, network_output)

def get_notes():
    """ Gets all notes and chords with their durations from MIDI files in the ./input/midi_songs directory """
    notes = []

    for file in glob.glob("input/midi_songs/chopin/*.mid"):
        midi = converter.parse(file)
        print(f"Parsing {file}")

        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            duration = element.duration.quarterLength

            if duration < 0.75:
                duration_class = 'short'
            elif duration < 1.5:
                duration_class = 'medium'
            else:
                duration_class = 'long'

            if isinstance(element, note.Note):
                note_str = f"{str(element.pitch)}_{duration_class}"
                notes.append(note_str)
            elif isinstance(element, chord.Chord):
                chord_str = f"{'.'.join(str(n) for n in element.normalOrder)}_{duration_class}"
                notes.append(chord_str)

    with open('transformer_outputs/model_notes/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, vocab_size):
    """ Prepares the sequences used by the Transformer """
    sequence_length = 25

    note_names = sorted(set(notes))

    note_to_int = dict((note, number) for number, note in enumerate(note_names))

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        input_sequence = notes[i:i + sequence_length]
        output_sequence = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in input_sequence])
        network_output.append(note_to_int[output_sequence])

    network_input = np.array(network_input)
    network_output = to_categorical(network_output, num_classes=vocab_size)

    return network_input, network_output

def create_transformer(network_input, vocab_size):
    """ Creates the Transformer model structure """
    import tensorflow as tf  # Import TensorFlow

    d_model = 256  # Embedding dimension
    num_heads = 8  # Number of attention heads
    dff = 512      # Feed-forward layer dimension

    # Obtain input_length from network_input
    input_length = network_input.shape[1]
    input_length = int(input_length)  # Ensure it's an integer

    # Debugging statements
    print(f"input_length: {input_length}")
    print(f"type(input_length): {type(input_length)}")

    inputs = Input(shape=(input_length,))

    # Embedding layer for notes
    embedding = Embedding(input_dim=vocab_size, output_dim=d_model)(inputs)

    # Positional Embedding Layer
    positions = tf.range(start=0, limit=input_length, delta=1)
    positions = positions[tf.newaxis, :]  # Shape: (1, input_length)
    position_embedding_layer = Embedding(input_dim=input_length, output_dim=d_model)
    position_embeddings = position_embedding_layer(positions)

    # Add positional embeddings to token embeddings
    x = embedding + position_embeddings  # Broadcasting over batch dimension

    # Rest of the Transformer block
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

def train_model(model, network_input, network_output):
    """ Trains the Transformer model """
    filepath = "transformer_outputs/weights/weights_transformer-{epoch:02d}-{loss:.4f}.keras"

    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=100, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    train_transformer()