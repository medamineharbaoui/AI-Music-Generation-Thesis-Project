""" This module generates notes for a MIDI file using the trained neural network with variable note lengths """

# Import necessary libraries
import pickle  # To load and save data in binary format
import numpy  # Library for numerical calculations and multidimensional arrays
from music21 import instrument, note, stream, chord  # Library for working with musical data
from keras.models import Sequential  # To create sequential models in Keras
from keras.layers import Dense, Dropout, LSTM, BatchNormalization as BatchNorm, Activation  # Neural network layers

def generate():
    """ Generates a piano MIDI file """
    # Load the notes used to train the model
    with open('thesis_models/LSTM/using_Transposed_Full51_dataset/Notes/LSTM_Full_51_notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch and duration names
    note_names = sorted(set(notes))
    n_vocab = len(note_names)  # Vocabulary size

    # Prepare input sequences for the neural network
    network_input, normalized_input = prepare_sequences(notes, note_names, n_vocab)

    # Create the neural network structure and load the trained weights
    model = create_network(normalized_input, n_vocab)

    # Generate notes using the trained model
    prediction_output = generate_notes(model, network_input, note_names, n_vocab)

    # Create a MIDI file from the generated notes
    create_midi(prediction_output)

def prepare_sequences(notes, note_names, n_vocab):
    """ Prepares the sequences used by the Neural Network """
    # Map notes to integers
    note_to_int = dict((note, number) for number, note in enumerate(note_names))

    sequence_length = 25  # Define the length of input sequences
    network_input = []  # List to store input sequences

    # Create input sequences from the notes
    for i in range(len(notes) - sequence_length):
        input_sequence = notes[i:i + sequence_length]  # Get a sequence of notes
        network_input.append([note_to_int[char] for char in input_sequence])  # Convert notes to integers

    n_patterns = len(network_input)  # Total number of patterns

    # Reshape the input to be compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input values by dividing by the vocabulary size
    normalized_input = normalized_input / float(n_vocab)

    return network_input, normalized_input  # Return the input sequences and normalized input

def create_network(network_input, n_vocab):
    """ Creates the neural network structure """
    model = Sequential()  # Initialize the sequential model

    # Add an LSTM layer with 512 units, with recurrent dropout and returning sequences
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    # Add another similar LSTM layer
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    # Add a third LSTM layer without returning sequences
    model.add(LSTM(512))
    # Add a batch normalization layer
    model.add(BatchNorm())
    # Add a dropout layer to reduce overfitting
    model.add(Dropout(0.3))
    # Add a dense layer with 256 units and ReLU activation
    model.add(Dense(256, activation='relu'))
    # Another batch normalization layer
    model.add(BatchNorm())
    # Another dropout layer
    model.add(Dropout(0.3))
    # Output layer with softmax activation
    model.add(Dense(n_vocab, activation='softmax'))
    # Compile the model with categorical crossentropy loss and RMSprop optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the trained weights into the model
    model.load_weights('thesis_models/LSTM/using_Transposed_Full51_dataset/Weights/weights-LSTM_Full_51-195-1.7494.keras')

    return model  # Return the created model

def generate_notes(model, network_input, note_names, n_vocab):
    """ Generates notes from the neural network based on a sequence of notes """
    # Use the first input sequence as the starting point
    pattern = network_input[0]

    # Map integers to notes
    int_to_note = dict((number, note) for number, note in enumerate(note_names))

    prediction_output = []  # List to store generated notes

    # Generate 500 notes
    for note_index in range(500):
        # Prepare the input for the model
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        # Predict the next note
        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)  # Get the index with the highest probability
        result = int_to_note[index]  # Convert the index to a note
        prediction_output.append(result)  # Add the note to the output

        # Update the pattern for the next prediction
        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output  # Return the generated notes

def create_midi(prediction_output):
    """ Converts the prediction output to notes and creates a MIDI file """
    offset = 0  # Offset for each note
    output_notes = []  # List to store notes and chords

    # Create note and chord objects based on the generated values
    for pattern in prediction_output:
        if '_' in pattern:
            # Split the pattern into pitch/chord and duration class
            pitch_duration, duration_class = pattern.split('_')

            # Determine the actual duration value based on the duration class
            if duration_class == 'short':
                duration = 0.5  # Representative value for 'short'
            elif duration_class == 'medium':
                duration = 1.0  # Representative value for 'medium'
            else:  # 'long'
                duration = 1.5  # Representative value for 'long'

            if '.' in pitch_duration or pitch_duration.isdigit():
                # The pattern is a chord
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
                # The pattern is a note
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

        # Increment the offset by the duration to avoid note overlap
        offset += duration

    # Create a music stream with the generated notes
    midi_stream = stream.Stream(output_notes)

    # Write the stream to a MIDI file
    midi_stream.write('midi', fp='thesis_generated_music/LSTM/DEMO2-LSTM_full51_generated_music.mid')
    print("MIDI file generated successfully")

if __name__ == '__main__':
    generate()  # Call the main function to generate the MIDI file