""" This module prepares data from MIDI files and feeds it to the neural network for training with variable note lengths """

# Import necessary libraries for processing and training
import glob  # To find all files matching a specific pattern
import pickle  # To serialize and deserialize Python objects
import numpy  # Library for numerical calculations and multidimensional arrays
from music21 import converter, instrument, note, chord  # Library for working with musical data
from keras.models import Sequential  # To create sequential models in Keras
from keras.layers import Dense, Dropout, LSTM, Activation  # Neural network layers
from keras.layers import BatchNormalization as BatchNorm  # For batch normalization
from keras.utils import to_categorical  # To convert labels to categorical format
from keras.callbacks import ModelCheckpoint  # To save the model during training


def train_network():
    """ Trains a Neural Network to generate music """
    notes = get_notes()  # Calls the function to obtain all notes and chords from MIDI files

    # Calculate the number of unique pitch and duration combinations
    n_vocab = len(set(notes))  # The size of the vocabulary

    # Prepare input and output sequences for the neural network
    network_input, network_output = prepare_sequences(notes, n_vocab)

    # Create the neural network structure
    model = create_network(network_input, n_vocab)

    # Train the model with the prepared sequences
    train_model(model, network_input, network_output)


def get_notes():
    """ Retrieves all notes and chords with their durations from MIDI files in the ./midi_songs directory """
    notes = []  # List to store notes and chords

    # Iterate through all MIDI files in the specified directory
    for file in glob.glob("input/midi_songs/*.mid"):
        midi = converter.parse(file)  # Convert the MIDI file into a music object

        print(f"Parsing {file}")  # Print the name of the file being parsed

        notes_to_parse = None  # Initialize the variable to store notes to parse

        try:
            # Attempt to separate parts by instrument
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()  # Take notes from the first part
        except:
            # If there are no parts, take all notes flat
            notes_to_parse = midi.flat.notes

        # Iterate through each element in the notes to parse
        for element in notes_to_parse:
            duration = element.duration.quarterLength  # Get the duration of the element in quarter notes

            # Classify the duration as 'short', 'medium', or 'long'
            if duration < 0.75:
                duration_class = 'short'
            elif duration < 1.5:
                duration_class = 'medium'
            else:
                duration_class = 'long'

            if isinstance(element, note.Note):
                # If the element is a note, combine the pitch and duration class
                note_str = f"{str(element.pitch)}_{duration_class}"
                notes.append(note_str)  # Add the note to the list
            elif isinstance(element, chord.Chord):
                # If the element is a chord, combine the pitches and duration class
                chord_str = f"{'.'.join(str(n) for n in element.normalOrder)}_{duration_class}"
                notes.append(chord_str)  # Add the chord to the list

    # Save the list of notes to a file using pickle
    with open('lstm_outputs/model_notes/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes  # Return the list of notes


def prepare_sequences(notes, n_vocab):
    """ Prepares the sequences used by the Neural Network """
    sequence_length = 10  # Define the length of input sequences

    # Get all unique pitch and duration combinations, sorted
    note_names = sorted(set(notes))

    # Create a dictionary mapping each note to an integer
    note_to_int = dict((note, number) for number, note in enumerate(note_names))

    network_input = []  # List to store input sequences
    network_output = []  # List to store corresponding output notes

    # Iterate through the notes to create pairs of input and output sequences
    for i in range(len(notes) - sequence_length):
        input_sequence = notes[i:i + sequence_length]  # Extract a sequence of notes
        output_sequence = notes[i + sequence_length]  # The next note to predict
        # Convert the notes in the input sequence to numbers using the dictionary
        network_input.append([note_to_int[char] for char in input_sequence])
        # Convert the output note to its corresponding number
        network_output.append(note_to_int[output_sequence])

    n_patterns = len(network_input)  # Total number of patterns

    # Reshape the input to be compatible with LSTM layers (n_patterns, sequence_length, 1)
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input values by dividing by the vocabulary size
    network_input = network_input / float(n_vocab)

    # Convert outputs to categorical format (one-hot encoding)
    network_output = to_categorical(network_output)

    return network_input, network_output  # Return the prepared input and output sequences


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
    # Add a batch normalization layer to speed up training
    model.add(BatchNorm())
    # Add a dropout layer to reduce overfitting
    model.add(Dropout(0.3))
    # Add a dense layer with 256 units and ReLU activation
    model.add(Dense(256, activation='relu'))
    # Another batch normalization layer
    model.add(BatchNorm())
    # Another dropout layer
    model.add(Dropout(0.3))
    # Output layer with softmax activation to predict the probability of each note in the vocabulary
    model.add(Dense(n_vocab, activation='softmax'))
    # Compile the model with categorical crossentropy loss and RMSprop optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model  # Return the created model


def train_model(model, network_input, network_output):
    """ Trains the neural network """
    # Define the file path where the model weights will be saved
    filepath = "lstm_outputs/weights/weights-improve-LSTM-{epoch:02d}-{loss:.4f}.keras"

    # Create a checkpoint to save the model each time a new best loss is achieved
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',  # Monitor training loss
        verbose=0,  # Do not display additional messages
        save_best_only=True,  # Save only if it is the best model so far
        mode='min'  # Mode to minimize loss
    )
    callbacks_list = [checkpoint]  # List of callbacks

    # Train the model with input and output data for 20 epochs and batch size of 128
    model.fit(network_input, network_output, epochs=20, batch_size=128, callbacks=callbacks_list)  # 200 epochs, 128 batch_size


if __name__ == '__main__':
    train_network()  # Call the main function to start training