"""
This script allows for command line running of a simple character level RNN model.  The architecture of the model is
modified from that used in this paper https://arxiv.org/pdf/1308.0850.pdf.  This model differs in that it only has four
hidden layers instead of seven.  Here we allow for a configurable amount of context, hidden layer breadth, and dropout.

You can run an example model with 650 hidden units in each layer, 40% dropout and 100 characters of context by typing:

python3 train_model.py --file_path /path/to/file/filename --model_name my_model --context_size 100 --n_hidden 650 \
--dropout .4  --epochs 40 --batch_size 128
"""

import argparse

import numpy

from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import concatenate as layers_concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


def get_slice(X):
    """Method to slice out last layer of an LSTM"""
    return X[:, -1, :]


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Trains a character level RNN on the specified text')
    parser.add_argument('--file_path', type=str, help='Path to text file')
    parser.add_argument('--model_name',  type=str, default='RNN', help='The name of the model to save')
    parser.add_argument('--context_size',  type=int, default=100, help='The size of the RNN context window')
    parser.add_argument('--n_hidden',  type=int, default=650, help='The number of hidden units in each layer')
    parser.add_argument('--dropout',  type=float, default=.4, help='The percent dropout to use')
    parser.add_argument('--epochs',  type=int, default=40, help='The number of epochs to train for')
    parser.add_argument('--batch_size',  type=int, default=128, help='The batch size to use for training')
    args = parser.parse_args()

    # Load in Raw Text
    text = open(args.file_path).read().lower()

    # extract the characters and create a map
    chars = sorted(list(set(text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # display the number of characters
    n_chars = len(text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)

    # prepare the data set of input to output pairs encoded as integers
    # Create an integer vector representing the input string
    # Create a label as the single next character in the string
    seq_length = args.context_size
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)

    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

    # normalize
    X /= float(n_vocab)

    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    # Create Graves Net https://arxiv.org/pdf/1308.0850.pdf figure 1 using Keras function api
    # Create the input layers we have a main
    main_input = Input(shape=(seq_length, 1), name='main_input')

    # Create the first LSTM layer based on the input alone
    lstm_1 = LSTM(args.n_hidden, return_sequences=True)(main_input)

    # Create a slice layer to pass output of lstm to the dense layer
    slice_1 = Lambda(get_slice)(lstm_1)
    drop_slice_1 = Dropout(args.dropout)(slice_1)

    # Joing first layer and input into second layer and pass this to a new lstm
    layer_1 = layers_concatenate([lstm_1, main_input])
    lstm_2 = LSTM(args.n_hidden, return_sequences=True)(layer_1)

    # Create a slice layer to pass output of lstm to the dense layer
    slice_2 = Lambda(get_slice)(lstm_2)
    drop_slice_2 = Dropout(args.dropout)(slice_2)

    # Join the second layer and input into a third layer
    layer_2 = layers_concatenate([lstm_2, main_input])
    lstm_3 = LSTM(args.n_hidden, return_sequences=True)(layer_2)
    drop_3 = Dropout(args.dropout)(lstm_3)
    slice_3 = Lambda(get_slice)(lstm_3)
    drop_slice_3 = Dropout(args.dropout)(slice_3)

    # Join the third layer and input into a fourth layer
    layer_3 = layers_concatenate([lstm_3, main_input])
    lstm_4 = LSTM(args.n_hidden)(layer_3)
    drop_4 = Dropout(args.dropout)(lstm_4)

    # And finally we add the main softmax layer based on all LSTM Layers
    output_layer = layers_concatenate([drop_slice_1, drop_slice_2, drop_slice_3, drop_4])
    main_output = Dense(y.shape[1], activation='softmax', name='main_output')(output_layer)

    # Define inputs and outputs of model
    model = Model(inputs=[main_input], outputs=[main_output])

    # Compile the final model
    model.compile(optimizer='adam', loss='categorical_crossentropy', clipvalues=1)

    # Define the check points
    model_to_save = args.model_name + "-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(model_to_save, monitor='loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # Fit the model
    model.fit([X], [y], epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks_list, verbose=2)
