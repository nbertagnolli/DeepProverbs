"""
This script allows for command line running of a simple character level RNN model.  The architecture of the model is
modified from that used in this paper https://arxiv.org/pdf/1308.0850.pdf.  This model differs in that it only has four
hidden layers instead of seven.  Here we allow for a configurable amount of context, hidden layer breadth, and dropout.

You can run an example model with 650 hidden units in each layer, 40% dropout and 100 characters of context by typing:

python3 train_model.py --file_path /path/to/file/filename --model_name my_model --context_size 100 --n_hidden 650 \
--dropout .4  --epochs 40 --batch_size 128
"""

import argparse
import csv

import numpy
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from model_functions import define_model


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

    # Create model spec file this file holds all values used for this run
    with open('{path}{file_name}_spec.csv'.format(path=args.file_path, file_name=args.model_name), 'w') as csv_file:
        field_names = ['argument', 'value']
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for arg, val in vars(args).items():
            writer.writerow({'argument': arg, 'value': val})

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
    X = X / float(n_vocab)

    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    model = define_model(y.shape[1], seq_length, args.n_hidden, args.dropout)

    # Define the check points
    model_to_save = args.model_name + "-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(model_to_save, monitor='loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # Fit the model
    model.fit([X], [y], epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks_list, verbose=2)
