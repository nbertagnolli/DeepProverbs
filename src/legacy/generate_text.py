from __future__ import print_function
"""
This script will generate text from a trained model based on an initial seed.  The seed needs to be of the same size
as that used to train the model.  It will then generate text character by character.

To run this script to generate 100 characters based on a random seed from the original havamal document type:

python generate_text.py --model_path /path/to/model/file --model_spec /path/to/model/spec --n_chars 100
"""

import argparse
import csv
import sys

import numpy as np
from keras.utils import np_utils

from model_functions import get_slice, define_model


def sample(preds, temperature=1.0):
    # type: (np.ndarray, float) -> np.ndarray
    """

    :param preds:
    :param temperature:
    :return:
    """
    preds = np.log(preds) / temperature
    dist = np.exp(preds) / np.sum(np.exp(preds))
    # Floating point stability errors for multinomial see https://github.com/numpy/numpy/issues/8317
    # return np.argmax(np.random.multinomial(1, preds, 1))
    choices = range(len(preds))
    return np.random.choice(choices, p=dist)


def generate_text(model, data, char_to_int, int_to_char, n_vocab, num_chars=1000, seed=None, temperature=1.0):
    # type: (keras.model.Model, np.ndarray, Dict[str, int], Dict[int, str], int, int, str, float) -> None
    """

    :param model:
    :param data:
    :param char_to_int:
    :param int_to_char:
    :param n_vocab:
    :param num_chars:
    :param seed:
    :param temperature:
    :return:
    """
    generated_text = ''
    if not seed:
        start = np.random.randint(0, len(data)-1)
        pattern = data[start]
    else:
        pattern = [char_to_int[value] for value in seed]
    print("Seed:")
    generated_text = ''.join([int_to_char[value] for value in pattern])
    print("\"", generated_text, "\"")
    # generate characters
    for i in range(num_chars):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = sample(prediction[0])
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        # sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        generated_text += result
    print("\nDone.")
    return generated_text


def main(model_path, spec_dict, num_chars_to_generate):
    # type: (str, Dict[str, Any], int) -> None
    """

    :param model_path:
    :param spec_dict:
    :param num_chars_to_generate:
    :return:
    """
    # Load in Raw Text
    text = open(spec_dict['file_path']).read().lower()

    # extract the characters and create a map
    chars = sorted(list(set(text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # display the number of characters
    n_chars = len(text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)

    # prepare the dataset of input to output pairs encoded as integers
    # Create an integer vector representing the input string of 100 characters for
    # every 100 character window.  Create a label as the single next character in the string
    seq_length = int(spec_dict['context_size'])
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
    X = np.reshape(dataX, (n_patterns, seq_length, 1))

    # normalize
    X = X / float(n_vocab)

    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    # Create the compiled Keras model with spec parameters
    model = define_model(y.shape[1], seq_length, int(spec_dict['n_hidden']), float(spec_dict['dropout']))

    # Load earlier weights
    model.load_weights(model_path)

    # Generate the text
    generated_text = generate_text(model, dataX, char_to_int, int_to_char, n_vocab, num_chars=num_chars_to_generate)

    return generated_text


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Trains a character level RNN on the specified text')
    parser.add_argument('--model_path', type=str, help='The path to model file')
    parser.add_argument('--model_spec',  type=str, help='The path to the model spec file')
    parser.add_argument('--n_chars',  type=int, default=100, help='The number of characters to generate')
    args = parser.parse_args()

    # Parse the spec file and extract model parameters
    with open(args.model_spec, mode='rU') as infile:
        reader = csv.reader(infile)
        spec_dict = {rows[0]: rows[1] for rows in reader}

    generated_text = main(args.model_path, spec_dict, args.n_chars)
    print(generated_text)
