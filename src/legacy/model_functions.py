
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import concatenate as layers_concatenate
from keras.models import Model


def get_slice(X):
    # type: (np.ndarray) -> np.ndarray
    """Method to slice out last layer of an LSTM"""
    return X[:, -1, :]


def define_model(output_size, seq_length, n_hidden, dropout):
    # type: (int, int, int, float) -> Model
    """This method defines the model architecture and compiles the model.
    The model architecuter can be found in https://arxiv.org/pdf/1308.0850.pdf figure 1
    our architecture only uses four layers with dropout in each layer's connection to the
    final dense layer.

    Args:
        output_size: The size of the output for the dense layer.  It will be the number of characters
        seq_length: The length of the context the RNN should use
        n_hidden: The number of hidden units in each layer
        dropout: The amount of dropout to use between [0, 1]

    Returns:
        A compiled Keras model.
    """
    # Create the input layers we have a main
    main_input = Input(shape=(seq_length, 1), name='main_input')

    # Create the first LSTM layer based on the input alone
    lstm_1 = LSTM(n_hidden, return_sequences=True)(main_input)

    # Create a slice layer to pass output of lstm to the dense layer
    slice_1 = Lambda(get_slice)(lstm_1)
    drop_slice_1 = Dropout(dropout)(slice_1)

    # Joing first layer and input into second layer and pass this to a new lstm
    layer_1 = layers_concatenate([lstm_1, main_input])
    lstm_2 = LSTM(n_hidden, return_sequences=True)(layer_1)

    # Create a slice layer to pass output of lstm to the dense layer
    slice_2 = Lambda(get_slice)(lstm_2)
    drop_slice_2 = Dropout(dropout)(slice_2)

    # Join the second layer and input into a third layer
    layer_2 = layers_concatenate([lstm_2, main_input])
    lstm_3 = LSTM(n_hidden, return_sequences=True)(layer_2)
    drop_3 = Dropout(dropout)(lstm_3)
    slice_3 = Lambda(get_slice)(lstm_3)
    drop_slice_3 = Dropout(dropout)(slice_3)

    # Join the third layer and input into a fourth layer
    layer_3 = layers_concatenate([lstm_3, main_input])
    lstm_4 = LSTM(n_hidden)(layer_3)
    drop_4 = Dropout(dropout)(lstm_4)

    # And finally we add the main softmax layer based on all LSTM Layers
    output_layer = layers_concatenate([drop_slice_1, drop_slice_2, drop_slice_3, drop_4])
    main_output = Dense(output_size, activation='softmax', name='main_output')(output_layer)

    # Define inputs and outputs of model
    model = Model(inputs=[main_input], outputs=[main_output])

    # Compile the final model
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model
