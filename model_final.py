from keras.layers import GRU, Input, Dense, TimeDistributed
from keras.models import Model
from keras.layers import Activation
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras import backend as K
from keras.layers import RepeatVector
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding


def custom_sparse_categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())

def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    
    #Config Hyperparameters
    K.clear_session()
    learning_rate = 0.01
    latent_dim = 128
    
    #Config Model
    inputs = Input(shape=input_shape[1:])
    embedding_layer = Embedding(input_dim=english_vocab_size,
                                output_dim=output_sequence_length,
                                mask_zero=False)(inputs)
    bd_layer = Bidirectional(GRU(output_sequence_length))(embedding_layer)
    encoding_layer = Dense(latent_dim, activation='relu')(bd_layer)
    decoding_layer = RepeatVector(output_sequence_length)(encoding_layer)
    output_layer = Bidirectional(GRU(latent_dim, return_sequences=True))(decoding_layer)
    outputs = TimeDistributed(Dense(french_vocab_size, activation='softmax'))(output_layer)
    
    #Create Model from parameters defined above
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=[custom_sparse_categorical_accuracy])
    
    return model