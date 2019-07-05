# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:09:56 2019

@author: Amir.Khan
"""
from flask import Flask, render_template,request,url_for
from flask_bootstrap import Bootstrap 
import numpy as np
import time
app = Flask(__name__)
Bootstrap(app)

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import os
from model_final import *
from googletrans import Translator
translator = Translator()
value = '';
actual = '';

def final_predictions(x, y, x_tk, y_tk, s):
    """
    Gets predictions using the final model
    :param x: Preprocessed English data
    :param y: Preprocessed French data
    :param x_tk: English tokenizer
    :param y_tk: French tokenizer
    """
    # TODO: Train neural network using model_final
    v = '';
    a = '';

    print(len(x_tk.word_index))
    print(len(y_tk.word_index))

    model = model_final(x.shape,
                        y.shape[1],
                       len(x_tk.word_index) + 1,
                       len(y_tk.word_index) + 1)
    
    model.load_weights('models/machine_translation/best_combine.hdf5')
    ## DON'T EDIT ANYTHING BELOW THIS LINE
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    if s == "new jersey is sometimes quiet during autumn , and it is snowy in april.":
        sentences = np.array([x[0]])
        predictions = model.predict(sentences, len(sentences))
        value = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])
        actual = french_sentences[0]
    elif s ==  "the united states is usually chilly during july , and it is usually freezing in november.":
        sentences = np.array([x[1]])
        predictions = model.predict(sentences, len(sentences))
        value = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])
        actual = french_sentences[1]

    elif s ==  "california is usually quiet during march , and it is usually hot in june .":
        sentences = np.array([x[2]])
        predictions = model.predict(sentences, len(sentences))
        value = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])
        actual = french_sentences[2]

    elif s == "the united states is sometimes mild during june , and it is cold in september .":
        sentences = np.array([x[3]])
        predictions = model.predict(sentences, len(sentences))
        value = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])
        actual = french_sentences[3]

    elif s == "your least liked fruit is the grape , but my least liked is the apple .":
        sentences = np.array([x[4]])
        predictions = model.predict(sentences, len(sentences))
        value = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])
        actual = french_sentences[4]

    elif s == "his favorite fruit is the orange , but my favorite is the grape .":
        sentences = np.array([x[5]])
        predictions = model.predict(sentences, len(sentences))
        value = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])
        actual = french_sentences[5]

    else:
        sentence = [x_tk.word_index[word] for word in s.split()]
        sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
        sentences = np.array([sentence[0],x[0]])
        predictions = model.predict(sentences, len(sentences))
        value = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])
        translations  = translator.translate(s, dest='fr')
        actual = translations.text

    v = value
    a = actual
    return v, a

def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')

french_sentences = load_data("data/small_vocab_fr")
english_sentences = load_data("data/small_vocab_en")


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """    
    try:
        tokenizer = Tokenizer(num_words=None, char_level=False)
        tokenizer.fit_on_texts(x)
        sequences = tokenizer.texts_to_sequences(x)
        return sequences, tokenizer
    except Exception:
        #TODO: implement proper logging
        print('Something wrong, please check.')


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    try:
        padded = pad_sequences(x, maxlen=length,
                              padding='post', truncating='post')
        return padded 
    
    except Exception:
        #TODO: implement proper logging
        print('Something wrong, please check.')



def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
    preprocess(english_sentences, french_sentences)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/analyse',methods=['POST'])
def analyse():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        print(rawtext)
        cv,av = final_predictions(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer,rawtext)
        print(cv)
        rv = cv.replace('<PAD>', '')
        end = time.time()
        final_time = end-start         
    return render_template('success_machinetranslation.html', received_text = rawtext, result=rv, actualv = av,final_time=final_time)
if __name__ == '__main__':
	app.run(debug=True)