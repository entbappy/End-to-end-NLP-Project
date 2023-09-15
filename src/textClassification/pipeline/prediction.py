import re
import os
import string
import nltk
from nltk.corpus import stopwords
import pickle
from tensorflow import keras
import keras
from keras.utils import pad_sequences
from pathlib import Path


class PredictionPipeline:
    def __init__(self):
        self.stemmer = nltk.SnowballStemmer("english")
        self.stopword = set(stopwords.words('english'))

    def clean_text(self,text):
        print(text)
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        print(text)
        text = [word for word in text.split(' ') if word not in self.stopword]
        text=" ".join(text)
        text = [self.stemmer.stem(word) for word in text.split(' ')]
        text=" ".join(text)
        return text


    def predict(self,test):
        test=[self.clean_text(test)]
        print(test)

        load_model=keras.models.load_model(Path('artifacts/model_trainer/model.h5'))

        with open(Path("artifacts/model_trainer/tokenizer.pickle"), 'rb') as handle:
            load_tokenizer = pickle.load(handle)

        seq = load_tokenizer.texts_to_sequences(test)
        padded = pad_sequences(seq, maxlen=300)
        print(seq)

        pred = load_model.predict(padded)

        print("pred", pred)
        if pred<0.5:
            print("no hate")
            return "no hate"
        else:
            print("hate and abusive")
            return "hate and abusive"