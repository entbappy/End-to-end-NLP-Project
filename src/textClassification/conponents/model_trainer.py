import pandas as pd
import pickle
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import os
import json
from textClassification.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):

        df = pd.read_csv(os.path.join(self.config.data_path,"main_df.csv"))
        df.tweet=df.tweet.astype(str)

        x = df['tweet']
        y = df['label']

        # Let's split the data into train and test
        x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 42)

        print(len(x_train),len(y_train))
        print(len(x_test),len(y_test))


        max_words = self.config.max_words
        max_len = self.config.max_len

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(x_train)


        sequences = tokenizer.texts_to_sequences(x_train)
        sequences_matrix = pad_sequences(sequences,maxlen=max_len)
        
        #saving tokenizer
        with open(os.path.join(self.config.root_dir,'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


        # Creating model architecture.
        model = Sequential()
        model.add(Embedding(self.config.max_words,100,input_length=self.config.max_len))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.summary()

        model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

        # starting model training
        model.fit(sequences_matrix,y_train,batch_size=self.config.batch_size,epochs = self.config.epochs,validation_split=self.config.validation_split)

        test_sequences = tokenizer.texts_to_sequences(x_test)
        test_sequences_matrix = pad_sequences(test_sequences,maxlen=max_len)

        # Model evaluation
        accr = model.evaluate(test_sequences_matrix,y_test)

        metrics = {"eval": accr}

        with open(os.path.join(self.config.root_dir,'metrics.json'), "w") as file:
            json.dump(metrics, file)


        # Let's save the mdoel.
        model.save(os.path.join(self.config.root_dir,'model.h5'))

