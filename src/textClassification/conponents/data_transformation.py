import pandas as pd
import nltk
import re
import os
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from textClassification.entity.config_entity import DataTransformationConfig
nltk.download('stopwords')



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.stemmer = nltk.SnowballStemmer("english")
        self.stopword = set(stopwords.words('english'))

    
    # Let's apply regex and do cleaning.
    def data_cleaning(self,words):
        words = str(words).lower()
        words = re.sub('\[.*?\]', '', words)
        words = re.sub('https?://\S+|www\.\S+', '', words)
        words = re.sub('<.*?>+', '', words)
        words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
        words = re.sub('\n', '', words)
        words = re.sub('\w*\d\w*', '', words)
        words = [word for word in words.split(' ') if words not in self.stopword]
        words=" ".join(words)
        words = [self.stemmer.stem(words) for word in words.split(' ')]
        words=" ".join(words)

        return words
    

    def clean_and_transform(self):
        imbalance_data = pd.read_csv(os.path.join(self.config.data_path,"imbalanced_data.csv"))
        imbalance_data.drop('id', axis=1, inplace=True)

        raw_data = pd.read_csv(os.path.join(self.config.data_path,"raw_data.csv"))
        raw_data.drop(['Unnamed: 0','count','hate_speech','offensive_language','neither'], axis=1, inplace=True)
        raw_data[raw_data['class'] == 0]["class"]=1
        raw_data["class"].replace({0:1},inplace=True)
        raw_data["class"].replace({2:0}, inplace = True)
        raw_data.rename(columns={'class':'label'},inplace =True)

        frame = [imbalance_data, raw_data]
        df = pd.concat(frame)


        df['tweet']=df['tweet'].apply(self.data_cleaning)

        df.to_csv(os.path.join(self.config.root_dir,'main_df.csv'), index=False)

        