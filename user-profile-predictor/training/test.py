import pickle
import nltk

import pandas as pd
from nltk.tokenize import RegexpTokenizer
import nltk
import random
from nltk.corpus import stopwords
import pickle

tokenizer = RegexpTokenizer(r'\w+')
lemma = nltk.wordnet.WordNetLemmatizer()
stopw = stopwords.words("english")


def load_file():
    df = pd.read_csv("../dataset/mbti_1.csv")
    print("Distribution............")
    print(df.groupby('type').count())
    return df


print(load_file())
