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


def tokenize(post):
    tokenized_df = tokenizer.tokenize(post)
    word_bag = [lemma.lemmatize(token.lower()) for token in tokenized_df if token not in stopw]
    return word_bag


def all_words(documents):
    all_words = []
    for (words, type) in documents:
        for word in words:
            all_words.append(word)
    return all_words


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


def feature_sets(documents):
    return [(find_features(document), category) for (document, category) in documents]


df = load_file()
categories = df['type'].unique()
print("Preprocessing..............")
df['tokenize_words'] = df.apply(lambda row: tokenize(row['posts']), axis=1)
df = df.dropna(axis=1, how='all')
documents = list(df[['tokenize_words', 'type']].itertuples(index=False, name=None))
random.shuffle(documents)
all_words = all_words(documents)
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[5:5000]
feature_sets = feature_sets(documents)

training_set = feature_sets[:500]
testing_set = feature_sets[500:]

print("Training..................")
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
print(classifier.show_most_informative_features())

f = open('classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

with open('classifier.pickle', 'rb') as f:
    clf = pickle.load(f)

f = open('word_features.pickle', 'wb')
pickle.dump(word_features, f)
f.close()

