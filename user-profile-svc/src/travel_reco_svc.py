import os
import pickle
import ssl
from flask_cors import CORS

import nltk
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('popular')

tokenizer = nltk.RegexpTokenizer(r'\w+')
lemma = nltk.wordnet.WordNetLemmatizer()

with open('stopw.pickle', 'rb') as f:
    stopw = pickle.load(f)

with open('classifier.pickle', 'rb') as f:
    clf = pickle.load(f)

with open('word_features.pickle', 'rb') as f:
    word_features = pickle.load(f)

with open('recommender_model.pickle', 'rb') as f:
    knn_model = pickle.load(f)

with open('dataset.pickle', 'rb') as f:
    final_df = pickle.load(f)

occupations_df = pd.read_csv('dataset/u.occupation')
types_df = pd.read_csv('dataset/u.type')
cities_df = pd.read_csv('dataset/packages_cities.csv')
items = pd.read_csv('dataset/u.item')


app = Flask(__name__)
CORS(app)
port = int(os.getenv("PORT"))


def tokenize(post):
    tokenized_df = tokenizer.tokenize(post)
    word_bag = [lemma.lemmatize(token.lower()) for token in tokenized_df if token not in stopw and not token.isdigit()]
    return word_bag


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


def personality_profile(blog_text):
    tokenize_words = tokenize(blog_text)
    features = find_features(tokenize_words)
    mbti_type = clf.classify(features)
    return mbti_type


def evaluate(age, package_id, Sporty, Adventurous, Artistic, Family, mbti_type_id, occupation_id, gender_id,
             total_rating_count):
    input = [[age, package_id, Sporty, Adventurous, Artistic, Family, mbti_type_id, occupation_id, gender_id,
              total_rating_count]]
    distance, indices = knn_model.kneighbors(np.asarray(input), n_neighbors=50)

    travel_package_list = set()
    for i, d in zip(indices.flatten(), distance.flatten()):
        data = final_df.iloc[i]
        df = items[items['ID'] == data['package_id']]
        package_city = cities_df[cities_df['ID'] == df['ID'].values[0]]['Package City'].values[0]
        if package_city not in travel_package_list:
            print(package_city, d)
            travel_package_list.add(package_city)
    return list(travel_package_list)


def personality_traits(mbti_type, family):
    Sporty = 0
    Adventurous = 0
    Artistic = 0
    Family = 0

    if mbti_type == 'infp' or mbti_type == 'enfp':
        Sporty = 1
    elif mbti_type == 'infj' or mbti_type == 'enfj':
        Artistic = 1
    if mbti_type.startswith('e'):
        Adventurous = 1
    if family == 'Married':
        Family = 1

    return Sporty, Adventurous, Artistic, Family


@app.route('/predict', methods=['POST'])
def index():
    content = request.json
    blog_text = ' '.join(content['blog'])
    mbti_type_text = personality_profile(blog_text)
    age = content['age']
    occupation_id = occupations_df[occupations_df['type'] == content['occupation']]['occupation_id'].values[0]
    gender_id = 1 if content['gender'] == 'male' else 2
    family = content['family']
    mbti_type_id = types_df[types_df['type'] == mbti_type_text]['mbti_type_id'].values[0]

    Sporty, Adventurous, Artistic, Family = personality_traits(mbti_type_text, family)

    ref_df = final_df[(final_df['Sporty'] == Sporty) & (final_df['Adventurous'] == Adventurous)
                      & (final_df['Artistic'] == Artistic) & (final_df['Family'] == Family)]

    total_rating_count = ref_df['total_rating_count'].mean()
    if len(ref_df[ref_df['total_rating_count'] == ref_df['total_rating_count'].max()]['package_id'].values) == 0:
        package_id = 0
    else:
        package_id = ref_df[ref_df['total_rating_count'] == ref_df['total_rating_count'].max()]['package_id'].values[0]

    return jsonify(
        mbti_text=mbti_type_text,
        packages=evaluate(age, package_id, Sporty, Adventurous, Artistic, Family, mbti_type_id, occupation_id,
                          gender_id,
                          total_rating_count))




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
