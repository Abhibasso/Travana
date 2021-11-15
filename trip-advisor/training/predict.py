import pickle

import numpy as np
import pandas as pd

age = 23
Sporty = 1
Adventurous = 1
Artistic = 1
Family = 0
mbti_type_id = 8
occupation_id = 4
gender_id = 2

package_id = 102
total_rating_count = 223

with open('recommender_model.pickle', 'rb') as f:
    knn_model = pickle.load(f)

with open('dataset.pickle', 'rb') as f:
    final_df = pickle.load(f)

input = [[age, package_id, Sporty, Adventurous, Artistic, Family, mbti_type_id, occupation_id, gender_id,
          total_rating_count]]
distance, indices = knn_model.kneighbors(np.asarray(input), n_neighbors=50)
# proba = knn_model.predict_proda(input)
# print(proba)
items = pd.read_csv('../dataset/u.item')

set = set()
for i, d in zip(indices.flatten(), distance.flatten()):
    data = final_df.iloc[i]
    df = items[items['ID'] == data['package_id']]
    package = df['Package Name'].values[0]
    if package not in set:
        print(df['Package Name'].values, d)
        set.add(package)
    data