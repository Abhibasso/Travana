import pickle

import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle

users = pd.read_csv('../dataset/u.user')

ratings = pd.read_csv('../dataset/u.data')

items = pd.read_csv('../dataset/u.item')

occupations = pd.read_csv('../dataset/u.occupation')

types = pd.read_csv('../dataset/u.type')

genders = pd.read_csv('../dataset/u.gender')

print('==============Users details==================')
print(users.shape)
print(users.head())

print('=============Ratings details==================')
print(ratings.shape)
print(ratings.head())

print('============Items details====================')
print(items.shape)
print(items.head())

users.plot(kind='bar', x='mbti_type', y='Age')
plt.show()
# users_y = users.groupby('mbti_type')['Age'].count()
# users_y.plot(kind='bar')
# plt.show()
# plt.show('Package Distribution')

sport_count = len(items.loc[items['Sporty'] == 1])
adv_count = len(items.loc[items['Adventurous'] == 1])
art_count = len(items.loc[items['Artistic'] == 1])
family_count = len(items.loc[items['Family'] == 1])

plt.bar(['Sporty', 'Adventurous', 'Artistic', 'Family'], [sport_count, adv_count, art_count, family_count])
plt.show('Package Distribution')

rating_count_df = ratings.groupby("package_id")["ratings"].count().reset_index() \
    .rename(columns={'ratings': 'total_rating_count'})
print(rating_count_df.head())

mean_rating = rating_count_df['total_rating_count'].mean();

print("================Training========================")
final_df = pd.merge(users, ratings, left_on='ID', right_on='user_id')
final_df.drop('ID', axis=1, inplace=True)
final_df.drop('user_id', axis=1, inplace=True)
final_df = pd.merge(final_df, items, left_on='package_id', right_on='ID')
final_df.drop('ID', axis=1, inplace=True)
final_df.drop('Package Name', axis=1, inplace=True)
final_df = pd.merge(final_df, types, left_on='mbti_type', right_on='type')
final_df.drop('mbti_type', axis=1, inplace=True)
final_df.drop('type', axis=1, inplace=True)
final_df = pd.merge(final_df, occupations, left_on='Occupation', right_on='type')
final_df.drop('Occupation', axis=1, inplace=True)
final_df.drop('type', axis=1, inplace=True)
final_df = pd.merge(final_df, genders, left_on='Gender', right_on='gender')
final_df.drop('Gender', axis=1, inplace=True)
final_df.drop('gender', axis=1, inplace=True)
final_df = pd.merge(final_df, rating_count_df, left_on='package_id', right_on='package_id')
final_df = shuffle(final_df)
final_df.drop('ratings', axis=1, inplace=True)
final_df = final_df[final_df['total_rating_count'] >= mean_rating]

print(final_df.head())

final_crs_matrix = csr_matrix(final_df.values)
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(final_crs_matrix)

f = open('recommender_model.pickle', 'wb')
pickle.dump(knn_model, f)
f.close()

f = open('dataset.pickle', 'wb')
pickle.dump(final_df, f)
f.close()
