import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# data preprocess
df = pd.read_csv("./Travel details dataset.csv").dropna(axis=0, how='any')  # 136 valid rows
df = df.reset_index(drop=True)


# dict = {}
# for key in nationality:
#     dict[key] = dict.get(key, 0) + 1
# a = sorted(dict.items(), key=lambda x: x[1], reverse=True)
# print(
#     len(a))  # since there are only 41 countries. No need to aggregate them into bigger group like European, Asian, North American

def get_group_and_scalers_and_kmeans(df):
    df_age_gender_nationality = df[['Traveler age', 'Traveler gender', 'Traveler nationality']]

    # normalize age
    column_age = df['Traveler age'].values.reshape(-1, 1)
    scaler_age = MinMaxScaler()
    column_age = scaler_age.fit_transform(column_age)
    df_age_gender_nationality = df_age_gender_nationality.drop('Traveler age', axis=1)
    df_age = pd.DataFrame(column_age, columns=['Traveler age'])
    df_age_gender_nationality = df_age_gender_nationality.join(df_age)

    # onehot gender
    column_gender = df['Traveler gender']
    scaler_gender = LabelBinarizer()
    column_gender = scaler_gender.fit_transform(column_gender)
    df_age_gender_nationality = df_age_gender_nationality.drop('Traveler gender', axis=1)
    df_gender = pd.DataFrame(column_gender, columns=['Traveler gender'] * len(column_gender[0]))
    df_age_gender_nationality = df_age_gender_nationality.join(df_gender)

    # onehot nationality
    column_nationality = df['Traveler nationality']
    scaler_nationality = LabelBinarizer()
    column_nationality = scaler_nationality.fit_transform(column_nationality)
    df_age_gender_nationality = df_age_gender_nationality.drop('Traveler nationality', axis=1)
    df_nationality = pd.DataFrame(column_nationality, columns=['Traveler nationality'] * len(column_nationality[0]))
    df_age_gender_nationality = df_age_gender_nationality.join(df_nationality)

    ndarray_age_gender_nationality = df_age_gender_nationality.to_numpy()
    # choose k using elbow method. K=60 is optimal. So there are 60 groups in total
    # inertia = []
    # for k in range(2, 100):
    #     kmeans = KMeans(init='k-means++', n_clusters=k, random_state=0, n_init=1).fit(ndarray_age_gender_nationality)
    #     inertia.append(kmeans.inertia_)
    # plt.plot(range(2, 100), inertia)
    # plt.xlabel('k')
    # plt.ylabel('inertia')
    # plt.show()

    kmeans = KMeans(init='k-means++', n_clusters=60, random_state=0, n_init=1).fit(ndarray_age_gender_nationality)
    column_group = kmeans.labels_
    df_group = pd.DataFrame(column_group, columns=['Traveler group'])
    return df_group, scaler_age, scaler_gender, scaler_nationality, kmeans


df_group, scaler_age, scaler_gender, scaler_nationality, kmeans = get_group_and_scalers_and_kmeans(df)
df_group_added = df.join(df_group)


# generate a dictionary that sores popular destinations of each group
def generate_group_dct(df_group_added):
    group_dct = {}
    for i in range(60):
        group_dct[i] = []
    for i in range(len(df_group_added)):
        current_row = df_group_added.iloc[i]
        current_group = current_row['Traveler group']
        current_destination = current_row['Destination']
        group_dct[current_group].append(current_destination)
    # print(group_dct)
    for i in range(60):
        destination_list = group_dct[i]
        dict_temp = {}
        for key in destination_list:
            dict_temp[key] = dict_temp.get(key, 0) + 1
        sorted_destinations = sorted(dict_temp.items(), key=lambda x: x[1], reverse=True)
        group_dct[i] = sorted_destinations
    return group_dct


group_dct = generate_group_dct(df_group_added)

################################################################### test
# only need age, gender, and nationality
new_customer1 = [[25, 'Male', 'Korean']]
df_new_customer1 = pd.DataFrame(new_customer1, columns=['Traveler age', 'Traveler gender', 'Traveler nationality'])

# onehot age
column_age = df_new_customer1['Traveler age'].values.reshape(-1, 1)
column_age = scaler_age.transform(column_age)
df_new_customer1 = df_new_customer1.drop('Traveler age', axis=1)
df_age = pd.DataFrame(column_age, columns=['Traveler age'])
df_new_customer1 = df_new_customer1.join(df_age)

# onehot gender
column_gender = df_new_customer1['Traveler gender']
column_gender = scaler_gender.transform(column_gender)
df_new_customer1 = df_new_customer1.drop('Traveler gender', axis=1)
df_gender = pd.DataFrame(column_gender, columns=['Traveler gender'])
df_new_customer1 = df_new_customer1.join(df_gender)

# onehot nationality
column_nationality = df_new_customer1['Traveler nationality']
column_nationality = scaler_nationality.transform(column_nationality)
df_new_customer1 = df_new_customer1.drop('Traveler nationality', axis=1)
df_nationality = pd.DataFrame(column_nationality, columns=['Traveler nationality'] * len(column_nationality[0]))
df_new_customer1 = df_new_customer1.join(df_nationality)

ndarray_new_customer1 = df_new_customer1.to_numpy()
group_of_new_customer1 = kmeans.predict(ndarray_new_customer1).item()
print(group_dct[group_of_new_customer1])
