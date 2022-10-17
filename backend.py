from importlib.util import module_from_spec
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder 
import keras
from tensorflow.keras import layers
import tensorflow as tf
import NN_Recommender
import pickle
import os

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "Neural Network",
          "Classification with Embedding Features"
)

XGB_param = {'booster' : 'gbtree',
              'eta':0.3,
              'gamma': 0.1,
              'max_depth': 10,
              'subsample':1,
              'scale_pos_weight':0.1
}
                          
            

# ------- Functions ------

def load_ratings():
    return pd.read_csv("ratings.csv")

def load_course_sims():
    return pd.read_csv("sim.csv")

def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df

def load_bow():
    return pd.read_csv("courses_bows.csv")

def load_course_genre():
    return pd.read_csv('course_genre.csv')

def load_users_profiles() :
    return pd.read_csv('profile_genre.csv')

def load_nn_model():
    nn_model = tf.keras.models.load_model('nn_trainedmodel') 
    return nn_model   

def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id

# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res

def generate_user_profile(user_id, course_genre_df, idx_id_dict):

    all_courses = set(idx_id_dict.values())
    ratings_df = load_ratings()
    user_courses_df = ratings_df[ratings_df['user'] == user_id]
    enrolled_course_ids = user_courses_df['item'].to_list()

    user_profile_df = course_genre_df[course_genre_df['COURSE_ID'].isin(enrolled_course_ids)]
    user_profile_vector = user_profile_df.iloc[:, 2:].to_numpy().sum(axis=0)*3
    unkown_courses = all_courses.difference(enrolled_course_ids)
    unknown_course_df = course_genre_df[course_genre_df['COURSE_ID'].isin(unkown_courses)]
    return enrolled_course_ids, unknown_course_df, user_profile_vector


def generate_recomendation_scores(user_id, course_genre_df, idx_id_dict):
    
    users = []
    courses = []
    scores = []
    
    enrolled_course_ids, unknown_course_df, user_profile_vector = generate_user_profile(user_id, 
                                                                                        course_genre_df, 
                                                                                        idx_id_dict)
    unknown_course_id = unknown_course_df['COURSE_ID'].values
    recom_scores = np.dot(unknown_course_df.iloc[:, 2:].to_numpy(), user_profile_vector)
    
    for i in range(len(unknown_course_id)) :
            users.append(user_id)
            courses.append(unknown_course_id[i])
            scores.append(recom_scores[i])
    return users, courses, scores

def combine_user_label(user_ids, labels):
    labels_df = pd.DataFrame(labels)
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    return cluster_df

def train_kmeans(n):

    users_profiles_df = load_users_profiles()
    scaler = StandardScaler()
    user_profile_matrix = users_profiles_df.iloc[:, 1:].to_numpy()
    scaled_matrix = scaler.fit_transform(user_profile_matrix)
    kmeans_model = KMeans(n_clusters=n, max_iter=200, n_init=5, random_state=123)
    kmeans_model.fit(scaled_matrix)
    pickle.dump(scaler, open('kmeans\ScalerModel.pkl', 'wb')) 
    pickle.dump(kmeans_model, open('kmeans\KMeansModel.pkl', 'wb')) 

def train_pca_kmeans(m, n):
    users_profiles_df = load_users_profiles()
    scaler = StandardScaler()
    user_profile_matrix = users_profiles_df.iloc[:, 1:].to_numpy()
    scaled_matrix = scaler.fit_transform(user_profile_matrix)
    PCA_model = PCA(n_components=m, random_state=123)
    PCA_model = PCA_model.fit(scaled_matrix)
    PCA_vector = PCA_model.transform(scaled_matrix)
    kmeans_model = KMeans(n_clusters=n, max_iter=200, n_init=5, random_state=123)
    kmeans_model.fit(PCA_vector)
    pickle.dump(scaler, open('PCA_kmeans\ScalerModel.pkl', 'wb'))
    pickle.dump(PCA_model, open('PCA_kmeans\PCA_KMeans.pkl', 'wb'))
    pickle.dump(kmeans_model, open('PCA_kmeans\KMeansModel.pkl', 'wb')) 

def user_item_idx_id(data):

    encoded_data = data.copy()
    # Mapping user ids to indices
    user_list = encoded_data["user"].unique().tolist()
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}
    
    # Mapping course ids to indices
    course_list = encoded_data["item"].unique().tolist()
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)}
    return user_id2idx_dict, user_idx2id_dict, course_id2idx_dict, course_idx2id_dict

def train_nn_recom(epochs=4):
    
    rating_df = load_ratings()
    num_users = len(rating_df['user'].unique())
    num_items = len(rating_df['item'].unique())
    user_id2idx_dict, user_idx2id_dict, course_id2idx_dict, course_idx2id_dict = user_item_idx_id(rating_df)
    encoded_data = rating_df.copy()
    # Convert original user ids to idx
    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    # Convert original course ids to idx
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    # Convert rating to int
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")
    scaler = MinMaxScaler()
    encoded_data['rating'] = scaler.fit_transform(encoded_data['rating'].values.reshape(-1, 1))
    X = encoded_data[['user', 'item']].values
    y = encoded_data['rating'].values

    nn_model = NN_Recommender.Neural_Network_Recommender(num_users, num_items)
    nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics='RootMeanSquaredError')
    nn_model.fit(X, y, epochs=epochs)
    print(nn_model.summary())
    nn_model.save(filepath='nn_trainedmodel', save_format='tf', overwrite=True, include_optimizer=True)
    print('Model Training Done Successfuly!')
    return nn_model

def clf_w_embeding(epochs=4):
    rating_df = load_ratings()
    user_id2idx_dict, user_idx2id_dict, course_id2idx_dict, course_idx2id_dict = user_item_idx_id(rating_df)
    nn_model = train_nn_recom(epochs=epochs)
    user_embd = nn_model.get_layer('user_embedding_layer').get_weights()[0]
    item_embd = nn_model.get_layer('item_embedding_layer').get_weights()[0]
    user_embd = pd.DataFrame(user_embd)
    item_embd = pd.DataFrame(item_embd)
    user_embd.index = user_embd.index.map(user_idx2id_dict)
    user_embd = user_embd.reset_index().rename(columns={'index':'user'})
    # save new user's embd profile
    user_embd[user_embd.user==user_embd.user.max()].to_csv('NewUser_embd.csv', index=False)
    item_embd.index = item_embd.index.map(course_idx2id_dict)
    item_embd = item_embd.reset_index().rename(columns={'index':'item'})
    item_embd.to_csv('item_embd.csv', index=False)
    embd_df = pd.merge(rating_df, user_embd, how='left', on='user')
    embd_df = embd_df.merge(item_embd, how='left', on='item')
    user_f = [f'{i}_x' for i in range(16)]
    item_f = [f'{j}_y' for j in range(16)]
    embd_f = [f'f{z}' for z in range(16)]
    embd_df[embd_f] = embd_df[user_f] + embd_df[item_f].values
    embd_df.drop((user_f + item_f), axis=1, inplace=True)
    y = embd_df['rating'].values
    y = LabelEncoder().fit_transform(y)
    X = embd_df.drop(['user', 'item', 'rating'], axis=1)
    XGB_model = XGBClassifier(**XGB_param, seed=rs)
    XGB_model.fit(X, y)
    pickle.dump(XGB_model, open('XGBoost\XGBoostModel.pkl', 'wb')) 


# Model training
def train(model_name, params):

    if model_name == models[2]:
       if 'Number_of_Clusters' in params : 
           n = params['Number_of_Clusters']
           train_kmeans(n)

    elif model_name == models[3]:
       m = params['Number_of_Components']
       n = params['Number_of_Clusters']
       train_pca_kmeans(m, n) 

    elif model_name == models[4]:  
        n = params['epochs']
        train_nn_recom(n) 

    elif model_name == models[5] :
        n = params['epochs']
        clf_w_embeding(n)         
           
    else :
      pass

# Prediction
def predict(model_name, user_ids, params):

    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    course_genre_df = load_course_genre()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            sim_threshold = 0.6
            if "sim_threshold" in params:
                sim_threshold = params["sim_threshold"] / 100.0
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)

        # User profile model
        elif model_name == models[1]:

            idx_id_dict, _ = get_doc_dicts()
            users, courses, scores = generate_recomendation_scores(user_id, course_genre_df, idx_id_dict)
        
        elif model_name == models[2]:

            ratings_df = load_ratings()
            users_profiles_df = load_users_profiles()
            users_ids = users_profiles_df.loc[:, users_profiles_df.columns == 'user']
            scaler = pickle.load(open('kmeans\ScalerModel.pkl', 'rb'))
            kmeans_model = pickle.load(open('kmeans\KMeansModel.pkl', 'rb'))
            labels = kmeans_model.labels_
            cluster_df = combine_user_label(users_ids, labels)
            cluster_df = pd.merge(ratings_df, cluster_df, on='user', how='inner').drop('rating', axis=1)
            cluster_item_df = cluster_df.groupby(['cluster', 'item']).size().reset_index().rename(columns={0:'enrollments'})
            # Create new user's profile
            enrolled_course_ids, _, user_profile_vector = generate_user_profile(user_id, 
                                                                                course_genre_df, 
                                                                                idx_id_dict)
                                                                                    
            scaled_vector = scaler.transform(user_profile_vector.reshape(-1, 14))
            user_cluster = kmeans_model.predict(scaled_vector)[0]
            cluster_item_df = cluster_item_df[cluster_item_df['cluster']==user_cluster]
            for user in user_ids :
                cluster_courses = cluster_item_df['item'].to_list()
                unenrolled_courses = [course for course in cluster_courses if course not in enrolled_course_ids]
                for course in unenrolled_courses :
                    enroll_count = cluster_item_df[cluster_item_df['item']==course]['enrollments'].values[0]

                    if enroll_count > 100 :
                       users.append(user)
                       courses.append(course)
                       scores.append(enroll_count)

        elif model_name == models[3]:

            ratings_df = load_ratings()
            users_profiles_df = load_users_profiles()
            users_ids = users_profiles_df.loc[:, users_profiles_df.columns == 'user']
            scaler = pickle.load(open('PCA_kmeans\ScalerModel.pkl', 'rb'))
            PCA_model = pickle.load(open('PCA_kmeans\PCA_KMeans.pkl', 'rb'))
            kmeans_model = pickle.load(open('PCA_kmeans\KMeansModel.pkl', 'rb'))
            labels = kmeans_model.labels_
            cluster_df = combine_user_label(users_ids, labels)
            cluster_df = pd.merge(ratings_df, cluster_df, on='user', how='inner').drop('rating', axis=1)
            cluster_item_df = cluster_df.groupby(['cluster', 'item']).size().reset_index().rename(columns={0:'enrollments'})
            # Create new user's profile
            enrolled_course_ids, _, user_profile_vector = generate_user_profile(user_id, 
                                                                                course_genre_df, 
                                                                                idx_id_dict)
                                                                                    
            scaled_vector = scaler.transform(user_profile_vector.reshape(-1, 14))
            PCA_vector = PCA_model.transform(scaled_vector)
            user_cluster = kmeans_model.predict(PCA_vector)[0]
            cluster_item_df = cluster_item_df[cluster_item_df['cluster']==user_cluster]
            for user in user_ids :
                cluster_courses = cluster_item_df['item'].to_list()
                unenrolled_courses = [course for course in cluster_courses if course not in enrolled_course_ids]
                for course in unenrolled_courses :
                    enroll_count = cluster_item_df[cluster_item_df['item']==course]['enrollments'].values[0]

                    if enroll_count > 100 :
                       users.append(user)
                       courses.append(course)
                       scores.append(enroll_count)

        elif model_name == models[4]:

            nn_model = load_nn_model()
            ratings_df = load_ratings()
            new_user_df = ratings_df[ratings_df.user==user_id]
            new_user_enrolled = new_user_df.item.to_list()
            new_user_unenrolled = list(set(ratings_df.item.values).difference(new_user_enrolled))
            new_user_df = pd.DataFrame({'user':user_id, 'item':new_user_unenrolled})
            user_id2idx_dict, user_idx2id_dict, course_id2idx_dict, course_idx2id_dict = user_item_idx_id(ratings_df)
            new_user_df['user'] = new_user_df['user'].map(user_id2idx_dict)
            new_user_df['item'] = new_user_df['item'].map(course_id2idx_dict)
            new_user_values = new_user_df.values
            y_pred = nn_model.predict(new_user_values)
            new_user_df['pred_score'] = y_pred
            new_user_df.sort_values(by='pred_score', ascending=False, inplace=True)
            new_user_df['user'] = new_user_df['user'].map(user_idx2id_dict)
            new_user_df['item'] = new_user_df['item'].map(course_idx2id_dict)
            users.append(user_id)
            courses.extend(new_user_df['item'].to_list())
            scores.extend(new_user_df['pred_score'].to_list())


        elif model_name == models[5]:
            rating_df = load_ratings()
            XGB_model = pickle.load(open('XGBoost\XGBoostModel.pkl', 'rb'))
            new_user_df = rating_df[rating_df.user==user_id]
            new_user_enrolled = new_user_df.item.to_list()
            new_user_unenrolled = list(set(rating_df.item.values).difference(new_user_enrolled))
            new_user_df = pd.DataFrame({'user':user_id, 'item':new_user_unenrolled})
            new_user_embd = pd.read_csv('NewUser_embd.csv')
            item_embd = pd.read_csv('item_embd.csv')
            unenrolled_item_embd = item_embd[item_embd.item.isin(new_user_unenrolled)]
            new_user_embd_profile = unenrolled_item_embd.copy()
            new_user_embd_profile.iloc[:,1:] = unenrolled_item_embd.iloc[:,1:] + new_user_embd.iloc[:,1:].values
            new_X = new_user_embd_profile.iloc[:, 1:].values
            pred_rate = XGB_model.predict_proba(new_X)[:, 1]
            new_user_embd_profile['pred_rate'] = pred_rate
            new_user_pred = new_user_embd_profile[['item','pred_rate']]
            new_user_pred.sort_values(by='pred_rate', ascending=False, inplace=True)
            users.append(user_id)
            courses.extend(new_user_pred['item'].to_list())
            scores.extend(new_user_pred['pred_rate'].to_list())

        else :
            pass

    res_dict['USER'] = users*len(courses)
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    res_df = res_df.sort_values(by='SCORE', ascending=False)
    if 'top_courses' in params :
        res_df = res_df[ : params['top_courses']]

    return res_df


