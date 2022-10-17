from functools import cache
import streamlit as st
import pandas as pd
import time
import backend as backend
from PIL import Image
import os
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode


# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)

# ------- Functions ------
# Load datasets
#@st.cache
def load_ratings():
    return backend.load_ratings()


@st.cache
def load_course_sims():
    return backend.load_course_sims()


@st.cache
def load_courses():
    return backend.load_courses()

@cache
def load_course_genre():
    return backend.load_course_genre()

@st.cache
def load_bow():
    return backend.load_bow()

    
# Initialize the app by first loading datasets
def init__recommender_app():

    col1, col2, col3 = st.columns([1,6,1])

    with col1:
         st.write('')

    with col2:
         st.title('Course Recommender System') 
         image = Image.open('recom.PNG')
         st.image(image, width=400) 

    with col3:
         st.write('')
         

    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_genre = load_course_genre()
        course_bow_df = load_bow()

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results


def train(model_name, params):

    if model_name == backend.models[0]:
        # Start training course similarity model
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_name, params)
        st.success('Done!')

    elif model_name == backend.models[1]:
        pass

    elif model_name == backend.models[2]:
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_selection, params)
            st.success('Done!')

    elif model_name == backend.models[3]:
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_selection, params)
            st.success('Done!')

    elif model_name == backend.models[4]:
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_selection, params)
            st.success('Model Training Done Successfuly!')     

    elif model_name == backend.models[5]:
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_selection, params)
            st.success('Model Training Done Successfuly!')                        
            
    else:
        pass


def predict(model_name, user_ids, params):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        time.sleep(0.5)
        res = backend.predict(model_name, user_ids, params)
    st.success('Recommendations generated!')
    return res


# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold

# User profile model
elif model_selection == backend.models[1]:
     # Add a slide bar for selecting top courses  
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=20,
                                    value=5, step=1)
    """"                             
    profile_sim_threshold = st.sidebar.slider('User Profile Similarity Threshold %',
                                              min_value=0, max_value=100,
                                              value=50, step=10)

    """
    params['top_courses'] = top_courses
# Clustering model
elif model_selection == backend.models[2]:

    cluster_no = st.sidebar.slider('Number_of_Clusters',
                                   min_value=0, max_value=20,
                                   value=18, step=1)
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=20,
                                    value=5, step=1)                                   
    params['Number_of_Clusters'] = cluster_no
    params['top_courses'] = top_courses

elif model_selection == backend.models[3]:

    component_no = st.sidebar.slider('Number_of_Components',
                                   min_value=1, max_value=14,
                                   value=9, step=1)

    cluster_no = st.sidebar.slider('Number_of_Clusters',
                                   min_value=1, max_value=20,
                                   value=18, step=1)
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=20,
                                    value=5, step=1)

    params['Number_of_Components'] = component_no 
    params['Number_of_Clusters'] = cluster_no 
    params['top_courses'] = top_courses   

elif model_selection == backend.models[4]:
    n_epochs = st.sidebar.slider('epochs',
                                    min_value=0, max_value=5,
                                    value=4, step=1)

    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=20,
                                    value=5, step=1)                                
    params['epochs'] = n_epochs
    params['top_courses'] = top_courses


elif model_selection == backend.models[5]:
    n_epochs = st.sidebar.slider('epochs',
                                    min_value=0, max_value=5,
                                    value=4, step=1)

    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=20,
                                    value=5, step=1)                                
    params['epochs'] = n_epochs
    params['top_courses'] = top_courses    

else:
    pass
   
   
# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')

# Start training process
if training_button:
    if (model_selection==backend.models[4]) or (model_selection==backend.models[5]):
       # train model with new user  
       new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
       user_ids = [new_id] 
       train(model_selection, params)

    else :
       train(model_selection, params)    
       
  
# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:

    if (model_selection==backend.models[4]) or (model_selection==backend.models[5]) :
        new_id = (load_ratings())['user'].max()
        user_ids = [new_id] 
        res_df = predict(model_selection, user_ids, params)
        res_df = res_df[['COURSE_ID', 'SCORE']]
        course_df = load_courses()
        res_df = pd.merge(res_df, course_df, on=["COURSE_ID"], how='inner').drop('COURSE_ID', axis=1)
        st.table(res_df) 

    else :

        # Create a new id for current user session
        new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
        user_ids = [new_id]
        res_df = predict(model_selection, user_ids, params)
        res_df = res_df[['COURSE_ID', 'SCORE']]
        course_df = load_courses()
        res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
        st.table(res_df)
