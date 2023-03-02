import streamlit as st
from surprise import KNNBasic
from surprise import Dataset
import pandas as pd
from surprise import Reader
import pickle
import numpy as np
# Load the dataset

with open('data/data_for_model/data_for_model.pkl', 'rb') as f:
    # Load the contents of the file using pickle.load()
    data_for_model = pickle.load(f)
    

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(data_for_model[['Username','BGGId', 'Rating']],reader)

# Define function to load data and build model
def build_model(df_new_user,time,genres,gameweight):
    # Create new user DataFrame
    with st.spinner('Building model...'):
        df_new_user['Username'] = 'demo_user'
        demo_df = pd.concat([data_for_model, df_new_user])
        df_final = pd.concat([data_for_model,demo_df])
        # Build model
        model = KNNBasic()
        reader = Reader(rating_scale=(1, 10))
        data_2 = Dataset.load_from_df(df_final[['Username', 'BGGId', 'Rating']], reader)
        train = data_2.build_full_trainset()
        model.fit(train)
        # Make predictions
        predict_list = [model.predict(uid='demo_user', iid=i) for i in not_user_list]
        est = [i.est for i in predict_list if i.details['was_impossible'] !=True]
        iids = [i.iid for i in predict_list if i.details['was_impossible'] !=True]
        demo_test_df = pd.DataFrame({'est': est, 'BGGId': iids})
        df_test_2 = pd.merge(demo_test_df, games_csv2, on='BGGId')
        if show_slider:
            percentile_1 = 33
            percentile_2 = 66
            if gameweight == 1:
                filtered_games_1 = df_test_2[df_test_2['MfgPlaytime'] < percentile_1]
                df_test_2 = filtered_games_1
            elif gameweight == 2:
                filtered_games_1 = df_test_2[df_test_2['MfgPlaytime'] < percentile_2]
                df_test_2 = filtered_games_1
            elif gameweight == 3:
                filtered_games_1 = df_test_2[df_test_2['MfgPlaytime'] > percentile_2]
                df_test_2 = filtered_games_1

            if genres == 'Yes':
                filtered_games =df_test_2[df_test_2['Cat:Family']==1]
                df_test_2 = filtered_games
            if time == 1:
                filtered_games_1 = df_test_2[df_test_2['MfgPlaytime'] < percentile_1]
                df_test_2 = filtered_games_1
            elif time == 2:
                filtered_games_1 = df_test_2[df_test_2['MfgPlaytime'] < percentile_2]
                df_test_2 = filtered_games_1
            elif time == 3:
                filtered_games_1 = df_test_2[df_test_2['MfgPlaytime'] > percentile_2]
                df_test_2 = filtered_games_1
        top_10 = df_test_2[['Name','Description','ImagePath','est']].sort_values('est', ascending=False).reset_index(drop=True).head(10)
        return top_10

# Load data
not_user_list = list(data_for_model['BGGId'].unique())
games_csv2 = pd.read_csv('data/no_null_user_ratings/games_after_2010.csv')
games_csv2 = games_csv2[games_csv2['YearPublished'] > 2016]
# Set up Streamlit app
st.title('Game Recommender')

# User input
game_names = []
game_ratings = []
show_slider = st.checkbox("Do you have more specific requirements for your recommendations?")
container = st.container()
if show_slider:
    genres = container.radio("Are you looking for a family orientated game?:", ("Yes", "No"))
    time = st.slider(f"On a scale of of 1 to 3 how much time are you willing to spend on a single game?:", min_value=1, max_value=3,key='time')
    gameweight = st.slider(f"On a scale of of 1 to 3 how much time are you willing to spend on a single game?:", min_value=1, max_value=3,key='gameweight')

for i in range(5):
    game_name = st.selectbox(f"Game {i+1}:", [''] + list(games_csv2['Name']))
    if game_name:
        game_rating = st.slider(f"Enter rating for {game_name}:", 1.0, 10.0, value=7.0, key=f"rating_{i}")
        game_names.append(game_name)
        game_ratings.append(game_rating)
    else:
        game_names.append('')
        game_ratings.append(0)

# Create a DataFrame from game_names and game_ratings
data_dict = {'BGGId': game_names, 'Rating': game_ratings}
df_new_user = pd.DataFrame(data_dict)

# Map BGGId from game_names to its corresponding value in games_csv2
df_new_user['BGGId'] = [games_csv2.loc[games_csv2['Name'] == names, 'BGGId'].iloc[0] if len(games_csv2.loc[games_csv2['Name'] == names, 'BGGId']) > 0 else None for names in game_names]

# Replace NaN values with -1
df_new_user['BGGId'] = df_new_user['BGGId'].fillna(-1)

# Add Username column
df_new_user['Username'] = 'demo_user'

# Build model and show top 10 recommendations
if st.button('Submit'):
    top_10 = build_model(df_new_user,time,genres,gameweight)

if 'top_10' in locals() and not top_10.empty:
    # Show table of top 10 recommendations with Name, Description, and Image columns
    st.write("Top 10 recommendations:")
    for i, (index, row) in enumerate(top_10.iterrows()):
        st.markdown(f"**{i+1}. {row['Name']}**")
        st.write(row['Description'])
        st.image(row['ImagePath'])