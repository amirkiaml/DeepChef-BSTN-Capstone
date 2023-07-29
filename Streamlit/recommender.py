import streamlit as st
import joblib
import time
import gdown #!pip install 
import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken #pip install
import openai   #pip install
from tenacity import retry, stop_after_attempt, wait_exponential
from heapq import nlargest
import sys

start_time = time.time()

# Create a title
st.title('DeepChef')
st.subheader('A Modern Recipe Recommender')

#st.write('DeepChef is ... ')

# Get user input and display similar recipes
user_input = st.text_area("What do you like?", height=200)
button = st.button("Submit")

# Data loading
st.cache(persist=True)
@st.cache_data(persist=True)  # Decorator allows us to cache the df in memory

def load_data():
    # URL to the pickled file on Google Drive
    url = 'https://drive.google.com/uc?export=download&id=1kfRBie2U1Ov9kRJAUhS8HgDg7D2JKuxC'
    
    # Download the pickled data from Google Drive
    output_file = 'recipes.pkl'
    gdown.download(url, output_file, quiet=False)
    
    # Read the pickled data into a DataFrame
    with open(output_file, 'rb') as f:
        recipes = pickle.load(f)

    return recipes

# Load the data using the function
df = load_data()

# Display the dataframe in the app
#st.dataframe(df.head(11))

# Get the embeddings
openai.api_key = "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"  # Replace with your OpenAI API key

def get_embedding(text, engine="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model=engine)
    embeddings = response['data'][0]['embedding']
    return embeddings

def most_similar_recipes(user_input, df):
    input_embeddings = get_embedding(user_input)
    similarities = []

    for index, row in df.iterrows():
        cur_embeddings = row['ada_embeddings']
        similarity = cosine_similarity(np.array(input_embeddings).reshape(1, -1), np.array(cur_embeddings).reshape(1, -1))[0][0]
        similarities.append(similarity)

    df['SimilarityRate'] = similarities
    return df.sort_values(['SimilarityRate'], ascending=False)[:5]

# Get user input and display similar recipes

if button:
    if user_input:
        
        top_5_similar_recipes = most_similar_recipes(user_input, df)
        
        def recipe_info(df,row_num):
        
            st.subheader(df['Name'].iloc[row_num])

            image_url = df['Images'].iloc[row_num][0]
            st.image(image_url, caption=df['Name'].iloc[row_num], use_column_width=True)
            
            st.write(df['Description'].iloc[row_num])

            with st.expander("Ingredients"):
                servings = int(df['RecipeServings'].iloc[row_num])
                st.warning(f"**This recipe serves {servings} people.**")
                
                for i,j in zip(df['ingred_quants'].iloc[row_num],df['ingred_items'].iloc[row_num]):
                    st.write(f"{i} {j}")

            with st.expander("Nutritional Facts"):
                nut_facts = int(df['RecipeServings'].iloc[row_num])
                st.warning(f"**These nutritional facts are for the recipe serving {nut_facts} people.**")

                st.write(f"Calories: {df['Calories'].iloc[row_num]} KCal")
                st.write(f"Fat Content: {df['FatContent'].iloc[row_num]} gr")
                st.write(f"Saturated Fat Content: {df['SaturatedFatContent'].iloc[row_num]} gr")
                st.write(f"Cholesterol Content: {df['CholesterolContent'].iloc[row_num]} mg")
                st.write(f"Sodium] Content: {df['SodiumContent'].iloc[row_num]} mg")
                st.write(f"Carbohydrate Content: {df['CarbohydrateContent'].iloc[row_num]} gr")
                st.write(f"Fiber Content: {df['FiberContent'].iloc[row_num]} gr")
                st.write(f"Sugar Content: {df['SugarContent'].iloc[row_num]} gr")
                st.write(f"Protein Content: {df['ProteinContent'].iloc[row_num]} gr")

            with st.expander("Instructions"):
                for i,line in enumerate(df['RecipeInstructions'].iloc[row_num]):
                    st.write(f'{i+1}: {line}')

        recipe_info(top_5_similar_recipes,0)
        recipe_info(top_5_similar_recipes,1)
        recipe_info(top_5_similar_recipes,2)
        recipe_info(top_5_similar_recipes,3)
        recipe_info(top_5_similar_recipes,4)


    
end_time = time.time()
duration = end_time - start_time

st.write(f'Time spent: {duration:.2f} seconds, i.e., {(duration/60):.2f} minutes')

