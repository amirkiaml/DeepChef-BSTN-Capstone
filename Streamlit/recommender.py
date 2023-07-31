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
image_url = 'https://drive.google.com/uc?export=download&id=1rgGyzAasBKQRgROPakamsi5EbzolkjCb'


# Display the image in Streamlit with a custom width (e.g., 300 pixels)
st.image(image_url, use_column_width=True)

st.title('DeepChef')
st.subheader('A Modern Recipe Recommender')


st.write("👨‍🍳 **DeepChef** is a recommender system created by [Amir Kiani](https://linkedin.com/in/amirhossein-kiani)\
          that uses user prompts consisting of ingredients, recipe instructions and even themes,\
        and retrieves recipes from food.com that are most similar to the prompt.\
         This app uses OpenAI's semantics embdedding models on the text data\
         of over 520,000 recipes scraped from [food.com](https://food.com). Learn more about DeepChef, its motivations,\
        limitations and codes at its [GitHub repository](https://github.com/amirkiaml/DeepChef-BSTN-Capstone).")

with st.expander("Sample Prompts"):
    st.write('There are various ways you can use DeepChef. For example ...')
    st.write('1. Banana muffin')
    st.write('2. Chicken burger sauce onion')
    st.write('3. A delicious and healthy\
              dish featuring grilled salmon fillets served with roasted\
              asparagus.')
    st.write('4. A simple and refreshing salad made with\
              fresh tomatoes, mozzarella cheese, basil leaves, and a \
             drizzle of balsamic glaze or olive oil.')
    st.write('5. (This is outight copied from https://themodernproper.com/30-best-ground-beef-recipes):\
             Easy Ground Beef Recipes with Few Ingredients Thai Basil\
              Beef (cover photo). A speedy, savory Thai beef basil stir\
              fry that’s just a bit spicy and really hits the spot. \
             Gingery Ground Beef (Soboro Donburi). Five ingredients, a\
              few minutes and a hot skillet, and you’ll be digging into\
              a delicious soboro donburi, a gingery ground beef that\
              reminds us that the best Japanese recipes are often the\
              simplest. Crock-Pot Taco Meat. Just three-ingredients and\
              the use of our trusty slow-cooker make this Crock-Pot beef\
              taco meat recipe an easy win. Taco Pizza. Two classic\
              family favorites, pizza and tacos, come together in this\
              super easy weeknight meal that is destined to become a\
              favorite in your home. Ground Beef Burger and Sandwich\
              Recipes Classic Cheeseburger with Secret Sauce. A perfectly\
              soft bun, quality, juicy ground beef that is seasoned well,\
              plenty of cheese, caramelized onions and a really good secret\
              sauce are the key to this truly classic cheeseburger recipe.\
              Patty Melt. As diner sandwiches go, the melty, meaty, oniony\
              perfection that is the patty melt reigns supreme—if you know,\
              you know. And if you don’t know, well, welcome to the club.\
              Meatball Sub Sandwich. Big juicy, tender meatballs simmered\
              in marinara sauce (store-bought or homemade—either works!)\
              stuffed into garlicky hoagie rolls, topped with mozzarella\
              and broiled to melty, bubbly perfection. Beef Sliders. These\
              juicy ground beef sliders—slathered with a homemade bright\
              and lemony sun-dried tomato mayo, spicy pickles, and fresh\
              arugula—prove that bigger doesn’t always equal better.\
              Low-Carb Bacon Burger with Guacamole. You’ll forget all about\
              the missing bun while devouring this juicy, keto-friendly\
              bacon guacamole burger loaded with veggies and jalapeno aioli!')


# Get user input and display similar recipes
user_input = st.text_area("🥑🥕What are you craving?🥑🥕", height=200)

# Provide a unique key for the button
button_key = "submit_button"

# Use st.button property called 'key' to treat it as a new button each time
# The label for the button remains the same, but it will be disabled after click
#st.warning('Please only click **once** and wait.')
st.error('🍅🚨Please click only **ONCE** and wait until the results show up before making a new query🚨🍅')
button = st.button("Submit", key=button_key,type='primary')
st.toast('Your recipes are being found!', icon='👩‍🍳')


# Data loading
st.cache(persist=True)
@st.cache_data(persist=True)  # Decorator allows us to cache the df in memory

def load_data():
    # URL to the pickled file on Google Drive
    url = 'https://drive.google.com/uc?id=1Xu1s427Goe787gCjRysCj5SWb8psbFH6&export=download'
    
    #10 percent (without reviews): 'https://drive.google.com/uc?export=download&id=1kfRBie2U1Ov9kRJAUhS8HgDg7D2JKuxC'
    #20 percent: 'https://drive.google.com/uc?id=1AZDhn3LhShjiHq7Bb9IKGYNbsr1YI4_D&export=download'
    #30 percent: 'https://drive.google.com/uc?id=1Fkyn_iv-P8JswsxHFROdMTjWoe-1t8Fq&export=download'

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
openai.api_key = "..."  # Replace with your OpenAI API key.

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

# Get user input and display similar recipes.

if user_input != '':
    if button:
        # Disable the button after it's clicked
        # button = st.button("Please wait...", key='wait', disabled=True)
        
        top_5_similar_recipes = most_similar_recipes(user_input, df)
        
        def recipe_info(df,row_num):
        
            st.subheader(f"{df['Name'].iloc[row_num]}")

            image_url = df['Images'].iloc[row_num][0]
            st.image(image_url, caption=df['Name'].iloc[row_num], use_column_width=True)
            
            st.write(f"{df['Description'].iloc[row_num]}. [Link to recipe]({df['url'].iloc[row_num]}).")

            with st.expander("Ingredients"):
                servings = int(df['RecipeServings'].iloc[row_num])
                if servings == 1:
                    st.warning(f"**This recipe serves {servings} person.**")
                else:
                    st.warning(f"**This recipe serves {servings} people.**")
                
                for i,j in zip(df['ingred_quants'].iloc[row_num],df['ingred_items'].iloc[row_num]):
                    st.write(f"{i} {j}")

            with st.expander("Nutritional Facts"):
                nut_facts = int(df['RecipeServings'].iloc[row_num])
                if nut_facts ==1:
                    st.warning(f"**These nutritional facts are for the recipe serving {nut_facts} person.**")
                else:
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

            with st.expander("Reviews"):
                review_count = int(df['ReviewCount'].iloc[row_num])
                if review_count == 1:
                    st.warning(f"Based on {review_count} review, this recipe is rated {int(df['CorrectAggregatedRating'].iloc[row_num])}/5.")
                    st.write('**Review:**')
                    st.write(df['Review'].iloc[row_num][0])
                else:
                    st.warning(f"Based on {review_count} reviews, this recipe is rated {int(df['CorrectAggregatedRating'].iloc[row_num])}/5.")
                    st.write('**Some reviews:**')
                    for i,line in enumerate(df['Review'].iloc[row_num]):
                        st.write(f'{i+1}: {line}')

        recipe_info(top_5_similar_recipes,0)
        recipe_info(top_5_similar_recipes,1)
        recipe_info(top_5_similar_recipes,2)
        recipe_info(top_5_similar_recipes,3)
        recipe_info(top_5_similar_recipes,4)
        st.toast('Enjoy!', icon='😍')

end_time = time.time()
duration = end_time - start_time

st.write(f'Time spent: {duration:.2f} seconds, i.e., {(duration/60):.2f} minutes')

