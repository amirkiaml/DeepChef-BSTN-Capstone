<div style="display: flex; justify-content: center;">
  <img src="https://github.com/amirkiaml/DeepChef-BSTN-Capstone/blob/main/Logo.png" alt="DeepChef Logo" width="200" height="200" />
</div>

# DeepChef: A Modern Recipe Recommendation
Amirhossein Kiani, Data Science Diploma Program @BrainStation

## 👨‍🍳 Welcome to DeepChef!

This project harnesses the power of Natural Language Processing (NLP) and unsupervised machine learning techniques to deliver a unique, user-centric recipe recommendation experience. By allowing users to input their desired prompts, DeepChef suggests recipes that best match their preferences, elevating personalization and user engagement. Dive into the repository to explore the inner workings of this system and discover its potential!

## 💡Suggested Use Cases 
Ideally, DeepChef offers a wide range of versatile use cases, catering to various user needs and preferences.

- **Personalized recipe recommendations:** Users could input their desired recipe ingredients, instructions or themes to receive recipe recommendations tailored to their tastes. This could help users discover new recipes that they might not have otherwise found and could enhance their overall culinary experience.
- **Cooking with Leftovers:** Users can input leftover ingredients, and the recommender can suggest creative ways to use them in new recipes.
- **Cuisine exploration:** Users can input ingredients and flavors from different cuisines they want to explore, and the recommender can suggest authentic recipes from those cuisines.
- **Content recommendation** for cooking websites: cooking websites could integrate and fully develop DeepChef to complement their recommendation systems, offering users recipe suggestions based on specific recipe preferences or cuisines. This could enhance user satisfaction and engagement, and ultimately drive more views and revenue for these websites.


## 🏗 Built With 
- `bertopic`
- `gdown`
- `jupyter`
- `spacy`
- `pandas`
- `openai`
- `re`
- `requests`
- `scikit-learn`
- `matplotlib`
- `nltk`
- `numpy`
- `seaborn`
- `selenium`
- `streamlit`
- `tiktoken`
- `umap`
- `wordcloud`
  
and more..

## 📚 Project Composition
The project is composed of 6 Jupyter Notebooks and a few supplementary files:

- Notebook 1: Part 1 - Scraping www.food.com
- Notebook 2: Part 2 - Basic Data Cleaning
- Notebook 3: Part 3 - Topic Modeling
- Notebook 4: Part 4 - EDA
- Notebook 5: Part 5 - Semantic Embeddings
- Notebook 6: Part 6 - Recommendation Systems
- (Supplementary Folder) 'Streamlit': contains the Python scripts and requirements file for the Streamlit app
- (Supplementary File 1) 'docs_and_topics_300.html': contains Bertopic's interactive topic visualization for 300 topics
- (Supplementary File 2) 'intertopic_dist_map.html': contains Bertopic's interactive intertopic visualization for 300 topics
- (Supplementary File 3) 'Final_Report.pdf': contains the final report of the project

Please note that to focus on the primary notebooks, simply follow the numbered sequence (Part 1, Part 2, etc.) for a streamlined experience without any distractions. All Jupyter notebooks have markdown introductions, table of contents, and commented codes for higher readability.

## 🎮 Streamlit app 
I have deployed the recommender system from Notebooks 5 and 6 into an online interactive app that can be accessed through this address: https://deepchef.streamlit.app.

Check out a demo video on DeepChef:

<!-- [![DeepChef](https://drive.google.com/uc?id=1vLItXqQDOEdFYyEd0Egnf6fkW8HXHo8x)](https://www.loom.com/share/350eed0ce28c43e297ea78e6ede7d694)-->
https://www.loom.com/share/350eed0ce28c43e297ea78e6ede7d694 

### 🔭 Limitations and Extensions

There are several ways that DeepChef can be extended to increase its performance and make it more personalized to users.

- In order to assure the app is up and running on Streamlit's website, I have deployed the recommender system only on 10% of the data that was originally curated for this purpose (see Notebook 6). I hope to be able to scale the model in the near future. Such extensions will give more data to the model to compare user inputs against, which means that there will be more accurate recommendations.
- Features other than merely the user's textual input can be incorporated into the recommender system, such as nutritional factors. For instance, the user not only can find recipes that are similar in content to what they look for, but they can also filter recipes based on their nutritional similarities, e.g., protein content, fat percentage, or fiber content. The initial blueprint of this layer of the recommender system has been coded up (see Notebook 6, 'Part 6'); I hope to incorporate these into the app in the future.
- We could incorporate user reviews (text data) or ratings (numbers) in finding recipes to users' tastes. This falls under the broad category of 'collaborative filtering' in recommender systems. For example, we could after the user has searched for a few queries (hence surpassing the 'cold start' phase), they can be recommended similar recipes, in terms of content, rating, reviews, or a blend of these, based on their own query history or similar users query histories.


## 💾 Datasets 
The two original datasets I've worked with, the recipes dataset and the reviews dataset, can be directly accessed through Kaggle.  The recipes dataset has been significantly enhanced and augmented using newly scraped data (from Notebook 1), the reviews dataset and feature engineering. The enhanced dataset will soon be uploaded on Kaggle for public use.

- The original datasets from Kaggle can be found [here](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews).
- The dataset used for creating the semantic recommendation system in Notebooks 5 and 6 can be downloaded from [here](https://www.dropbox.com/scl/fi/165kyme2d72iitpg1wbtd/recipes_with_ada_embeddings.pkl?rlkey=8331x1vqq7zjn6hov6mjw1srb&dl=1).
- The dataset used for the streamlit app can be downloaded from [here](https://drive.google.com/uc?id=1Xu1s427Goe787gCjRysCj5SWb8psbFH6&export=download).
