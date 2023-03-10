![image](https://tbgddotblog.files.wordpress.com/2019/01/bgg-snap.png)

# Recommendation system for users of board game selling and review website "BoardGameGeeks.com".
---



# Executive Summary: Collaborative Filtering Recommendation System for board game reommendations using the "K-Nearest Neighbour" predictive model from Sci-Kit Suprise library 
## Background
The boardgame industry has been fast [growing](https://www.globenewswire.com/news-release/2022/07/19/2482068/0/en/Board-Games-Market-to-Attain-Value-of-30-93-Billion-By-2028-Thanks-to-Increased-Popularity-of-Online-Gaming-and-Entry-of-New-OTT-platforms-In-Board-Gaming.html) in recent years, being valued at 13.75 billion US Dollars in 2021 and projected to grow to 30.93 Billion by 2028. [BoardGameGeek](https://boardgamegeek.com/) is one of the largest platforms for selling, buying, reviewing and discussing your favourite board games. In 2019, it reached [2 million active users](https://boardgamegeek.com/thread/2147066/2000000-users) and has only continued to grow as a platform since. In order to compete with today's highly competitive online [landscape](https://www.forbes.com/sites/forbesbusinessdevelopmentcouncil/2022/03/23/the-attention-economy-standing-out-among-the-noise/?sh=4b755c127fda), our projects aims to attract even more of the fast growing board game market into one of the most popular web forums for the discussion, purchasing and sale of board games, via a personalised user machine-learning based recommender system which would add much value to current and potential future users by increasing user engagement as well as having a system to incentivise users to browse products that would be the most relevant to them.

## Overview:

### Problem Statement
The objective of this project was to model our user and game reviews from the website BoardGameGeeks.com, in order to predict for and generate a list of game recommendations, which provides the user a customised experience according to games they have previously reviewed. As our focus in on attracting new market share, a great way to stay up-to-date with the latest trends and innovations in the industry would be to have a model more focused on modern games which we will define as games published in the last 5 years (2017 to 2021).

### General information
Our model uses 1,008,727  unique user and review combinations. The model was built using scikit-suprise, an open-source machine learning framework. 
The final chosen model "KNNBasic" is a basic collaborative filtering model based on preferences of similar users to offer recommendations to a target user.Our final model had an rmse value of 1.07 and an overall precision@10 with threshold 7.5 of 0.698, a strong result for us to confidently say that the recommenders generated by our model will be relevant to whichever user interacts with our model.

This repo contains 7 notebooks:

* [EDA Part 1](./code/EDA%20Part1.ipynb) Takes a more general look at our dataset before deciding on our focal points such as subsetting the data.
* [EDA Part 2](./code/EDA%20Part2.ipynb) Focuses more on deciding how to deal with the filter our dataset before modelling.
* [baseline](./code/basline.ipynb) The start point for our modelling, to check start with our two most basic models "NormalPredictor" and "Baseline" as recommended by sci-kit suprise documentation.
* [All-Knns](./code/all-knns.ipynb) Takes a look at the metrics of the scikit suprise K-Nearest Neigbhors inspired models available: "KNNBasic", "KNNZScore", "KNNBaseline" and "KNNwithMeans".
* [Advanced_Models](./code/advanced_models.ipynb) Takes a look at the more advanced models such as SVD, NonNegative Matrix Factorization, Slope One and Co-clustering.
* [Final Model](./code/final_model.ipynb) After much cross validation the final model has been decide based on the most relevant metrics to the problem statement. Examining the final model's predictions we then move on to deployement. 
* [Deployment](./code/final_model.ipynb) Our code that deploys our model using streamlit.



## Data Dictionary:

### User Reviews(csv)

Feature|Description
---|---
BGGId |Ids associated with each board game
Rating| Rating out of 10 given by the user
Username| Username of the user

### Games(csv)
The following description was pulled straight from the kaggle dataset:
Feature|Description
---|---|
Name | Name of game
Description | Description, stripped of punctuation and lemmatized
YearPublished | First year game published
GameWeight | Game difficulty/complexity
AvgRating | Average user rating for game
BayesAvgRating | Bayes weighted average for game (x # of average reviews applied)
StdDev | Standard deviation of Bayes Avg
MinPlayers | Minimum number of players
MaxPlayers | Maximun number of players
ComAgeRec | Community's recommended age minimum
LanguageEase | Language requirement
BestPlayers | Community voted best player count
GoodPlayers | List of community voted good plater counts
NumOwned | Number of users who own this game
NumWant | Number of users who want this game
NumWish | Number of users who wishlisted this game
NumWeightVotes | ? Unknown
MfgPlayTime | Manufacturer Stated Play Time
ComMinPlaytime | Community minimum play time
ComMaxPlaytime | Community maximum play time
MfgAgeRec | Manufacturer Age Recommendation
NumUserRatings | Number of user ratings
NumComments | Number of user comments
NumAlternates | Number of alternate versions
NumExpansions | Number of expansions
NumImplementations | Number of implementations
IsReimplementation | Binary - Is this listing a reimplementation?
Family | Game family
Kickstarted | Binary - Is this a kickstarter?
ImagePath | Image http:// path
Rank:boardgame | Rank for boardgames overall
Rank:strategygames | Rank in strategy games
Rank:abstracts | Rank in abstracts
Rank:familygames | Rank in family games
Rank:thematic | Rank in thematic
Rank:cgs | Rank in card games
Rank:wargames | Rank in war games
Rank:partygames | Rank in party games
Rank:childrensgames | Rank in children's games
Cat:Thematic | Binary is in Thematic category
Cat:Strategy | Binary is in Strategy category
Cat:War | Binary is in War category
Cat:Family | Binary is in Family category
Cat:CGS | Binary is in Card Games category
Cat:Abstract | Binary is in Abstract category
Cat:Party | Binary is in Party category
Cat:Childrens | Binary is in Childrens category


## Model Training and Evaluation:
For our final model, we utilised the "BGGId", "Rating" and "Username" columns from the "user reviews(csv)" file. and used the "Name" column from the "Games(csv)" to subsitute the BGGIds when looking at the final results.

Every model provided from the scikit suprise documentation
https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html
except for "matrix_factorization.SVDpp" was evaluated with a K-Fold 5 cross validation on rmse, precision@k(k=10,threshold=7.5) and recall@k(k=10,threshold=7.5). 

A threshold of 7.5 as we found that 7, what one would consider a reasonably above average score in our dataset was 50th percentile, making our model inaccurate in its reporting thus we moved it up to 7.5, a slightly higher percentile of 60.

With our business context in mind, we priortised the model with a higher ability to recommend with a predicted score of 7.5 and above, as well as the highest recall@k for the highest variety of games being shown, this is due to the fact that the rmse difference between models was neglible.

|                                  |   precision_at_k |   recall_at_k |   average_rmse |
|:---------------------------------|-----------------:|--------------:|---------------:|
| KNNBasic                         |        0.698465  |    0.428652   |        1.07193 |
| SVD                              |        0.698177  |    0.390089   |        1.02047 |
| KNNBaseline                      |        0.688521  |    0.356677   |        1.01619 |
| Baseline                         |        0.68613   |    0.367994   |        1.02057 |
| KNNWithZScore                    |        0.683345  |    0.367066   |        1.03202 |
| Slope One                        |        0.679875  |    0.360517   |        1.01751 |
| Co-clustering                    |        0.665922  |    0.333746   |        1.04566 |
| KNNWithMeans                     |        0.66181   |    0.334689   |        1.03238 |
| Normal_predictor                 |        0.465807  |    0.314209   |        1.85339 |
| NonNegative Matrix Factorization |        0.0874943 |    0.00955713 |        1.78454 |



## Model Deployment:
We will be deploying our model to streamlit and will be available for demonstration. Our model uses KNNBasic to predict games that would be relevant to the user and returns the top 10 games sorted by predicted rating.
There is a fitler that allows you to have a more customised experience via a checkbox that opts into additional filters on the "MfgPlaytime", "Cat:Family" and "GameWeight", which are the upper range of a game, a binary on if the game is family friendly and the difficult respectively.



## Conclusion:
Our model is a powerful tool that is capable of providing highly accurate recommendations for users, making it an invaluable asset to the platform. As users become more active and leave more reviews, our model's ability to provide insightful and relevant recommendations will only increase, resulting in even greater user satisfaction.

Furthermore, by leveraging our model to increase user engagement, the platform stands to benefit tremendously. With its ability to provide solid recommendations based on users' prior reviews, our model can help to create a more personalized and engaging experience for users, ultimately leading to greater retention and loyalty.

We picked our strongest recall@k score in order to increase variety within our recommendations that would be relavant to users, as a lower recall@k despite a higher rmse would mean a lower variety of games being recommended i.e. leading to similar recommendations across all users not as tailored to each individual. 

Future work could involve a model that can account for more than the users and reviews, such as pytorch's recommendation module, as well as using big data tools such as PySpark that allows us to train models on large datasets.  


---
The links to the dataset, streamlit app and website are below<br>
App : https://siazachtj-capstone-codestreamlit-5es7vw.streamlit.app/
<br>
Website: https://boardgamegeek.com/
<br>
Dataset : https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek
<br>
Data: https://drive.google.com/drive/folders/16ny9q1lS7C3g7_AsOu3AeDXQsv1Pd8ty?usp=sharing
