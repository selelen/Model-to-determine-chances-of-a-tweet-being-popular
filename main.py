# -*- coding: utf-8 -*-

# UTILS

from datetime import datetime
import time

import pandas as pd
import numpy as np 
import twint
import nest_asyncio
from sklearn.linear_model import LinearRegression

from pathlib import Path
import plotnine as pn

from mizani.formatters import percent_format
from sklearn.metrics import mean_squared_error, r2_score

"""**Configurations**"""

pd.set_option('display.max_columns', 500)

"""**Functions**"""

def return_relationship_between(x_likes,x_retweets,y_likes,y_retweets,variable):
    
    # Calculate the means of likes for tweets with and without video
  mean_likes = pd.DataFrame({'Tweet with '+variable: [x_likes.mean()],
                            'Tweet without '+variable: [y_likes.mean()]})

  mean_likes_melted = pd.melt(mean_likes, var_name=variable, value_name='Mean Likes')

  # Create the bar chart
  ggplot_mean_likes = pn.ggplot(mean_likes_melted, pn.aes(x=variable, y='Mean Likes')) + \
                    pn.geom_bar(stat='identity', fill='red') + \
                    pn.labs(title='Relation between tweet with ' +variable+ ' and likes', 
                            x='', y='Mean number of likes') + \
                            pn.theme(figure_size=(4, 4))   
                               


  # Calculate the means of retweets for tweets with and without video
  mean_retweets = pd.DataFrame({'Tweet with '+variable: [x_retweets.mean()],
                            'Tweet without '+variable: [y_retweets.mean()]})

  mean_retweets_melted = pd.melt(mean_retweets, var_name=variable, value_name='Mean Retweets')

  # Create the bar chart
  ggplot_mean_retweets = pn.ggplot(mean_retweets_melted, pn.aes(x=variable, y='Mean Retweets')) + \
                    pn.geom_bar(stat='identity', fill='blue') + \
                    pn.labs(title='Relation between tweet with ' +variable+ ' and retweets', 
                            x='', y='Mean number of retweets') + \
                    pn.theme(figure_size=(4, 4)) 

  return ggplot_mean_likes, ggplot_mean_retweets

"""# DATASET CREATION"""

user_details = pd.read_csv(f'2021-08-11-handles-data.csv').drop_duplicates('id')
tweet_data = pd.read_csv('2021-08-11-2021-08-12-2021-08-19-tweets-data.csv')

copy_tweet_data=tweet_data
tweet_data_for_merge = copy_tweet_data.drop(columns=['username']).rename(columns={'id': 'tweet_id'})
merged_data_tweet_users = pd.merge(tweet_data_for_merge, user_details, left_on='user_id', right_on='id')

"""# ANALYSIS"""

#Here we are determinating the difference between the likes and retweets a tweet gets whether it has or not the following features:
# -Has a video
# -Has a photo
# -Has hashtags
# -Comes from a verified account
# -Is replying to another tweet

# Creating a list for the filtered data
filtered_data=[]

#by tweets with and without video
video_likes = merged_data_tweet_users.loc[merged_data_tweet_users['video'] == True, 'nlikes']
no_video_likes = merged_data_tweet_users.loc[merged_data_tweet_users['video'] == False, 'nlikes']
video_retweets = merged_data_tweet_users.loc[merged_data_tweet_users['video'] == True, 'nretweets']
no_video_retweets = merged_data_tweet_users.loc[merged_data_tweet_users['video'] == False, 'nretweets']

filtered_data.append([video_likes,video_retweets,no_video_likes,no_video_retweets,'Video'])

#by tweets with and without verified user
verified_likes = merged_data_tweet_users.loc[merged_data_tweet_users['verified'] == True, 'nlikes']
no_verified_likes = merged_data_tweet_users.loc[merged_data_tweet_users['verified'] == False, 'nlikes']
verified_retweets = merged_data_tweet_users.loc[merged_data_tweet_users['verified'] == True, 'nretweets']
no_verified_retweets = merged_data_tweet_users.loc[merged_data_tweet_users['verified'] == False, 'nretweets']

filtered_data.append([verified_likes,verified_retweets,no_verified_likes,no_verified_retweets,'verified user'])

#by tweets with and without photo
photo_likes =  merged_data_tweet_users.loc[merged_data_tweet_users['photos']!="[]", 'nlikes']
no_photo_likes = merged_data_tweet_users.loc[merged_data_tweet_users['photos']=="[]", 'nlikes']
photo_retweets = merged_data_tweet_users.loc[merged_data_tweet_users['photos']!="[]", 'nretweets']
no_photo_retweets = merged_data_tweet_users.loc[merged_data_tweet_users['photos']=="[]", 'nretweets']

filtered_data.append([photo_likes,photo_retweets,no_photo_likes,no_photo_retweets,'Photos'])

#by tweets with and without hashtags
hashtags_likes =  merged_data_tweet_users.loc[merged_data_tweet_users['hashtags']!="[]", 'nlikes']
no_hashtags_likes = merged_data_tweet_users.loc[merged_data_tweet_users['hashtags']=="[]", 'nlikes']
hashtags_retweets = merged_data_tweet_users.loc[merged_data_tweet_users['hashtags']!="[]", 'nretweets']
no_hashtags_retweets = merged_data_tweet_users.loc[merged_data_tweet_users['hashtags']=="[]", 'nretweets']

filtered_data.append([hashtags_likes,hashtags_retweets,no_hashtags_likes,no_hashtags_retweets,'Hashtags'])

#by tweets with and without reply_to
reply_to_likes =  merged_data_tweet_users.loc[merged_data_tweet_users['reply_to']!="[]", 'nlikes']
no_reply_to_likes = merged_data_tweet_users.loc[merged_data_tweet_users['reply_to']=="[]", 'nlikes']
reply_to_retweets = merged_data_tweet_users.loc[merged_data_tweet_users['reply_to']!="[]", 'nretweets']
no_reply_to_retweets = merged_data_tweet_users.loc[merged_data_tweet_users['reply_to']=="[]", 'nretweets']

filtered_data.append([reply_to_likes,reply_to_retweets,no_reply_to_likes,no_reply_to_retweets,'reply_to'])

for x_likes,x_retweets,y_likes,y_retweets,variable in filtered_data:

  ggplot_mean_likes,ggplot_mean_retweets=return_relationship_between(x_likes,x_retweets,y_likes,y_retweets,variable)

  # Show the bar chart
  print(ggplot_mean_likes,ggplot_mean_retweets)

#Getting the required likes to know if a tweet is popular. It should be the mean of nlikes without the maximum and minimum value 
likes_required=merged_data_tweet_users[(merged_data_tweet_users['nlikes']!=merged_data_tweet_users.nlikes.max()) & (merged_data_tweet_users['nlikes']!=merged_data_tweet_users.nlikes.min())].nlikes.mean()

#the required retweets to know if a tweet is popular. It should be the mean of nretweet without the maximum and minimum value 
retweets_required=merged_data_tweet_users[(merged_data_tweet_users['nretweets']!=merged_data_tweet_users.nretweets.max()) & (merged_data_tweet_users['nretweets']!=merged_data_tweet_users.nretweets.min())].nretweets.mean()

#Getting the tweets with more likes and retweets than the mean 
popular_tweets= merged_data_tweet_users[(merged_data_tweet_users['nlikes']>likes_required) & (merged_data_tweet_users['nretweets']>retweets_required)]

#Getting the most common values for the day of the week in which the popular tweets were created
mode_day=popular_tweets['day'].value_counts().head(2)

#Print correlation between the day of the week and the number of likes/retweets just to show I am not making up this data cause it really affects the visualizations and because of that the interactions.

corr_day=pn.ggplot(merged_data_tweet_users, pn.aes(x='day', y='nlikes')) + \
  pn.geom_point(alpha=0.2) + \
  pn.labs(title='Correlation between day of the week and number of likes',
       x='Day', y='Number of likes')
print(corr_day)
corr_day_rt=pn.ggplot(popular_tweets, pn.aes(x='day', y='nretweets')) + \
  pn.geom_point(alpha=0.2) + \
  pn.labs(title='Correlation between day of the week and number of retweets',
       x='Day', y='Number of retweets')
print(corr_day_rt)

#Getting the more common hours in which a popular tweet was created. Not the same amount of users in twitter at 03:00 than at 20:00

mode_hours=popular_tweets['hour'].value_counts().head(4)

#Printing correlation between hours and number of likes/retweets
corr_hours=pn.ggplot(popular_tweets, pn.aes(x='hour', y='nlikes')) + \
  pn.geom_point(alpha=0.2) + \
  pn.labs(title='Correlation between hour and number of likes',
       x='Hour', y='Number of likes')
print(corr_hours)
corr_hour_rt=pn.ggplot(popular_tweets, pn.aes(x='hour', y='nretweets')) + \
  pn.geom_point(alpha=0.2) + \
  pn.labs(title='Correlation between hour and number of retweets',
       x='Hour', y='Number of retweets')
print(corr_hour_rt)

#Getting most used three languages for popular tweets
top_three_languages = popular_tweets['language'].value_counts().head(3)

#Printing correlation between language and number of likes
corr_language=pn.ggplot(popular_tweets, pn.aes(x='language', y='nlikes')) + \
  pn.geom_point(alpha=0.2) + \
  pn.labs(title='Correlation between language and number of likes',
       x='Language', y='Number of likes')
print(corr_language)

corr_language_rt=pn.ggplot(popular_tweets, pn.aes(x='language', y='nretweets')) + \
  pn.geom_point(alpha=0.2) + \
  pn.labs(title='Correlation between language and number of retweets',
       x='Language', y='Number of retweets')
print(corr_language_rt)


#Mean of followers of the users that created popular tweets without min and max
mean_followers=popular_tweets[(popular_tweets['followers']!=popular_tweets.followers.max()) & (popular_tweets['followers']!=popular_tweets.followers.min())].followers.mean()

#Graph to show the number of number of likes per day, hour and followings_per_followers
graph_data = popular_tweets.groupby('day').nlikes.median().reset_index()
graph = pn.ggplot(graph_data, pn.aes(x='day', y='nlikes')) + pn.geom_line() # + pn.xlim(0, 52)
graph.draw();

graph_data = popular_tweets.groupby('hour').nlikes.median().reset_index()
graph = pn.ggplot(graph_data, pn.aes(x='hour', y='nlikes')) + pn.geom_line() # + pn.xlim(0, 52)
graph.draw();



#Same but with retweets
graph_data = popular_tweets.groupby('day').nretweets.median().reset_index()
graph = pn.ggplot(graph_data, pn.aes(x='day', y='nretweets')) + pn.geom_line() # + pn.xlim(0, 52)
graph.draw();

graph_data = popular_tweets.groupby('hour').nretweets.median().reset_index()
graph = pn.ggplot(graph_data, pn.aes(x='hour', y='nretweets')) + pn.geom_line() # + pn.xlim(0, 52)
graph.draw();

"""# DATA TRANSFORMATION AND MODEL CREATION"""

#Strings

#Based on the analysis, these are the conditions a tweet has to follow to be popular, so we create variables for each one
cond1 = merged_data_tweet_users['hashtags'] != "[]"
cond2 = merged_data_tweet_users['photos'] != "[]"
cond3 = merged_data_tweet_users['reply_to'] == "[]"
cond4 = merged_data_tweet_users['language'].isin(top_three_languages.index.tolist()) ==True
cond5 = merged_data_tweet_users['hour'].isin(mode_hours.index.tolist()) ==True
cond6 = merged_data_tweet_users['day'].isin(mode_day.index.tolist()) ==True
cond7 = merged_data_tweet_users['url'] != "[]"
cond8 = merged_data_tweet_users['followers'] >= mean_followers
cond9 = merged_data_tweet_users['nlikes']>(likes_required + likes_required/10)
cond10=merged_data_tweet_users['nretweets']>(retweets_required + retweets_required/10)
cond11 = merged_data_tweet_users['verified'] == True
cond12 = merged_data_tweet_users['video'] == True

#Except for the boolean type ones, we create a new column in order to know if they are acomplishing the corresponding condition

merged_data_tweet_users['hashtags_bool'] = 0
merged_data_tweet_users['photos_bool'] = 0
merged_data_tweet_users['reply_to_bool'] = 0
merged_data_tweet_users['language_bool'] = 0
merged_data_tweet_users['hour_bool'] = 0
merged_data_tweet_users['day_bool'] = 0
merged_data_tweet_users['url_bool'] = 0
merged_data_tweet_users['followers_bool'] = 0

#If they meet the conditions then we set the value to the corresponding column to 1
merged_data_tweet_users.loc[cond1,'hashtags_bool'] = 1
merged_data_tweet_users.loc[cond2,'photos_bool'] = 1
merged_data_tweet_users.loc[cond3,'reply_to_bool'] = 1
merged_data_tweet_users.loc[cond4,'language_bool'] = 1
merged_data_tweet_users.loc[cond5,'hour_bool'] = 1
merged_data_tweet_users.loc[cond6,'day_bool'] = 1
merged_data_tweet_users.loc[cond7,'url_bool'] = 1
merged_data_tweet_users.loc[cond8,'followers_bool'] = 1

#Here we change the conditions for the string columns because it couldn't evaluate it properly and use instead the new columns created

cond1 = merged_data_tweet_users['hashtags_bool'] == 1
cond2 = merged_data_tweet_users['photos_bool'] ==1
cond3 = merged_data_tweet_users['reply_to_bool'] ==1
cond7 = merged_data_tweet_users['url_bool'] ==1




# Define the list of conditions
conditions = [cond1, cond2, cond3, cond4, cond5, cond6, cond7,cond8,cond11,cond12]

#Creating the new column popular, wich will be the target of our model, and by default we set it at 0 
merged_data_tweet_users['popular'] = 0.0

# Set values for popular column based on the conditions they meet. 
for i in range(len(conditions)):
    merged_data_tweet_users.loc[conditions[i], 'popular'] += 1.0
   

# Normalize popular values to be between 0 and 1
merged_data_tweet_users['popular'] = merged_data_tweet_users['popular'] / len(conditions)

#So if the data meets the conditions of nlikes and nretweets it is popular 100%
merged_data_tweet_users.loc[cond9 & cond10,'popular'] = 1.0

# Splitting and shuffling with indexes in order to create three datasets
idx = merged_data_tweet_users               # Dataset
id_train = int(len(idx) * 0.8)              # Train 80%
id_valid = int(len(idx) * (0.8 + 0.05))     # Valid 5%, Test 15%
train, val, test = np.split(idx, (id_train, id_valid))

# Define X and y for linear regression

#The features 
X = ['video', 'hashtags_bool', 'reply_to_bool', 'url_bool', 'photos_bool', 'verified', 'hour_bool', 'day_bool', 'language_bool', 'followers_bool']
#The target
y = 'popular'

# Create linear regression model and fit to training data
model = LinearRegression()
model.fit(train[X],train[y]) 

# Predict popular column of validation data based on X
predictions = model.predict(test[X]) 
test['predictions'] = predictions

"""# Initial evaluation


"""

# The coefficients
print("Coefficients: \n", model.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(test[y], predictions))

# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(test[y], predictions))

#Difference between values
test['difference'] = (test[y] - test.predictions)

#Info about the columns
print(test[[y, 'predictions', 'difference']].describe())
print('\n')
print(test[[y, 'predictions', 'difference']].corr())

graph_data = pd.DataFrame({'realidad' : test[y], 'predicciones' : predictions})

graph = pn.ggplot(graph_data, pn.aes(x='realidad', y='predicciones')) + pn.geom_point() + pn.geom_smooth(method="lm", color='red')
graph.draw();

from sklearn.metrics import median_absolute_error

test['baseline'] = test[y].mean()

print(median_absolute_error(test[y], test['baseline']))
print(median_absolute_error(test[y], predictions))