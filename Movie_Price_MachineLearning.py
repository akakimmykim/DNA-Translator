import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

df = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@movie_data.csv", index_col=0)
profitable_vector = np.zeros((df.shape[0],1))
profitable_count = 0

regression_target = 'revenue'
classification_target = 'profitable'

# Add profitable classification to dataframe
for row in df.index:
    if df['revenue'][row] > df['budget'][row]:
        profitable_vector[row] = 1
        profitable_count += 1
df['profitable'] = profitable_vector


# Remove incomplete data
df.replace(to_replace=[np.inf, np.NINF], value=np.nan, inplace=True)
df.dropna(inplace=True)

# Strip All Genres
genre_list = []
genre_list = np.asarray(genre_list)
for row in df.index:
    genre_list = np.append(genre_list,((df['genres'][row]).split(", ")))
genre_list = np.unique(genre_list)

genre_check = np.zeros((df.shape[0],1))
# Add Column Genres and classify
for genre in genre_list:
    df[genre] = genre_check
    for row in df.index:
        if (genre in df['genres'][row]):
            df.loc[row, genre] = 1
            
# Statistical Analysis
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]
plotting_variables = ['budget', 'popularity', regression_target]
axes = pd.plotting.scatter_matrix(df[plotting_variables], alpha=0.15, \
       color=(0,0,0), hist_kwds={"color":(0,0,0)}, facecolor=(1,0,0))
    
#plt.show()
for covariate in outcomes_and_continuous_covariates:
    for row in df.index:
        df.loc[row, covariate] = np.log10(1+df.loc[row, covariate])
print(df[outcomes_and_continuous_covariates].skew())