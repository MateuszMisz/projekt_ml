from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from time import time

# %%
df = pd.read_csv('./res/df_inne_choosen')
X = df.drop('Moda', axis=1)
y = df['Moda']
print(df.columns)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

from sklearn.model_selection import GridSearchCV

param_dict = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [2, 4,8, 10, None],
    'min_samples_split': [2, 3,5,8],
    'min_samples_leaf': [2, 7, 8, 9, 10],
    'max_features': [0.5, 0.8, 0.2, None],
    'bootstrap': [True],
    'n_estimators': [100,200],
    'min_impurity_decrease': [float(x) for x in list(np.linspace(0, 1, 5))],
    'class_weight': ['balanced', 'balanced_subsample']

}
# %%
t=time()
searcher = GridSearchCV(verbose=True, estimator=RandomForestClassifier(), param_grid=param_dict, scoring="f1_weighted",
                        n_jobs=-1, cv=5)
searcher.fit(X, y)
print('czas: ',time()-t)
print('najlepsze parametry: ', searcher.best_params_)
print('najlepsze f1: ', searcher.best_score_)
