from sklearn.ensemble import RandomForestClassifier
from time import time
import pandas as pd
df=pd.read_csv('./res/df_inne_choosen')
X=df.drop('Moda',axis=1)
y=df['Moda']
t=time()
rfc=RandomForestClassifier()
rfc.fit(X,y)
print('czas dla n_e=100',time()-t)
t=time()
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X,y)
print('czas dla n_e=200',time()-t)
t=time()
rfc=RandomForestClassifier(n_estimators=300)
rfc.fit(X,y)
print('czas dla n_e=300',time()-t)