from joblib import load
from pathlib import Path
import pandas as pd
from argparse import ArgumentParser
parser=ArgumentParser()
parser.add_argument('-i',"--input",help="plik wejsciowy, plik z danymi w formacie csv")
parser.add_argument('-s','--show_classifiers',action="store_true",help='wypisuje dostepne klasyfikatory')
parser.add_argument('-c','--classifiers',type=str,help='nazwa klasyfikatora ktory bedzie uzyty (wymagane chyba ze uzyto -s)')
parser.add_argument('-o','--output',type=str,help='plik wyjsciowy, jesli nie podany, wypisuje na wyjscie standardowe')
args=parser.parse_args()
MODELS_DIR='./'
decision_tree=load(Path(MODELS_DIR)/'ostateczne_drzewo_decyzyjne.joblib')
svc=load(Path(MODELS_DIR)/'ostateczne_svc.joblib')
random_forest=load(Path(MODELS_DIR)/'ostatecz_random_forest.joblib')
logistic_regression=load(Path(MODELS_DIR)/'ostateczne_logistic_regresion.joblib')
multinomialNB=load(Path(MODELS_DIR)/'ostateczne_multinomial_naive_bayes.joblib')
lasso_attributes=['Wolny_czas__spie', 'Sniadanie__nie', 'Zajecie__Czytanie',
       'Paznokcie__nie', 'Wakacje__Praca']
gini_no_soki_attributes=[ 'Wzrost', 'gry_czas', 'Zajecie__sluchanie muzyki',
       'Wakacje__Praca', 'Rod_prz_sam__Przyjaciele', 'Napoj__soki',
       'Napoj__woda']
entropy_attributes=[ 'Muzyka__Rock', 'Miasto__Poznan', 'Zajecie__Sport',
       'Media_u__TikTok', 'Paznokcie__nie', 'Wakacje__Praca',
       'Wartosc__Kariera', 'Napoj__woda']
models_and_attributes={
    'drzewo_decyzyjne': (decision_tree,entropy_attributes),
    'svc': (svc,lasso_attributes),
    'random_forest': (random_forest,lasso_attributes),
    'logistic_regression': (logistic_regression,gini_no_soki_attributes),
    'multinomialnb':(multinomialNB,gini_no_soki_attributes),
}

if args.show_classifiers:
    print('modele:\twymagane columny:')
    for key,value in models_and_attributes.items():
        model_name,required_columns=key,value[1]
        print(model_name,required_columns,sep='\t')
    exit()
data=pd.read_csv(args.input)
choosen_classifiers=args.classifiers
try:
    df=data[models_and_attributes[choosen_classifiers][1]]
except:
    print('brakuje kolumny, potrzebne kolumny:',*models_and_attributes[choosen_classifiers][1],sep=' ')
    exit()
X=df
classifier=models_and_attributes[choosen_classifiers][0]
predict_y=classifier.predict(X)
print(*predict_y,sep=',')