import pandas as pd#pandasのインポート
import datetime#元データの日付処理のためにインポート
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV#データ分割用
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,precision_score,recall_score,f1_score
from sklearn.metrics import r2_score#決定係数求める用
import matplotlib.pyplot as plt
#%matplotlib inline
import joblib #モデルの保存、ロード用
import pickle#モデルの保存、ロード用
from pandas import Series, DataFrame
import sklearn
import pathlib
from sklearn import datasets, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import *
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
import glob
import random
import os
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
import openpyxl
import pandas as pd#pandasのインポート
import datetime#元データの日付処理のためにインポート
import dice_ml

features_list = ['Vktgs', 'wind_speed','gravi_fts', 'elev_stick', 'elron_stick','rudder_stick','elev_surf', 'ailrn_surf', 'ruddr_surf','nwhel,steer','trim','flap' ,'pitch', 'roll','hding','   alpha,__deg ','   _beta,__deg ', '   hpath,__deg ', '   vpath,__deg ','   _slip,__deg ', '   __alt,ftmsl ', '   ___on,runwy ', '   _dist,___ft ', '   thro1,_part ', '   power,_1,hp ']

df_list_test = []
df_list_train = []
files = glob.glob('C:\\Users\\rikua\\Documents\\ml_run_fdToWl\\ml_all\\now_all\\*')
test = [files[1],files[5],files[7],files[9],files[11],files[14],files[17],files[19],files[22],files[24]]

for file in files:
    if file in test:
        sub_df = pd.read_csv(file)
        df_list_test.append(sub_df)
    else :
        sub_df = pd.read_csv(file)
        df_list_train.append(sub_df)

df_list_train = pd.concat(df_list_train)
df_list_test = pd.concat(df_list_test)

df_list_train = df_list_train.replace({"workload":{1:0, 2:0,3:0,4:1,5:1}})
df_list_test = df_list_test.replace({"workload":{1:0, 2:0,3:0,4:1,5:1}})

train_x = df_list_train.drop("workload",axis=1)
train_y= df_list_train[["workload"]]
test_x=df_list_test.drop("workload",axis=1)
test_y= df_list_test[["workload"]]

sm = SMOTE()

print(train_x)
print(test_x)

# train_x, train_y = sm.fit_resample(train_x, np.ravel(train_y))
# test_x, test_y = sm.fit_resample(test_x, np.ravel(test_y))



with open('RDFmodeltwo0825.pickle', mode='rb') as f:  # with構文でファイルパスとバイナリ読み来みモードを設定
    model = pickle.load(f)                  # オブジェクトをデシリアライズ

d = dice_ml.Data(dataframe = pd.concat([test_x, test_y], axis=1),# データは、変数とアウトカムの両方が必要
                 continuous_features = features_list, #　連続変数の指定
                 outcome_name = "workload")

m = dice_ml.Model(model=model, 
                  backend="sklearn")

exp = dice_ml.Dice(d, m)

query = pd.read_csv("query.csv")
print(query)
dice_exp = exp.generate_counterfactuals(query ,total_CFs=1 ,desired_class="opposite")
dice_exp.visualize_as_dataframe(show_only_changes=True)