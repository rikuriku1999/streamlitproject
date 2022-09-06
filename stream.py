import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import dice_ml


get = st.sidebar.radio("oke", ["dogs","cats"])

with open('RDFmodel0824.pickle', mode='rb') as f:  # with構文でファイルパスとバイナリ読み来みモードを設定
    model = pickle.load(f)                  # オブジェクトをデシリアライズ

uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df

if st.button('predict'):
    pred = model.predict(df)
    pred
    pred_num = 2
    image = Image.open(str(pred_num) + ".png")
    st.image(image, caption = "サンプル", use_column_width = True)


# with open('RDFmodeltwo0825.pickle', mode='rb') as f:  # with構文でファイルパスとバイナリ読み来みモードを設定
#     model = pickle.load(f)                  # オブジェクトをデシリアライズ

# d = dice_ml.Data(dataframe = pd.concat([test_x, test_y], axis=1),# データは、変数とアウトカムの両方が必要
#                  continuous_features = list(train_x.drop("CHAS", axis=1).columns), #　連続変数の指定
#                  outcome_name = "price")

# m = dice_ml.Model(model=model_logi, 
#                   backend="sklearn")

# exp = dice_ml.Dice(d, m)