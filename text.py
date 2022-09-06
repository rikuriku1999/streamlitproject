#最適なパラメータを入れる配列
max_depth_array=[]
max_features_array=[]
min_samples_split_array=[]
n_estimators_array=[]
n_jobs_array=[]

for i in poms:
    #テストデータと訓練データを読み取り
    df_train2=pd.read_csv("train_total.csv",index_col=0)
    df_test2=pd.read_csv("test_total.csv",index_col=0)
    
    #時間のデータ(time,sleep_date)
    df_train2_timedata=df_train2[["time","sleep_date"]]
    df_test2_timedata=df_test2[["time","sleep_date"]]

    #時間のデータを消す(標準化するため)
    df_train2=df_train2.drop(["time","sleep_date","Snow"],axis=1)
    df_test2=df_test2.drop(["time","sleep_date","Snow"],axis=1)

    print(df_train2.shape)
    print(df_test2.shape)
    
    #標準化しないとき
    train_dataset=df_train2
    test_dataset=df_test2

    #標準化を行う
    #(統計処理を基に戻す用の統計量・関数)
    #テストデータの標準化に用いる
    stats_train = df_train2.describe()#訓練データのデータを表示
    stats_train = stats_train.transpose()#転置
    
    #使わない列を削除
    df_train2=train_dataset.drop(["id","height","weight","T-A_class","D-D_class","A-H_class","V_class","F_class","C_class","caffeine_yes","drinking_yes","less_1cup","1cup_3cups","3cups_6cups","6cups_9cups","more_9cups"], axis=1)
    df_test2=test_dataset.drop(["id","height","weight","T-A_class","D-D_class","A-H_class","V_class","F_class","C_class","caffeine_yes","drinking_yes","less_1cup","1cup_3cups","3cups_6cups","6cups_9cups","more_9cups"], axis=1)
    for j in poms:
        if i!=j:
            df_train2=df_train2.drop([j],axis=1)
            df_test2=df_test2.drop([j],axis=1)
            
    #特徴量
    name=df_train2.columns
    
    #訓練データとテストデータに分ける
    x_train = df_train2.drop(i, axis=1).values
    y_train= df_train2[i].values
    x_test=df_test2.drop(i, axis=1).values
    y_test= df_test2[i].values
    
    #データを整形する
    Xtrain2 = np.array(x_train,dtype=int)
    Xtest2 = np.array(x_test,dtype=int)
    y_train2 = np.array(y_train,dtype=int)
    y_test2 = np.array(y_test,dtype=int)
    
    #グリッドサーチ
    #https://www.sejuku.net/blog/64455
    #https://qiita.com/nanairoGlasses/items/93d764f549943d42d5e6
    search_params = {
         'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],#決定木の数
          'max_features'      : [3, 5, 10, 15, 20],
          'n_jobs'            : [1],
          'min_samples_split' : [3, 5, 10, 20, 30, 40, 50, 100],
          'max_depth'         : [3, 5, 10, 20, 30, 40, 50, 100]#決定木最大の深さ
    }
    # ランダムサーチ(パラメータ範囲指定)用のパラメータ 1~100
    #paramR = {'n_estimators':np.arange(100)}
    
    #RFR_raw  = RFR(random_state=0)#通常のランダムフォレスト
    RFR_grid = GridSearchCV(estimator=RFR(random_state=0),param_grid=search_params, scoring='r2', cv=3,verbose=True,n_jobs=-1)       # 並列処理)#グリッドサーチ・ランダムフォレスト
    #RFR_rand = RandomizedSearchCV(estimator=RFR(random_state=0), param_distributions=paramR, scoring='r2', cv=3,verbose=True,n_jobs=-1)# ランダムサーチ・ランダムフォレスト
    
    # 各モデルに学習を行わせる。
    #RFR_raw.fit (Xtrain2, y_train2)
    #print('通常のランダムフォレストモデルにおける n_estimators         :  %d'  %RFR_raw.n_estimators)
    model=RFR_grid.fit(Xtrain2, y_train2)
    
    #最適なモデルの選択-----------------------------------------------------------------------------------------------------------------
    best_model = RFR_grid.best_estimator_
    best_model_params=RFR_grid.best_params_
    
    max_depth_ = best_model_params["max_depth"]
    max_features_=best_model_params["max_features"]
    min_samples_split_ = best_model_params["min_samples_split"]
    n_estimators_ = best_model_params["n_estimators"]
    n_jobs_=best_model_params["n_jobs"]
    
    #配列に入れる
    max_depth_array.append(best_model_params["max_depth"])
    max_features_array.append(best_model_params["max_features"])
    min_samples_split_array.append(best_model_params["min_samples_split"])
    n_estimators_array.append(best_model_params["n_estimators"])
    n_jobs_array.append(best_model_params["n_jobs"])
    
    #モデルの構築
    model2 = RFR(max_depth = max_depth_, max_features=max_features_,min_samples_split= min_samples_split_,n_estimators=n_estimators_,n_jobs=n_jobs_,random_state=0,verbose=1)

    # モデルのコンパイル
    #model_.compile(loss='mse',
    #              optimizer=tf.keras.optimizers.Adam(learning_rate=lr_),
    #              metrics=['mae', 'mse'])

    # モデルの学習
    history_ = model2.fit(Xtrain2, y_train2)
    
    #-------------------------------------------------------------------------------------------------------------------------
    
    print(i+'_のグリッドサーチ・ランダムフォレストモデルにおける n_estimators   :  %d'  %RFR_grid.best_estimator_.n_estimators)
    #RFR_rand.fit(Xtrain2, y_train2)
    #print('ランダムサーチ・ランダムフォレストモデルにおける n_estimators  :  %d'  %RFR_rand.best_estimator_.n_estimators)
    """
    #https://arakan-pgm-ai.hatenablog.com/entry/2019/08/13/000000
    # テストデータで予測実行
    pred_train = model2.predict(Xtest2)
    # R2決定係数で評価
    print(i+"_のR2決定係数")
    pred_train=np.array(pred_train, dtype='int64')
    r2 = r2_score(y_test2 , pred_test)
    #print(r2)
    """
    # 特徴量の重要度を取得
    
    feature = RFR_grid.best_estimator_.feature_importances_
    # 特徴量の名前ラベルを取得
    label = df_train2.columns[0:]
    # 特徴量の重要度順（降順）に並べて表示
    indices = np.argsort(feature)[::-1]
    for i in range(len(feature)):
        print(str(i + 1) + "   " +
              str(label[indices[i]]) + "   " + str(feature[indices[i]]))
        
    # 予測値を計算(精度を確認)
    #https://tekenuko.hatenablog.com/entry/2016/09/20/222453
    y_train_pred = model2.predict(Xtrain2)#.の前にモデルを書く
    y_test_pred =model2.predict(Xtest2)
    # MSEの計算 
    print("MSE")
    print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train2, y_train_pred), mean_squared_error(y_test2, y_test_pred)) )
    # R^2の計算
    print("R^2")
    print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train2, y_train_pred), r2_score(y_test2, y_test_pred)) )
    
    # モデルを保存する(https://localab.jp/blog/save-and-load-machine-learning-models-in-python-with-scikit-learn/)
    #filename = 'RDF_regression'+str(i)+'finalized_model.sav'
    #joblib.dump(model2, open(filename, 'wb'))
            
    # 実際の値と予測値の比較グラフ
    plt.subplot(121, facecolor='white')
    #plt_label = [i for i in range(1, 32)]
    plt.plot(y_test2, color='blue')
    plt.plot(y_test_pred, color='red')
    # 特徴量の重要度の棒グラフ
    plt.subplot(122, facecolor='white')
    plt.title('特徴量の重要度')
    plt.bar(
        range(
            len(feature)),
        feature[indices],
        color='blue',
        align='center')
    plt.xticks(range(len(feature)), label[indices], rotation=45)
    plt.xlim([-1, len(feature)])
    plt.tight_layout()
    # グラフの表示
    plt.show()