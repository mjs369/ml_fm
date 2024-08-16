import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from sklearn.metrics import accuracy_score

def preprocess_training_data(data):
    data = data.dropna(subset=['勘定科目'])
    
    X = data[['摘要', '税・売上仕入種別', '税率', '課税/売上額（円）', '課税/預り消費税（円）Ａ',
              '課税/仕入額（円）', '課税/支払い消費税（円）Ｂ', '非課税/売上額（円）', '非課税/仕入額（円）',
              '免税/売上額（円）', '免税/仕入額（円）', '不課税/売上額（円）', '不課税/仕入額（円）']]
    y = data['勘定科目']
    
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    
    label_encoders = {}
    for col in ['税・売上仕入種別', '税率']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    tfidf = TfidfVectorizer(max_features=10000)
    X_text = tfidf.fit_transform(X['摘要'])
    
    X_numeric = X.drop(columns=['摘要']).reset_index(drop=True)
    X_combined = pd.concat([pd.DataFrame(X_text.toarray()), X_numeric], axis=1)
    
    return X_combined, y, label_encoders, tfidf, le_y

def preprocess_production_data(data, label_encoders, tfidf):
    numeric_columns = ['課税/売上額（円）', '課税/預り消費税（円）Ａ', '課税/仕入額（円）',
                       '課税/支払い消費税（円）Ｂ', '非課税/売上額（円）', '非課税/仕入額（円）',
                       '免税/売上額（円）', '免税/仕入額（円）', '不課税/売上額（円）', '不課税/仕入額（円）']
    
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    X = data[['摘要', '税・売上仕入種別', '税率'] + numeric_columns]
    
    for col in ['税・売上仕入種別', '税率']:
        le = label_encoders[col]
        unknown_labels = set(X[col]) - set(le.classes_)
        if unknown_labels:
            st.warning(f"Column '{col}' contains unseen labels: {unknown_labels}")
        
        X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    X_text = tfidf.transform(X['摘要'])
    
    X_numeric = X.drop(columns=['摘要']).reset_index(drop=True)
    X_combined = pd.concat([pd.DataFrame(X_text.toarray()), X_numeric], axis=1)
    
    return X_combined

def aggregate_store_data(store_data):
    store_data['税率'] = store_data['税率'].fillna(0)
    
    aggregated_data = store_data.groupby(
        ['税・売上仕入種別', '税率', '予測勘定科目', '店舗コード', '店舗名称', '計上年月日']
    ).agg(
        {col: 'sum' for col in store_data.columns if col not in ['税・売上仕入種別', '税率', '予測勘定科目', '店舗コード', '店舗名称', '計上年月日', '摘要']}
    ).reset_index()
    
    max_summary_count = 10
    summary_group = store_data.groupby(
        ['税・売上仕入種別', '税率', '予測勘定科目', '店舗コード', '店舗名称', '計上年月日']
    )['摘要'].apply(list).reset_index()

    for i in range(max_summary_count):
        col_name = f'使用摘要{i+1}'
        summary_group[col_name] = summary_group['摘要'].apply(lambda x: x[i] if i < len(x) else None)
    
    summary_group = summary_group.drop(columns=['摘要'])
    
    aggregated_data = pd.merge(aggregated_data, summary_group, on=['税・売上仕入種別', '税率', '予測勘定科目', '店舗コード', '店舗名称', '計上年月日'])

    category_order_tax = ['課税売上', '課税仕入', '非課税売上', '非課税仕入', '免税売上', '免税仕入', '不課税売上', '不課税仕入']
    aggregated_data['税・売上仕入種別'] = pd.Categorical(aggregated_data['税・売上仕入種別'], categories=category_order_tax, ordered=True)
    
    category_order_rate = ['１０', '＊８','８', 0]
    aggregated_data['税率'] = pd.Categorical(aggregated_data['税率'], categories=category_order_rate, ordered=True)
    
    aggregated_data = aggregated_data.sort_values(by=['予測勘定科目', '税・売上仕入種別', '税率']).reset_index(drop=True)
    
    return aggregated_data

def save_to_zip(aggregated_results, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for store_name, aggregated_data in aggregated_results.items():
            csv_file_path = f"{store_name}_result.csv"
            aggregated_data.to_csv(csv_file_path, index=False, encoding='cp932')
            zipf.write(csv_file_path, arcname=csv_file_path)
            os.remove(csv_file_path)

# Streamlit UI
st.title('会計データ予測モデル')

st.header('学習データのアップロード')
training_file = st.file_uploader("学習データのCSVファイルをアップロードしてください", type='csv')

st.header('本番データのアップロード')
production_file = st.file_uploader("本番データのCSVファイルをアップロードしてください", type='csv')

if st.button('予測を実行'):
    if training_file is not None and production_file is not None:
        # 学習データの前処理
        training_data = pd.read_csv(training_file, encoding='cp932')
        X_combined, y, label_encoders, tfidf, le_y = preprocess_training_data(training_data)

        # 学習データとテストデータの分割
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

        # LightGBMデータセットの作成
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # パラメータ設定
        params = {
            'objective': 'multiclass',
            'num_class': len(set(y)), 
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 50,
            'min_data_in_leaf': 30,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        # モデル学習
        model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=1000, callbacks=[lgb.early_stopping(stopping_rounds=10)])

        # 本番データの前処理
        production_data = pd.read_csv(production_file, encoding='cp932', header=1)
        production_data = production_data.drop(['会社コード', '会社名称', 'ディストリクトコード', 'ディストリクト名称', '営業所コード', '営業所名称','指定年月','伝票No'], axis=1)
        production_data.columns = production_data.columns.str.strip()
        production_data = production_data[production_data['店舗名称'].notna()]
        production_data = production_data[production_data['店舗名称'].str.strip() != '']
        production_data = production_data[production_data['店舗名称'] != '店舗名称']
        values_to_drop = [np.nan, '小計', '合計', "◆消費税預り金残高推移", "前月残高", "消費税預け金戻し", "当月消費税納付金額", "◆課税売上割合"]
        production_data = production_data[~production_data['摘要'].isin(values_to_drop)]

        # 店舗ごとにデータをグループ化し、予測を実行
        store_groups = production_data.groupby('店舗名称')
        store_results = {}
        for store_name, store_data in store_groups:
            X_prod_combined = preprocess_production_data(store_data, label_encoders, tfidf)
            y_prod_pred = model.predict(X_prod_combined)
            y_prod_pred_max = np.argmax(y_prod_pred, axis=1)
            store_data['予測勘定科目'] = le_y.inverse_transform(y_prod_pred_max)
            store_results[store_name] = store_data

        # 集計処理
        aggregated_results = {}
        for store_name, store_data in store_results.items():
            aggregated_data = aggregate_store_data(store_data)
            aggregated_results[store_name] = aggregated_data

        # 結果をZIPファイルに保存
        zip_file_path = 'aggregated_results.zip'
        save_to_zip(aggregated_results, zip_file_path)

        # ZIPファイルのダウンロードリンクを表示
        with open(zip_file_path, 'rb') as f:
            st.download_button('結果をダウンロード', f, file_name=zip_file_path)
    else:
        st.warning("学習データと本番データの両方をアップロードしてください。")
