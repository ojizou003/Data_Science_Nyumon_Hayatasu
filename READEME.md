# 「Python x データサイエンス入門」 はやたす youtube

2023/10/15 ~ 10/16

## ① データサイエンスとは？必要なスキルと7つの流れを解説

### データサイエンスとは何か

データサイエンスとは、ビジネスにおける課題を、データを使って解決すること

### データサイエンスで必要になる知識

- ロジカルシンキング（仮説 ｰ> 検証）
- 課題解決したい業界の知識（ドメイン知識）
- コミュニケーションの能力
- 機械学習
- プログラミング
- 数学
- 統計学

### データサイエンスの流れ

1. 目的・課題の特定
2. データの取得・収集
3. データ理解・可視化
4. データの加工・前処理
5. 機械学習モデルの作成
   - ディープラーニングを使わないとき → scikit-learn
   - ディープラーニングを使うとき → TensorFlow
6. 評価・テスト
7. レポーティング、アプリケーション化 ..Flask,Django

今回の学習は2.～6.

## ② Kaggle と Google Colabolatory を使って無料でデータ分析を始めよう

### データの取得・収集

- [x] Kaggleにユーザー登録
- [x] KaggleからTitanic のデータダウンロード(test.csv,train.csv)
- [x] Googleドライブにダウンロードしたデータと今回のテキストをアップロード
- [x] Google Colab で読み込むために Googleドライブをマウント

## ③ Python・Pandasを使ったデータの読み込み

### データ理解・データの可視化

タイタニック号のデータを理解することから始める

- どんなデータが入っているんだろう？
- このデータには、どんな特徴があるんだろう？
- 欠けているデータはあるのかな？

#### 必要ライブラリのインポート

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

%matplotlib inline
```

これらのライブラリはデータ分析するとき、ほぼマストで必要になる  
各ライブラリの特徴

- numpy : 行列計算や数値を扱いやすくするライブラリで、pandasと合わせて使うことが多い
- pandas : データを扱いやすくするライブラリ
- matplotlib : グラフを作成するためのライブラリ
- seaborn : グラフをキレイかつ簡潔に書くためのライブラリ

#### データの読み込み

Pandasを使ってデータを読み込む

```python
dir_path = '/content/drive/MyDrive/DataScience/titanic/' # google colab

# 学習データの読み込み
train_df = pd.read_csv(dir_path + 'train.csv')

# テストデータの読み込み
test_df = pd.read_csv(dir_path + 'test.csv')
```

#### データの中身の確認

```python
# 学習データの先頭5行を確認してみる
train_df.head()
```

```python
# テストデータの先頭5行を確認してみる
test_df.head()
```

```python
# データフレームの大きさの確認
print(train_df.shape)
print(test_df.shape)
```

## ④ データの可視化と欠損値の確認

### データの可視化と欠損地の確認

データの特徴の把握

```python
# 学習データとテストデータを縦に連結する
df = pd.concat([train_df,test_df],ignore_index=True)
```

```python
# 最後の5行を確認
df.tail()
```

#### データの可視化（グラフ化）

- 性別 : どんな内訳になっているのか？
- チケットのクラス : どんな階級の人が多いのか？  
性別やチケットのクラスなど、各カテゴリーの数値を確認したいときは、棒グラフ(barplot)を使ってあげると良い

```python
# ①性別ごとにグループ分けして、②各性別ごとにカウントする  （aggregate(集計)）
df.groupby('Sex').agg({'Sex':'count'})
```

```python
# 集計結果を変数tmpに格納する
temp = df.groupby('Sex').agg({'Sex':'count'}).rename(columns={'Sex': 'count_sex'})
```

matplotlibで棒グラフを作成するには、plt.bar(x軸で使う列(=カラム), y軸で使う列(=カラム))と書いてあげる

```python
# グラフの大きさを設定
plt.figure(figsize=(5,3))

# 性別の数を確認してみる
plt.bar(temp.index,temp.count_sex)
plt.show()
```

```python
# pandasで棒グラフを作成する
temp.plot(kind='bar',figsize=(5,3))
plt.show()
```

Seabornで表示する

```python
# グラフの大きさを設定
plt.figure(figsize=(5,3))

# 性別の数を確認してみる
sns.countplot(data=df,x='Sex')
plt.show()
```

#### 欠損値の確認

```python
# データ内の欠損値を確認する
df.isnull().sum()
```

## ⑤ 必須項目「欠損値の補完」を習得

### データの加工・前処理・特徴量エンジニアリング

特徴量エンジニアリング ..機械学習でよい結果を出すために、新しいカラムを作ること

```python
# Embarkedの欠損値を確認する
df.Embarked.isnull().sum()
```

```python
# 元データをコピー
df2 = df.copy()

# 欠損値の補完
df2.Embarked = df2.Embarked.fillna('S')
```

```python
# 年齢の最小値と最大値を確認
print(df2.Age.max())
print(df2.Age.min())
```

```python
# ヒストグラムを作成する
sns.displot(df2.Age,bins=8,kde=False)
plt.show()
```

```python
# 年齢の平均値と中央値を確認する
print(df2.Age.mean()) #平均値
print(df2.Age.median()) #中央値
```

```python
# 年齢の欠損値を、計算しておいた中央値で補完する
df3.Age = df3.Age.fillna(age_median)
```

## ⑥ カテゴリカル変数の数値変換(one-hot encoding)をマスターしよう

### カテゴリカル変数の数値変換

- ワンホットエンコーディング  
ワンホットエンコーディングは、各カテゴリーに対して別のカラムを準備して、該当する部分には1, そうではない部分には0を振り分ける方法(いわゆるフラグ付)  

    たとえば、S, C, Qというカテゴリーを持つ乗船した港の情報であれば、以下のような変換になる

```python
 \  S   C   Q  
S   1   0   0  
Q   0   0   1  
S   1   0   0  
C   0   1   0  
```

- ラベルエンコーディング  
各カテゴリーを純粋に数値変換する方法

```python
S   0  
Q   2  
S   0  
C   1
```

```python
# まずはワンホットエンコーディングしてみる
pd.get_dummies(df4.Embarked)
```

```python
# ワンホットの結果を変数tmp_embarkedに格納する
tmp_embarked = pd.get_dummies(df4.Embarked,prefix="Embarked")
```

```python
# 元のデータフレームにワンホット結果を横に連結して、変数df5に格納する
df5 = pd.concat([df4,tmp_embarked], axis=1).drop(columns=['Embarked'])
```

## ⑦ 機械学習の必修科目 学習データとテストデータに分割しよう

### 学習用データとテストデータに分割する

機械学習をするときは、学習データとテストデータに分ける

```python
# 学習データに分割した結果を変数trainに格納する
train = df5[~df5.Survived.isnull()] # ~は打消し

# テストデータに分割した結果を変数trainに格納する
test = df5[df5.Survived.isnull()]

```

### 学習データを「学習に使うカラム(=特徴量)」と「正解(=目的変数)」に分割する

```python
# 正解をy_trainに格納する
y_train = train.Survived #yはベクトルなので小文字

# 特徴量をX_trainに格納する
X_train = train.drop(columns=['Survived']) # Xは行列なので大文字
```

## ⑧ 機械学習モデル「決定木」で予測してみよう

### 機械学習を使って予測する

```python
# ライブラリのインポート #決定木モデルの作成以外でScikit-learnを使わないので。
from sklearn import tree
```

```python
# 決定木モデルの準備
model = tree.DecisionTreeClassifier()
```

分類と回帰  

- 分類 : ラベルを予測すること
    例 : 今回のように生存するか否かを当てる
- 回帰 : 数値を予測すること
    例 : 明日の株価(=数値)を当てる

```python
# 決定木モデルの作成
model.fit(X_train, y_train)
```

### 作成したモデルを使って予測する

```Python
# 作成した決定木モデルを使った予測をおこなう
y_pred = model.predict(test)
```

```Python
# テストデータと予測結果の大きさを確認する
len(test),len(y_pred)
```

```Python
# 予測結果をテストデータに反映する(カラムを追加しy_predを代入)
test['Survived'] = y_pred
```

## ⑨ 決定木モデルの予測結果(精度)を確認しよう

### 予測結果の精度を確認する

```Python
# 提出用のデータマートを作成する
pred_df = test[['PassengerId','Survived']].set_index('PassengerId')
```

```Python
# 予測結果を整数に変換する
pred_df.Survived = pred_df.Survived.astype(int)
```

```Python
# CSVの作成
pred_df.to_csv('submission_v1.csv', index_label=['PassengerId'])
```

Kaggleに提出し、精度の確認  
[結果](result.png) 68.899%
