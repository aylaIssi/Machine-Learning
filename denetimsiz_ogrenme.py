################################
# GÖZETİMSİZ ÖĞRENME İLE MÜŞTERİ SEGMENTASYONU
################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama
# stratejileri belirlemek istiyor. Buna yönelik olarak müşterilerin
# davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre
# gruplar oluşturulacak.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel
# (hem online hem offline alışveriş yapan) olarak yapan müşterilerin
# geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı
# (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# Görev 1: Veriyi Hazırlama
###############################################################

# aykırı değerlerin baskılanması
# online-offline birleştirmesiyle değişken azaltma
# rfm değişkenlerinin oluşturulması
# tenure, recency değişkenlerinin oluşturulması


#imports

# pip install yellowbrick
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' %x)

# Adım 1: flo_data_20K.csv verisini okutunuz.
# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
# Not: Tenure (Müşterinin yaşı),
# Recency (en son kaç gün önce alışveriş yaptığı)
# gibi yeni değişkenler oluşturabilirsiniz.

df_ = pd.read_csv('flo_data_20k.csv', index_col=0)
df = df_.copy()

df.head()
df.shape
# (19945, 12)

df.info()
#  #   Column                             Non-Null Count  Dtype
# ---  ------                             --------------  -----
#  0   order_channel                      19945 non-null  object
#  1   last_order_channel                 19945 non-null  object
#  2   first_order_date                   19945 non-null  object
#  3   last_order_date                    19945 non-null  object
#  4   last_order_date_online             19945 non-null  object
#  5   last_order_date_offline            19945 non-null  object
#  6   order_num_total_ever_online        19945 non-null  float64
#  7   order_num_total_ever_offline       19945 non-null  float64
#  8   customer_value_total_ever_offline  19945 non-null  float64
#  9   customer_value_total_ever_online   19945 non-null  float64
#  10  interested_in_categories_12        19945 non-null  object

df.describe().T
#                                       count    mean     std    min     25%  \
# order_num_total_ever_online       19945.000   3.111   4.226  1.000   1.000
# order_num_total_ever_offline      19945.000   1.914   2.063  1.000   1.000
# customer_value_total_ever_offline 19945.000 253.923 301.533 10.000  99.990
# customer_value_total_ever_online  19945.000 497.322 832.602 12.990 149.980
#                                       50%     75%       max
# order_num_total_ever_online         2.000   4.000   200.000
# order_num_total_ever_offline        1.000   2.000   109.000
# customer_value_total_ever_offline 179.980 319.970 18119.140
# customer_value_total_ever_online  286.460 578.440 45220.130

df.isnull().any()
# False


# outlierları baskılama
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    return up_limit

def replace_with_thresholds(dataframe, variable):
    up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

df.describe().T
df_.describe().T

# Tarih değişkenlerini date'e çevirme

time_cols = [col for col in df.columns if 'date' in col]
df[time_cols] = df[time_cols].apply(pd.to_datetime)

df.info()


# Değişkenlerin indirgenmesi ve yeni değişkenlerin oluşturulması

df['total_customer_value'] = df['customer_value_total_ever_offline'] + \
                       df['customer_value_total_ever_online']

df['total_order'] = df['order_num_total_ever_online'] + \
                        df['order_num_total_ever_offline']

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)


# RFM - CLTV Metriklerinin hazırlanması

rfm = df.groupby("master_id").agg({"last_order_date": lambda last_order_date: (today_date - last_order_date.max()).days,
                             "total_order": lambda total_order: total_order,
                             "total_customer_value": lambda total_price: total_price})

rfm.columns = ['recency', 'frequency', 'monetary']


# RFM score
rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1])

rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5,
                                 labels=[1,2,3,4,5])

rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5])

rfm['rfm_score'] = rfm['recency_score'].astype(str) + \
                   rfm['frequency_score'].astype(str) + \
                   rfm['monetary_score'].astype(str)

rfm['rfm_score'] = rfm['rfm_score'].astype(int)
rfm = rfm.drop(['recency_score', 'frequency_score', 'monetary_score'], axis=1)


# tenure
df["tenure_weekly"] = ((today_date - df["first_order_date"]).dt.days)/7

df = df.merge(rfm, on= 'master_id', how='left')
df.columns

# Index(['order_channel', 'last_order_channel', 'first_order_date',
#        'last_order_date', 'last_order_date_online', 'last_order_date_offline',
#        'order_num_total_ever_online', 'order_num_total_ever_offline',
#        'customer_value_total_ever_offline', 'customer_value_total_ever_online',
#        'interested_in_categories_12', 'total_customer_value', 'total_order',
#        'tenure_weekly', 'recency', 'frequency', 'monetary', 'rfm_score'],
#       dtype='object')


###############################################################
# Görev 2: K-Means ile Müşteri Segmentasyonu
###############################################################

df.dtypes
# order_channel                                object
# last_order_channel                           object
# first_order_date                     datetime64[ns]
# last_order_date                      datetime64[ns]
# last_order_date_online               datetime64[ns]
# last_order_date_offline              datetime64[ns]
# order_num_total_ever_online                 float64
# order_num_total_ever_offline                float64
# customer_value_total_ever_offline           float64
# customer_value_total_ever_online            float64
# interested_in_categories_12                  object
# total_customer_value                        float64
# total_order                                 float64
# recency                                       int64
# frequency                                   float64
# monetary                                    float64
# tenure_weekly                               float64
# category                                     object


# Adım 1: Değişkenleri standartlaştırınız.

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
cat_cols = [col for col in df.columns if df[col].dtypes in ["object"]]
time_cols = [col for col in df.columns if 'date' in col]


### model_df
model_df = df[['total_customer_value', 'total_order', 'recency', 'tenure_weekly', 'rfm_score']]
model_num_cols = [col for col in model_df.columns if model_df[col].dtypes in ["int32","int64", "float64"]]


## model_df için korelasyon analizi
model_df.corr()

fig, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(model_df[model_num_cols].corr(), annot=True, fmt=".2f", ax=ax)
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

# Adım 2: Optimum küme sayısını belirleyiniz.

ssd = []
K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(model_df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show(block=True)
from sklearn.cluster import KMeans
kmeans = KMeans()
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
elbow = KElbowVisualizer(kmeans, k=(2, 30))
elbow.fit(model_df)
elbow.show(block=True)

elbow.elbow_value_
# 7


# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.

#### K means uygulaması

model_df.describe().T

sc = MinMaxScaler((0,1))
model_df = sc.fit_transform(model_df)
model_df[0:5]

kmeans = KMeans(n_clusters = elbow.elbow_value_, random_state=42).fit(model_df)
kmeans.get_params()
kmeans.inertia_
# 567.0885067982533

clusters = kmeans.labels_

model_df.shape
#Out[32]: (19945, 4)
df.shape
#Out[33]: (19945, 17)
df_.shape
#Out[34]: (19945, 11)

# Ham veri setindeki cluster çalışması
df_['segment'] = clusters
df_['segment'] = df_['segment'] + 1

df_[df_['segment'] == 1]
# [3546 rows x 12 columns]

df_[df_['segment'] == 2]
# [3780 rows x 12 columns]

for n in range(1,8):
    print(len(df_[df_['segment'] == n]))
# 3546
# 3780
# 3352
# 1271
# 1006
# 3608
# 3382


# Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.

#final df
df_.groupby('segment').agg(['count', 'mean', 'median'])

#tüm değişkenler
df['segment'] = clusters
df['segment'] = df['segment'] + 1
df.groupby('segment')[num_cols].agg(['count', 'mean', 'median'])


###############################################################
# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
###############################################################

# Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.

# model_df
from scipy.cluster.hierarchy import linkage
hc_average = linkage(model_df, "average")

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
dend = dendrogram(hc_average,
           truncate_mode="lastp",
           p=8,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=0.7, color='r', linestyle='--')
plt.show(block=True)
#5

# Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")
hi_clusters = cluster.fit_predict(model_df)

df["hi_cluster_no"] = hi_clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

#clusters = kmeans.labels_

#df["kmeans_cluster_no"] = df["kmeans_cluster_no"]  + 1
#df["kmeans_cluster_no"] = clusters

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure_weekly"]]
final_df["hi_cluster_no"] = hi_clusters
final_df.head()

# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.

final_df.groupby('hi_cluster_no').agg(['count', 'mean', 'median'])