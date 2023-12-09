import textwrap
import pickle
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_pickle("data\description_df.pkl")

embed_2d = pickle.load(open("data\embeddings_bert_embed_2d.pkl", "rb"))
labels = pickle.load(open("data\embeddings_bert_cluster_labels.pkl", "rb"))
df['description'] = ["{}".format(textwrap.fill(df['description'][i]).replace("\n", "<br>")) for i in range(len(df))]
num_clusters = 3000
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
labels = kmeans.fit_predict(embed_2d)
fig = px.scatter(
    x=embed_2d[:, 0], y = embed_2d[:, 1],
    color=labels,
    hover_data=[df['cmpdname'], df['description']]
)
fig.show()