import pickle
import pandas as pd
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt
import openai
import time
from IPython.display import clear_output

openai.api_key = "Private Key"


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

df = pd.read_pickle('data\df_pseudo_1k_missing.pkl')
df = df[df['description'] != ""].reset_index()

embeddings_list = []
st = time.time()

for i in range(100000, len(df)):
  if (i+1) % 100 == 0:
    time_diff = (time.time()-st)/60
    pct = round((i+1)/(len(df)), 5)
    eta = (time_diff / pct) - time_diff
    clear_output()
    print(f"{i+1} / {len(df)} finished, {time_diff // 60 } hours and {time_diff % 60} minutes done, ETA: {eta // 60} hours and {eta % 60} minutes, PCT done: {pct*100}%")

  embeddings_list.append(get_embedding(df['description'][i]))

# df['description'].apply(lambda x: get_embedding(x)).tolist()
# embeddings_list = df.apply(lambda row: get_embedding(row['description'] if pd.notnull(row['description']) else row['cmpdname']), axis=1).tolist()

# import pickle
# file = open("drive/MyDrive/embeddings_openai_save_59999.pickle", "rb")
# embeddings = pickle.load(file)
# file.close()

# import numpy as np
# np.array(embeddings).shape

file = open("data\embeddings_openai.pickle", "wb")
pickle.dump(embeddings_list, file)
file.close()