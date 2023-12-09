import pandas as pd
import numpy as np
import pickle
from tenacity import retry, wait_random_exponential, stop_after_attempt
import time
from IPython.display import clear_output

import openai
openai.api_key = "Private Key"


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

df = pd.read_pickle('data\description_df.pkl')

embeddings_list = []
st = time.time()
total_char_num = 0

for i in range(len(df)):
  total_char_num += len(df['description'][i])
  if (i+1) % 20000 == 0:
    file = open(f"data\embeddings_openai_save_{i}.pickle", "wb")
    pickle.dump(embeddings_list, file)
    file.close()
  if (i+1) % 100 == 0:
    time_diff = (time.time()-st)/60
    pct = round((i+1)/(len(df)), 5)
    eta = (time_diff / pct) - time_diff
    token_num = total_char_num / 4
    accumulated_price = 0.0001 * token_num / 1000
    clear_output()
    print(f"{i+1} / {len(df)} finished, {time_diff} minutes done, ETA: {eta}, PCT done: {pct*100}%, price: ${accumulated_price}/$5")
  embeddings_list.append(get_embedding(df['description'][i]))

print("Dumping")
pickle.dump(embeddings_list, open('data\openai_embeddings.pkl'))
print("Done")
