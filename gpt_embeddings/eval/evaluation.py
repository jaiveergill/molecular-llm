import openai
import pandas as pd

new = pd.read_pickle('question_new.pkl')
df = pd.read_pickle('df_pseudo_1k_missing.pkl')

bbbp = pd.read_csv('GNN_BBBP_Property_Final.csv')

for i in range(len(df)):
  df['fullDescription'][i] += bbbp['Permeability Description'][i]

openai.api_key = 'Private key'


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from deeplake.core.vectorstore import VectorStore
import os

from sentence_transformers import SentenceTransformer

bert_model = SentenceTransformer('recobo/chemical-bert-uncased')

def create_qa(embedding_url='', df_url='', embedding_model='', embedding_function='', column='description', db_store_path='db', llm='gpt-3.5-turbo-16k', embeddings_finished=True):
  import shutil
  print(f"Loading df from {df_url}...")
  df = pickle.load(open(f'{df_url}', 'rb'))
  print("df loaded...")

  print("=" * 50)
  print()
  if embeddings_finished:
    print("Embeddings pre-loaded")
    print(f"Loading embeddings from {embedding_url}...")
    embeddings = np.array(pickle.load(open(f'{embedding_url}', 'rb')))
    print("Embeddings loaded")
  else:
    print("Embeddings not pre-loaded")
    print("Creating embeddings...")
    embeddings = bert_model.encode(df[column])
    print("Embeddings created")

  source_text = 'data.txt'

  CHUNK_SIZE = 1000
  chunked_text = [df[column][i] for i in range(len(df))]

  print("=" * 50)
  print()

  print(f"Removing previous db from '{db_store_path}'")
  shutil.rmtree('db', ignore_errors=True)

  print(f"Creating VectorStore from {db_store_path}...")
  vector_store = VectorStore(
      path = db_store_path,
  )

  vector_store.add(text = chunked_text,
                  embedding = embeddings,
                  metadata = [{"source": source_text}]*len(chunked_text))


  print()
  print("=" * 50)
  db = DeepLake(dataset_path=db_store_path, read_only=True, embedding_function=embedding_function)
  print("Creating Retriever...")
  retriever = db.as_retriever()
  retriever.search_kwargs['distance_metric'] = 'cos'
  retriever.search_kwargs['k'] = 20

  print("=" * 50)
  print()
  print(f"Creating model as {llm}...")
  model = ChatOpenAI(model=llm, openai_api_key=openai.api_key) # 'gpt-3.5-turbo',
  qa = RetrievalQA.from_llm(model, retriever=retriever)
  print("END")
  print("=" * 50)
  print()
  return qa

def embedding_function(texts):
  pass

def embed_query(query):
    return bert_model.encode(query)
    # response = openai.Embedding.create(
    #   input=query,
    #   model="text-embedding-ada-002"
    # )
    # return response['data'][0]['embedding']

embedding_function.embed_query = embed_query

import numpy as np
import pickle
import datetime as dt

qa_full = create_qa(embedding_url='chem_bert_embeddings_full_description_1k_missing.pkl', df_url='df_pseudo_1k_missing.pkl', embedding_model=bert_model, embedding_function=embedding_function, column='fullDescription', db_store_path='db_full2', llm='gpt-3.5-turbo-16k')
def evaluate(query):
  response1 = qa_full.run("Create a comprehensive description for the following molecule including important properties. You can include density, solubilty, reactivity, and appearance among other properites you deem important. If there is no information about it in the context, never say nothing; instead use prior knowledge. If there is none, relate it to similar compounds and list these steps to make an educated guess on the properties of the compound" + query)
  response2 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Create a comprehensive description for the following molecule including important properties. You can include density, solubilty, reactivity, and appearance among other properites you deem important. If there is no information about it in the context, never say nothing; instead use prior knowledge. If there is none, relate it to similar compounds and list these steps to make an educated guess on the properties of the compound"},
                          {"role": "user", "content": query}
                ])
  return response1, response2['choices'][0]['message']['content']

import time
def calcProcessTime(starttime, cur_iter, max_iter):

    telapsed = time.time() - starttime
    testimated = (telapsed/(cur_iter+1))*(max_iter)

    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

    lefttime = testimated-telapsed  # in seconds

    return (int(telapsed), int(lefttime), finishtime)

new['chembert_results'] = [0 for i in range(len(new))]
new['base_gpt_results'] = [0 for i in range(len(new))]
st = time.time()
for i in range(len(new)):
  if i % 50 == 0:
    pickle.dump(new, open(f'new{i}.pkl', 'wb'))
  query = new['compdname'][i]
  try:
    response1, response2 = evaluate(query)
    new['chembert_results'][i] = response1
    new['base_gpt_results'][i] = response2
  except Exception as e:
    print("Retrying", e)
    i -= 1
    continue
    
  prstime = calcProcessTime(st,i,len(new))
  print(f"time elapsed: %s(s), time left: %s(s), estimated finish time: %s, i: {i}"%prstime)
  o = response1.split('\n')[0]
  print(f"latest response: {o}")

pickle.dump(new, open('eval_results_FINAL_chem.pkl', 'wb'))