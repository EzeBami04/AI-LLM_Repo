from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
import os
import pandas as pd
import logging
from langchain_openai import OpenAI
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from langchain_openai import OpenAIEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import  VectorParams, Distance

import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

openai_api_key = os.getenv("open_ai")
openai.api_key = openai_api_key

llm = OpenAI(model="gpt-4o", api_key=openai_api_key, temperature=0.7)
client = QdrantClient(host="localhost", port=6333)

default_args = {
    'owner': 'Trace',
    'email_on_failure': False,
    'email_on_retry': False
    }

collection = "trace"

@task
def load():
    df = pd.read_json("./data/traceminerals_data.json")
    if df.empty:
        raise ValueError("No data found in the specified directory.")
   
    texts = df['text'].astype(str).tolist()  
    return texts

def get_embedding(text: str):
    res = OpenAIEmbeddings(model="text-embeddings-large").embed_documents(text)
    return res[0] if res else None

@task
def vector_index(texts: list):

    if not client.get_collection(collection_name=collection, ignore_missing=True):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    parser = SentenceSplitter(chunk_size=500, chunk_overlap=10)
    docs = [Document(text=t) for t in texts]
    nodes = []
    for doc in docs:
        nodes.extend(parser.get_nodes_from_documents([doc]))

    ids = []
    vectors = []
    payloads = []

    for idx, node in enumerate(nodes):
        try:
            embedding = get_embedding(node.text)
            ids.append(idx)
            vectors.append(embedding)
            payloads.append({"text": node.text})
        except Exception as e:
            logging.error(f"Embedding failed for item {idx}: {e}")

    client.upsert(
        collection_name=collection,
        points=models.Batch(
            ids=ids,
            vectors=vectors,
            payloads=payloads
        )
    )
    logging.info(f"Upserted {len(ids)} vectors into collection '{collection}'.")

@dag(
    dag_id='trace_v1',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False
)
def trace_v1():
    texts = load()
    vector_index(texts)

trace_v1()
