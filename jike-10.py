import os
import json
import openai
from llama_index.llms import AzureOpenAI
from llama_index import set_global_service_context
from llama_index.embeddings import OpenAIEmbedding
from llama_index import VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage, SimpleDirectoryReader, QuestionAnswerPrompt
import logging
import sys
import azure_openai as azure
from llama_index.storage.docstore import SimpleDocumentStore

#### llama_index 中使用openAI与openAI是不一样的，生成向量和chat需要两个不同的deployment_name
#### https://gpt-index.readthedocs.io/en/latest/examples/customization/llms/AzureOpenAI.html

# 文件存储 @link https://docs.llamaindex.ai/en/stable/core_modules/data_modules/storage/docstores.html


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

api_key = os.getenv("AZURE_OPENAI_KEY")
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_type = "azure"
api_version = "2023-05-15"

llm = AzureOpenAI(
    model="gpt-3.5-turbo",
    engine=azure.deployment_name,
    api_key=api_key,
    api_base=api_base,
    api_type=api_type,
    api_version=api_version,
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=azure.embedding_deployment_name,
    api_key=api_key,
    api_base=api_base,
    api_type=api_type,
    api_version=api_version,
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)
set_global_service_context(service_context)

# documents = SimpleDirectoryReader('./data/mr_fujino').load_data()
# index = VectorStoreIndex.from_documents(documents)
# index.storage_context.persist('mr_fujino.json')

storage_context  = StorageContext.from_defaults(persist_dir = 'mr_fujino.json')
index = load_index_from_storage(storage_context )

query = "鲁迅先生去哪里学的医学"
query_engine = index.as_query_engine()
answer = query_engine.query(query)

print(answer.get_formatted_sources())
print("query was:", query)
print("answer was:", answer)

