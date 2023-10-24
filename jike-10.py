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
from llama_index.prompts import PromptTemplate


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


# 从文本直接创建index
def create_index():
    documents = SimpleDirectoryReader('./data/mr_fujino').load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist('mr_fujino.json')
    return index

# 从文件读取index
def create_index_from_storage(persist_dir = 'mr_fujino.json'):
    storage_context  = StorageContext.from_defaults(persist_dir = persist_dir)
    index = load_index_from_storage(storage_context)
    return index

def create_index_image():
    filename_fn = lambda filename: {'file_name': filename}
    receipt_reader = SimpleDirectoryReader(input_dir='./data/receipts', file_metadata=filename_fn)
    documents = receipt_reader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist('mr_fujino_receipts')
    return index

# 定义回答模板
def get_text_qa_template():
    text_qa_template_str = (
        "下面的“我”指的是鲁迅先生 \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "根据这些信息，请回答问题: {query_str} \n"
        "如果您不知道的话，请回答不知道\n"
    )
    text_qa_template = PromptTemplate(text_qa_template_str)
    return text_qa_template

def query_simplify(query):
    index = create_index_from_storage()

    query_engine = index.as_query_engine()
    answer = query_engine.query(query)

    print(answer.get_formatted_sources())
    print("query was:", query)
    print("answer was:", answer)

def query_with_template(query):
    index = create_index_from_storage()

    text_qa_template = get_text_qa_template()
    query_engine = index.as_query_engine(text_qa_template = text_qa_template)
    answer = query_engine.query(query)

    print(answer.get_formatted_sources())
    print("query was:", query)
    print("answer was:", answer)


def query_in_images():
    from llama_index.readers.file.base import DEFAULT_FILE_EXTRACTOR, ImageParser

    # index = create_index_image()
    index = create_index_from_storage(persist_dir = 'mr_fujino_receipts')
    query_engine = index.as_query_engine()

    # query = 'When was the last time I went to McDonald\'s and how much did I spend. Also show me the receipt from my visit.'
    query = "how many food I buy in McDonald? and show me the price of them"
    # query = "how much did I spend in McDonald? and show me the price of them"
    answer = query_engine.query(query)

    print("query was:", query)
    print("answer was:", answer)

# query_simplify('鲁迅先生去哪里学的医学')
# query_simplify('请问林黛玉和贾宝玉是什么关系')
# query_with_template('请问林黛玉和贾宝玉是什么关系')
query_in_images()