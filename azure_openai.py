#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
from openai.embeddings_utils import get_embeddings
import backoff

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

deployment_name = "gpt-35"

embedding_deployment_name = "gpt-Embedding"

def get_openai():
  return openai

def chat_with_openai(messages):
  response = openai.ChatCompletion.create(
    engine = deployment_name,
    messages = messages
  )
  return response['choices'][0]['message']['content']

def complete_with_openai(prompt):
  response = openai.Completion.create(
    engine = deployment_name,
    prompt = prompt
  )
  return response.choices[0].text

def get_embeddings_with_openai(prompts, batch_size):
  embeddings = []
  for i in range(0, len(prompts), batch_size):
    batch = prompts[i: i + batch_size]
    data = openai.Embedding.create(input=batch, engine=embedding_deployment_name).data
    # return 

    # data = get_embeddings(list_of_text=batch, engine=embedding_deployment_name)
    embeddings += [d["embedding"] for d in data]
  return embeddings