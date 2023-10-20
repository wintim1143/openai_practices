#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

deployment_name = "gpt-35"

conversation=[{"role": "system", "content": "你是一个由openai训练出来的能力强大的ai机器人，能快速回答用户提出的问题。如果你不知道答案，你可以直接跟用户数'我不清楚'"}]

while True:
    user_input = input('input:')      
    conversation.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        engine=deployment_name, # The deployment name you chose when you deployed the GPT-35-turbo or GPT-4 model.
        messages=conversation
    )

    conversation.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
    print("\n" + response['choices'][0]['message']['content'] + "\n")