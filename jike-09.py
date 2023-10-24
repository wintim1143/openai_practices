import azure_openai as azure
import pandas as pd
import numpy as np

def generate_by_prompt(prompt):
    messages = [{ "role": "system", "content": "你是一个有OpenAI训练的智能AI" }]
    messages.append({ "role": "user", "content": prompt });
    return azure.chat_with_openai(messages)

def generate_product():
    prompt = """请你生成50条淘宝网里的商品的标题，每条在30个字左右，品类是3C数码产品，标题里往往也会有一些促销类的信息，每行一条。"""
    data = generate_by_prompt(prompt);
    product_names = data.strip().split('\n');
    df = pd.DataFrame({ 'product_name': product_names })
    df = df[df['product_name'] != '']
    df.product_name = df.product_name.apply(lambda x: x.split('.')[1].strip())
    df.head()

    clothes_prompt = """请你生成50条淘宝网里的商品的标题，每条在30个字左右，品类是女性的服饰箱包等等，标题里往往也会有一些促销类的信息，每行一条。"""
    clothes_data = generate_by_prompt(clothes_prompt)
    clothes_product_names = clothes_data.strip().split('\n')
    clothes_df = pd.DataFrame({'product_name': clothes_product_names})
    clothes_df = clothes_df[clothes_df['product_name'] != '']
    clothes_df.product_name = clothes_df.product_name.apply(lambda x: x.split('.')[1].strip())
    clothes_df.head()

    df = pd.concat([df, clothes_df], axis=0)
    df = df.reset_index(drop=True)
    print(df)
    return df



def generate_embeddings():
    df = generate_product()
    batch_size = 10
    prompts = df.product_name.tolist()
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    embeddings = []
    for batch in prompt_batches:
        batch_embeddings = azure.get_embeddings_with_openai(prompts=batch, batch_size=batch_size)
        embeddings += batch_embeddings
    df['embeddings'] = embeddings
    df.to_parquet("data/taobao_product_title.parquet", index=False)

# generate_embeddings()