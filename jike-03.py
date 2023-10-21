import azure_openai as azure

prompt = '请你用朋友的语气回复给到客户，并称他为“亲”，他的订单已经发货在路上了，预计在3天之内会送达，订单号2021AEDG，我们很抱歉因为天气的原因物流时间比原来长，感谢他选购我们的商品。'
messages = [{"role": "system", "content": "你是一个由openai训练出来的智能ai，能快速解答用户提出的问题"}]
messages.append({ "role": "user", "content": prompt });

response = azure.get_openai().ChatCompletion.create(
  engine = azure.deployment_name,
  messages = messages
)

print("\n" + response['choices'][0]['message']['content'] + "\n")