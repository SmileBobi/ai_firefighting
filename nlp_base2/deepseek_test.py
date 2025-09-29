# 第一种访问方式：OpenAI
# from openai import OpenAI

# client = OpenAI(
#     base_url="http://localhost:11434/v1",  # Ollama默认API地址
#     api_key="test"  # 必填字段但会被忽略，随便填写即可
# )

# response = client.chat.completions.create(
#     model="deepseek-r1:7b",  # 替换为已下载的模型名称
#     messages=[
#         {"role": "system", "content": "你是一个有帮助的助手"},
#         {"role": "user", "content": "用50字解释量子计算"}
#     ],
#     temperature=0.7,
#     max_tokens=1024
# )

# print(response.choices[0].message.content)


# 第二种方式：requests
# import requests

# url = "http://localhost:11434/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json"
# }
# data = {
#     "model": "deepseek-r1:7b",
#     "messages": [{"role": "user", "content": "用50字解释量子计算"}]
# }
# response = requests.post(url, headers=headers, json=data)
# print(response.json())

# 方式三：ollama API
## 1、generate方式
import ollama

client = ollama.Client(host='http://localhost:11434')
# print('----------')
# models = client.list()
# print('Available models:', models)

# print('----------')
# response = client.generate(
# model="deepseek-r1:7b", # 指定模型名称
# prompt='你是谁'
# )
# print(response['response'])

## 2、chat方式
# response = client.chat(
# model="deepseek-r1:7b", # 指定模型名称
# messages=[{'role': 'user', 'content': '你是谁'}]
# )
# print(response['message']['content'])

response = client.chat(
model="qwen:0.5b", # 指定模型名称
messages=[{'role': 'user', 'content': '你是谁'}]
)
print(response['message']['content'])
