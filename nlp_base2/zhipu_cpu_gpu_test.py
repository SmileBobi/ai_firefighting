from modelscope import AutoTokenizer, AutoModel, snapshot_download
import time
import os
# print(os.environ.get("MODELSCOPE_CACHE"))  # 检查环境变量值

model_dir = snapshot_download("zhipuAI/chatglm3-6b",revision = "v1.0.0")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

start_time = time.time()
# model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)                # 运行时间: 388.28605008125305 秒
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()    # 运行时间: 25.71428571428571 秒
model = model.eval()
response,history = model.chat(tokenizer,"如何测试程序运行时间",history=[])  
print(response)

end_time = time.time()
print("运行时间:",end_time - start_time,"秒")


