import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv
# 1. 读取项目根目录里的 .env
load_dotenv()
# 2. 从 .env 里拿配置
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model = os.getenv("LLM_DEFAULT_MODEL")
# 3. 先看看配置有没有成功读取
print("API Key 是否读取成功：",bool(api_key))
print("API 地址：", base_url)
print("模型：", model)
# 4. 拼接真正请求大模型的地址
url = f"{base_url.rstrip('/')}/chat/completions"
# 5. 准备发送给大模型的数据
data = {
    "model": model,
    "messages":[{
        "role":"user",
        "content":"只回答两个字：成功"
    }]
}
# 6. 创建 HTTP 请求
request = urllib.request.Request(url, json.dumps(data).encode("utf-8"),
                                 headers={"Authorization": f"Bearer {api_key}",
                                 "Content-Type":"application/json"},
                                 method="POST")
# 7. 真正发送请求
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))

    print("API 调用成功")
    print("模型回答：", result["choices"][0]["message"]["content"])

except urllib.error.HTTPError as e:
    print("API 调用失败")
    print("状态码：", e.code)
    print("错误信息：", e.read().decode("utf-8"))

except Exception as e:
    print("发生其他错误：", e)