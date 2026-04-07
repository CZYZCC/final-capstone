import requests
import json
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

def check_deepseek_balance():
    # 填入你的 API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    url = "https://api.deepseek.com/user/balance"
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    print("正在查询 DeepSeek 账户余额...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ 查询成功！你的账户信息如下：")
        # 格式化打印返回的 JSON 数据
        print(json.dumps(data, indent=4, ensure_ascii=False))
    else:
        print(f"\n❌ 查询失败！状态码: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    check_deepseek_balance()