
# Please make sure the requests library is installed
# pip install requests
import os
import base64
import requests


# API_URL 及 TOKEN 请访问 https://aistudio.baidu.com/paddleocr/task，在 API 调用示例中获取。
API_URL = "http://localhost:8080"
TOKEN = "test_token_123"

file_path = "../images/image_10.png"
input_filename = os.path.splitext(os.path.basename(file_path))[0]

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

headers = {
    "Authorization": f"token {TOKEN}",
    "Content-Type": "application/json"
}

required_payload = {
    "file": file_data,
    "fileType": 1,  # For PDF documents, set `fileType` to 0; for images, set `fileType` to 1
}

optional_payload = {
    "useDocOrientationClassify": True,
    "useDocUnwarping": True,
    "useTextlineOrientation": True,
}

payload = {**required_payload, **optional_payload}

response = requests.post(f"{API_URL}/ocr", json=payload, headers=headers)

assert response.status_code == 200
result = response.json()["result"]

print("\n========== OCR Results ==========")
for i, res in enumerate(result["ocrResults"]):
    print(f"\n[Result {i+1}]")
    print(res["prunedResult"])