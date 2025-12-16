# DeepX OCR Server

基于 Crow 框架的 OCR HTTP 服务。

## 编译

```bash
cd /home/deepx/Desktop/OCR
bash build.sh
```

## 启动

```bash
./build/bin/ocr_server
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-p, --port` | 服务端口 | 8080 |
| `-t, --threads` | 线程数 | 4 |
| `-v, --vis-dir` | 可视化输出目录 | output/vis |
| `-h, --help` | 帮助 | - |

## API

### POST /ocr

OCR 识别接口。

**请求头**

```
Content-Type: application/json
Authorization: token <任意字符串>
```

**请求体**

```json
{
    "file": "<base64编码的图像>",
    "visualize": true
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | string | 是 | Base64 编码的图像 |
| visualize | bool | 否 | 生成可视化图像，默认 false |

**响应**

```json
{
    "code": 0,
    "msg": "success",
    "data": {
        "results_num": 2,
        "results": [
            {
                "text": "识别的文字",
                "confidence": 0.98,
                "box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            }
        ],
        "vis_url": "/static/vis/ocr_vis_xxx.jpg"
    }
}
```

### GET /health

健康检查。

```json
{
    "status": "healthy",
    "service": "DeepX OCR Server",
    "version": "1.0.0"
}
```

### GET /static/vis/\<filename\>

访问可视化图像。

## 测试

```python
import requests
import base64

with open("test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://localhost:8080/ocr",
    headers={
        "Content-Type": "application/json",
        "Authorization": "token test"
    },
    json={"file": img_b64, "visualize": True}
)
print(resp.json())
```

## 输出

- 可视化图像: `output/vis/`
- 日志: `logs/deepx_ocr.log`
