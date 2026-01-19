# DeepX OCR Server Web UI

基于 Gradio 框架的 OCR 在线演示界面，用于 DeepX OCR Server。

## 📋 目录

- [功能特性](#-功能特性)
- [前置条件](#-前置条件)
- [安装和设置](#-安装和设置)
- [运行方式](#-运行方式)
- [环境变量配置](#-环境变量配置)
- [目录结构](#-目录结构)
- [使用说明](#-使用说明)
- [常见问题](#-常见问题)

## ✨ 功能特性

- **支持多种文件格式**: JPG, PNG, JPEG, PDF
- **图像处理选项** (Module Selection): 
  - 图像方向矫正 (Image Orientation Correction)
  - 图像扭曲矫正 (Image Distortion Correction)
  - 文本行方向矫正 (Text Line Orientation Correction)
- **OCR 参数调整** (OCR Settings): 
  - 文本检测像素阈值 (Text Detection Pixel Threshold): 0~1
  - 文本检测框阈值 (Text Detection Box Threshold): 0~1
  - 扩张系数 (Expansion Coefficient): 1.0~3.0
  - 文本识别置信度阈值 (Text Recognition Score Threshold): 0~1
- **PDF 处理** (PDF Settings): 
  - 可调节渲染 DPI (72-300)，默认 150
  - 可设置最大处理页数 (1-100)，默认 10
- **结果展示**: 
  - 可视化 OCR 结果图像 (OCR Tab)
  - JSON 格式数据 (JSON Tab)
  - 完整结果 ZIP 下载 (包含 OCR 图像、原始图像、JSON 数据)
- **响应式 UI**: 
  - 侧边栏折叠功能 (HIDE/SHOW LEFT MENU)
  - 移动端适配
  - 自定义 PaddleOCR 风格主题

## 🔧 前置条件

### 1. 运行 OCR Server

此 Web UI 需要与后端 OCR 服务器通信，请先启动 OCR 服务器：

```bash
cd /home/deepx/Desktop/ocr_demo

# 设置环境变量
source ./set_env.sh 1 2 1 3 2 4

# 启动服务
cd build_Release
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/../3rd-party/pdfium/lib ./bin/ocr_server --port 8080
```

### 2. 验证服务运行

```bash
curl http://localhost:8080/health
# 响应: {"status": "healthy", "service": "DeepX OCR Server", "version": "1.0.0"}
```

### 3. 系统要求

- **Python**: 3.10 或更高版本
- **内存**: 最少 2GB RAM
- **磁盘空间**: 约 500MB

## 📦 安装和设置

### 1. 进入 WebUI 目录

```bash
cd /home/deepx/Desktop/ocr_demo/server/webui
```

### 2. 创建 Python 虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境 (Linux/macOS)
source venv/bin/activate

# 激活虚拟环境 (Windows)
venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**主要依赖** (requirements.txt):
- `pillow>=10.0.0`: 图像处理
- `requests>=2.31.0`: HTTP 通信
- `gradio==5.30.0`: Web UI 框架

## 🚀 运行方式

### 1. 基本运行 (本地 OCR Server)

当 OCR 服务器运行在 `localhost:8080` 时：

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行应用
python app.py
```

### 2. 指定自定义 OCR Server URL

当 OCR 服务器运行在不同主机或端口时：

```bash
# 通过环境变量指定 API URL
export API_URL="http://192.168.1.100:8080/ocr"
export API_BASE="http://192.168.1.100:8080"
python app.py
```

### 3. 访问 Web 界面

服务启动后，在浏览器中访问：

```
http://localhost:7860
```

## 🔑 环境变量配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `API_URL` | OCR API 端点 | `http://localhost:8080/ocr` |
| `API_BASE` | OCR 服务器基础 URL | `http://localhost:8080` |
| `API_TOKEN` | API 认证令牌 | `deepx_token` |

### 配置示例

```bash
# 方式1: 命令行设置 (推荐)
export API_URL="http://192.168.1.100:8080/ocr"
export API_BASE="http://192.168.1.100:8080"
export API_TOKEN="your_token_here"
python app.py

# 方式2: 使用脚本文件
cat > run_webui.sh << EOF
#!/bin/bash
export API_URL="http://192.168.1.100:8080/ocr"
export API_BASE="http://192.168.1.100:8080"
export API_TOKEN="your_token_here"
python app.py
EOF
chmod +x run_webui.sh
./run_webui.sh
```

## 📁 目录结构

```
webui/
├── app.py              # 主应用 (Gradio UI)
├── requirements.txt    # Python 依赖
├── README.md           # 本文档
├── examples/           # 图片示例文件 (8 个)
│   ├── ancient_demo.png
│   ├── handwrite_ch_demo.png
│   ├── handwrite_en_demo.png
│   ├── japan_demo.png
│   ├── magazine.png
│   ├── pinyin_demo.png
│   ├── research.png
│   └── tech.png
├── examples_pdf/       # PDF 示例文件 (10 个)
│   ├── 1251647.pdf
│   ├── 3M-7770.pdf
│   ├── 438417-cap-prr-receipt.pdf
│   ├── 6275314-011414-Board-Meeting-Minutes-Approved.pdf
│   ├── BVRC_Meeting_Minutes_2024-04.pdf
│   ├── jresv101n1p69_A1b.pdf
│   ├── meeting_minutes_september_30_2020.pdf
│   ├── MiscMssLempereur_27.pdf
│   ├── physics0409110.pdf
│   └── Yinglish_Mikado Song Text comparison...pdf
└── res/                # 资源文件
    └── img/            # Banner 图片资源
        ├── deepx-baidu-pp-banner.png
        └── DEEPX-Banner-CES-2026-01.png
```

## 🎯 使用说明

### 1. 上传文件
- **拖拽上传**: 将文件拖拽到 "📁 Input File" 上传区域
- **点击上传**: 点击上传区域选择文件
- **示例选择**: 
  - 点击 "📷 Image Examples" 下方的示例图片
  - 点击 "📄 PDF Examples" 下方的示例 PDF

### 2. 调整参数 (⚙️ Settings)
- **Module Selection (模块选择)**:
  - Image Orientation Correction: 图像方向矫正
  - Image Distortion Correction: 图像扭曲矫正
  - Text Line Orientation Correction: 文本行方向矫正
- **OCR Settings (OCR 参数)**:
  - Text Detection Pixel Threshold (0.30): 文本检测像素阈值
  - Text Detection Box Threshold (0.60): 文本检测框阈值
  - Expansion Coefficient (1.5): 扩张系数
  - Text Recognition Score Threshold (0.00): 文本识别置信度阈值
- **PDF Settings (PDF 设置)**:
  - PDF Render DPI (150): 渲染分辨率
  - PDF Max Pages (10): 最大处理页数

### 3. 解析文档
- 点击 "🚀 Parse Document" 按钮开始 OCR 处理
- 处理过程中会显示加载动画

### 4. 查看结果 (📋 Results)
- **OCR Tab**: 带检测框的可视化图像，多页时左侧显示缩略图
- **JSON Tab**: 结构化的识别结果数据
- **下载**: 点击 "📦 Download Full Results (ZIP)" 打包下载所有结果

### 5. 展开结果视图
- 点击左侧边缘的 "HIDE LEFT MENU" 按钮可隐藏左侧菜单，全屏查看结果
- 再次点击 "SHOW LEFT MENU" 可恢复左侧菜单

## 🐛 常见问题

### 1. 连接 OCR Server 失败

**症状**: `API 请求失败` 或 `OCR processing failed` 错误

**解决方案**:
```bash
# 检查 OCR 服务器状态
curl http://localhost:8080/health

# 如果服务器未运行，启动它
cd /home/deepx/Desktop/ocr_demo/build_Release
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/../3rd-party/pdfium/lib ./bin/ocr_server --port 8080
```

### 2. 端口冲突

**症状**: `Address already in use` 错误

**解决方案**:
```bash
# 检查占用端口 7860 的进程
lsof -i :7860

# 终止占用进程
kill -9 <PID>

# 或修改 app.py 末尾的 server_port 使用不同端口
```

### 3. 依赖安装失败

**解决方案**:
```bash
# 升级 pip
pip install --upgrade pip

# 单独安装依赖
pip install pillow>=10.0.0
pip install requests>=2.31.0
pip install gradio==5.30.0
```

### 4. PDF 处理内存不足

**症状**: 处理大型 PDF 时程序崩溃或响应缓慢

**解决方案**:
- 降低 PDF Render DPI (建议 150)
- 减少 PDF Max Pages (建议 10 以内)
- A4 页面 @ 150 DPI 约占用 8.7MB/页

### 5. 示例文件无法显示

**解决方案**:
```bash
# 确保 examples/ 和 examples_pdf/ 目录存在且有文件
ls -la examples/
ls -la examples_pdf/

# 确保文件权限正确
chmod 644 examples/*
chmod 644 examples_pdf/*
```

