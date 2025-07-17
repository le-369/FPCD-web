# 🚗 假车牌车辆检测系统

一个基于 Flask 的前端 Web 应用程序，用于检测假车牌车辆并监控驾驶员安全（如安全带使用情况），支持 9 种功能，并提供可扩展接口，便于用户进一步开发和定制，适用于智能交通监控场景。

## 📌 项目简介

假车牌车辆——使用重复车牌的车辆——通常通过智能交通系统中的卡口摄像头检测。这些车辆的车牌号码相同，但外观特征（如**颜色**、**品牌**、**型号**和**车身形状**）存在差异。

本项目基于之前的[推理算法](https://github.com/le-369/Fake-plate-car-detection)进行了扩展，开发了一个基于 Flask 的 Web 应用程序，提供直观的前端界面，支持 9 种功能，包括车牌检测、车辆属性分析和安全带检测等。系统采用模块化设计，预留了可扩展接口，允许用户根据需求添加新功能。

## 📦 前置条件与安装

### 📥 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/le-369/FPCD-web.git
   cd FPCD-web
   ```

2. 安装依赖项：
   ```bash
   pip install -r requirements.txt
   ```

3. 下载预训练模型权重：
   - 访问 Google Drive 文件夹：[模型权重](https://drive.google.com/drive/folders/10aQi223yn6hZEjfldmuow7iQrSsMu_uW?usp=drive_link)
   - 下载所有权重文件并将其放置在项目目录的 `./weights` 文件夹中。
   - 确保权重文件的路径与推理脚本中的预期路径一致（例如 `./weights/car_belt.pth`、`./weights/car_brand.pth` 等）。

## 📂 项目结构

```bash
├── inference/             # 推理脚本模块（如车牌检测、车辆属性分析）
├── static/                # 静态文件（如 CSS、JavaScript、图片）
├── templates/             # HTML 模板文件（Flask 渲染的 Web 页面）
├── weights/               # 下载的模型权重文件夹
├── utils.py               # 工具函数（例如预处理、后处理）
├── start_server.py        # Flask Web 服务器启动脚本
├── requirements.txt       # 依赖包列表
└── README.md              # 项目文档
```

## 🖼️ 系统框架

![图1](frame.bmp "系统流程")


## 🚀 如何运行

### 启动 Web 服务器

1. 确保权重文件已下载至 `./weights` 文件夹。
2. 运行 Flask 服务器：
   ```bash
   python start_server.py
   ```
3. 打开浏览器，访问 `http://localhost:5000`（默认地址，可能因 `start_server.py` 配置而异）。
4. 通过 Web 界面上传图像，执行检测任务。

## 📈 功能特性

系统支持以下 9 种功能（包括但不限于）：
1. 车牌检测与 OCR
2. 车辆品牌识别
3. 车辆颜色分类
4. 车辆类型检测
5. 安全带使用检测
6. 实时套牌结果 (500ms,CPU)
7. 历史记录查询
8.  检测结果导出 (JSON 格式)
9.  可扩展接口

系统提供模块化架构，用户可通过扩展接口开发新功能，例如添加新的车辆属性检测或集成其他 AI 模型。


## 🛠️ 扩展开发

系统预留了可扩展接口，允许用户添加新功能：
- **添加新检测模块**：在 `inference` 文件夹中创建新脚本，遵循现有模块的输入/输出格式。
- **自定义 Web 界面**：修改 `templates` 中的 HTML 文件或 `static` 中的 CSS/JavaScript。
- **API 扩展**：在 `start_server.py` 中添加新的 Flask 路由以支持额外功能。

## 📮 联系方式

如有问题、反馈或合作意向，请联系： 
📧 [HELLOLE_369@126.com]