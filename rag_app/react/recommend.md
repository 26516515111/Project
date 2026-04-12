# 项目启动指南 (Frontend & Backend Startup Guide)

本文档推荐了前后端分离架构下的本地启动方法。

## 1. 后端启动 (Backend)

后端使用 Python 和 FastAPI 提供 API 服务，推荐在项目根目录 (`rag_app`) 下使用虚拟环境运行。

### 1.1 安装依赖

请确保你处于项目根目录下：

```bash
# 1. 创建虚拟环境 (可选但强烈推荐)
python -m venv .venv

# 2. 激活虚拟环境
# Windows 命令行 / PowerShell:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. 安装依赖包
pip install -r requirements.txt
```

### 1.2 启动服务

运行以下命令启动 FastAPI 后端服务：

```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```
服务启动后：
- API 服务地址：`http://localhost:8000`
- Swagger 接口文档：`http://localhost:8000/docs`

> 注意：如果不需要前后端分离，项目中也提供了一个基于 Streamlit 的独立 UI，可以通过在根目录运行 `streamlit run app_streamlit.py` 启动。

---

## 2. 前端启动 (Frontend)

前端使用 React + Vite 构建，所有相关代码均在 `react` 文件夹中。

### 2.1 安装依赖

请确保你的终端路径已切换到 `react` 文件夹下：

```bash
cd react

# 使用 npm 安装依赖
npm install
# 或者使用 yarn / pnpm
# yarn install
# pnpm install
```

### 2.2 启动服务

运行以下命令启动前端开发服务器：

```bash
npm run dev
```
服务启动后，Vite 将在终端输出本地访问地址（通常为 `http://localhost:5173`），你可以直接在浏览器中打开此地址进行访问。前端会自动与后端的 `http://localhost:8000` 端口进行交互。