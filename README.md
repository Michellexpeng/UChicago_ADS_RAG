# UChicago ADS Q&A — Frontend

React + Vite + Tailwind frontend that connects to the FastAPI RAG backend.

## 快速启动

```bash
# 1. 安装依赖
npm install

# 2. 启动开发服务器
npm run dev
```

打开 http://localhost:5173

## 后端配置

默认连接 `http://localhost:8000`。  
如需更改，编辑 `.env` 文件：

```
VITE_API_URL=http://your-backend-url:8000
```

## 文件结构

```
src/
├── App.tsx                    # 主入口，API 调用逻辑
├── main.tsx                   # React 挂载点
├── styles/index.css           # Tailwind + 全局样式
└── components/
    ├── ChatMessage.tsx        # 消息气泡 + 来源链接
    ├── ChatInput.tsx          # 输入框 + 发送按钮
    └── SampleQuestions.tsx    # 左侧问题面板
```

## 生产构建

```bash
npm run build   # 输出到 dist/
```
