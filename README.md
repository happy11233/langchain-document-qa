# 智能文档问答系统

我的作业：基于 LangChain 和阿里云通义千问的智能文档问答系统

## 功能特点

- 📄 支持多种文档格式（PDF、Word、TXT、Markdown）
- 🤖 基于大语言模型的智能问答
- 💬 支持多轮对话
- 📚 支持多文档联合问答
- 🔍 显示答案来源
- 🔐 用户自定义 API Key（保护隐私）

## 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

## 运行

```bash
streamlit run app.py
```

## 使用方法

1. 在左侧边栏输入你的阿里云 API Key
2. 点击 "登录系统"
3. 上传文档
4. 开始提问

详细使用说明请查看 [使用说明.md](使用说明.md)

## 项目结构

```
Q&A system based on LangChain/
├── app.py                      # 主程序
├── core/                       # 核心模块
│   ├── document_processor.py  # 文档处理
│   ├── vector_store.py         # 向量存储
│   └── qa_engine.py            # 问答引擎
├── utils/                      # 工具函数
│   └── file_utils.py           # 文件工具
├── data/                       # 数据目录
│   └── chroma_db/              # 向量数据库
├── requirements.txt            # 依赖列表
└── 使用说明.md                 # 使用说明
```

## 技术栈

- **前端框架**: Streamlit
- **LLM 框架**: LangChain
- **向量数据库**: ChromaDB
- **大语言模型**: 阿里云通义千问
- **文档加载器**: PyPDF、docx2txt、UnstructuredMarkdown

## 注意事项

- 需要自己的阿里云 API Key
- 使用 API 会产生费用，但是一般会送一些额度用来测试
- API Key 不会被保存到文件，只在会话中有效

## License

MIT
