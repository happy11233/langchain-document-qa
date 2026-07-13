# 智能文档问答系统

我的作业：基于 LangChain 和阿里云通义千问的智能文档问答系统

## 功能特点

- 📄 支持多种文档格式（PDF、Word、TXT、Markdown、CSV、Excel）
- 🤖 基于大语言模型的智能问答
- 💬 支持多轮对话
- 📚 支持多文档联合问答
- 🔍 显示答案来源
- 🧩 支持 metadata 过滤检索
- 🔎 支持混合检索（向量检索 + 关键词检索）
- 🎯 支持本地 rerank 重排序
- 🛠️ 支持轻量 Agent 工具调用（文档列表、表格概览、计算器）
- 📈 支持 RAG 评估用例批量测试
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
3. 上传文档，可填写部门、分类、标签等 metadata
4. 选择检索模式、过滤条件、是否启用 rerank/Agent 工具
5. 开始提问
6. 可在页面底部运行 RAG 评估

详细使用说明请查看 [使用说明.md](使用说明.md)

## 项目结构

```
Q&A system based on LangChain/
├── app.py                      # 主程序
├── core/                       # 核心模块
│   ├── document_processor.py   # 文档处理
│   ├── vector_store.py         # 向量存储
│   ├── qa_engine.py            # 问答引擎
│   ├── reranker.py             # 本地重排序
│   ├── agent_tools.py          # Agent 工具调用
│   └── evaluation.py           # RAG 评估
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
- **表格处理**: pandas、openpyxl、xlrd

## 增强 RAG 能力

### metadata 过滤

上传文档时可以设置：

- 部门：例如 `财务`、`销售`、`人事`
- 分类：例如 `制度`、`表格`、`产品`
- 标签：例如 `价格,库存`

提问时可以在侧边栏按这些 metadata 过滤检索范围。注意：这是演示级过滤，不等同于企业级登录鉴权或 RBAC。

### 混合检索

系统支持三种检索模式：

- 混合检索：同时使用向量相似度和关键词命中
- 向量检索：只使用语义相似度
- 关键词检索：只使用本地关键词匹配

混合检索适合同时覆盖“语义相近”和“字段/编号/文件名精确命中”的场景。

### rerank 重排序

系统内置一个轻量本地 rerank，会结合关键词覆盖、短语命中、文件名/表头命中和上游检索分数重新排序候选片段。它不是专用 cross-encoder 模型，但适合教学和轻量 demo。

### Agent 工具调用

目前内置三个工具：

- 文档列表：回答“知识库里有哪些文档”
- 表格概览：回答“有哪些表格/字段/工作表”
- 计算器：处理简单四则运算

### CSV/Excel 表格知识

上传 CSV/Excel 后，系统会生成：

- 表格摘要片段：文件名、工作表、行列数、字段、样例数据
- 表格数据片段：按批次把行数据转换成可检索文本

适合做表格问答和字段发现，不适合替代数据库级精确统计。如果要做大规模聚合、联表、写回新 Excel，建议增加专门的表格工具服务。

### RAG 评估

页面底部可以粘贴评估用例，支持 JSON、JSONL、CSV。字段：

- `question`：问题
- `expected_keywords`：期望答案包含的关键词，CSV 中多个关键词用 `|` 分隔
- `expected_source`：期望命中的来源文件名

系统会输出关键词命中率、来源命中率和每题召回片段。

## 注意事项

- 需要自己的阿里云 API Key
- 使用 API 会产生费用，但是一般会送一些额度用来测试
- API Key 不会被保存到文件，只在会话中有效
- 本项目仍是单机 Demo，不包含企业登录、部门权限硬隔离、审计日志和生产部署能力

## License

MIT
