import ast
import operator


class AgentToolManager:
    """面向知识库问答的轻量工具调用层。

    这里不用依赖模型函数调用能力，而是用可解释的规则触发工具；
    后续如果接支持 tool calling 的模型，可以把这些方法直接包装成 LangChain tools。
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def run(self, question, filters=None):
        question = question or ""
        tool_calls = []

        if self._should_list_documents(question):
            tool_calls.append({
                "tool": "list_documents",
                "input": {},
                "output": self.list_documents(filters)
            })

        if self._should_describe_tables(question):
            tool_calls.append({
                "tool": "describe_tables",
                "input": {},
                "output": self.describe_tables(filters)
            })

        if self._should_calculate(question):
            expression = self._extract_expression(question)
            if expression:
                tool_calls.append({
                    "tool": "calculator",
                    "input": {"expression": expression},
                    "output": self.calculate(expression)
                })

        return tool_calls

    def list_documents(self, filters=None):
        summaries = self.vector_store.get_source_summaries(filters=filters)
        if not summaries:
            return "当前过滤条件下没有找到文档。"

        lines = ["当前知识库文档:"]
        for item in summaries:
            lines.append(
                f"- {item['source_file']} | 类型: {item.get('file_type', 'unknown')} | "
                f"片段数: {item['chunks']} | 部门: {item.get('department', '未设置')} | "
                f"分类: {item.get('category', '未设置')}"
            )
        return "\n".join(lines)

    def describe_tables(self, filters=None):
        tables = self.vector_store.get_table_summaries(filters=filters)
        if not tables:
            return "当前过滤条件下没有找到 CSV/Excel 表格。"

        lines = ["表格知识概览:"]
        for table in tables:
            lines.append(
                f"- {table.get('source_file')} / {table.get('sheet_name', 'CSV')}: "
                f"{table.get('row_count', '?')} 行, {table.get('column_count', '?')} 列, "
                f"字段: {table.get('columns', '')}"
            )
        return "\n".join(lines)

    def calculate(self, expression):
        try:
            tree = ast.parse(expression, mode="eval")
            result = self._eval_ast(tree.body)
            return f"{expression} = {result}"
        except Exception as exc:
            return f"计算失败: {exc}"

    def _eval_ast(self, node):
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.Mod: operator.mod,
        }
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](self._eval_ast(node.left), self._eval_ast(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](self._eval_ast(node.operand))
        raise ValueError("只支持数字四则运算")

    def _should_list_documents(self, question):
        keywords = ["有哪些文档", "列出文档", "知识库里有什么", "上传了什么", "文档列表", "文件列表"]
        return any(keyword in question for keyword in keywords)

    def _should_describe_tables(self, question):
        keywords = ["excel", "csv", "表格", "工作表", "字段", "列名", "有哪些表", "数据表"]
        lowered = question.lower()
        return any(keyword in lowered for keyword in keywords)

    def _should_calculate(self, question):
        return any(keyword in question for keyword in ["计算", "等于", "+", "-", "*", "/", "%"])

    def _extract_expression(self, question):
        allowed = set("0123456789.+-*/()% ")
        expression = "".join(ch for ch in question if ch in allowed).strip()
        return expression if any(ch.isdigit() for ch in expression) else ""
