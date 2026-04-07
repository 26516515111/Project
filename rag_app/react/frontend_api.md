# RAG 前端对接文档（增量上传 + SSE 流式问答）

## 接口总览

- 基础地址：`http://localhost:8000`
- 增量上传：`POST /rag/incremental/upload`
- 流式问答：`POST /rag/query/stream`
- 普通问答（可选）：`POST /rag/query`

## 1. 增量上传接口

- URL：`POST /rag/incremental/upload`
- Content-Type：`multipart/form-data`
- 用途：上传用户增量文件并触发索引更新

参数表：

- `user_id`（string，必填）：用户标识，用于会话和增量文件目录隔离
- `file`（file，必填）：增量文档，支持 `.txt`、`.md`、`.pdf`

请求示例（curl）：

```bash
curl -X POST "http://localhost:8000/rag/incremental/upload" \
  -F "user_id=u001" \
  -F "file=@./data/docs/new_case.md"
```

成功响应示例：

```json
{
  "ok": true,
  "user_id": "u001",
  "file_name": "1719999999_new_case.md",
  "saved_path": "data/docs/incremental/u001/1719999999_new_case.md",
  "message": "incremental file indexed"
}
```

失败响应示例：

```json
{
  "detail": "unsupported file type: .docx. allowed: ['.md', '.pdf', '.txt']"
}
```

## 2. 流式问答接口（SSE JSON 事件）

- URL：`POST /rag/query/stream`
- Content-Type：`application/json`
- Accept：`text/event-stream`
- 用途：逐 token 返回答案，并在末尾返回参考文献/图谱/结束状态

请求参数（JSON Body）：

- `user_id`（string，必填）：用户标识（映射为会话 ID）
- `question`（string，必填）：用户问题
- `top_k`（number，可选，默认 5）：检索数量
- `use_kg`（boolean，可选，默认 true）：是否启用知识图谱
- `use_llm`（boolean，可选，默认 true）：是否启用 LLM 生成（关闭后使用抽取式回答）
- `use_history`（boolean，可选，默认 true）：是否启用历史对话
- `enable_decompose`（boolean，可选）：是否启用问题分解
- `enable_retrieval_optimization`（boolean，可选，默认 true）：是否启用检索增强（前检索分解 + 后检索重排）

请求示例：

```json
{
  "user_id": "u001",
  "question": "主柴油机报警如何排查？",
  "top_k": 5,
  "use_kg": true,
  "use_llm": true,
  "enable_retrieval_optimization": true,
  "use_history": true
}
```

## 3. SSE 事件字段

### 3.1 `meta`

- 作用：请求元信息
- `data` 字段：
  - `user_id`（string）
  - `question`（string）
  - `stream`（boolean）

示例：

```text
event: meta
data: {"user_id":"u001","question":"主柴油机报警如何排查？","stream":true}
```

### 3.2 `token`

- 作用：增量答案片段
- `data` 字段：
  - `text`（string）

示例：

```text
event: token
data: {"text":"先检查报警来源回路..."}
```

### 3.3 `references`

- 作用：返回参考文献列表
- `data` 字段：
  - `citations`（array）
    - `doc_id`（string）
    - `source`（string）
    - `score`（number）
    - `text`（string）

示例：

```text
event: references
data: {"citations":[{"doc_id":"xxx","source":"manual.md","score":0.82,"text":"..."}]}
```

### 3.4 `kg`（可选）

- 作用：返回知识图谱三元组
- `data` 字段：
  - `triplets`（array<object>）

### 3.5 `done`

- 作用：流结束标记
- `data` 字段：
  - `finish_reason`（string）
  - `elapsed_ms`（number）
  - `citation_count`（number）

示例：

```text
event: done
data: {"finish_reason":"stop","elapsed_ms":1280,"citation_count":4}
```

### 3.6 `error`

- 作用：错误事件
- `data` 字段：
  - `message`（string）
  - `type`（string）

## 4. 前端状态管理建议

- `token`：追加到 `answer`
- `references`：设置 `citations`
- `kg`：设置 `kgTriplets`
- `done`：结束 loading
- `error`：展示错误并结束 loading

建议统一状态结构：

```json
{
  "answer": "",
  "citations": [],
  "kgTriplets": [],
  "meta": null,
  "done": null,
  "error": null,
  "loading": true
}
```

## 5. 调试与联调

- Swagger：`http://localhost:8000/docs`
- OpenAPI：`http://localhost:8000/openapi.json`
- Python 最小联调：

```bash
python test_react.py --user-id u001 --question "主机报警如何排查"
```
