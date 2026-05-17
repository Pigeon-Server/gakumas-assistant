"""LLM 决策模块 — 基于大语言模型的培育策略系统。

模块结构:
  config.py           — LLM 配置（从 ConfigService 读取）
  client.py           — OpenAI API 客户端封装
  llm_strategy.py     — 统一决策入口（编排层）
  session_state.py    — 局内会话状态管理
  message_builder.py  — Jinja2 提示词渲染 + 消息组装
  llm_caller.py       — LLM 调用 + 响应解析工具
  decision_dumper.py  — 决策 JSON dump（离线调试）
  insight_store.py    — 策略洞察 CRUD + 检索
  insight_generator.py — 后台洞察生成 + LLM 自审
  summary_store.py    — 旧系统 stub（已废弃）
  prompt_renderer.py  — Jinja2 模板渲染引擎
  prompts/            — 提示词模板文件
"""
