"""
Агент на Anthropic API с function calling.

Отвечает на вопросы о складе на естественном языке, вызывая инструменты:
- поиск по каталогу (RAG)
- прогноз спроса по SKU
- расчёт ABC-XYZ класса
"""

import os
import json
import anthropic
import pandas as pd

from ml.forecast import predict_next_period
from rag.retriever import search_catalog

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

TOOLS = [
    {
        "name": "search_catalog",
        "description": "Поиск по каталогу запчастей на естественном языке",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Поисковый запрос"}},
            "required": ["query"],
        },
    },
    {
        "name": "predict_next_period",
        "description": "Прогноз спроса на следующий период для конкретной SKU",
        "input_schema": {
            "type": "object",
            "properties": {"sku": {"type": "string", "description": "Артикул SKU"}},
            "required": ["sku"],
        },
    },
]


def run_tool(name: str, tool_input: dict, history_df: pd.DataFrame) -> str:
    if name == "search_catalog":
        return json.dumps(search_catalog(tool_input["query"]), ensure_ascii=False)
    if name == "predict_next_period":
        sku_df = history_df[history_df["sku"] == tool_input["sku"]]
        forecast = predict_next_period(sku_df, tool_input["sku"])
        return json.dumps({"sku": tool_input["sku"], "forecast": forecast}, ensure_ascii=False)
    return json.dumps({"error": f"unknown tool {name}"})


def ask_agent(question: str, history_df: pd.DataFrame) -> str:
    """Отправляет вопрос агенту, обрабатывает вызовы инструментов, возвращает финальный ответ."""
    messages = [{"role": "user", "content": question}]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=TOOLS,
        messages=messages,
    )

    while response.stop_reason == "tool_use":
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": run_tool(block.name, block.input, history_df),
            }
            for block in tool_use_blocks
        ]
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    return "\n".join(text_blocks)
