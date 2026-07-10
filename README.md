# AI Warehouse Analyst

Чат-бот и аналитическая платформа для управления складскими запасами и каталогом запчастей: прогнозирование спроса, ABC-XYZ классификация, RAG-поиск по каталогу SKU и естественно-языковые ответы на вопросы о запасах.

## Зачем этот проект

Пет-проект вырос из реальных задач по управлению запасами в ритейле/логистике (ABC-XYZ анализ, расчёт точки перезаказа, прогнозирование спроса для каталога ~2 300–9 600 SKU). Цель — собрать в одном инструменте:

- ML-прогнозирование спроса по SKU
- RAG-поиск по каталогу запчастей на естественном языке
- API + UI для доступа к аналитике без Excel

## Стек

| Слой | Технологии |
|---|---|
| Backend / API | FastAPI, asyncio |
| LLM / агент | Anthropic API (function calling / tool use) |
| RAG | ChromaDB, эмбеддинги |
| ML | scikit-learn (RandomForest, GradientBoosting), pandas, NumPy, joblib |
| UI | Streamlit |
| Данные | 1С ERP экспорты (Excel/CSV), SKU и каталог запчастей |
| Тесты | pytest |

## Архитектура

```
Пользователь → Streamlit UI → FastAPI backend
                                  ├── ML-модуль (прогноз спроса)
                                  ├── RAG-модуль (ChromaDB, поиск по каталогу)
                                  └── Anthropic API (function calling, ответы на естественном языке)
```

## Возможности

- **Прогнозирование спроса**: EWM-baseline, RandomForest, GradientBoosting; функция `predict_next_period()` для прогноза на следующий период по SKU
- **ABC-XYZ классификация**: автоматическая сегментация каталога по объёму продаж и стабильности спроса
- **RAG-поиск**: вопросы на естественном языке по каталогу запчастей ("какие аналоги есть у артикула X?")
- **Function calling агент**: LLM вызывает инструменты (поиск по каталогу, расчёт прогноза, точки перезаказа) для ответа на бизнес-вопросы

## Быстрый старт

```bash
git clone https://github.com/XID18/ai-warehouse-analyst.git
cd ai-warehouse-analyst
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env  # добавить ANTHROPIC_API_KEY

uvicorn app.main:app --reload       # backend на http://localhost:8000
streamlit run app/streamlit_app.py  # UI на http://localhost:8501
```

## Структура проекта

```
ai-warehouse-analyst/
├── app/
│   ├── main.py              # FastAPI приложение
│   ├── streamlit_app.py     # UI
│   └── agent.py             # Anthropic function calling агент
├── ml/
│   ├── forecast.py          # predict_next_period(), EWM/RF/GB модели
│   └── abc_xyz.py           # ABC-XYZ классификация
├── rag/
│   └── retriever.py         # ChromaDB индексация и поиск
├── data/                    # примеры данных (без реальных 1С-выгрузок)
├── tests/
│   └── test_forecast.py
├── requirements.txt
└── README.md
```

## Статус

🚧 В разработке — пет-проект для отработки связки Python / ML / FastAPI / RAG на реальных задачах управления запасами.

## Автор

Иракли Хубулашвили — Data/BI Analyst, 5+ лет в аналитике данных (ритейл, логистика).
[LinkedIn] · [Email]
