from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from ml.forecast import predict_next_period
from ml.abc_xyz import classify_abc, classify_xyz, combine_abc_xyz
from app.agent import ask_agent

app = FastAPI(title="AI Warehouse Analyst")


class ChatRequest(BaseModel):
    question: str


class ForecastRequest(BaseModel):
    sku: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    history_df = pd.read_csv("data/sales_history.csv")
    answer = ask_agent(req.question, history_df)
    return {"answer": answer}


@app.post("/forecast")
async def forecast(req: ForecastRequest):
    history_df = pd.read_csv("data/sales_history.csv")
    sku_df = history_df[history_df["sku"] == req.sku]
    result = predict_next_period(sku_df, req.sku)
    return {"sku": req.sku, "forecast": result}


@app.get("/abc-xyz")
async def abc_xyz():
    sales_df = pd.read_csv("data/sales_summary.csv")
    history_df = pd.read_csv("data/sales_history.csv")

    abc = classify_abc(sales_df)
    xyz = classify_xyz(history_df)
    combined = combine_abc_xyz(abc, xyz)

    return combined.to_dict("records")
