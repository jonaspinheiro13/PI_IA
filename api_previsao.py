from pathlib import Path
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd

from prever_ia import prever_intervalo

ARQUIVO_MODELO = Path(r".\modelos\modelo_demanda_ifood.joblib").resolve()

app = FastAPI(title="API Previsões (Local)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "modelo_existe": ARQUIVO_MODELO.exists()}

@app.get("/prever")
def prever(
    inicio: str = Query(..., description="AAAA-MM-DD"),
    fim: str = Query(..., description="AAAA-MM-DD"),
    modo: str = Query("agregado", regex="^(agregado|diario)$"),
):
    if not ARQUIVO_MODELO.exists():
        raise HTTPException(500, f"Modelo não encontrado: {ARQUIVO_MODELO}")

    try:
        pd.to_datetime(inicio).normalize()
        pd.to_datetime(fim).normalize()
    except Exception:
        raise HTTPException(422, "Datas inválidas. Use AAAA-MM-DD.")

    df = prever_intervalo(str(ARQUIVO_MODELO), inicio, fim, modo=modo)

    return df.to_dict(orient="records")

app.mount("/web", StaticFiles(directory="web", html=True), name="web")
