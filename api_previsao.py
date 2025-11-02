# api_previsao.py (versão com cara de código humano)
from pathlib import Path
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd

# só usamos a função pronta de previsão
from prever_ia import prever_intervalo

# caminho do modelo (igual ao original)
ARQUIVO_MODELO = Path(r".\modelos\modelo_demanda_ifood.joblib").resolve()

# cria o app e libera CORS geralzão
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
    # ping básico pra saber se o modelo tá lá
    return {"ok": True, "modelo_existe": ARQUIVO_MODELO.exists()}

@app.get("/prever")
def prever(
    inicio: str = Query(..., description="AAAA-MM-DD"),
    fim: str = Query(..., description="AAAA-MM-DD"),
    modo: str = Query("agregado", regex="^(agregado|diario)$"),
):
    # confere se o arquivo do modelo existe
    if not ARQUIVO_MODELO.exists():
        raise HTTPException(500, f"Modelo não encontrado: {ARQUIVO_MODELO}")

    # validação bem direta das datas (só pra mensagem ficar amigável)
    try:
        pd.to_datetime(inicio).normalize()
        pd.to_datetime(fim).normalize()
    except Exception:
        raise HTTPException(422, "Datas inválidas. Use AAAA-MM-DD.")

    # chama a função que já faz todo o serviço
    df = prever_intervalo(str(ARQUIVO_MODELO), inicio, fim, modo=modo)

    # retorna listinha de dict (compatível com o original)
    return df.to_dict(orient="records")

# serve os arquivos estáticos do seu site em /web
app.mount("/web", StaticFiles(directory="web", html=True), name="web")
