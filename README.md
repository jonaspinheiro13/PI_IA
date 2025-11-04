# Versão do python a ser utilizada: Python 3.12.8

# Codigos para rodar o programa:
# 1-Treinar a IA
python treino_ia.py --itens_csv .\dados\Pedidos-Produtos-ifood-072024-082025-CSV.csv `
                    --saida .\modelos\modelo_demanda_ifood.joblib
                    
(Roda o codigo no terminal com a venv ativada)


# 2-Codigo para rodar o prever_ia.py individual do site
**Agregado (soma do mês por produto)**
python prever_ia.py --modelo .\modelos\modelo_demanda_ifood.joblib `
                    --inicio 2025-11-01 --fim 2025-11-30 `
                    --modo agregado `
                    --saida .\saidas\prev_2025-11_agregado.csv
                    
**Diário (linha por dia e produto)**
python prever_ia.py --modelo .\modelos\modelo_demanda_ifood.joblib `
                    --inicio 2025-11-01 --fim 2025-11-30 `
                    --modo diario `
                    --saida .\saidas\prev_2025-11_diario.csv

(Roda os codigos no terminal com a venv ativada)


# 3-Codigo para subir a API
python -m uvicorn api_previsao:app --reload --port 8000
**link para visualizar o site após subir a API**
http://127.0.0.1:8000/web/

(Roda o codigo no terminal com a venv ativada)

# Roda o codigo no terminal com a venv ativada
