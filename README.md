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
(Roda o codigo no terminal com a venv ativada)

**link para visualizar o site após subir a API**

http://127.0.0.1:8000/web/

# Estrutura do projeto:
O projeto possui três arquivos principais.
O treino_ia.py é responsável por treinar a inteligência artificial utilizando o histórico de vendas. Ao final do processo, ele gera dois arquivos: o .joblib, que representa a “memória” da IA — ou seja, tudo o que ela aprendeu —, e o arquivo de avaliação, que serve para medir a qualidade e o desempenho do aprendizado.
O prever_ia.py utiliza o modelo .joblib para realizar as previsões de vendas e salva os resultados na pasta saidas/, quando executado individualmente pelo terminal.
Por fim, o api_previsao.py disponibiliza as previsões por meio de uma API, permitindo que o site acesse e exiba os dados de forma visual. Ele se comunica com o prever_ia.py para gerar as previsões e, em seguida, envia essas informações ao painel do site.
