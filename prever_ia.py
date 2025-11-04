# -*- coding: utf-8 -*-
import argparse, joblib
import pandas as pd
import numpy as np

def adicionar_atributos_tempo(df, coluna_data="data"):
    df["dia_semana"] = df[coluna_data].dt.weekday
    df["semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7.0)
    df["semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7.0)
    df["dia_mes"] = df[coluna_data].dt.day
    df["semana_ano"] = df[coluna_data].dt.isocalendar().week.astype(int)
    df["mes"] = df[coluna_data].dt.month
    df["ano"] = df[coluna_data].dt.year
    df["fim_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["distancia_dia5"] = (df[coluna_data].dt.day - 5).abs().clip(0, 15)
    return df

def prever_recursivo(historico_produto, data_inicio, data_fim, modelo, recursos):
    datas = pd.date_range(data_inicio, data_fim, freq="D")
    hist = historico_produto.sort_values("data").copy()
    serie = hist.set_index("data")["quantidade"].copy()
    linhas = []

    for d in datas:
        linha = pd.DataFrame({"data": [d]})
        linha = adicionar_atributos_tempo(linha, "data")
        linha["tendencia"] = (d - hist["data"].min()).days

        for atraso in [1, 7, 14, 28]:
            linha[f"lag_{atraso}"] = serie.get(d - pd.Timedelta(days=atraso), np.nan)

        for janela in [7, 14, 28]:
            vizinhos = [serie.get(d - pd.Timedelta(days=i), np.nan) for i in range(1, janela + 1)]
            viz_validos = [v for v in vizinhos if pd.notna(v)]
            linha[f"media_{janela}"] = float(np.mean(viz_validos)) if viz_validos else np.nan

        linha = linha[recursos].copy()

        if (modelo is None) or linha.isna().all(axis=None):
            base = hist.tail(28)["quantidade"].mean()
            yhat = float(base if pd.notna(base) else hist["quantidade"].mean())
        else:
            linha = linha.fillna(hist["quantidade"].mean())
            yhat = float(max(0.0, modelo.predict(linha.values)[0]))

        linhas.append({"data": d, "quantidade_prevista": yhat})
        serie.loc[d] = yhat

    return pd.DataFrame(linhas)

def carregar_modelo(caminho_modelo):
    pacote = joblib.load(caminho_modelo)
    return pacote["modelos"], pacote["metadados"], pacote["historico"]

def prever_intervalo(caminho_modelo, inicio, fim, modo="agregado"):
    modelos_por_produto, metadados, historico = carregar_modelo(caminho_modelo)
    recursos = metadados["recursos"]
    data_inicio = pd.to_datetime(inicio).normalize()
    data_fim = pd.to_datetime(fim).normalize()
    pedacos = []
    for produto, pacote in modelos_por_produto.items():
        hist_prod = historico[historico["Produto"] == produto][["data", "Produto", "quantidade"]]
        prev = prever_recursivo(hist_prod, data_inicio, data_fim, pacote["modelo"], recursos)
        prev["produto"] = produto
        pedacos.append(prev)

    saida = pd.concat(pedacos, ignore_index=True)

    if modo == "agregado":
        return (
            saida.groupby("produto", as_index=False)["quantidade_prevista"]
            .sum()
            .sort_values("quantidade_prevista", ascending=False)
        )
    else:
        return saida.sort_values(["produto", "data"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", required=True, help="Caminho para o artefato .joblib")
    parser.add_argument("--inicio", required=True, help="AAAA-MM-DD")
    parser.add_argument("--fim", required=True, help="AAAA-MM-DD")
    parser.add_argument("--modo", choices=["agregado", "diario"], default="agregado")
    parser.add_argument("--saida", required=True)
    args = parser.parse_args()

    df_prev = prever_intervalo(args.modelo, args.inicio, args.fim, modo=args.modo)
    df_prev.to_csv(args.saida, index=False)
    print("OK! Previsões em:", args.saida)

if __name__ == "__main__":
    main()
