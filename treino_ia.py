# -*- coding: utf-8 -*-
# treino_ia.py (versão "menos chique")
# faz o mesmo que o original: treina por produto e salva um .joblib

import argparse, csv, joblib
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.experimental import enable_hist_gradient_boosting  # deixa aqui mesmo
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# tentei não mudar nada de importante, só o "jeitão" do código


def ler_csv_robusto(caminho):
    # tenta vários jeitos porque csv pt-br é cheio de pegadinha
    for sep in [",", ";"]:
        for enc in ["utf-8", "latin1", "cp1252"]:
            try:
                dfz = pd.read_csv(
                    caminho,
                    sep=sep,
                    engine="python",
                    encoding=enc,
                    quoting=csv.QUOTE_MINIMAL,
                    on_bad_lines="skip"
                )
                if dfz.shape[1] >= 3:
                    return dfz
            except Exception:
                # se der ruim, ignora e tenta o próximo
                pass

    # último recurso: abre na unha e separa por vírgula
    bruto = open(caminho, "r", encoding="utf-8", errors="ignore").read().splitlines()
    linhas = [linha.split(",") for linha in bruto]
    cab = linhas[0]
    corpo = linhas[1:]
    return pd.DataFrame(corpo, columns=cab)


def adicionar_atributos_tempo(df, coluna_data="data"):
    # basicão de features de tempo
    df["dia_semana"] = df[coluna_data].dt.weekday
    df["semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7.0)
    df["semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7.0)
    df["dia_mes"] = df[coluna_data].dt.day
    df["semana_ano"] = df[coluna_data].dt.isocalendar().week.astype(int)
    df["mes"] = df[coluna_data].dt.month
    df["ano"] = df[coluna_data].dt.year
    df["fim_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    # "ciclo de pagamento" meio tosco, mas funciona
    df["distancia_dia5"] = (df[coluna_data].dt.day - 5).abs().clip(0, 15)
    return df


def construir_lags_e_medias(df_produto, coluna_alvo="quantidade", lags=[1, 7, 14, 28], janelas=[7, 14, 28]):
    # ordena por data só pra garantir
    dfp = df_produto.sort_values("data").copy()

    # lags (valores defasados)
    for atraso in lags:
        nome_col = f"lag_{atraso}"
        dfp[nome_col] = dfp[coluna_alvo].shift(atraso)

    # médias móveis
    for janela in janelas:
        nome_media = f"media_{janela}"
        dfp[nome_media] = dfp[coluna_alvo].rolling(janela, min_periods=1).mean()

    # tendência simples: dias desde o começo
    dfp["tendencia"] = (dfp["data"] - dfp["data"].min()).dt.days
    return dfp


def carregar_series_diarias(caminho_itens_csv):
    itens = ler_csv_robusto(caminho_itens_csv)

    # limpa cabeçalho meio troncho
    itens.columns = [c.strip() for c in itens.columns]

    # confere as colunas que a gente precisa
    colunas_que_quero = {"DT_Abert", "Produto", "QTD"}
    if not colunas_que_quero.issubset(set(itens.columns)):
        raise RuntimeError(f"Esperava colunas {colunas_que_quero}. Recebi: {list(itens.columns)}")

    # arruma data/hora e quantidade
    itens["_datahora"] = pd.to_datetime(itens["DT_Abert"], format="%d/%m/%Y %H:%M", errors="coerce")
    itens["data"] = pd.to_datetime(itens["_datahora"].dt.date)
    itens["QTD"] = pd.to_numeric(itens["QTD"], errors="coerce").fillna(0.0)

    # vira diário por produto
    diario = (
        itens.groupby(["data", "Produto"], dropna=False)["QTD"]
        .sum()
        .reset_index()
        .rename(columns={"QTD": "quantidade"})
    )

    # gera um calendário completo por produto (pra preencher zeros)
    todas_datas = pd.date_range(diario["data"].min(), diario["data"].max(), freq="D")
    produtos = diario["Produto"].dropna().unique().tolist()

    base_tudo = pd.MultiIndex.from_product([todas_datas, produtos], names=["data", "Produto"]).to_frame(index=False)
    completo = (
        base_tudo.merge(diario, on=["data", "Produto"], how="left")
        .fillna({"quantidade": 0.0})
    )

    # mete as features de tempo
    completo = adicionar_atributos_tempo(completo, "data")

    # cria lags/médias por produto (um de cada vez mesmo)
    pedacos = []
    for prod in produtos:
        dfp = completo[completo["Produto"] == prod].copy()
        dfp = construir_lags_e_medias(dfp, coluna_alvo="quantidade")
        pedacos.append(dfp)

    juntao = pd.concat(pedacos, ignore_index=True)
    return juntao


def treinar_modelos(df_completo):
    modelos_por_produto = {}
    aval_linhas = []

    # lista dos campos que vão pro X
    recursos = [
        "dia_semana", "semana_sin", "semana_cos", "dia_mes", "semana_ano", "mes", "ano",
        "fim_de_semana", "distancia_dia5", "tendencia",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "media_7", "media_14", "media_28"
    ]

    # roda produto por produto
    for produto, dfp in df_completo.groupby("Produto", sort=False):
        dfp = dfp.copy()

        # corta o último mês pra validação
        corte = dfp["data"].max() - relativedelta(months=1)
        dfp["eh_treino"] = dfp["data"] <= corte

        treino = dfp[dfp["eh_treino"]].dropna(subset=recursos + ["quantidade"])

        # se tiver muito pouco dado, nem treina
        if len(treino) < 30:
            modelos_por_produto[produto] = {"modelo": None}
            aval_linhas.append({"produto": produto, "mae_holdout_ultimo_mes": np.nan})
            continue

        X_tr = treino[recursos].values
        y_tr = treino["quantidade"].values

        # modelo padrão que já tava ok
        mdl = HistGradientBoostingRegressor(max_depth=None, learning_rate=0.08, max_iter=600)
        mdl.fit(X_tr, y_tr)

        # calcula MAE no último mês (se tiver dado)
        mae = np.nan
        valid = dfp[~dfp["eh_treino"]].dropna(subset=recursos + ["quantidade"])
        if not valid.empty:
            yhat = mdl.predict(valid[recursos].values)
            mae = mean_absolute_error(valid["quantidade"].values, yhat)

        modelos_por_produto[produto] = {"modelo": mdl}
        aval_linhas.append({"produto": produto, "mae_holdout_ultimo_mes": mae})

    avaliacao_df = pd.DataFrame(aval_linhas)

    # uns metadados só pra saber o que rolou
    metadados = {
        "recursos": recursos,
        "treinado_em": datetime.utcnow().isoformat() + "Z",
        "periodo_treino_min": str(df_completo["data"].min().date()),
        "periodo_treino_max": str(df_completo["data"].max().date()),
        "qtd_produtos": int(df_completo["Produto"].nunique()),
    }

    return modelos_por_produto, metadados, avaliacao_df


def salvar_artefato(modelos_por_produto, metadados, historico, caminho_saida):
    # pacote completo: modelos + infos + histórico enxuto
    pacote = {
        "modelos": modelos_por_produto,
        "metadados": metadados,
        "historico": historico[["data", "Produto", "quantidade"]].copy()
    }
    joblib.dump(pacote, caminho_saida)


def main():
    # mesma CLI do original
    parser = argparse.ArgumentParser()
    parser.add_argument("--itens_csv", required=True, help="Caminho para CSV de Itens (Pedidos-Produtos)")
    parser.add_argument("--saida", required=True, help="Caminho do artefato .joblib")
    args = parser.parse_args()

    # sequência igualzinha
    df_completo = carregar_series_diarias(args.itens_csv)
    modelos_por_produto, metadados, avaliacao_df = treinar_modelos(df_completo)
    salvar_artefato(modelos_por_produto, metadados, df_completo, args.saida)

    # salva csv de avaliação do mesmo jeito
    caminho_avaliacao = args.saida.replace(".joblib", "_avaliacao.csv")
    avaliacao_df.to_csv(caminho_avaliacao, index=False)

    # prints iguais
    print("OK. Artefato salvo em:", args.saida)
    print("Avaliação por produto:", caminho_avaliacao)


if __name__ == "__main__":
    main()
