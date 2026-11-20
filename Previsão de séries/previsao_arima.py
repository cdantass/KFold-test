import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

CAMINHO_CSV = Path("datasets") / "yahoo_stock.csv"

if not CAMINHO_CSV.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {CAMINHO_CSV.resolve()}"
    )

df = pd.read_csv(CAMINHO_CSV, index_col=False, sep=",")

df.drop(
    ["High", "Low", "Open", "Volume", "Adj Close"],
    inplace=True,
    axis=1
)

df.index = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.drop("Date", inplace=True, axis=1)

serie_original = df.iloc[:, 0]
serie_size = len(serie_original)
intervalo = 15

print(serie_original.tail())
print("Tamanho da série:", serie_size)


def previsao_arima(serie, intervalo, parametros):
    p, d, q = parametros

    serie_treino = serie.iloc[:-intervalo]
    serie_real = serie.iloc[-intervalo:]

    model = ARIMA(
        serie_treino.values,
        order=(p, d, q),
        trend="n"
    )

    model_fit = model.fit()
    previsto_vals = model_fit.forecast(steps=intervalo)

    previsto = Series(
        previsto_vals,
        index=serie_real.index
    )

    erro = np.abs(
        (previsto.values - serie_real.values)
        / serie_real.values
    )

    erro_medio = np.mean(erro)

    return abs(erro_medio), previsto, model_fit


v_p = [0, 1, 2, 3]
v_d = [1, 2]
v_q = [0, 1, 2, 3]

best_vlerro = 99
best_param = []
best_prev = None
modelo = None

for p, d, q in itertools.product(v_p, v_d, v_q):
    if p == 0 and q == 0:
        continue

    parametros = [p, d, q]

    try:
        vlerro, serie_prevista, modelo_atual = previsao_arima(
            serie_original,
            intervalo,
            parametros
        )
    except Exception as e:
        print("Falha com", parametros, "->", e)
        continue

    print(
        "ARIMA",
        parametros,
        "MAPE:",
        round(vlerro * 100, 4),
        "%"
    )

    if vlerro < best_vlerro:
        best_vlerro = vlerro
        best_param = parametros
        best_prev = serie_prevista
        modelo = modelo_atual

if best_prev is None or modelo is None:
    raise RuntimeError(
        "Não foi possível ajustar nenhum modelo ARIMA."
    )

x = best_prev.index
y = best_prev.values

plt.figure(figsize=(16, 8))

serie_original[
    serie_size - 2 * intervalo:serie_size
].plot(
    label="Série Real"
)

best_prev.plot(label="ARIMA Model")

plt.fill_between(
    x,
    y * 0.95,
    y * 1.05,
    color="g",
    alpha=0.1
)

plt.fill_between(
    x,
    y * 0.80,
    y * 1.20,
    color="r",
    alpha=0.1
)

plt.title(
    "MAPE Encontrado na Previsão (ARIMA): "
    + str(round(best_vlerro * 100, 2))
    + "%"
)

plt.xlabel("Data")
plt.ylabel("Valor de fechamento")
plt.legend()
plt.tight_layout()
plt.show()

print("Melhores Parametros: ")
print("(p, d, q) =", best_param)
print("MAPE =", round(best_vlerro * 100, 4), "%")

print("Analise de Ruido: ")

residuos = np.asarray(modelo.resid)
diferenciacao = best_param[1]

if len(residuos) > diferenciacao:
    residuos = residuos[diferenciacao:]

residuos = residuos[np.isfinite(residuos)]

teste_ruido = jarque_bera(residuos)

print("Estatística Jarque-Bera:", teste_ruido[0])
print("P-valor:", teste_ruido[1])
print("Assimetria:", teste_ruido[2])
print("Curtose:", teste_ruido[3])