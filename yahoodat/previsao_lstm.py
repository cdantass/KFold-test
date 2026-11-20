import itertools
import os
import random
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import Series
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import jarque_bera
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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


def cria_janelas(valores, janela):
    X, y = [], []

    for i in range(janela, len(valores)):
        X.append(valores[i - janela:i])
        y.append(valores[i])

    return np.array(X), np.array(y)


def previsao_lstm(serie, intervalo, parametros):
    janela, unidades, dropout, epocas = parametros

    serie_treino = serie.iloc[:-intervalo]
    serie_real = serie.iloc[-intervalo:]

    retornos = np.diff(np.log(serie_treino.values))

    scaler = StandardScaler()
    retornos_esc = scaler.fit_transform(
        retornos.reshape(-1, 1)
    ).ravel()

    X, y = cria_janelas(retornos_esc, janela)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = keras.Sequential([
        layers.Input(shape=(janela, 1)),
        layers.LSTM(unidades),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    parada = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X,
        y,
        epochs=epocas,
        batch_size=32,
        validation_split=0.1,
        callbacks=[parada],
        verbose=0,
        shuffle=False
    )

    janela_atual = list(retornos_esc[-janela:])
    ret_previstos_esc = []

    for _ in range(intervalo):
        entrada = np.array(
            janela_atual[-janela:]
        ).reshape(1, janela, 1)

        prox = float(
            model.predict(entrada, verbose=0)[0, 0]
        )

        ret_previstos_esc.append(prox)
        janela_atual.append(prox)

    ret_previstos = scaler.inverse_transform(
        np.array(ret_previstos_esc).reshape(-1, 1)
    ).ravel()

    ultimo_preco = serie_treino.values[-1]
    precos = ultimo_preco * np.exp(np.cumsum(ret_previstos))

    previsto = Series(
        precos,
        index=serie_real.index
    )

    erro = np.abs(
        (previsto.values - serie_real.values)
        / serie_real.values
    )

    erro_medio = np.mean(erro)
    residuos = y - model.predict(X, verbose=0).ravel()

    return abs(erro_medio), previsto, model, residuos


v_janela = [10, 20]
v_unidades = [32, 64]
v_dropout = [0.0, 0.2]
v_epocas = [50]

best_vlerro = 99
best_param = []
best_prev = None
best_residuos = None

for janela, unidades, dropout, epocas in itertools.product(
    v_janela,
    v_unidades,
    v_dropout,
    v_epocas
):
    parametros = [janela, unidades, dropout, epocas]

    try:
        vlerro, serie_prevista, modelo_atual, residuos = previsao_lstm(
            serie_original,
            intervalo,
            parametros
        )
    except Exception as e:
        print("Falha com", parametros, "->", e)
        continue

    print(
        "LSTM",
        parametros,
        "MAPE:",
        round(vlerro * 100, 4),
        "%"
    )

    if vlerro < best_vlerro:
        best_vlerro = vlerro
        best_param = parametros
        best_prev = serie_prevista
        best_residuos = residuos

if best_prev is None or best_residuos is None:
    raise RuntimeError(
        "Não foi possível ajustar nenhum modelo LSTM."
    )

x = best_prev.index
y = best_prev.values

plt.figure(figsize=(16, 8))

serie_original[
    serie_size - 2 * intervalo:serie_size
].plot(label="Série Real")

best_prev.plot(label="LSTM Model")

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
    "MAPE Encontrado na Previsão (LSTM): "
    + str(round(best_vlerro * 100, 2))
    + "%"
)

plt.xlabel("Data")
plt.ylabel("Valor de fechamento")
plt.legend()
plt.tight_layout()
plt.show()

print("============ Melhores Parametros ============")
print("[janela, unidades, dropout, epocas] =", best_param)
print("MAPE =", round(best_vlerro * 100, 4), "%")

print("============ Analise de Ruido ============")

best_residuos = np.asarray(best_residuos, dtype=float)
best_residuos = best_residuos[np.isfinite(best_residuos)]

teste_ruido = jarque_bera(best_residuos)

print("Estatística Jarque-Bera:", teste_ruido[0])
print("P-valor:", teste_ruido[1])
print("Assimetria:", teste_ruido[2])
print("Curtose:", teste_ruido[3])