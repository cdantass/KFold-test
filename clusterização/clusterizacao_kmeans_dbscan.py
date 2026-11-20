import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
PASTA = Path(__file__).resolve().parent
CAMINHO_CSV = PASTA / "datasets" / "yahoo_stock.csv"
PASTA_RESULTADO = PASTA / "resultado"
PASTA_RESULTADO.mkdir(exist_ok=True)

NOMES_CLASSE = {0: "Baixa", 1: "Alta"}
N_CLASSES = len(NOMES_CLASSE)

sns.set_theme(style="whitegrid")


def salvar_e_mostrar(nome_arquivo):
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADO / nome_arquivo, dpi=150, bbox_inches="tight")
    plt.show()

if not CAMINHO_CSV.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {CAMINHO_CSV}")

base = pd.read_csv(CAMINHO_CSV, parse_dates=["Date"])

retorno_diario = base["Close"].pct_change()

features_df = pd.DataFrame(
    {
        "Retorno_Anterior": retorno_diario.shift(1) * 100,
        "Gap_Abertura": (base["Open"] / base["Close"].shift(1) - 1) * 100,
        "Amplitude_Anterior": (
            (base["High"] - base["Low"]) / base["Close"]
        ).shift(1) * 100,
        "Var_Volume_Anterior": base["Volume"].pct_change().shift(1) * 100,
        "Volatilidade_5d": retorno_diario.rolling(5).std().shift(1) * 100,
        "Retorno_5d": base["Close"].pct_change(5).shift(1) * 100,
    }
)

classe = (base["Close"] > base["Open"]).astype(int)

validos = features_df.notna().all(axis=1)
features_df = features_df[validos].reset_index(drop=True)
classe = classe[validos].reset_index(drop=True).to_numpy()
datas = base.loc[validos, "Date"].reset_index(drop=True)

nomes_features = list(features_df.columns)

print("=" * 60)
print("Base:", CAMINHO_CSV.name)
print("Período:", datas.min().date(), "->", datas.max().date())
print("Registros usados:", len(features_df))
print("Features:", nomes_features)
print("Distribuição das classes:")
for k, v in zip(*np.unique(classe, return_counts=True)):
    print(f"  {k} ({NOMES_CLASSE[k]}): {v}")
print("=" * 60)

X = StandardScaler().fit_transform(features_df.values)

pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_ * 100

ks = range(2, 9)
inercias, silhuetas = [], []
for k in ks:
    km_tmp = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
    inercias.append(km_tmp.inertia_)
    silhuetas.append(silhouette_score(X, km_tmp.labels_))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(list(ks), inercias, marker="o")
ax[0].set_title("Método do cotovelo (K-Means)")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inércia")
ax[1].plot(list(ks), silhuetas, marker="o", color="darkorange")
ax[1].set_title("Silhouette por k (K-Means)")
ax[1].set_xlabel("k")
ax[1].set_ylabel("Silhouette")
salvar_e_mostrar("01_kmeans_cotovelo_silhouette.png")

kmeans = KMeans(n_clusters=N_CLASSES, random_state=SEED, n_init=10).fit(X)
cluster_kmeans = kmeans.labels_

MIN_SAMPLES = 2 * X.shape[1]
PERCENTIL_EPS = 90

dist, _ = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X).kneighbors(X)
k_dist = np.sort(dist[:, -1])
eps_final = float(np.percentile(k_dist, PERCENTIL_EPS))
ms_final = MIN_SAMPLES

plt.figure(figsize=(7, 4))
plt.plot(k_dist)
plt.axhline(eps_final, color="red", linestyle="--",
            label=f"eps escolhido = {eps_final:.2f}")
plt.title(f"Gráfico de k-distância (k = {MIN_SAMPLES}) para escolha do eps")
plt.xlabel("Pontos ordenados")
plt.ylabel(f"Distância ao {MIN_SAMPLES}º vizinho")
plt.legend()
salvar_e_mostrar("02_dbscan_k_distancia.png")

cluster_dbscan = DBSCAN(eps=eps_final, min_samples=ms_final).fit_predict(X)

n_clusters_db = len(set(cluster_dbscan)) - (1 if -1 in cluster_dbscan else 0)
n_ruido = int((cluster_dbscan == -1).sum())

print("\nDBSCAN escolhido:")
print(f"  eps = {eps_final:.3f} | min_samples = {ms_final}")
print(f"  clusters = {n_clusters_db} | ruído = {n_ruido} "
      f"({n_ruido / len(X) * 100:.1f}%)")
print("  tamanhos:", {int(k): int(v) for k, v in
                      zip(*np.unique(cluster_dbscan, return_counts=True))})

rotulos_classe = np.array([NOMES_CLASSE[c] for c in classe])
rotulos_km = np.array([f"Cluster {c}" for c in cluster_kmeans])
rotulos_db = np.array(
    ["Ruído" if c == -1 else f"Cluster {c}" for c in cluster_dbscan]
)

def dispersao(ax, x, y, rotulos, titulo):
    for r in sorted(set(rotulos)):
        m = rotulos == r
        ax.scatter(x[m], y[m], s=15, alpha=0.7, label=r)
    ax.set_title(titulo)
    ax.legend(fontsize=8)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
dispersao(ax[0], X_pca[:, 0], X_pca[:, 1], rotulos_classe, "Classes")
dispersao(ax[1], X_pca[:, 0], X_pca[:, 1], rotulos_km, "K-Means")
dispersao(ax[2], X_pca[:, 0], X_pca[:, 1], rotulos_db, "DBSCAN")
salvar_e_mostrar("03_dispersao_pca.png")

df_corr = features_df.copy()
df_corr["Classe"] = classe
df_corr["KMeans"] = cluster_kmeans
df_corr["DBSCAN"] = cluster_dbscan

plt.figure(figsize=(10, 8))
sns.heatmap(
    df_corr.corr("spearman"),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlação entre variáveis")
salvar_e_mostrar("04_heatmap_correlacao.png")

perfil_km = pd.DataFrame(
    X,
    columns=nomes_features
).groupby(cluster_kmeans).mean()

perfil_db = pd.DataFrame(
    X,
    columns=nomes_features
).groupby(cluster_dbscan).mean()

perfil = pd.concat([perfil_km, perfil_db])

plt.figure(figsize=(10, 5))
sns.heatmap(
    perfil,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Perfil dos clusters")
salvar_e_mostrar("05_heatmap_perfil_clusters.png")

def acuracia(y_true, y_cluster):
    clusters = sorted(set(y_cluster))
    classes = sorted(set(y_true))

    matriz = np.array([
        [
            np.sum((y_cluster == cl) & (y_true == cs))
            for cs in classes
        ]
        for cl in clusters
    ])

    lin, col = linear_sum_assignment(-matriz)
    return matriz[lin, col].sum() / len(y_true)

def metricas(y):
    return {
        "Acurácia": acuracia(classe, y),
        "ARI": adjusted_rand_score(classe, y),
        "NMI": normalized_mutual_info_score(classe, y),
        "Homogeneidade": homogeneity_score(classe, y),
        "Completude": completeness_score(classe, y),
        "V-Measure": v_measure_score(classe, y),
        "Silhouette": silhouette_score(X, y)
        if len(set(y)) > 1 else np.nan
    }

tabela_metricas = pd.DataFrame({
    "K-Means": metricas(cluster_kmeans),
    "DBSCAN": metricas(cluster_dbscan)
}).round(4)

base_majoritaria = np.bincount(classe).max() / len(classe)

print("\nMÉTRICAS")
print(tabela_metricas)

ax = tabela_metricas.drop(index="Silhouette").plot(
    kind="barh",
    figsize=(10, 5)
)

ax.axvline(
    base_majoritaria,
    color="gray",
    linestyle="--",
    label=f"Classe majoritária ({base_majoritaria:.2f})"
)

ax.set_xlim(0, 1)
ax.legend()

for c in ax.containers:
    ax.bar_label(c, fmt="%.2f")

salvar_e_mostrar("06_metricas_clusters_vs_classes.png")

tabela_metricas.to_csv(
    PASTA_RESULTADO / "metricas_clusters_vs_classes.csv"
)

print(f"\nResultados salvos em: {PASTA_RESULTADO}")