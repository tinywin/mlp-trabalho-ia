"""
Versão final melhorada:
 - Multiestilo (Agressivo, Carregador, Visionário, Suporte, Consistente, Pipoqueiro, Duelista, Equilibrado)
 - Solo Kills entra de verdade nas regras
 - Sinergia avançada por time (estilos + performance) para definir campeão IA e vice
 - MVP IA baseado em DPM/KDA/KP + estilos
 - Relatório .txt com todas as seções
 - CSV com todas as previsões
 - Avaliação com hold-out + validação cruzada (k-fold com pipeline)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
    # StratifiedKFold para CV estratificada
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================
# Hiperparâmetros / Constantes
# ==========================

RANDOM_STATE = 42
TEST_SIZE = 0.3
HIDDEN_LAYERS = (128, 64)
MAX_ITER = 3000
CV_FOLDS = 5

# ==========================
# Utilitários básicos
# ==========================


def ts() -> str:
    """Timestamp simples para nomes de arquivos."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_outputs() -> str:
    """Garante a existência da pasta de outputs e retorna o caminho absoluto."""
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "outputs"))
    os.makedirs(out, exist_ok=True)
    return out


# ==========================
# Carregamento e preparação
# ==========================


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Lê o CSV original e:
      - converte colunas numéricas (removendo vírgula e '%')
      - garante Solo Kills numérico
      - preenche NaNs numéricos com mediana
    """
    df = pd.read_csv(csv_path)

    # Colunas que devem permanecer texto/categóricas
    keep_text = {"PlayerName", "TeamName", "Position", "Country", "FlashKeybind"}

    # Tentar converter tudo que for numérico
    for col in df.columns:
        if col in keep_text:
            continue
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace("%", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Solo Kills tem que existir e ser numérica
    if "Solo Kills" in df.columns:
        df["Solo Kills"] = pd.to_numeric(df["Solo Kills"], errors="coerce").fillna(0)

    # Preencher NaNs numéricos com mediana
    num_cols = [c for c in df.columns if c not in keep_text]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    # Remover qualquer linha ainda quebrada
    df = df.dropna().reset_index(drop=True)
    return df


def compute_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    """Calcula médias e percentis usados nas regras de estilo."""
    col_map = {
        "DPM": "DPM",
        "KDA": "KDA",
        "WPM": "Avg WPM",
        "VSPM": "VSPM",
        "KP%": "KP%",
        "GPM": "GoldPerMin",
        "GD@15": "GD@15",
        "Avg Deaths": "Avg deaths",
        "Solo Kills": "Solo Kills",
    }
    for k, col in col_map.items():
        if col not in df.columns:
            raise KeyError(f"Coluna necessária não encontrada: '{col}' (regra para '{k}')")

    thr = {
        "mean_DPM": df[col_map["DPM"]].mean(),
        "mean_KDA": df[col_map["KDA"]].mean(),
        "mean_WPM": df[col_map["WPM"]].mean(),
        "mean_VSPM": df[col_map["VSPM"]].mean(),
        "mean_KP": df[col_map["KP%"]].mean(),
        "mean_GPM": df[col_map["GPM"]].mean(),
        "mean_SoloKills": df[col_map["Solo Kills"]].mean(),
        "mean_deaths": df[col_map["Avg Deaths"]].mean(),
        "p75_DPM": df[col_map["DPM"]].quantile(0.75),
        "p75_KDA": df[col_map["KDA"]].quantile(0.75),
        "p75_SoloKills": df[col_map["Solo Kills"]].quantile(0.75),
    }
    thr["col_map"] = col_map
    return thr


# ==========================
# Regras de estilo (multiestilo)
# ==========================


def compute_estilos(df: pd.DataFrame, thr: Dict[str, float]) -> Tuple[pd.Series, pd.Series]:
    """
    Retorna:
      - Estilos: string multiestilo ("Agressivo, Duelista", etc.)
      - Estilo_Primario: primeiro estilo pela prioridade

    Prioridade: Carregador > Agressivo > Visionário > Suporte > Consistente > Pipoqueiro > Duelista > Equilibrado
    """
    cm = thr["col_map"]
    DPM = df[cm["DPM"]]
    KDA = df[cm["KDA"]]
    WPM = df[cm["WPM"]]
    VSPM = df[cm["VSPM"]]
    KP = df[cm["KP%"]]
    GPM = df[cm["GPM"]]
    GD15 = df[cm["GD@15"]]
    deaths = df[cm["Avg Deaths"]]
    SK = df[cm["Solo Kills"]]

    styles_list: List[str] = []
    primary_list: List[str] = []

    for i in range(len(df)):
        tags: List[str] = []

        # Carregador
        if (
            DPM.iloc[i] > thr["p75_DPM"]
            and GPM.iloc[i] > thr["mean_GPM"]
            and KDA.iloc[i] > thr["mean_KDA"]
        ):
            tags.append("Carregador")

        # Agressivo
        if (
            DPM.iloc[i] > thr["mean_DPM"]
            and (KP.iloc[i] > thr["mean_KP"] or SK.iloc[i] > thr["mean_SoloKills"])
        ):
            tags.append("Agressivo")

        # Visionário
        if (
            VSPM.iloc[i] > thr["mean_VSPM"]
            and WPM.iloc[i] > thr["mean_WPM"]
            and DPM.iloc[i] < thr["mean_DPM"]
        ):
            tags.append("Visionário")

        # Suporte
        if (
            KP.iloc[i] > thr["mean_KP"]
            and WPM.iloc[i] > thr["mean_WPM"]
            and GPM.iloc[i] < thr["mean_GPM"]
        ):
            tags.append("Suporte")

        # Consistente
        if (KDA.iloc[i] > thr["p75_KDA"]) and (deaths.iloc[i] < thr["mean_deaths"]):
            tags.append("Consistente")

        # Pipoqueiro
        if (GD15.iloc[i] < 0) and (deaths.iloc[i] > thr["mean_deaths"]):
            tags.append("Pipoqueiro")

        # Duelista
        if (SK.iloc[i] > thr["p75_SoloKills"]) and (DPM.iloc[i] > thr["mean_DPM"]):
            tags.append("Duelista")

        if not tags:
            styles_list.append("Equilibrado")
            primary_list.append("Equilibrado")
        else:
            styles_list.append(", ".join(tags))
            for p in [
                "Carregador",
                "Agressivo",
                "Visionário",
                "Suporte",
                "Consistente",
                "Pipoqueiro",
                "Duelista",
            ]:
                if p in tags:
                    primary_list.append(p)
                    break

    return (
        pd.Series(styles_list, index=df.index, name="Estilos"),
        pd.Series(primary_list, index=df.index, name="Estilo_Primario"),
    )


# ==========================
# Features para o MLP
# ==========================


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepara X (features), y (rótulo primário) e meta (nome/time)."""
    meta = df[["PlayerName", "TeamName"]].copy()
    y = df["Estilo_Primario"].copy()

    X = df.drop(
        columns=["PlayerName", "TeamName", "Estilo_Primario", "Estilos", "Position"],
        errors="ignore",
    ).copy()

    le_country = LabelEncoder()
    le_flash = LabelEncoder()

    if "Country" in X.columns:
        X["Country"] = le_country.fit_transform(X["Country"].astype(str))
    if "FlashKeybind" in X.columns:
        X["FlashKeybind"] = le_flash.fit_transform(X["FlashKeybind"].astype(str))

    non_numeric = X.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric:
        print(f"Aviso: removendo colunas não numéricas não previstas: {non_numeric}")
        X = X.drop(columns=non_numeric, errors="ignore")

    return X, y, meta


def standardize_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
):
    """Split estratificado + padronização (hold-out)."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        print("Aviso: classe rara demais para stratify. Realizando split sem estratificação.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def build_mlp(random_state: int = RANDOM_STATE) -> MLPClassifier:
    """Cria o MLP com os hiperparâmetros padrão."""
    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        max_iter=MAX_ITER,
        random_state=random_state,
    )
    return clf


def train_mlp(X_train_scaled, y_train) -> MLPClassifier:
    """Treina o MLP no conjunto de treino já padronizado."""
    clf = build_mlp()
    clf.fit(X_train_scaled, y_train)
    return clf


def evaluate_cv(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE, cv: int = CV_FOLDS) -> dict:
    """
    Avalia o mesmo MLP usando validação cruzada k-fold com pipeline
    (StandardScaler + MLP) e StratifiedKFold.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", build_mlp(random_state=random_state)),
        ]
    )
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, cv=skf)
    return {
        "cv_folds": cv,
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
    }


# ==========================
# Avaliação + gráficos
# ==========================


def evaluate_and_plot(y_test, y_pred, labels, out_dir: str, run_id: str):
    """Calcula métricas e gera a matriz de confusão."""
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report_text = classification_report(
        y_test, y_pred, labels=labels, target_names=labels, zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão - Estilos (primário)")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"confusion_matrix_estilo_{run_id}.png")
    plt.savefig(cm_path)
    plt.close()

    return {
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "classification_report": report_text,
        "confusion_matrix_path": cm_path,
    }


def tokenize_estilos(estilos_series: pd.Series) -> pd.Series:
    """Transforma strings multiestilo em contagem de tags individuais."""
    all_tags: List[str] = []
    for s in estilos_series.astype(str):
        parts = [p.strip() for p in s.split(",") if p.strip()]
        all_tags.extend(parts)
    if not all_tags:
        return pd.Series(dtype=int)
    return pd.Series(all_tags).value_counts().sort_index()


def plot_style_counts(
    counts: pd.Series,
    out_dir: str,
    prefix: str,
    title_bar: str,
    title_pie: str,
    run_id: str,
) -> Tuple[str, str]:
    """Gera gráfico de barras e pizza para contagem de estilos."""
    counts_df = counts.reset_index()
    counts_df.columns = ["Estilo", "count"]

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=counts_df,
        x="Estilo",
        y="count",
        hue="Estilo",
        palette="viridis",
        legend=False,
    )
    plt.ylabel("Quantidade")
    plt.xlabel("Estilo")
    plt.title(title_bar)
    plt.xticks(rotation=15)
    plt.tight_layout()
    bar_path = os.path.join(out_dir, f"{prefix}_bar_{run_id}.png")
    plt.savefig(bar_path)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
    plt.title(title_pie)
    plt.tight_layout()
    pie_path = os.path.join(out_dir, f"{prefix}_pie_{run_id}.png")
    plt.savefig(pie_path)
    plt.close()

    return bar_path, pie_path


def summarize_results(
    meta: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    out_dir: str,
    all_labels: List[str] | None,
    run_id: str,
):
    """Resumo de distribuição de estilos previstos (primário)."""
    result_df = meta.copy()
    result_df["Estilo Real (primário)"] = list(y_true)
    result_df["Estilo Previsto"] = list(y_pred)

    counts = result_df["Estilo Previsto"].value_counts().sort_index()
    if all_labels is not None:
        counts = counts.reindex(all_labels, fill_value=0)

    counts_df = counts.reset_index()
    counts_df.columns = ["Estilo", "count"]

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=counts_df,
        x="Estilo",
        y="count",
        hue="Estilo",
        palette="viridis",
        legend=False,
    )
    plt.ylabel("Quantidade")
    plt.xlabel("Estilo Previsto")
    plt.title("Contagem de Estilos Previsto (primário)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    bar_path = os.path.join(out_dir, f"estilos_bar_{run_id}.png")
    plt.savefig(bar_path)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
    plt.title("Proporção de Estilos Previsto (primário)")
    plt.tight_layout()
    pie_path = os.path.join(out_dir, f"estilos_pie_{run_id}.png")
    plt.savefig(pie_path)
    plt.close()

    return result_df, bar_path, pie_path


def dominant_style_per_team(result_df: pd.DataFrame) -> pd.DataFrame:
    """Estilo predominante por time."""
    dom = (
        result_df.groupby(["TeamName", "Estilo Previsto"])
        .size()
        .reset_index(name="count")
        .sort_values(["TeamName", "count"], ascending=[True, False])
    )
    return dom.groupby("TeamName").head(1).reset_index(drop=True)


def format_header(title: str, subtitle: str = "") -> str:
    border = "=" * 41
    lines = [border, f"     🧠 {title}", f"     {subtitle}", border]
    return "\n".join(lines)


def format_class_distribution(series: pd.Series) -> str:
    parts = [f"{idx}: {val}" for idx, val in series.items()]
    return " | ".join(parts)


# ==========================
# Sinergia de time e MVP IA
# ==========================


def compute_team_synergy(pred_df: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a pontuação de sinergia por time combinando:
      - Style score (estilos) e
      - Performance score (estatísticas médias do time)

    1) Style score:
       - Diversidade de estilos core (Carregador, Agressivo, Visionário, Suporte, Consistente, Duelista):
         diversity_score = min(distinct_styles / 6 * 2.0, 2.0)  # até +2
       - Cobertura de papéis core:
         +0.6 se tiver Carregador
         +0.6 se tiver Visionário
         +0.6 se tiver Suporte
         +0.4 se tiver Consistente  (limitado a no máx. +2.2)
       - Profundidade de Carregador:
         +0.5 por Carregador extra além do primeiro (máx. +1.0)
       - Penalidade de Pipoqueiro:
         -0.5 por Pipoqueiro (limitado a no mínimo -2.0)
       - Bônus de duplas:
         +0.6 se tiver Carregador e Suporte
         +0.6 se tiver Carregador e Visionário
         +0.3 se tiver Agressivo e Duelista  (máx. +1.5)

       style_score = diversity_score + core_score + carry_depth_score + pair_score + pipo_penalty

    2) Performance score (por time):
       - Calculado em cima das médias por time:
         avg_kda, avg_dpm, avg_gd15
       - Cada uma é normalizada (z-score) entre os times.
       - perf_score = 0.4*z_kda + 0.4*z_dpm + 0.2*z_gd15

    3) Score final da IA:
       synergy_score = 0.7 * style_score + 0.3 * perf_score

    Também retornamos colunas auxiliares:
      - style_score, perf_score, num_carregador, num_consistente, distinct_styles, avg_kda, avg_deaths
    """
    # ===== performance agregada por time =====
    # Média por time de KDA, DPM, GD@15 e mortes
    team_stats = (
        df_full.groupby("TeamName")
        .agg(
            avg_kda=("KDA", "mean"),
            avg_dpm=("DPM", "mean"),
            avg_gd15=("GD@15", "mean"),
            avg_deaths=("Avg deaths", "mean"),
        )
        .reset_index()
    )

    def zscore(series: pd.Series) -> pd.Series:
        mu = series.mean()
        sd = series.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sd

    team_stats["z_kda"] = zscore(team_stats["avg_kda"])
    team_stats["z_dpm"] = zscore(team_stats["avg_dpm"])
    team_stats["z_gd15"] = zscore(team_stats["avg_gd15"])

    perf_by_team = team_stats.set_index("TeamName")[
        ["avg_kda", "avg_dpm", "avg_gd15", "avg_deaths", "z_kda", "z_dpm", "z_gd15"]
    ].to_dict(orient="index")

    rows = []

    for team, grp in pred_df.groupby("TeamName"):
        all_styles: List[str] = []
        for s in grp["Estilos"].astype(str).tolist():
            parts = [p.strip() for p in s.split(",") if p.strip()]
            all_styles.extend(parts)

        vc = pd.Series(all_styles).value_counts() if all_styles else pd.Series(dtype=int)

        num_car = int(vc.get("Carregador", 0))
        num_cons = int(vc.get("Consistente", 0))
        num_vis = int(vc.get("Visionário", 0))
        num_sup = int(vc.get("Suporte", 0))
        num_pipo = int(vc.get("Pipoqueiro", 0))
        has_agressivo = vc.get("Agressivo", 0) > 0
        has_duelista = vc.get("Duelista", 0) > 0

        core_styles = {
            "Carregador",
            "Agressivo",
            "Visionário",
            "Suporte",
            "Consistente",
            "Duelista",
        }
        styles_set = {k for k in vc.index.tolist() if k in core_styles}
        distinct_styles = len(styles_set)

        # ===== style_score =====
        # Diversidade
        diversity_score = min(distinct_styles / 6.0 * 2.0, 2.0)

        # Cobertura de papéis core
        core_score = 0.0
        if num_car > 0:
            core_score += 0.6
        if num_vis > 0:
            core_score += 0.6
        if num_sup > 0:
            core_score += 0.6
        if num_cons > 0:
            core_score += 0.4
        core_score = min(core_score, 2.2)

        # Profundidade de Carregador (mais de um carry forte é bom, mas com retorno decrescente)
        extra_carries = max(num_car - 1, 0)
        carry_depth_score = min(extra_carries * 0.5, 1.0)

        # Penalidade de Pipoqueiro
        pipo_penalty = -0.5 * num_pipo
        pipo_penalty = max(pipo_penalty, -2.0)

        # Bônus de duplas
        pair_score = 0.0
        if (num_car > 0) and (num_sup > 0):
            pair_score += 0.6
        if (num_car > 0) and (num_vis > 0):
            pair_score += 0.6
        if has_agressivo and has_duelista:
            pair_score += 0.3
        pair_score = min(pair_score, 1.5)

        style_score = diversity_score + core_score + carry_depth_score + pair_score + pipo_penalty

        # ===== perf_score =====
        perf = perf_by_team.get(
            team,
            {
                "avg_kda": 0.0,
                "avg_dpm": 0.0,
                "avg_gd15": 0.0,
                "avg_deaths": float("inf"),
                "z_kda": 0.0,
                "z_dpm": 0.0,
                "z_gd15": 0.0,
            },
        )

        avg_kda = float(perf["avg_kda"])
        avg_deaths = float(perf["avg_deaths"])
        z_kda = float(perf["z_kda"])
        z_dpm = float(perf["z_dpm"])
        z_gd15 = float(perf["z_gd15"])

        perf_score = 0.4 * z_kda + 0.4 * z_dpm + 0.2 * z_gd15

        # ===== score final =====
        synergy_score = 0.7 * style_score + 0.3 * perf_score

        rows.append(
            {
                "TeamName": team,
                "style_score": style_score,
                "perf_score": perf_score,
                "synergy_score": synergy_score,
                "num_carregador": num_car,
                "num_consistente": num_cons,
                "distinct_styles": distinct_styles,
                "avg_kda": avg_kda,
                "avg_deaths": avg_deaths,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "synergy_score",
                "style_score",
                "num_carregador",
                "num_consistente",
                "distinct_styles",
                "avg_kda",
                "avg_deaths",
                "TeamName",
            ],
            ascending=[False, False, False, False, False, False, True, True],
        )
        .reset_index(drop=True)
    )


def select_ia_champion(synergy_df: pd.DataFrame) -> Tuple[str, Dict]:
    """Escolhe o time campeão segundo a sinergia total (estilos + performance)."""
    if synergy_df.empty:
        return "", {}
    max_score = synergy_df["synergy_score"].max()
    top = synergy_df[synergy_df["synergy_score"] == max_score].copy()
    top = top.sort_values(
        [
            "style_score",
            "num_carregador",
            "num_consistente",
            "distinct_styles",
            "avg_kda",
            "avg_deaths",
            "TeamName",
        ],
        ascending=[False, False, False, False, False, True, True],
    )
    champ_row = top.iloc[0]
    return str(champ_row["TeamName"]), champ_row.to_dict()


def select_mvp(
    df_full: pd.DataFrame,
    pred_df: pd.DataFrame,
    champ_team: str,
) -> Tuple[str, str, float]:
    """
    MVP do time campeão:
      - Score = 0.6*zDPM + 0.3*zKDA + 0.1*zKP + bônus por estilo (Carregador/Agressivo)
      - Escolhe dentro do time campeão
    """
    base_stats = df_full[["PlayerName", "TeamName", "DPM", "KDA", "KP%"]].copy()
    merged = pred_df.merge(base_stats, on=["PlayerName", "TeamName"], how="left")

    team_df = merged[merged["TeamName"] == champ_team].copy()
    if team_df.empty:
        return "", "", float("nan")

    def zscore(series: pd.Series) -> pd.Series:
        mu = series.mean()
        sd = series.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sd

    zD_all = zscore(df_full["DPM"])
    zK_all = zscore(df_full["KDA"])
    zKP_all = zscore(df_full["KP%"])

    z_all = pd.DataFrame(
        {
            "PlayerName": df_full["PlayerName"],
            "TeamName": df_full["TeamName"],
            "zD": zD_all,
            "zK": zK_all,
            "zKP": zKP_all,
        }
    )

    merged = merged.merge(z_all, on=["PlayerName", "TeamName"], how="left")

    def bonus_styles(s: str) -> float:
        parts = [p.strip() for p in str(s).split(",") if p.strip()]
        b = 0.0
        if "Carregador" in parts:
            b += 0.1
        if "Agressivo" in parts:
            b += 0.05
        return b

    merged["score_mvp"] = (
        0.6 * merged["zD"].fillna(0)
        + 0.3 * merged["zK"].fillna(0)
        + 0.1 * merged["zKP"].fillna(0)
        + merged["Estilos"].apply(bonus_styles)
    )

    team_scores = merged[merged["TeamName"] == champ_team].copy()

    def is_carry_or_aggr(s: str) -> bool:
        parts = [p.strip() for p in str(s).split(",") if p.strip()]
        return ("Carregador" in parts) or ("Agressivo" in parts)

    elegiveis = team_scores[team_scores["Estilos"].apply(is_carry_or_aggr)]
    if not elegiveis.empty:
        pick = elegiveis.sort_values("score_mvp", ascending=False).iloc[0]
    else:
        pick = team_scores.sort_values("score_mvp", ascending=False).iloc[0]

    return str(pick["PlayerName"]), str(pick["Estilos"]), float(pick["DPM"])


# ==========================
# main()
# ==========================


def main():
    out_dir = ensure_outputs()
    csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "player_statistics_cleaned_final.csv")
    )
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV não encontrado em {csv_path}")

    run_id = ts()  # um id único pra essa execução

    df = load_and_clean(csv_path)
    thresholds = compute_thresholds(df)

    estilos_multi, estilo_primario = compute_estilos(df, thresholds)
    df["Estilos"] = estilos_multi
    df["Estilo_Primario"] = estilo_primario

    # Multiestilo: distribuição geral
    def split_styles(s: str) -> List[str]:
        return [p.strip() for p in str(s).split(",") if p.strip()]

    tokens = df["Estilos"].apply(split_styles)
    all_tokens = [t for sub in tokens for t in sub]
    classes_multi = sorted(set(all_tokens))
    dist_multi = pd.Series(all_tokens).value_counts().sort_index()

    # Distribuição do rótulo primário
    dist_primary = df["Estilo_Primario"].value_counts().sort_index()

    # Plots multiestilo
    multi_counts_all = tokenize_estilos(df["Estilos"])
    multi_bar_path, multi_pie_path = plot_style_counts(
        multi_counts_all,
        out_dir,
        prefix="estilos_multi",
        title_bar="Contagem de Estilos (multiestilo)",
        title_pie="Proporção de Estilos (multiestilo)",
        run_id=run_id,
    )

    # Features + treino
    X, y, meta = build_features(df)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
    ) = standardize_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    clf = train_mlp(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    # Avaliação adicional com validação cruzada (k-fold) no dataset completo
    cv_res = evaluate_cv(X, y, random_state=RANDOM_STATE, cv=CV_FOLDS)

    labels = list(classes_multi)
    eval_res = evaluate_and_plot(y_test, y_pred, labels, out_dir, run_id=run_id)

    result_df_test, bar_path, pie_path = summarize_results(
        meta.loc[y_test.index], y_test, y_pred, out_dir, all_labels=labels, run_id=run_id
    )

    # Previsão para TODOS os jogadores
    X_all_scaled = scaler.transform(X)
    y_pred_all = clf.predict(X_all_scaled)
    result_df_all = meta.copy()
    result_df_all["Estilo Primário (treino)"] = list(y)
    result_df_all["Estilo Previsto"] = list(y_pred_all)
    result_df_all = result_df_all.merge(
        df[["PlayerName", "TeamName", "Estilos"]], on=["PlayerName", "TeamName"], how="left"
    )

    # Salvar CSV completo
    pred_csv_path = os.path.join(out_dir, f"predicoes_completas_{run_id}.csv")
    result_df_all.to_csv(pred_csv_path, index=False, encoding="utf-8")

    # Dominância por time com conjunto completo
    dom_all = dominant_style_per_team(
        result_df_all[["TeamName", "Estilo Previsto"]]
    )

    header_text = format_header(
        "CLASSIFICAÇÃO DE ESTILO DE JOGO", "2024 LoL Championship Player Stats"
    )

    sec1 = [
        "\n1️⃣ Dados e pré-processamento",
        f" - Dados carregados com sucesso: {len(df)} jogadores",
        " - Colunas categóricas codificadas: Country, FlashKeybind",
        f" - Classes criadas (multiestilo): {classes_multi}",
        "",
        "📊 Distribuição multiestilo (todas as tags):",
        format_class_distribution(dist_multi),
        "",
        "📦 Distribuição do rótulo de treino (primário):",
        format_class_distribution(dist_primary),
    ]

    sec2 = [
        "\n2️⃣ Treinamento",
        f" - Camadas ocultas: {HIDDEN_LAYERS}",
        " - Função de ativação: ReLU",
        f" - Épocas máximas: {MAX_ITER}",
    ]

    sec3 = [
        "\n3️⃣ Avaliação",
        f"Acurácia (hold-out): {eval_res['accuracy']:.4f}",
        f"Precisão: {eval_res['precision_weighted']:.4f} | Recall: {eval_res['recall_weighted']:.4f} | F1: {eval_res['f1_weighted']:.4f}",
        "",
        f"Validação cruzada ({cv_res['cv_folds']} folds):",
        f" - Acurácia média: {cv_res['cv_mean']:.4f}",
        f" - Desvio padrão: {cv_res['cv_std']:.4f}",
        "",
        "Relatório de classificação (hold-out):",
        eval_res["classification_report"],
    ]

    sec4 = [
        "\n4️⃣ Resultados",
        f" - Matriz de confusão (rótulo de treino): {eval_res['confusion_matrix_path']}",
        f" - Distribuição de estilos previstos (barras): {bar_path}",
        f" - Proporção de estilos previstos (pizza): {pie_path}",
        f" - Distribuição multiestilo (barras): {multi_bar_path}",
        f" - Proporção multiestilo (pizza): {multi_pie_path}",
        f" - CSV com todas as previsões: {pred_csv_path}",
    ]

    # Principais times (usando TODAS as previsões)
    top_teams = result_df_all["TeamName"].value_counts().head(5).index.tolist()
    lines_top = ["\n🏆 Principais resultados:"]
    for team in top_teams:
        row = dom_all[dom_all["TeamName"] == team]
        if not row.empty:
            estilo_dom = row.iloc[0]["Estilo Previsto"]
            lines_top.append(f" - {team}: predominância “{estilo_dom}”")

    # Destaques curiosos (baseado em todo o dataset)
    lines_fun = ["\n😂 Destaques curiosos:"]
    pipo_all = (
        result_df_all.groupby(["TeamName", "Estilo Previsto"])
        .size()
        .reset_index(name="count")
    )
    pipo_all = pipo_all[pipo_all["Estilo Previsto"] == "Pipoqueiro"]
    if not pipo_all.empty:
        top_pipo = pipo_all.sort_values("count", ascending=False).head(1).iloc[0]
        lines_fun.append(
            f" - {top_pipo['TeamName']} lidera em 'Pipoqueiro' (segundo a IA)"
        )
    else:
        lines_fun.append(" - Ninguém 'Pipoqueiro' hoje – sem pipocadas! 😅")

    most_common = result_df_all["Estilo Previsto"].value_counts().idxmax()
    lines_fun.append(f" - Estilo mais comum previsto: {most_common}")

    # 5️⃣ Lista completa de jogadores
    sorted_res = result_df_all.sort_values(
        ["TeamName", "PlayerName"], ascending=[True, True]
    ).reset_index(drop=True)

    sec5_list = ["\n5️⃣ Lista de jogadores e estilos previstos"]
    for i, row in enumerate(sorted_res.itertuples(index=False, name="Row"), start=1):
        estilos_multi_row = getattr(row, "Estilos")
        sec5_list.append(
            f"{i}. {row.PlayerName} ({row.TeamName}) — {estilos_multi_row}"
        )

    def count_styles(s: str) -> int:
        return len([p for p in str(s).split(",") if p.strip()])

    n_styles = sorted_res["Estilos"].apply(count_styles)
    c1 = int((n_styles == 1).sum())
    c2 = int((n_styles == 2).sum())
    c3p = int((n_styles >= 3).sum())
    sec5_list.append(
        f"Jogadores com 1 estilo: {c1} | 2 estilos: {c2} | 3+ estilos: {c3p}"
    )

    # 6️⃣ Cerimônia Final IA
    synergy_df = compute_team_synergy(
        result_df_all[["PlayerName", "TeamName", "Estilos"]], df
    )
    champ_team, champ_details = select_ia_champion(synergy_df)
    vice_team = str(synergy_df.iloc[1]["TeamName"]) if len(synergy_df) > 1 else ""

    sec6 = ["\n6️⃣ Cerimônia Final"]
    if champ_team:
        sec6 += [
            f"🥇 Segundo a IA, o time mais completo (estilos + performance) é: {champ_team}",
            f" - Pontuação de sinergia: {champ_details.get('synergy_score', 'N/A'):.3f}",
            f" - Estilos distintos: {champ_details.get('distinct_styles', 'N/A')} | Carregadores: {champ_details.get('num_carregador', 'N/A')} | Consistentes: {champ_details.get('num_consistente', 'N/A')}",
        ]
        if vice_team:
            sec6.append(f"🥈 Vice-campeão técnico segundo a IA: {vice_team}")

        mvp_player, mvp_style, mvp_dpm = select_mvp(
            df, result_df_all[["PlayerName", "TeamName", "Estilos"]], champ_team
        )
        if mvp_player:
            sec6.append(
                f"🏅 MVP segundo a IA: {mvp_player} ({champ_team}) — {mvp_style}, DPM={mvp_dpm:.1f}"
            )
        else:
            sec6.append("🏅 MVP segundo a IA: não foi possível determinar")
    else:
        sec6.append("Não foi possível determinar o campeão pela sinergia.")

    sec6 += [
        "\n🎭 Encerramento:",
        "Seja qual for o resultado técnico,",
        "o Campeão real do Worlds 2024 é a T1.",
        "MVP real: Faker (T1).",
    ]

    # 7️⃣ Limitações e Trabalhos Futuros
    sec7 = [
        "\n7️⃣ Limitações e Trabalhos Futuros",
        "",
        "Apesar dos resultados interessantes, este modelo tem algumas limitações importantes:",
        "",
        "- Tamanho da amostra: o conjunto de dados possui apenas 81 jogadores. Isso é pouco para um modelo de rede neural, o que pode tornar a acurácia sensível a pequenas alterações no split de treino e teste.",
        "- Rótulos heurísticos: os estilos de jogo (Carregador, Agressivo, Visionário, Suporte, Consistente, Pipoqueiro, Duelista, Equilibrado) não vieram rotulados no dataset original. Eles foram definidos a partir de regras manuais (heurísticas) com base em estatísticas como DPM, KDA, KP%, visão e Solo Kills. Ou seja, o modelo aprende a reproduzir essas regras, e não um 'rótulo oficial' dado por analistas humanos.",
        "- Multiestilo vs. rótulo único: na prática, vários jogadores recebem múltiplos estilos (por exemplo, um jogador pode ser ao mesmo tempo Carregador e Duelista). Porém, para treinar o MLP, foi necessário escolher apenas um estilo primário por jogador. Um modelo multi-rótulo (multi-label) poderia representar melhor essa sobreposição de papéis.",
        "- Contexto de série e draft: o modelo não leva em conta informações de draft (campeões escolhidos), adversário e contexto de série (MD3, MD5, fase de grupos, mata-mata). Ele trabalha apenas com médias agregadas do jogador no campeonato, o que simplifica muito a realidade competitiva.",
        "",
        "Como trabalhos futuros, seria interessante:",
        "",
        "- Testar modelos mais adequados a multi-rótulo (por exemplo, um classificador binário por estilo ou modelos baseados em One-vs-Rest).",
        "- Ajustar os limiares de classificação por função (Top, Jungle, Mid, ADC, Suporte), em vez de usar os mesmos percentis para todos os jogadores.",
        "- Incorporar mais variáveis contextuais, como estatísticas por matchup, fase de jogo ou tipo de composição do time.",
        "- Comparar a MLP com modelos mais simples (por exemplo, árvores de decisão e random forests) para avaliar se a complexidade da rede neural é realmente necessária para esse problema.",
    ]

    # 8️⃣ Interpretação dos resultados da IA x campeão real
    sec8 = [
        "\n8️⃣ Interpretação dos resultados da IA",
        "",
        "Em algumas execuções do modelo, a IA aponta times como Gen.G, Bilibili Gaming ou Weibo Gaming como os mais 'equilibrados e taticamente completos', principalmente por concentrarem jogadores com alto DPM, boa KDA e papéis bem definidos (Carregadores, Visionários e Consistentes).",
        "",
        "Com a versão atual, a IA leva em conta tanto a composição de estilos (style_score) quanto a performance média do time no campeonato (perf_score). Isso aproxima o 'campeão técnico' da realidade dos números, mas ainda assim não captura toda a complexidade de uma série MD5.",
        "",
        "Na realidade competitiva, o campeão do Worlds 2024 foi a T1, com o Faker como principal referência. Isso evidencia uma diferença importante: o modelo enxerga apenas números médios por jogador, enquanto o resultado real depende de fatores que não estão no dataset, como adaptação de draft, pressão de palco, leitura de série MD5, sinergia em momentos decisivos e o famoso 'clutch' em jogos-chave.",
        "",
        "Em resumo, a IA mostra quais times e jogadores se destacam estatisticamente, mas o título da T1 lembra que, em esportes eletrônicos, nem sempre o campeão técnico é o campeão da taça. O modelo ajuda a contar parte da história; o servidor, o palco e o Faker cuidam do resto.",
    ]

    # Unificar tudo no relatório
    relatorio_partes = [
        header_text,
        *sec1,
        *sec2,
        *sec3,
        *sec4,
        *lines_top,
        *lines_fun,
        *sec5_list,
        *sec6,
        *sec7,
        *sec8,
    ]
    relatorio_texto = "\n".join(relatorio_partes)

    print(relatorio_texto)

    report_path = os.path.join(out_dir, f"relatorio_estilos_{run_id}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(relatorio_texto)

    print("\nRelatório salvo em:", report_path)


if __name__ == "__main__":
    main()