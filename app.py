from __future__ import annotations

from pathlib import Path
import json
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_TITLE = "VBC Analytics — version recruteur"
DATA_DIR = Path(__file__).parent / "data" / "processed"
GITHUB_URL = "https://github.com/jamesm-parisv/vbc-analytics"

PR_BINS = [0, 4.5, 6.5, 8.5, 10.5, 13, np.inf]
PR_LABELS = ["< 4.5", "4.5–6.5", "6.5–8.5", "8.5–10.5", "10.5–13", "13+"]
PARIS_LATITUDE = 48.8566
PARIS_LONGITUDE = 2.3522

st.set_page_config(page_title=APP_TITLE, page_icon="🎲", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
div[data-testid="stMetricValue"] {font-size: 1.7rem;}
.vbc-hero {
    padding: 2rem 2.2rem;
    border-radius: 1.4rem;
    background: linear-gradient(135deg, #101828 0%, #27364f 52%, #6941c6 100%);
    color: white;
    margin-bottom: 1.25rem;
    box-shadow: 0 18px 45px rgba(16, 24, 40, 0.20);
}
.vbc-eyebrow {text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.78rem; opacity: 0.8; margin-bottom: 0.45rem;}
.vbc-title {font-size: clamp(2.3rem, 6vw, 4.2rem); font-weight: 850; line-height: 1.0; margin-bottom: 0.6rem;}
.vbc-subtitle {max-width: 900px; font-size: 1.08rem; line-height: 1.55; opacity: 0.93;}
.vbc-badges {margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 0.55rem;}
.vbc-badge {border: 1px solid rgba(255,255,255,0.28); background: rgba(255,255,255,0.12); border-radius: 999px; padding: 0.35rem 0.75rem; font-size: 0.88rem;}
.vbc-card {border: 1px solid rgba(16, 24, 40, 0.10); border-radius: 1rem; padding: 1rem 1.1rem; background: rgba(255,255,255,0.78); margin-bottom: 0.85rem;}
.vbc-card h4 {margin: 0 0 0.35rem 0; color: #101828;}
.vbc-card p {margin: 0; color: #344054; line-height: 1.55;}
.vbc-insight-card {border: 1px solid rgba(16, 24, 40, 0.10); border-radius: 1rem; padding: 1.1rem 1.2rem; background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.96)); margin-top: 0.25rem; min-height: 235px; box-shadow: 0 10px 28px rgba(16,24,40,0.06);}
.vbc-insight-kicker {display:inline-block; font-size:0.72rem; font-weight:800; letter-spacing:0.08em; text-transform:uppercase; color:#6941c6; background:#f4f3ff; border-radius:999px; padding:0.22rem 0.55rem; margin-bottom:0.65rem;}
.vbc-insight-card h3 {margin: 0 0 0.45rem 0; font-size: 1.22rem; color:#101828; line-height:1.2;}
.vbc-insight-lead {margin: 0 0 0.65rem 0; color:#101828; line-height: 1.42; font-size: 1.04rem; font-weight:700;}
.vbc-insight-card p {margin: 0; color:#344054; line-height: 1.55; font-size: 0.96rem;}
.vbc-insight-note {margin-top:0.75rem; padding-left:0.75rem; border-left:3px solid #d6bbfb; color:#475467; font-size:0.92rem; line-height:1.45;}
.vbc-small {color:#667085; font-size:0.92rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name)
    if "TournamentDate" in df.columns:
        df["TournamentDate"] = pd.to_datetime(df["TournamentDate"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    return {
        "matches": load_csv("matches.csv"),
        "pairs": load_csv("match_pairs.csv"),
        "tournaments": load_csv("tournament_summary.csv"),
        "players": load_csv("player_overall.csv"),
        "player_tournament": load_csv("player_tournament.csv"),
    }


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_paris_weather(start_date: str, end_date: str) -> pd.DataFrame:
    params = urllib.parse.urlencode(
        {
            "latitude": PARIS_LATITUDE,
            "longitude": PARIS_LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min",
            "timezone": "Europe/Paris",
        }
    )
    url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    daily = payload.get("daily", {})
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(daily.get("time", []), errors="coerce"),
            "Température moyenne Paris (°C)": daily.get("temperature_2m_mean", []),
            "Température min Paris (°C)": daily.get("temperature_2m_min", []),
            "Température max Paris (°C)": daily.get("temperature_2m_max", []),
        }
    ).dropna(subset=["Date"])


def pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.1%}"


def num(x: float | int | None, decimals: int = 0) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}"


def corr_value(df: pd.DataFrame, x: str, y: str) -> float:
    clean = df[[x, y]].dropna()
    if len(clean) < 2 or clean[x].nunique() < 2 or clean[y].nunique() < 2:
        return np.nan
    return float(clean[x].corr(clean[y]))


def add_regression_line(fig, df: pd.DataFrame, x: str, y: str, name: str = "Tendance linéaire") -> None:
    clean = df[[x, y]].dropna()
    if len(clean) < 2 or clean[x].nunique() < 2:
        return
    slope, intercept = np.polyfit(clean[x], clean[y], 1)
    xs = np.linspace(clean[x].min(), clean[x].max(), 50)
    ys = slope * xs + intercept
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name))


def chart(fig, key: str, height: int = 390) -> None:
    fig.update_layout(margin=dict(l=10, r=10, t=48, b=10), height=height, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True, key=key)


def compact_chart(fig, key: str, height: int = 235) -> None:
    fig.update_layout(
        margin=dict(l=6, r=6, t=38, b=6),
        height=height,
        legend_title_text="",
        font=dict(size=11),
    )
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})


def build_tournament_stats(matches: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    t = (
        matches.groupby(["TournamentDate", "TournamentFile"], as_index=False)
        .agg(
            Joueurs=("PlayerName", "nunique"),
            PR_moyen=("PR", "mean"),
            Decisions=("Decisions", "sum"),
            Lignes_joueur=("MatchID", "count"),
        )
        .sort_values("TournamentDate")
    )
    p = (
        pairs[pairs["QualityStatus"].eq("OK")]
        .groupby(["TournamentDate", "TournamentFile"], as_index=False)
        .agg(Matchs=("MatchID", "count"), Victoire_meilleur_PR=("BetterPRWon", "mean"))
    )
    t = t.merge(p, on=["TournamentDate", "TournamentFile"], how="left")
    t["Tournoi"] = t["TournamentDate"].dt.strftime("%Y-%m-%d")
    return t


def pr_category(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=PR_BINS, labels=PR_LABELS, right=False, include_lowest=True)


def add_global_pr_categories(matches: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    lookup = players.set_index("PlayerName")["AvgPR"].to_dict()
    out = matches.copy()
    out["PR_global_joueur"] = out["PlayerName"].map(lookup)
    out["PR_global_adversaire"] = out["OpponentName"].map(lookup)
    out["Catégorie joueur"] = pr_category(out["PR_global_joueur"])
    out["Catégorie adversaire"] = pr_category(out["PR_global_adversaire"])
    return out


def build_category_vs_opponent(matches: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    categorized = add_global_pr_categories(matches, players)
    clean = categorized.dropna(subset=["Catégorie joueur", "Catégorie adversaire", "PR"])
    return (
        clean.groupby(["Catégorie joueur", "Catégorie adversaire"], observed=False)
        .agg(PR_moyen=("PR", "mean"), Taux_victoire=("Victory", "mean"), Matchs=("MatchID", "count"))
        .reset_index()
    )


def build_tournament_category_distribution(player_tournament: pd.DataFrame) -> pd.DataFrame:
    dist = player_tournament.copy()
    dist["Catégorie PR"] = pr_category(dist["AvgPR"])
    dist["Tournoi"] = dist["TournamentDate"].dt.strftime("%Y-%m-%d")
    return (
        dist.groupby(["Tournoi", "TournamentDate", "Catégorie PR"], observed=False)
        .agg(Joueurs=("PlayerName", "nunique"))
        .reset_index()
        .sort_values(["TournamentDate", "Catégorie PR"])
    )


def build_global_category_progression(matches: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    categorized = add_global_pr_categories(matches, players)
    progression = (
        categorized.dropna(subset=["Catégorie joueur"])
        .groupby(["TournamentDate", "Catégorie joueur"], observed=False)
        .agg(PR_moyen=("PR", "mean"), Matchs=("MatchID", "count"))
        .reset_index()
        .sort_values("TournamentDate")
    )
    progression["Tournoi"] = progression["TournamentDate"].dt.strftime("%Y-%m-%d")
    return progression


def build_weather_joined(matches: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame | None:
    tournaments = build_tournament_stats(matches, pairs)
    start_date = tournaments["TournamentDate"].min().date().isoformat()
    end_date = tournaments["TournamentDate"].max().date().isoformat()
    try:
        weather = load_paris_weather(start_date, end_date)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None
    tournaments["Date"] = tournaments["TournamentDate"].dt.normalize()
    joined = tournaments.merge(weather, on="Date", how="left")
    return joined.dropna(subset=["Température moyenne Paris (°C)", "PR_moyen"])


def fig_better_pr_donut(pairs: pd.DataFrame, title: str = "Le meilleur PR gagne-t-il ?") -> go.Figure:
    outcome = pairs[pairs["QualityStatus"].eq("OK")].dropna(subset=["BetterPRWon"]).copy()
    outcome["Résultat"] = np.where(outcome["BetterPRWon"].eq(1), "Meilleur PR gagne", "Meilleur PR ne gagne pas")
    counts = outcome["Résultat"].value_counts().rename_axis("Résultat").reset_index(name="Matchs")
    fig = px.pie(counts, names="Résultat", values="Matchs", hole=0.45, title=title)
    fig.update_traces(textposition="inside", textinfo="percent")
    return fig


def fig_pr_by_round(matches: pd.DataFrame, title: str = "PR moyen par ronde") -> go.Figure:
    round_pr = matches.groupby("Round", as_index=False).agg(PR_moyen=("PR", "mean"), Matchs=("MatchID", "count")).sort_values("Round")
    fig = px.line(round_pr, x="Round", y="PR_moyen", markers=True, title=title, labels={"Round": "Ronde", "PR_moyen": "PR moyen"})
    return fig


def fig_duration_pr(matches: pd.DataFrame, title: str = "PR moyen selon la longueur du match") -> go.Figure:
    duration = matches.dropna(subset=["Decisions", "PR"]).copy()
    duration["Tranche de décisions"] = pd.cut(
        duration["Decisions"],
        bins=[0, 80, 120, 160, 200, 260, np.inf],
        labels=["<80", "80–120", "120–160", "160–200", "200–260", "260+"],
        include_lowest=True,
        right=False,
    )
    duration_stats = (
        duration.groupby("Tranche de décisions", observed=False)
        .agg(PR_moyen=("PR", "mean"), Lignes=("MatchID", "count"))
        .reset_index()
    )
    fig = px.bar(
        duration_stats,
        x="Tranche de décisions",
        y="PR_moyen",
        text=duration_stats["PR_moyen"].map(lambda v: f"{v:.2f}" if pd.notna(v) else ""),
        title=title,
        labels={"PR_moyen": "PR moyen", "Tranche de décisions": "Décisions"},
    )
    return fig


def fig_weather_pr(matches: pd.DataFrame, pairs: pd.DataFrame, title: str = "PR moyen vs température") -> go.Figure | None:
    joined = build_weather_joined(matches, pairs)
    if joined is None or joined.empty:
        return None
    fig = px.scatter(
        joined,
        x="Température moyenne Paris (°C)",
        y="PR_moyen",
        size="Joueurs",
        hover_name="Tournoi",
        title=title,
        labels={"PR_moyen": "PR moyen"},
    )
    add_regression_line(fig, joined, "Température moyenne Paris (°C)", "PR_moyen")
    return fig


def fig_category_heatmap(matches: pd.DataFrame, players: pd.DataFrame, value: str = "PR_moyen", title: str = "PR moyen selon catégorie joueur/adversaire") -> go.Figure:
    stats = build_category_vs_opponent(matches, players)
    pivot = stats.pivot(index="Catégorie joueur", columns="Catégorie adversaire", values=value).reindex(index=PR_LABELS, columns=PR_LABELS)
    text = stats.pivot(index="Catégorie joueur", columns="Catégorie adversaire", values="Matchs").reindex(index=PR_LABELS, columns=PR_LABELS)
    fig = px.imshow(
        pivot,
        text_auto=".2f" if value == "PR_moyen" else ".0%",
        aspect="auto",
        title=title,
        labels={"x": "Catégorie adversaire", "y": "Catégorie joueur", "color": "PR moyen" if value == "PR_moyen" else "Taux de victoire"},
    )
    fig.update_traces(customdata=text, hovertemplate="Joueur: %{y}<br>Adversaire: %{x}<br>Valeur: %{z}<br>Matchs: %{customdata}<extra></extra>")
    return fig


def fig_category_distribution(player_tournament: pd.DataFrame, normalize: bool = False, title: str = "Répartition des catégories PR par tournoi") -> go.Figure:
    dist = build_tournament_category_distribution(player_tournament)
    if normalize:
        totals = dist.groupby("Tournoi")["Joueurs"].transform("sum")
        dist["Part"] = dist["Joueurs"] / totals.replace(0, np.nan)
        fig = px.bar(dist, x="Tournoi", y="Part", color="Catégorie PR", title=title, labels={"Part": "Part des joueurs"})
        fig.update_yaxes(tickformat=".0%")
    else:
        fig = px.bar(dist, x="Tournoi", y="Joueurs", color="Catégorie PR", title=title, labels={"Joueurs": "Nombre de joueurs"})
    return fig


def hero() -> None:
    st.markdown(
        """
<div class="vbc-hero">
    <div class="vbc-eyebrow">Projet portfolio Data Analyst</div>
    <div class="vbc-title">VBC Analytics</div>
    <div class="vbc-subtitle">
        Analyse des 18 premiers tournois Videau Backgammon Championship : performance PR,
        résultats de match, fatigue, météo, progression du plateau et limites statistiques.
    </div>
    <div class="vbc-badges">
        <span class="vbc-badge">Python</span>
        <span class="vbc-badge">Pandas</span>
        <span class="vbc-badge">Streamlit</span>
        <span class="vbc-badge">Plotly</span>
        <span class="vbc-badge">Data cleaning</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def tab_report(data: dict[str, pd.DataFrame]) -> None:
    matches, pairs = data["matches"], data["pairs"]
    ok_pairs = pairs[pairs["QualityStatus"].eq("OK")].copy()
    tournaments = build_tournament_stats(matches, pairs)

    st.markdown("## Rapport")
    st.markdown("""**James MacNaughtan**  
[jamesmacnaughtan@gmail.com](mailto:jamesmacnaughtan@gmail.com)""")

    if GITHUB_URL:
        st.link_button("Voir le code sur GitHub", GITHUB_URL)
    else:
        st.info("Lien GitHub à ajouter : renseigner la constante `GITHUB_URL` dans `app.py` après création du dépôt public.")

    st.markdown(
        """
### Contexte
Les tournois **Videau Backgammon Championship** représentent le pôle compétition du club de backgammon **Paris Videau**, qui fédère plus de 200 membres. Ils se jouent environ six fois par an.

Chaque joueur joue généralement **4 ou 5 matchs**. Les matchs sont enregistrés, transcrits avec **Extreme Gammon**, exportés puis soumis à la **Backgammon Masters Awarding Body (BMAB)**, l’organisation mondiale qui octroie les grades de Master, Grand-Master, etc. Les joueurs expérimentés participent pour alimenter leur profil BMAB, mesurer leur niveau technique et tenter d’obtenir un titre.

Les données des **18 premiers tournois VBC** ont été collectées, nettoyées et modélisées pour répondre à quatre questions :

- Existe-t-il une relation entre bon **PR** et taux de victoire ?
- Quel est l’impact de la fatigue, de la température, de la longueur du match et du niveau adverse ?
- Les VBC attirent-ils ou génèrent-ils des joueurs plus performants ?
- Ces tournois sont-ils un outil efficace pour améliorer sa performance ?
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tournois", num(matches["TournamentDate"].nunique()))
    c2.metric("Joueurs", num(matches["PlayerName"].nunique()))
    c3.metric("Matchs reconstruits OK", num(len(ok_pairs)))
    c4.metric("Victoire du meilleur PR", pct(ok_pairs["BetterPRWon"].mean()))

    left, right = st.columns(2)
    with left:
        fig = px.line(tournaments, x="Tournoi", y="PR_moyen", markers=True, title="PR moyen par tournoi", labels={"PR_moyen": "PR moyen"})
        chart(fig, "report_pr_by_tournament")
    with right:
        fig = px.bar(tournaments, x="Tournoi", y="Joueurs", title="Participants par tournoi", labels={"Joueurs": "Nombre de joueurs"})
        chart(fig, "report_players_by_tournament")


def insight_text(number: int, title: str, lead: str, body: str, note: str | None = None) -> None:
    note_html = f'<div class="vbc-insight-note">{note}</div>' if note else ""
    st.markdown(f"""
<div class="vbc-insight-card">
    <div class="vbc-insight-kicker">Insight #{number}</div>
    <h3>{title}</h3>
    <div class="vbc-insight-lead">{lead}</div>
    <p>{body}</p>
    {note_html}
</div>
""", unsafe_allow_html=True)


def tab_insights(data: dict[str, pd.DataFrame]) -> None:
    matches, pairs = data["matches"], data["pairs"]
    players = data["players"]
    player_tournament = data["player_tournament"]
    ok_pairs = pairs[pairs["QualityStatus"].eq("OK")].copy()
    better_pr_rate = ok_pairs["BetterPRWon"].mean()

    round_pr = matches.groupby("Round", as_index=False).agg(PR_moyen=("PR", "mean"), Matchs=("MatchID", "count")).sort_values("Round")
    best_round = round_pr.sort_values("PR_moyen").iloc[0]
    worst_round = round_pr.sort_values("PR_moyen", ascending=False).iloc[0]

    st.markdown("## Insights clés")
    st.caption("Lecture rapide du rapport : chaque conclusion est placée directement à côté du graphique qui la soutient.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Victoire du meilleur PR", pct(better_pr_rate))
    c2.metric("Meilleure ronde PR", f"R{int(best_round['Round'])} · {best_round['PR_moyen']:.2f}")
    c3.metric("Ronde la plus difficile", f"R{int(worst_round['Round'])} · {worst_round['PR_moyen']:.2f}")

    st.divider()

    # 1 — Texte à gauche, visuel à droite
    text_col, chart_col = st.columns([1.04, 1])
    with text_col:
        insight_text(
            1,
            "PR et réussite sportive",
            "Le meilleur PR gagne plus souvent, mais pas massivement.",
            "Les joueurs avec le meilleur PR remportent environ 56,4 % des matchs. Le signal est positif, mais limité par des écarts de PR souvent faibles et par des matchs courts.",
            "À retenir : le PR compte, mais le backgammon garde une forte variance sur un seul match.",
        )
    with chart_col:
        fig = fig_better_pr_donut(pairs, title="Matchs gagnés par le meilleur PR")
        compact_chart(fig, "insight_1_donut", height=275)

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    # 2 — Visuel à gauche, texte à droite, puis deux graphiques larges en dessous
    chart_col, text_col = st.columns([1, 1.04])
    with chart_col:
        compact_chart(fig_pr_by_round(matches, title="PR moyen par ronde"), "insight_2_round", height=300)
    with text_col:
        insight_text(
            2,
            "Fatigue, météo et longueur",
            "La fatigue est le signal le plus visible.",
            "La pire performance apparaît sur le troisième match, souvent juste après la pause déjeuner, tandis que la meilleure est observée sur le premier match. La température semble avoir un impact faible. La longueur du match est plus ambiguë.",
            "À surveiller : les matchs plus longs peuvent aussi refléter des joueurs moins forts qui utilisent le videau plus tardivement.",
        )

    weather_col, duration_col = st.columns(2)
    with weather_col:
        weather_fig = fig_weather_pr(matches, pairs, title="PR moyen vs température moyenne à Paris")
        if weather_fig is not None:
            compact_chart(weather_fig, "insight_2_weather", height=315)
        else:
            st.info("Graphique météo indisponible hors connexion.")
    with duration_col:
        compact_chart(fig_duration_pr(matches, title="PR moyen selon longueur du match"), "insight_2_duration", height=315)

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    # 3 — Texte à gauche, visuel à droite
    text_col, chart_col = st.columns([1.04, 1])
    with text_col:
        insight_text(
            3,
            "Niveau de l’adversaire",
            "Le niveau adverse explique le PR - mais uniquement chez les très forts.",
            "Le niveau de l’adversaire n’a que d’impact net sur la performance des joueurs les plus forts, qui affichent des performances étincelantes entre eux mais qui souffrent contre les joueurs intermédiaires. Parmi les autres, la relation est beaucoup moins claire.",
            "À retenir : les joueurs élites gèrent mal la baisse de précision chez les joueurs avancés et intermédiaires.",
        )
    with chart_col:
        compact_chart(fig_category_heatmap(matches, players, value="PR_moyen", title="PR moyen : catégorie vs adversaire"), "insight_3_heatmap", height=285)

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    # 4 — Visuel à gauche, texte à droite
    chart_col, text_col = st.columns([1, 1.04])
    with chart_col:
        compact_chart(fig_category_distribution(player_tournament, normalize=True, title="Répartition des catégories PR"), "insight_4_categories", height=285)
    with text_col:
        insight_text(
            4,
            "Niveau du plateau",
            "Les VBC attirent de plus en plus de joueurs forts.",
            "Le nombre de joueurs au niveau M3 ou plus fort, soit PR < 6,50, augmente régulièrement. En revanche, il n’y a pas encore de développement clair du PR moyen dans les catégories de joueurs.",
            "Interprétation prudente : les tournois semblent attirer progressivement de bons joueurs plutôt que les créer directement.",
        )

def tab_key_charts(data: dict[str, pd.DataFrame]) -> None:
    matches, pairs = data["matches"], data["pairs"]
    players = data["players"]
    player_tournament = data["player_tournament"]
    ok_pairs = pairs[pairs["QualityStatus"].eq("OK")].copy()
    tournaments = build_tournament_stats(matches, pairs)

    st.markdown("## Graphiques clés")
    st.caption("Version détaillée des visuels utilisés dans les insights, avec quelques vues complémentaires pour comprendre le rapport.")

    st.markdown("### 1. PR et taux de victoire")
    left, right = st.columns(2)
    with left:
        chart(fig_better_pr_donut(pairs), "key_better_pr_pie")
    with right:
        ok_pairs["Écart PR"] = ok_pairs["PR_Difference_Abs"]
        ok_pairs["Tranche d'écart"] = pd.cut(ok_pairs["Écart PR"], bins=[0,1,2,3,5,8,12,np.inf], labels=["0–1","1–2","2–3","3–5","5–8","8–12","12+"], include_lowest=True, right=False)
        diff_stats = ok_pairs.groupby("Tranche d'écart", observed=False).agg(Matchs=("MatchID", "count"), Taux=("BetterPRWon", "mean")).reset_index()
        fig = px.bar(diff_stats, x="Tranche d'écart", y="Taux", text=diff_stats["Taux"].map(lambda v: f"{v:.0%}" if pd.notna(v) else ""), title="Victoire du meilleur PR selon l’écart de PR", labels={"Taux": "Taux de victoire"})
        fig.update_yaxes(tickformat=".0%", range=[0,1])
        chart(fig, "key_better_pr_by_gap")

    st.markdown("### 2. Fatigue, météo et durée")
    chart(fig_pr_by_round(matches, title="Effet fatigue : PR moyen par ronde"), "key_round_pr")

    weather_fig = fig_weather_pr(matches, pairs, title="PR moyen vs température moyenne à Paris")
    if weather_fig is not None:
        chart(weather_fig, "key_weather_pr_wide")
    else:
        st.warning("Données météo indisponibles pour l’instant. Ce graphique nécessite une connexion internet.")

    chart(fig_duration_pr(matches, title="PR moyen selon la longueur du match"), "key_duration_pr")

    st.markdown("### 3. Catégorie joueur vs catégorie adversaire")
    left, right = st.columns(2)
    with left:
        chart(fig_category_heatmap(matches, players, value="PR_moyen", title="PR moyen selon catégorie joueur/adversaire"), "key_category_pr_heatmap", height=450)
    with right:
        chart(fig_category_heatmap(matches, players, value="Taux_victoire", title="Taux de victoire selon catégorie joueur/adversaire"), "key_category_win_heatmap", height=450)

    st.markdown("### 4. Niveau du plateau et progression")
    left, right = st.columns(2)
    with left:
        chart(fig_category_distribution(player_tournament, normalize=True, title="Répartition des catégories PR par tournoi"), "key_category_distribution")
    with right:
        progression = build_global_category_progression(matches, players)
        fig = px.line(progression, x="Tournoi", y="PR_moyen", color="Catégorie joueur", markers=True, title="Progression du PR moyen selon catégorie globale", labels={"PR_moyen": "PR moyen"})
        chart(fig, "key_category_progression")

    st.markdown("### Vues complémentaires")
    left, right = st.columns(2)
    with left:
        fig = px.line(tournaments, x="Tournoi", y="PR_moyen", markers=True, title="PR moyen global par tournoi", labels={"PR_moyen": "PR moyen"})
        chart(fig, "key_tournament_pr_trend")
    with right:
        player_tournament = player_tournament.copy()
        player_tournament["M3 ou mieux"] = player_tournament["AvgPR"] < 6.5
        m3 = player_tournament.groupby("TournamentDate", as_index=False).agg(Joueurs=("PlayerName", "nunique"), Joueurs_M3=("M3 ou mieux", "sum"))
        m3["Part M3 ou mieux"] = m3["Joueurs_M3"] / m3["Joueurs"]
        m3["Tournoi"] = m3["TournamentDate"].dt.strftime("%Y-%m-%d")
        fig = px.line(m3, x="Tournoi", y="Part M3 ou mieux", markers=True, title="Part des joueurs M3 ou mieux (PR < 6,50)", labels={"Part M3 ou mieux": "Part des joueurs"})
        fig.update_yaxes(tickformat=".0%")
        chart(fig, "key_m3_share")


def tab_limits() -> None:
    st.markdown("## Limites statistiques")
    st.markdown(
        """
Cette section est importante pour une lecture recruteur : elle montre que l’analyse ne se limite pas aux graphiques et que les résultats sont interprétés avec prudence.

- **Taille d’échantillon encore modeste** : 18 tournois constituent une bonne base exploratoire, mais certaines analyses deviennent fragiles lorsqu’on filtre par joueur, ronde, catégorie PR, température ou rivalité.
- **Matchs courts** : beaucoup de matchs sont joués sur 7 à 9 points. Sur ces formats, un joueur peut mieux jouer techniquement mais perdre à cause de la variance.
- **Variance naturelle du backgammon** : le résultat d’un match dépend aussi des dés, du cube, du score et de la dynamique de match. Le PR mesure la qualité technique, pas la garantie de victoire.
- **Biais de sélection** : les VBC attirent des joueurs motivés par la compétition et les titres BMAB. L’amélioration du plateau peut donc refléter une attraction de bons joueurs plutôt qu’un effet d’entraînement.
- **Effet des joueurs très réguliers** : les participants les plus présents pèsent davantage dans les moyennes globales. Une analyse future devrait séparer réguliers et occasionnels.
- **Corrélation ≠ causalité** : fatigue, température, durée de match et niveau adverse sont des associations exploratoires. Elles ne prouvent pas une relation causale.
- **Proxy imparfaits** : le nombre de décisions sert d’approximation de la durée/complexité, et la météo extérieure de Paris ne garantit pas la température réelle dans la salle.
"""
    )

    st.markdown("## Ce que ce projet démontre")
    st.markdown(
        """
- Nettoyage et normalisation de données issues de fichiers CSV hétérogènes.
- Construction d’un modèle analytique : joueur-ronde, match reconstruit, tournoi, joueur.
- Contrôles qualité : alias joueurs, anomalies numériques, matchs incomplets.
- Analyse exploratoire et visualisation interactive.
- Capacité à communiquer les résultats, les limites et les prochaines étapes.
"""
    )


def main() -> None:
    data = load_data()
    hero()
    report, insights, charts_tab, limits = st.tabs(["Rapport", "Insights", "Graphiques clés", "Limites & code"])
    with report:
        tab_report(data)
    with insights:
        tab_insights(data)
    with charts_tab:
        tab_key_charts(data)
    with limits:
        tab_limits()


if __name__ == "__main__":
    main()
