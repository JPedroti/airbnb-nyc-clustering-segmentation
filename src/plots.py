"""Gráficos compartilhados pelos notebooks e scripts (matplotlib, PNG estático, tema claro).

Convenções: as cores categóricas seguem uma ordem fixa e acompanham a categoria (nunca o ranking);
as marcas são finas e a grade é discreta; há legenda sempre que existem duas ou mais séries. Em
gráficos de dispersão (mapa, PCA), no máximo três cores aparecem juntas. Acima disso, o gráfico vira
pequenos múltiplos, com um painel por categoria destacado sobre o restante em cinza.
"""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from src import config

# Paleta de referência (modo claro), na ordem validada para daltonismo.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT_GRAY = "#d9d8d1"  # pontos de contexto nos pequenos múltiplos
NOISE_GRAY = "#898781"  # ruído do DBSCAN: sem identidade de segmento

# Latitude média de NYC: 1 grau de longitude vale cerca de cos(40,73°) de 1 grau de latitude.
NYC_MEAN_LATITUDE = 40.73


def apply_style() -> None:
    """Aplica o tema a todos os gráficos da sessão."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=SERIES_COLORS),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.dpi": 100,
        "savefig.dpi": 150,
    })


def fmt_int(value: float) -> str:
    """Inteiro com separador de milhar em português (21.660)."""
    return f"{value:,.0f}".replace(",", ".")


def fmt_pct(share: float, decimals: int = 1) -> str:
    """Fração em percentual com vírgula decimal (0,443 -> 44,3%)."""
    return f"{share * 100:.{decimals}f}%".replace(".", ",")


def plain_log_axis(axis) -> None:
    """Eixo em escala log com rótulos 1, 2, 5, 10, 20, 50... em vez de notação científica."""
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda value, _: fmt_int(value) if value >= 1 else f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())


def category_colors(order: Sequence[str]) -> dict[str, str]:
    """Cor fixa por categoria, na ordem dada (no máximo 8 categorias)."""
    if len(order) > len(SERIES_COLORS):
        raise ValueError("Mais de 8 categorias: agrupe a cauda em 'Outros' ou use pequenos múltiplos.")
    return dict(zip(order, SERIES_COLORS))


def save_figure(fig: plt.Figure, name: str) -> None:
    """Salva em reports/figures/<name>.png."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", bbox_inches="tight")


def set_geographic_aspect(ax: plt.Axes) -> None:
    """Corrige a proporção do mapa: grau de longitude é menor que grau de latitude em NYC."""
    ax.set_aspect(1 / np.cos(np.deg2rad(NYC_MEAN_LATITUDE)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")


def scatter_small_multiples(
    frame: pd.DataFrame,
    x: str,
    y: str,
    column: str,
    order: Sequence,
    colors: dict | None = None,
    titles: dict | None = None,
    ncols: int = 3,
    point_size: float = 1.0,
    geographic: bool = False,
    panel_size: tuple[float, float] = (4.2, 4.4),
) -> plt.Figure:
    """Um painel por categoria: a categoria em cor, o restante dos pontos em cinza claro.

    Evita pôr mais de três cores no mesmo gráfico de dispersão. Cada painel informa n e a participação.
    `titles` permite trocar o rótulo da categoria (ex.: id do cluster -> nome do segmento).
    """
    colors = category_colors(order) if colors is None else colors
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False)
    total = len(frame)
    for ax, category in zip(axes.flat, order):
        subset = frame[frame[column] == category]
        ax.scatter(frame[x], frame[y], s=point_size, color=CONTEXT_GRAY, linewidths=0, rasterized=True)
        ax.scatter(subset[x], subset[y], s=point_size, color=colors[category], linewidths=0, rasterized=True)
        label = category if titles is None else titles.get(category, category)
        ax.set_title(f"{label}\nn = {fmt_int(len(subset))} ({fmt_pct(len(subset) / total)})", fontsize=9.5)
        if geographic:
            set_geographic_aspect(ax)
        else:
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        ax.grid(False)
    for ax in list(axes.flat)[len(order):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def map_small_multiples(
    frame: pd.DataFrame,
    column: str,
    order: Sequence,
    colors: dict | None = None,
    titles: dict | None = None,
    ncols: int = 3,
    point_size: float = 1.0,
) -> plt.Figure:
    """Mapas longitude x latitude (coordenadas originais, proporção corrigida), um por categoria."""
    return scatter_small_multiples(frame, "longitude", "latitude", column, order, colors=colors, titles=titles,
                                   ncols=ncols, point_size=point_size, geographic=True)


def boxplot_by_group(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value: str,
    group: str,
    order: Sequence,
    log_scale: bool = False,
    color: str = SERIES_COLORS[0],
    tick_labels: Sequence[str] | None = None,
) -> None:
    """Boxplot de `value` por `group`, com bigodes em p5 e p95 e sem pontos extremos.

    Uma única série: todas as caixas na mesma cor (a categoria já está no eixo).
    """
    data = [frame.loc[frame[group] == g, value].dropna().to_numpy() for g in order]
    parts = ax.boxplot(
        data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.55,
        medianprops={"color": INK, "linewidth": 1.5},
        whiskerprops={"color": INK_MUTED, "linewidth": 1},
        capprops={"color": INK_MUTED, "linewidth": 1},
        boxprops={"linewidth": 0},
    )
    for box in parts["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.55)
    ax.set_xticks(range(1, len(order) + 1), [str(g) for g in order] if tick_labels is None else list(tick_labels))
    ax.set_ylabel(value)
    if log_scale:
        ax.set_yscale("log")
        plain_log_axis(ax.yaxis)
    ax.grid(axis="x", visible=False)


def stacked_share_bars(
    ax: plt.Axes,
    shares: pd.DataFrame,
    colors: dict[str, str] | None = None,
    label_min: float = 8.0,
) -> None:
    """Barras horizontais 100% empilhadas: linhas de `shares` = barras, colunas = segmentos (em %).

    Rótulo dentro do segmento só quando ele tem pelo menos `label_min` pontos percentuais.
    """
    colors = category_colors(list(shares.columns)) if colors is None else colors
    left = np.zeros(len(shares))
    y = np.arange(len(shares))
    for column in shares.columns:
        widths = shares[column].to_numpy()
        ax.barh(y, widths, left=left, color=colors[column], edgecolor=SURFACE, linewidth=2,
                height=0.7, label=column)
        for yi, (start, width) in enumerate(zip(left, widths)):
            if width >= label_min:
                ax.text(start + width / 2, yi, f"{width:.0f}%", ha="center", va="center",
                        fontsize=8, color=INK)
        left += widths
    ax.set_yticks(y, list(shares.index))
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% dos anúncios")
    ax.grid(axis="y", visible=False)
    ax.legend(ncols=min(len(shares.columns), 5), loc="upper center", bbox_to_anchor=(0.5, -0.18))
