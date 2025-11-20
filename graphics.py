# Все библиотеки

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_ind, levene, mannwhitneyu,shapiro, \
    f_oneway, kruskal, chi2_contingency, fisher_exact
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pingouin as pg
import sys
import os
from ydata_profiling import ProfileReport
from matplotlib import rcParams
from matplotlib import font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
import matplotlib as mpl
from typing import List, Dict, Optional, Any
import warnings

mpl.rcParams['font.family'] = 'HSE Sans'
plt.rcParams['figure.dpi'] = 300

warnings.filterwarnings('ignore')


def __info__():
    text = ("В этой библиотеке:\n - Столбчатая диаграмма (plot_bar_chart)\n - линейную диаграмму (просто y от x, plot_line_chart)\n"
            " - Столбчатая диаграмма с успешной частью (plot_bar_chart_with_success)\n - круговая диаграмма (norm_pie)\n"
            " - Гисотграмма (может выделить выбросы; norm_hist)\n - Хитмапа (plot_heatmap)\n - Скаттерплот (norm_scatter)\n"
            " - Матрица корреляций (mixed_corr_heatmaps)\n\n"
            "Для подробной информации по каждой функе - func.__doc__")
    print(text)

def plot_bar_chart(x, y, title=None, color='#4A90E2', bgcolor='#F5F5F5', figsize=(8, 5)):
    """
    Рисует столбчатую диаграмму с подписями значений над столбцами.

    Must-Have:
    x : list/array-like — подписи для оси X
    y : list/array-like — высоты столбцов
    title : str — заголовок графика

    Лучше не трогай:
    color : str — цвет столбцов (hex или имя)
    bgcolor : str — цвет фона графика
    figsize : pair - figsize
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Фон
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)

    # Построение столбцов
    bars = ax.bar(x, y, color=color, edgecolor='black')

    # Подписи над столбцами
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # отступ в пикселях
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, color='black')

    # Заголовок
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Убираем рамку сверху и справа (минимализм)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_line_chart(x, y, title=None, points_values=True, color='#4A90E2', bgcolor='#F5F5F5', figsize=(8, 5), marker='o'):
    """
    Рисует линейный график с подписями значений в каждой точке.

    Must-Have:
    x : list/array-like — координаты по оси X (одинаковая длина с y)
    y : list/array-like — координаты по оси Y
    title : str | None — заголовок (можно не задавать)

    Если не нужны подписи над точками (значение y), то сделай:
    points_values = False

    Лучше не трогай:
    color : str — цвет линии (hex или имя)
    bgcolor : str — цвет фона области
    figsize : tuple — размер графика
    marker : str — маркер для точек

    На вход должны идти последовательности одинаковой длины.
    Если x несортирован — линия пойдёт именно в том порядке, в котором ты задал.
    Подписи ставятся прямо над точками.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Фон
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)

    # Построение линии
    ax.plot(x, y, color=color, marker=marker, linewidth=2)

    # Подписи точек
    if points_values:
        for xi, yi in zip(x, y):
            ax.annotate(f'{yi:.2f}',
                        xy=(xi, yi),
                        xytext=(0, 5),  # смещение текста
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, color='black')

    # Заголовок
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Убираем рамки сверху и справа
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_bar_chart_with_success(
    x, y, success,
    title=None,
    color='#4A90E2',
    success_color='#50E3C2',
    bgcolor='#F5F5F5',
    figsize=(8, 5)
):
    """
    Рисует составной столбчатый график: «успешная» часть у основания + остаток сверху.
    Показывает процент успеха внутри каждого столбца и общее значение над ним.

    Must-Have:
    x : list/array-like — подписи оси X
    y : list/array-like — общие значения (неотрицательные)
    success : list/array-like — успешные значения (каждое <= соответствующего y)

    Уточнения:
    – success и y должны быть одной длины.
    – Если success > y — график будет некорректен (логической проверки нет, не допускай мусор на вход).

    Лучше не трогай:
    title : str | None — заголовок
    color : str — цвет «верхней» части столбца
    success_color : str — цвет успешной части
    bgcolor : str — фон графика
    figsize : tuple — размер фигуры

    Функция автоматически строит stacked-bar: success рисуется снизу,
    остальная часть достраивается сверху. Подписи процентов стоят над успехом,
    общий итог — над верхушкой столбца.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Фон
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)

    # Часть успеха у основания
    success_bars = ax.bar(x, success, color=success_color, edgecolor='black')

    # Остаток над успехом
    remaining = [total - succ for total, succ in zip(y, success)]
    ax.bar(x, remaining, bottom=success, color=color, edgecolor='black')

    # Подписи процентов над успешной частью
    for i, (succ, total, bar) in enumerate(zip(success, y, success_bars)):
        if total > 0:
            percent = succ / total * 100
            ax.annotate(f'{percent:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, succ),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, color='black')

    # Подписи общего значения сверху
    for xi, total in zip(x, y):
        ax.annotate(f'{total:.0f}',
                    xy=(xi, total),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, color='black', fontweight='bold')

    # Заголовок
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Минимализм
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def norm_pie(
    series,
    colors=None,
    bgcolor="#F5F5F5",
    title=None,
    figsize=(7, 7),
    label_fontsize=12
):
    """
    Рисует аккуратную круговую диаграмму по категориальным данным Series.
    Отображает количество в каждой категории прямо в подписи сектора.

    Must-Have:
    series : pandas.Series — категориальные данные.
              Будет взят value_counts(), порядок — по убыванию частот.

    Уточнения:
    – Значения должны быть сравнимыми/хешируемыми (обычный Series с категориальными или строками).
    – Если категорий больше 6 и colors не задан — лишние просто не получат уникальный цвет.

    Лучше не трогай:
    colors : list | None — цвета для секторов (обрезаются по количеству категорий)
    bgcolor : str — фон всего графика
    title : str | None — заголовок
    figsize : tuple — размер фигуры
    label_fontsize : int — размер шрифта подписей

    Функция сама формирует подписи вида "Категория (число)", равняет диаграмму в круг
    и использует минималистичную палитру с белыми разделителями между секторами.
    """

    counts = series.value_counts()
    labels = [f"{cat} ({cnt})" for cat, cnt in zip(counts.index, counts.values)]

    # Дефолтная палитра (максимум 6 категорий)
    if colors is None:
        colors = ["#4A90E2", "#6A5ACD", "#7B68EE", "#9370DB", "#8A2BE2", "#6A0DAD"]

    fig, ax = plt.subplots(figsize=figsize)

    # Фон
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)

    wedges, texts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors[:len(counts)],
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2)
    )

    # Настроим подписи
    for text in texts:
        text.set_fontsize(label_fontsize)
        text.set_color("black")
        text.set_fontweight("bold")

    # Заголовок
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    # Минимализм
    ax.set(aspect="equal")  # круг, а не овал
    plt.tight_layout()
    plt.show()




def norm_hist(
    series,
    bins=10,
    color='#4A90E2',
    edgecolor='black',
    bgcolor='#F5F5F5',
    title=None,
    figsize=(8, 5),
    podpisi=True,
    outliers_min=None,
    outliers_max=None,
    outlier_color='#6a0dad',
    outliers_text='Выбросы',
):
    """
    Рисует гистограмму для числовых данных Series с опциональным выделением выбросов.
    Может подписывать высоты столбцов и красить столбцы, попавшие под outliers_min / outliers_max.

    Must-Have:
    series : pandas.Series — числовые данные.
              Допускаются float/int, пропуски желательно заранее выкинуть.
    title : str | None — заголовок

    Выбросы (или просто выделить цветом):
    outliers_min : numeric - ставишь минимальный порог, меньше которого будет считататься выбросом
    outliers_max : numeric - ставишь максимальный порог, больше которого будет считататься выбросом
    outliers_text : str | "Выбросы" - текст для легенды

    Уточнения:
    – bins управляет количеством столбцов (обычный параметр matplotlib).
    – Выбросы определяются по границам корзин:
      если столбце полносью ≤ outliers_min или ≥ outliers_max — корзина считается выбросной.
    – Если заданы пороги и хоть один столбец попадает в условие — будет легенда.

    Подписи значений:
    podpisi : bool — подписывать числовое значение или нет

    Лучше не трогай:
    bins : int | 10 - кол-во столбцов
    color : str — цвет обычных столбцов
    outlier_color : str — цвет столбцов-выбросов
    edgecolor : str — границы столбцов
    bgcolor : str — фон
    figsize : tuple — размер графика
    outliers_min / outliers_max : float | None — пороги для определения выбросов
    outliers_text : str — текст в легенде

    Функция сама считает частоты, красит каждый столбец по условию,
    ставит подписи только для count > 0 и добавляет легенду, если выбросы реально присутствуют.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Фон
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)

    # Строим гистограмму
    counts, bin_edges, patches = ax.hist(series, bins=bins, edgecolor=edgecolor)

    has_outliers = False  # флаг для проверки, есть ли выбросы

    # Красим колонки
    for count, left, right, patch in zip(counts, bin_edges[:-1], bin_edges[1:], patches):
        if (outliers_max is not None and left >= outliers_max) or \
           (outliers_min is not None and right <= outliers_min):
            patch.set_facecolor(outlier_color)
            has_outliers = True
        else:
            patch.set_facecolor(color)

        # Подписи значений
        if podpisi and count > 0:
            ax.annotate(f'{int(count)}',
                        xy=((left + right) / 2, count),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, color='black')

    # Заголовок
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Легенда (только если заданы пороги)
    if (outliers_min is not None or outliers_max is not None) and has_outliers:
        outlier_patch = mpatches.Patch(color=outlier_color, label=outliers_text)
        ax.legend(handles=[outlier_patch], fontsize=10, frameon=False)

    # Минимализм
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# при необходимости определите эти цвета глобально (как в основном коде)
BG = "#F5F5F5"      # фон всей фигуры
PURPLE = "#6a0dad"  # насыщенный фиолетовый (max)
_LIGHT_PURPLE = "#F2E7FB"  # слабый лавандовый (min)

def plot_heatmap(
    data,
    x_labels=None,
    y_labels=None,
    title: str = None,
    cmap=None,                # если None — используем лавандо->фиолет
    bgcolor: str = BG,
    figsize: tuple = (8, 6),
    annot: bool = True,
    fmt: str = '.2f',
    cbar_label: str = None,
    vmin=None,
    vmax=None,
    center=None,
    linewidths: float = 0.8,  # тонкие "gaps" между ячейками
    linecolor: str = None,
    show: bool = True
):
    """
    Аккуратная heatmap с небольшими промежутками между ячейками и фиолетовой палитрой.
    Возвращает (fig, ax).
    """
    if linecolor is None:
        linecolor = bgcolor

    data = np.array(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be 2D (matrix-like)")

    nrows, ncols = data.shape

    # fallback для пустых данных
    if np.all(np.isnan(data)):
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = 1.0

    # default cmap: слабый лавандовый -> насыщенный фиолетовый
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("lav_purple", [_LIGHT_PURPLE, PURPLE])
    # попытка задать цвет для masked/NaN
    try:
        cmap.set_bad(bgcolor)
    except Exception:
        pass

    # пределы (игнорируем NaN)
    if vmin is None:
        try:
            vmin = np.nanmin(data)
        except ValueError:
            vmin = 0.0
    if vmax is None:
        try:
            vmax = np.nanmax(data)
        except ValueError:
            vmax = 1.0

    # нормировка
    if center is not None:
        norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # маскируем NaN — pcolormesh использует cmap.set_bad для masked
    mdata = np.ma.masked_invalid(data)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)

    # координаты для pcolormesh: (ncols+1) x (nrows+1)
    X = np.arange(ncols + 1)
    Y = np.arange(nrows + 1)

    # pcolormesh рисует "плитки" — edgecolors=bg задаёт видимые промежутки
    mesh = ax.pcolormesh(X, Y, mdata, cmap=cmap, norm=norm,
                         edgecolors=bgcolor, linewidth=linewidths, shading='auto')

    # инвертируем Y так, чтобы [0,0] был в верхнем левом углу (как в imshow)
    ax.set_xlim(0, ncols)
    ax.set_ylim(nrows, 0)

    # colorbar
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=10)

    # подписи осей (центры ячеек)
    if x_labels is None:
        x_labels = [str(i) for i in range(ncols)]
    if y_labels is None:
        y_labels = [str(i) for i in range(nrows)]

    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels(y_labels, fontsize=9)

    # подписи в ячейках
    if annot:
        denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        for i in range(nrows):
            for j in range(ncols):
                val = data[i, j]
                if np.isnan(val):
                    continue
                try:
                    text = format(val, fmt)
                except Exception:
                    text = f"{val:.2f}"
                # относительная яркость (0..1) для выбора цвета текста
                rel = (val - vmin) / denom
                text_color = 'white' if rel > 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', fontsize=9, color=text_color)

    # заголовок
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10, backgroundcolor=bgcolor)

    # минимализм — убрать верх/правую рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax

def norm_scatter(
    x,
    y,
    c=None,
    colors=None,
    bgcolor="#F5F5F5",
    title=None,
    fontweight='regular',
    figsize=(7, 5),
    alpha=0.8,
    size=50
):
    """
    Рисует минималистичный scatter plot.
    Поддерживает как обычное рассеяние, так и цветовую группировку по категориям.

    Must-Have:
    x, y : array-like — координаты точек одинаковой длины.

    Уточнения:
    – Если c не задан, все точки одной краской.
    – Если c задан, каждая категория получает свой цвет, подписи выводятся сбоку.
    – categories приводятся к строкам, так что смешанные типы не ломают график.

    Лучше не трогай:
    colors : list | None — палитра (циклически используется при большем числе категорий)
    bgcolor : str — фон графика
    title : str | None — заголовок
    fontweight : str — стиль надписи заголовка
    figsize : tuple — размер графика
    alpha : float — прозрачность точек
    size : int — размер точек

    Функция автоматически раскрашивает категории, рисует числа объектов в каждой,
    поддерживает равномерный минималистичный стиль и аккуратные подписи сбоку вместо стандартной легенды.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Фон
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)

    # Палитра
    if colors is None:
        colors = ["#4A90E2", "#6A5ACD", "#7B68EE", "#9370DB", "#8A2BE2", "#6A0DAD"]

    if c is None:
        # без категорий — все одной краской
        ax.scatter(x, y, s=size, alpha=alpha, color=colors[0], edgecolor="black", linewidth=0.5)
    else:
        categories = pd.Series(c).astype(str)
        unique_cats = categories.unique()
        for i, cat in enumerate(unique_cats):
            mask = categories == cat
            ax.scatter(
                x[mask],
                y[mask],
                s=size,
                alpha=alpha,
                color=colors[i % len(colors)],
                edgecolor="black",
                linewidth=0.5,
                label=f"{cat} ({mask.sum()})"
            )
        # Подписи вместо легенды
        for i, cat in enumerate(unique_cats):
            ax.text(
                1.02, 0.95 - i*0.05,
                f"{cat} ({(categories==cat).sum()})",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                color=colors[i % len(colors)],
                ha="left", va="center"
            )

    # Заголовок
    if title:
        ax.set_title(title, fontsize=14, fontweight=fontweight)

    # Минимализм
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()




# Универсальная функцию для построения трёх heatmap'ов для смешанных типов данных.
# - числовые-числовые: корреляция (pearson/spearman) в [-1,1]
# - категориальные-категориальные: Cramér's V в [0,1]
# - числовые-категориальные: eta-squared (ANOVA effect size) в [0,1]
#
# Как интерпретировать резульаты:
# - pearson/spearman или eta-squared -> 0.1/0.3/0.5/0.8 -> слабая/средняя/сильная/оч сильная
# - Cramér's V -> 0.01/0.06/0.14 -> слабая/средняя/сильная
#
# Функция отображает три отдельных графика (по одному на каждую метрику) и возвращает
# соответствующие DataFrame'ы.
#


# -----------------------
# Метрики (твои функции)
# -----------------------
def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    x2 = x[mask]
    y2 = y[mask]
    if x2.size == 0:
        return np.nan
    table = pd.crosstab(x2, y2)
    if table.size == 0:
        return np.nan
    n = table.values.sum()
    r, c = table.shape
    k = min(r - 1, c - 1)
    if k <= 0 or n == 0:
        return 0.0
    chi2, p, dof, expected = chi2_contingency(table, correction=False)
    return float(np.sqrt(chi2 / (n * k)))

def _eta_squared(numeric: pd.Series, categ: pd.Series) -> float:
    mask = numeric.notna() & categ.notna()
    x = numeric[mask]
    g = categ[mask]
    if x.size == 0:
        return np.nan
    overall_mean = x.mean()
    groups = x.groupby(g)
    ss_between = ((groups.size() * (groups.mean() - overall_mean) ** 2).sum())
    ss_total = ((x - overall_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    return float(ss_between / ss_total)

# -----------------------
# Визуальные хелперы
# -----------------------
BG = "#F5F5F5"      # фон всей фигуры
MID = "#EAEAEA"     # цвет в нуле (чуть темнее фона — заметно, но деликатно)
PURPLE = "#6a0dad"  # цвет на −1 и +1

def _make_cmap_symmetric(purple=PURPLE, mid=MID):
    # симметричная: purple <- mid -> purple
    cmap = LinearSegmentedColormap.from_list("sym_purple", [purple, mid, purple])
    cmap.set_bad(BG)  # masked/NaN -> фон
    return cmap

def _make_cmap_pos(purple=PURPLE, mid=MID):
    # для 0..1 (градиент от mid к purple)
    cmap = LinearSegmentedColormap.from_list("pos_purple", [mid, purple])
    cmap.set_bad(BG)
    return cmap

CMAP_SYM = _make_cmap_symmetric()
CMAP_POS = _make_cmap_pos()

def _mask_upper_triangle_df(df: pd.DataFrame):
    # mask True — те ячейки, которые хотим скрыть
    arr = np.triu(np.ones(df.shape, dtype=bool))
    return arr

# -----------------------
# Основная функция
# -----------------------
def mixed_corr_heatmaps(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    method_num: str = "mixed",
    decimals: int = 2,
    figsize_each: tuple = (6,6),
    empty_cell_color: str = "#EEDDFF",
    show: bool = True
) -> Dict[str, Any]:
    """
    Построение трех heatmap'ов: числовые×числовые (симметрично, маскируем сверху),
    категориальные×категориальные (симметрично, маскируем сверху), и числовые×категориальные (полная матрица).

    method_num - 'mixed' or 'pearson' or 'spearman'. used mixed

    Оформление: фон #F5F5F5, ноль — чуть темнее фона, +1/-1 — фиолетовый
    Маленькие промежутки между ячейками; лишние линии и сетки убраны.
    """
    # ----- выбор колонок -----
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
    else:
        categorical_cols = [c for c in categorical_cols if c in df.columns]
    numeric_cols = [c for c in df.columns if c not in categorical_cols]

    # ----- матрицы -----
    # 1) числовые-числовые
    if len(numeric_cols) > 0:
        corr_num = pd.DataFrame(index=numeric_cols, columns=numeric_cols, dtype=float)
        for i in numeric_cols:
            for j in numeric_cols:
                if i == j:
                    corr_num.loc[i, j] = 1.0
                else:
                    # Спирмен
                    v1, spearman_p = spearmanr(df[i], df[j], nan_policy='omit')
                    # Пирсон
                    v2, pearson_p = pearsonr(df[i], df[j])
                    if method_num == "mixed":
                        corr_num.loc[i, j] = v1 if abs(v1) > abs(v2) else v2
                    elif method_num == "pearson":
                        corr_num.loc[i, j] = v2
                    elif method_num == "spearman":
                        corr_num.loc[i, j] = v1
                    else:
                        print("Warning: Wrong method_num. must be 'mixed' or 'pearson' or 'spearman'. used mixed")
                        corr_num.loc[i, j] = v1 if abs(v1) > abs(v2) else v2

    else:
        corr_num = pd.DataFrame(index=[], columns=[])

    # 2) категориальные-категориальные
    if len(categorical_cols) > 0:
        cramers = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)
        for i in categorical_cols:
            for j in categorical_cols:
                if i == j:
                    cramers.at[i, j] = 1.0
                else:
                    cramers.at[i, j] = _cramers_v(df[i], df[j])
        cramers = cramers.astype(float)
    else:
        cramers = pd.DataFrame(index=[], columns=[])

    # 3) числовые-категориальные
    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        eta = pd.DataFrame(index=numeric_cols, columns=categorical_cols, dtype=float)
        for num in numeric_cols:
            for cat in categorical_cols:
                eta.at[num, cat] = _eta_squared(df[num], df[cat])
        eta = eta.astype(float)
    else:
        eta = pd.DataFrame(index=numeric_cols, columns=categorical_cols)

    figs = []

    # ----- plotting helper (seaborn) -----
    def _plot_matrix(matrix: pd.DataFrame, title: str, vmin, vmax, cmap, mask_upper: bool):
        if matrix.size == 0:
            return None
        fig, ax = plt.subplots(figsize=figsize_each)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        # mask: True -> скрыть (sns.heatmap ожидает mask с True где скрывать)
        mask = None
        if mask_upper:
            mask = _mask_upper_triangle_df(matrix)

        # linewidths + linecolor==BG создают тонкие пробелы между клетками
        sns.heatmap(
            data=matrix,
            mask=mask,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0 if (vmin < 0 and vmax > 0) else None,
            annot=matrix.round(decimals),
            fmt=f'.{decimals}f',
            annot_kws={"size": 8},
            linewidths=0.8,            # небольшой gap
            linecolor=BG,              # gap цвет = фон -> выглядит как пустые клетки
            cbar=True,
            square=False,
            ax=ax,
            xticklabels=matrix.columns,
            yticklabels=matrix.index
        )

        if mask_upper:
            for (i, j), val in np.ndenumerate(mask):
                if val:  # если эта ячейка была замаскирована
                    ax.add_patch(plt.Rectangle(
                        (j, i), 1, 1,
                        facecolor=empty_cell_color,
                        edgecolor=BG,
                        linewidth=0.8
                    ))

        # оформление: убрать спины, убрать сетки, сделать подписи аккуратно
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_title(title, pad=10, backgroundcolor=BG)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ----- постройка графиков (правильные mask_upper для каждого) -----
    # num×num — симметрично, маскируем диагональ и выше
    if corr_num.size != 0:
        figs.append(_plot_matrix(corr_num, f'Числовые vs Числовые (максимум из пирсона и спирмена:) )', -1, 1, CMAP_SYM, mask_upper=True))

    # cat×cat — симметрично, маскируем диагональ и выше
    if cramers.size != 0:
        figs.append(_plot_matrix(cramers, "Категориальные vs Категориальные (Cramér's V)", 0, 1, CMAP_POS, mask_upper=True))

    # num×cat — несимметрично, показываем всё
    if eta.size != 0:
        figs.append(_plot_matrix(eta, "Числовые vs Категориальные (eta-squared)", 0, 1, CMAP_POS, mask_upper=False))

    # удалить возможные None из списка (если какая-то матрица была пустой)
    figs = [f for f in figs if f is not None]

    return {
        'num_num': corr_num,
        'cat_cat': cramers,
        'num_cat': eta,
        'figs': figs
    }