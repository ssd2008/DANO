# Все библиотеки

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_ind, levene, mannwhitneyu,shapiro, \
    f_oneway, kruskal, chi2_contingency, fisher_exact, pearsonr, spearmanr
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
import matplotlib as mpl
import warnings

mpl.rcParams['font.family'] = 'HSE Sans'
plt.rcParams['figure.dpi'] = 300

warnings.filterwarnings('ignore')

import graphics

def ztest(counts, nobs): # counts = кол-во успехов, nobs - размеры выборок (nobs >= counts)
    from statsmodels.stats.proportion import proportions_ztest
    z_stat, p_value = proportions_ztest(counts, nobs, alternative='two-sided')
    print("z =", z_stat, "p-value =", p_value)
    return z_stat, p_value

def print_correlations(x, y):
    # Спирмен
    spearman_corr, spearman_p = spearmanr(x, y, nan_policy='omit')
    # Пирсон
    pearson_corr, pearson_p = pearsonr(x, y)

    print(f"Корреляция Спирмена: {spearman_corr:.2f}, p-value = {spearman_p:.4f}")
    print(f"Корреляция Пирсона: {pearson_corr:.2f}, p-value = {pearson_p:.4f}")


def normal_means(x):
    if x.shape[0] > 70:
        return True
    p_shapiro = shapiro(x)[-1]
    return p_shapiro > 0.05


def equal_var(*s):
    p_levene = levene(*s)[-1]
    if p_levene > 0.05:
        print('Дисперсии выборок равны')
    else:
        print('Дисперсии выборок НЕ равны')
    return p_levene


def compare_mean(df, num_var, cat_var):
    groups_count = df[cat_var].nunique()
    series_list = [df[df[cat_var] == g][num_var] for g in df[cat_var].unique()]
    test_type = 'parametric'

    # Проверка нормальности
    norm = all(normal_means(s) for s in series_list)

    if not norm:
        if groups_count == 2:
            print("U-критерий Манна-Уитни")
            p = mannwhitneyu(series_list[0], series_list[1])[1]
        else:
            print("Kruskal-Wallis Test")
            test_type = 'non parametric'
            p = kruskal(*series_list)[1]
            if p <= 0.05:
                print("Post-hoc: Dunn test (p-values, Bonferroni corrected):")
                posthoc = sp.posthoc_dunn(df, val_col=num_var, group_col=cat_var, p_adjust='bonferroni')
                print(posthoc)
    else:
        p_equal_var = equal_var(*series_list)
        print("p_equal_var:", p_equal_var)
        if p_equal_var > 0.05:
            if groups_count == 2:
                print("Т-критерий Стьюдента")
                p = ttest_ind(series_list[0], series_list[1], equal_var=True)[1]
            else:
                print("One-Way ANOVA")
                test_type = 'non parametric'
                p = f_oneway(*series_list)[1]
                if p <= 0.05:
                    print("Post-hoc: Tukey HSD:")
                    tukey = pairwise_tukeyhsd(endog=df[num_var], groups=df[cat_var], alpha=0.05)
                    print(tukey)
        else:
            if groups_count == 2:
                print("Т-критерий Уэлча")
                p = ttest_ind(series_list[0], series_list[1], equal_var=False)[1]
            else:
                print("Welch's ANOVA")
                test_type = 'non parametric'
                welch = pg.welch_anova(dv=num_var, between=cat_var, data=df)
                p = welch.loc[0, 'p-unc']
                if p <= 0.05:
                    print("Post-hoc: Games-Howell test:")
                    gh = pg.pairwise_gameshowell(dv=num_var, between=cat_var, data=df)
                    print(gh)

    print("p-value:", p)
    if p <= 0.05:
        print("Средние статистически значимо отличаются")
        if test_type == 'parametric':
            agg = 'mean'
        else:
            agg = 'median'
        agg_values = df.groupby(cat_var)[num_var].agg(agg)
        graphics.plot_bar_chart(agg_values.index, agg_values.values, title=f"{agg} {num_var} по {cat_var}")
    else:
        print("Средние статистически значимо НЕ отличаются")

def chi_square_test(df, col1, col2, correction=False):
    """
    χ²-тест независимости для двух категориальных переменных.

    df : pd.DataFrame — таблица данных
    col1, col2 : str — названия категориальных столбцов
    correction : bool — Yates' continuity correction (по умолчанию False)

    Выводит:
        - наблюдаемые частоты
        - ожидаемые частоты
        - χ²-статистику, p-value, степени свободы
    """
    # таблица сопряжённости
    contingency_table = pd.crosstab(df[col1], df[col2])

    # χ²-тест
    chi2, p, dof, expected = chi2_contingency(contingency_table, correction=correction)

    print("Наблюдаемые частоты:")
    print(contingency_table, "\n")

    print("Ожидаемые частоты (если бы переменные были независимы):")
    print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns), "\n")

    print(f"χ² = {chi2:.4f}, p-value = {p:.4f}, df = {dof}")
    if p <= 0.05:
        print("Связь между переменными статистически значима.")
        graphics.plot_heatmap(contingency_table, title="елки иголки")

    else:
        print("Нет статистически значимой связи между переменными.")

    return chi2, p, dof, expected

