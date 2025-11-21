import time

# Все библиотеки

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_ind, levene, mannwhitneyu,shapiro, \
    f_oneway, kruskal, chi2_contingency, fisher_exact
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
import tests


def series_to_nparray(series):
    return series.to_numpy()

def series_to_list(series):
    return series.tolist()

def filter_by_datetime_range(df, datetime_col, time_min="", time_max=""):
    """
    подаешь на вход:
    df - dataframe
    datetime_col - название столбца datetime (str)
    time_min/time_max - границы, по которым надо обрезать (str, "20-01-2025" или "23-01-2025 13:30:00"),
    если не нужно обрезать по концу - не подавай его
    """
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col], dayfirst=True, errors='coerce')

    t_min = pd.to_datetime(time_min, dayfirst=True, errors='coerce') if time_min else None
    t_max = pd.to_datetime(time_max, dayfirst=True, errors='coerce') if time_max else None

    mask = pd.Series([True] * len(df))

    if t_min is not None:
        mask &= (df[datetime_col] >= t_min)

    if t_max is not None:
        mask &= (df[datetime_col] <= t_max)

    return df.loc[mask].copy()

def one_hot_encode(df: pd.DataFrame, cat_cols, drop_first=True):
    """
    Принимает DataFrame и список категориальных столбцов.
    Возвращает новый DataFrame, где указанные колонки заменены
    на их one-hot-представление.
    """
    # Проверка: все переданные колонки существуют
    missing = [c for c in cat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Колонок {missing} нет в датафрейме")

    encoded = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)

    return encoded

"""
Две функции-«функи-хэлперы» для олимпиад по анализу данных:
 - fit_predict_logistic(learn, predict, target, return_proba=False, thresh=0.5, plot=True)
 - fit_predict_linear(learn, predict, target, return_residuals=False, plot=True)

Условия: все колонки числовые, NaN нет.

Функции обучают модель на learn и возвращают pd.Series предсказаний для строк из predict.
Функции печатают диагностическую информацию (коэффициенты + p-value и др.) и строят графики.
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def _compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Возвращает DataFrame с VIF для каждой переменной в X (предполагается, что X уже содержит константу).
    Для колонки 'const' VIF обычно бессмысленен — его можно игнорировать.
    """
    vif_data = []
    cols = X.columns
    for i in range(len(cols)):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.nan
        vif_data.append((cols[i], vif))
    return pd.DataFrame(vif_data, columns=["variable", "VIF"])


def fit_predict_logistic(
    learn: pd.DataFrame,
    predict: pd.DataFrame,
    target: str,
    return_proba: bool = False,
    thresh: float = 0.5,
    plot: bool = True,
) -> pd.Series:
    """
    Обучает логистическую регрессию (Logit из statsmodels по умолчанию), печатает:
      - таблицу коэффициентов с p-value;
      - ROC-кривую и ROC-AUC (по обучающей выборке);
      - метрики Accuracy, Recall, Precision, F1 (с формулами и значениями, посчитанными по обучающей выборке);
      - confusion matrix;
      - VIF (для признаков), если возможно;
    Возвращает pd.Series меток (0/1) для predict (index сохраняется).

    Параметры:
      learn, predict: DataFrame (все числовые, без NaN). В learn должна быть колонка target.
      target: имя целевой колонки в learn.
      return_proba: если True, функция вернёт pd.Series вероятностей вместо меток.
      thresh: порог для перевода вероятности в метку (по умолчанию 0.5).
      plot: показывать ли график ROC и другие графики.

    Примечание по порогу: Да — если предсказанная вероятность >= thresh, метка = 1, иначе 0.
    """

    # --- подготовка данных
    if target not in learn.columns:
        raise ValueError(f"Целевая колонка '{target}' отсутствует в learn")

    X_train = learn.drop(columns=[target])
    y_train = learn[target].astype(int)
    X_pred = predict.copy()

    # добавляем константу (intercept)
    X_train_c = add_constant(X_train, has_constant='add')
    X_pred_c = add_constant(X_pred, has_constant='add')

    # --- попытка подогнать statsmodels Logit
    sm_result = None
    used_sklearn = False
    try:
        model = sm.Logit(y_train, X_train_c)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sm_result = model.fit(disp=False, maxiter=100)
    except Exception as e:
        # иногда Logit не сходится или 'PerfectSeparation'. Попытка sklearn как запасной вариант.
        warnings.warn(f"statsmodels.Logit не удался ({e}), переключаюсь на sklearn LogisticRegression без p-value")
        used_sklearn = True

    if used_sklearn:
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train.values, y_train.values)
        proba_train = clf.predict_proba(X_train.values)[:, 1]
        proba_pred = clf.predict_proba(X_pred.values)[:, 1]

        # коэффициенты в sklearn
        try:
            coef = pd.Series(np.r_[clf.intercept_, clf.coef_.ravel()], index=X_train_c.columns)
        except Exception:
            coef = pd.Series(clf.coef_.ravel(), index=X_train.columns)
        pvalues = pd.Series([np.nan] * len(coef), index=coef.index)
    else:
        proba_train = sm_result.predict(X_train_c)
        proba_pred = sm_result.predict(X_pred_c)
        coef = sm_result.params
        pvalues = sm_result.pvalues

    # --- метрики на обучающей выборке
    y_pred_train = (proba_train >= thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_train, y_pred_train).ravel()

    # вычисление метрик
    acc = accuracy_score(y_train, y_pred_train)
    rec = recall_score(y_train, y_pred_train)
    prec = precision_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)
    try:
        roc_auc = roc_auc_score(y_train, proba_train)
    except Exception:
        roc_auc = np.nan

    # --- вывод результатов
    print("=== Logistic regression diagnostics ===")
    # коэффициенты + p-values
    coef_table = pd.DataFrame({"coef": coef, "p_value": pvalues})
    print("\nCoefficients with p-values:")
    print(coef_table)

    # VIF
    try:
        vif_df = _compute_vif(X_train_c)
        print("\nVIF (variance inflation factor):")
        print(vif_df)
    except Exception as e:
        print(f"Не удалось вычислить VIF: {e}")

    # confusion matrix and metrics with formulas
    print("\nConfusion matrix (train):")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    print("\nMetrics (train) with formulas:")
    print(f"Accuracy = (TP + TN) / (TP + TN + FP + FN) = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn}) = {acc:.4f}")
    print(f"Recall = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {rec:.4f}")
    print(f"Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {prec:.4f}")
    print(f"F1-score = 2 * Precision * Recall / (Precision + Recall) = {f1:.4f}")
    print(f"ROC AUC (train) = {roc_auc:.4f}")

    # порог
    print(f"По умолчанию метка = 1 при вероятности >= {thresh}, иначе 0.")

    # ROC plot
    if plot:
        try:
            fpr, tpr, _ = roc_curve(y_train, proba_train)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve (train)')
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            warnings.warn(f"Не удалось построить ROC: {e}")

    # --- предсказание для predict
    if return_proba:
        return pd.Series(proba_pred, index=predict.index)
    else:
        pred_labels = (proba_pred >= thresh).astype(int)
        return pd.Series(pred_labels, index=predict.index)


def fit_predict_linear(
    learn: pd.DataFrame,
    predict: pd.DataFrame,
    target: str,
    return_residuals: bool = False,
    plot: bool = True,
) -> pd.Series:
    """
    Обучает линейную регрессию (OLS из statsmodels), печатает:
      - таблицу коэффициентов с p-value;
      - MSE, RMSE, MAE, MAD и R^2 на обучающей выборке;
      - diagnostic plots: residuals vs fitted, Q-Q plot;
      - VIF для признаков (если возможно), Durbin-Watson.
    Возвращает pd.Series предсказаний для predict (index сохраняется).

    Пояснение R^2: R^2 = 1 - SS_res / SS_tot, где SS_res = sum((y - y_pred)^2),
    SS_tot = sum((y - y_mean)^2). Он показывает долю дисперсии целевой переменной,
    объяснённую моделью. R^2 = 1 означает идеальную подгонку, R^2 = 0 — модель не лучше,
    чем константная модель; R^2 < 0 — модель хуже константы.
    """

    if target not in learn.columns:
        raise ValueError(f"Целевая колонка '{target}' отсутствует в learn")

    X_train = learn.drop(columns=[target])
    y_train = learn[target].astype(float)
    X_pred = predict.copy()

    X_train_c = add_constant(X_train, has_constant='add')
    X_pred_c = add_constant(X_pred, has_constant='add')

    model = sm.OLS(y_train, X_train_c)
    res = model.fit()

    # coefficients and p-values
    coef = res.params
    pvalues = res.pvalues
    coef_table = pd.DataFrame({"coef": coef, "p_value": pvalues})

    # predictions on train
    y_train_pred = res.predict(X_train_c)
    residuals = y_train - y_train_pred

    # metrics
    mse = mean_squared_error(y_train, y_train_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, y_train_pred)
    mad = np.median(np.abs(residuals - np.median(residuals)))  # median absolute deviation of residuals
    r2 = res.rsquared

    print("=== Linear regression diagnostics ===")
    print("\nCoefficients with p-values:")
    print(coef_table)

    print("\nErrors on train:")
    print(f"MSE = {mse:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print(f"MAE = {mae:.6f}")
    print(f"MAD (median absolute deviation of residuals) = {mad:.6f}")
    print(f"R^2 = {r2:.6f}")

    print("\nR^2 пояснение: R^2 = 1 - SS_res/SS_tot, где SS_res = sum((y - y_pred)^2), SS_tot = sum((y - mean(y))^2).\n" +
          "Он показывает долю дисперсии целевой переменной, объяснённую моделью. Чем ближе к 1, тем лучше объяснение (но возможен overfit).")

    # VIF
    try:
        vif_df = _compute_vif(X_train_c)
        print("\nVIF:")
        print(vif_df)
    except Exception as e:
        print(f"Не удалось вычислить VIF: {e}")

    # Durbin-Watson
    try:
        dw = durbin_watson(res.resid)
        print(f"\nDurbin-Watson (autocorrelation of residuals) = {dw:.4f}")
    except Exception as e:
        print(f"Не удалось вычислить Durbin-Watson: {e}")

    # diagnostic plots
    if plot:
        try:
            # Residuals vs Fitted
            plt.figure(figsize=(6, 4))
            plt.scatter(y_train_pred, residuals)
            plt.axhline(0, linestyle='--')
            plt.xlabel('Fitted values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Fitted (train)')
            plt.grid(True)
            plt.show()

            # Q-Q
            sm.qqplot(residuals, line='45', fit=True)
            plt.title('Q-Q plot of residuals (train)')
            plt.show()
        except Exception as e:
            warnings.warn(f"Не удалось построить diagnostic plots: {e}")

    # предсказание для predict
    y_pred = res.predict(X_pred_c)
    preds = pd.Series(y_pred, index=predict.index)

    if return_residuals:
        # если пользователь хочет residuals — возвращаем их в виде Series с индексом train
        return preds, pd.Series(residuals, index=learn.index)
    else:
        return preds