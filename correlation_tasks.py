"""Задачи второй части лабораторной: корреляционный анализ."""

from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

import pandas as pd


def gender_to_corr_dict(df, gender):
    corr = (
        df[["MRI_Count", "FSIQ", "VIQ", "PIQ", "Weight", "Height"]]
        .corr(method="pearson")
        .to_dict(orient="dict")
    )["MRI_Count"]
    corr.pop("MRI_Count")
    return corr


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    csv_path = data.csv_path
    df = pd.read_csv(csv_path, sep="\t", na_values=["NaN"])
    misses = df.isna().sum().to_dict()
    for column in misses:
        if misses[column] > 0:
            df[column] = df[column].fillna(df[column].mean())
    men = len(df[df["Gender"] == "Male"])
    women = len(df[df["Gender"] == "Female"])

    men_pearson = gender_to_corr_dict(df, "Male")
    women_pearson = gender_to_corr_dict(df, "Female")
    max_pearson = max(max(men_pearson.values()), max(women_pearson.values()))

    return BrainCorrelationSummary(men, women, men_pearson, women_pearson, max_pearson)


analyze_brain_correlations(BrainDataInput("brainsize.txt"))
