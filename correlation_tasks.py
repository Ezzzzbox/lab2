"""Задачи второй части лабораторной: корреляционный анализ."""

from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

import pandas as pd


def count_pearson(df, corr_prms, trgt):
    return {
        corr_prms[i]: float(df[corr_prms[i]].corr(df[trgt], method="pearson"))
        for i in range(len(corr_prms))
    }


def cmp_notisna(prevmax, value):
    if not pd.isna(value):
        if value > prevmax:
            return True
    return False


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    csv_path = data.csv_path
    df = pd.read_csv(csv_path, sep="\t")

    men_df = df[df["Gender"] == "Male"]
    women_df = df[df["Gender"] == "Female"]
    corr_prms = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    men_pearson = count_pearson(men_df, corr_prms, "MRI_Count")
    women_pearson = count_pearson(women_df, corr_prms, "MRI_Count")

    max_pearson = 0
    max_pearson_prm = ""
    for key in men_pearson:
        if cmp_notisna(max_pearson, men_pearson[key]):
            max_pearson = men_pearson[key]
            max_pearson_prm = key
        if cmp_notisna(max_pearson, women_pearson[key]):
            max_pearson = women_pearson[key]
            max_pearson_prm = key

    return BrainCorrelationSummary(
        len(men_df), len(women_df), men_pearson, women_pearson, max_pearson_prm
    )


analyze_brain_correlations(BrainDataInput("brainsize.txt"))
