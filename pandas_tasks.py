"""Задачи первой части лабораторной: Pandas и Titanic."""

from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = pd.read_csv(data.csv_path)
    pass_per_column = df.isna().sum().to_dict()
    older_30 = len(df[df["Age"] > 30])
    mean_age_by_pclass = df.groupby("Pclass")["Age"].mean()
    survived = df.groupby("Pclass")["Survived"]
    survived_dale = (survived.sum() / survived.count()).to_dict()
    top_5_fare = df.nlargest(5, "Fare")["Fare"].tolist()

    return TitanicSummary(
        row_count=len(df),
        missing_by_column=pass_per_column,
        adults_over_30_count=older_30,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survived_dale,
        highest_fares=top_5_fare,
    )


analyze_titanic(TitanicInput("titanic.csv"))
