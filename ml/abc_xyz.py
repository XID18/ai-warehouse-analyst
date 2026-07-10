"""
ABC-XYZ классификация каталога SKU.

ABC — по доле в объёме продаж (кумулятивный вклад).
XYZ — по стабильности спроса (коэффициент вариации).
"""

import pandas as pd


def classify_abc(df: pd.DataFrame, value_col: str = "revenue") -> pd.DataFrame:
    """
    df: одна строка на SKU, с колонкой value_col (например, выручка за период).
    Возвращает df с колонкой 'abc_class' (A/B/C).
    """
    df = df.sort_values(value_col, ascending=False).copy()
    df["cum_share"] = df[value_col].cumsum() / df[value_col].sum()

    def classify(cum_share: float) -> str:
        if cum_share <= 0.80:
            return "A"
        elif cum_share <= 0.95:
            return "B"
        return "C"

    df["abc_class"] = df["cum_share"].apply(classify)
    return df


def classify_xyz(df: pd.DataFrame, sku_col: str = "sku", period_col: str = "period", qty_col: str = "qty") -> pd.DataFrame:
    """
    df: история продаж по SKU и периодам.
    Возвращает df — одна строка на SKU, с колонкой 'xyz_class' (X/Y/Z)
    на основе коэффициента вариации спроса.
    """
    stats = df.groupby(sku_col)[qty_col].agg(["mean", "std"]).reset_index()
    stats["cv"] = stats["std"] / stats["mean"]

    def classify(cv: float) -> str:
        if cv <= 0.10:
            return "X"
        elif cv <= 0.25:
            return "Y"
        return "Z"

    stats["xyz_class"] = stats["cv"].apply(classify)
    return stats[[sku_col, "cv", "xyz_class"]]


def combine_abc_xyz(abc_df: pd.DataFrame, xyz_df: pd.DataFrame, sku_col: str = "sku") -> pd.DataFrame:
    """Объединяет ABC и XYZ классификации в единую матрицу (например, AX, BZ, CY)."""
    merged = abc_df.merge(xyz_df, on=sku_col)
    merged["abc_xyz"] = merged["abc_class"] + merged["xyz_class"]
    return merged
