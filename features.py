"""Shared feature engineering for the Online Shoppers purchase-intent project.

Imported by BOTH the notebook and the Streamlit app, so the saved joblib
pipeline can be loaded anywhere (pickle stores functions by reference).
"""
import numpy as np
import pandas as pd

RAW_FEATURES = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]

DURATION_COLS = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]

# Codes (not quantities) -> treated as categories
CAT_COLS = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]

NUM_COLS = [
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates",
    "PageValues", "SpecialDay", "Weekend",
    "TotalPages", "TotalDuration", "AvgTimePerPage", "HasPageValue",
]

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering (no statistics learned from data -> no leakage)."""
    X = X.copy()
    # -1 appears in some duration cells in the raw data -> clip to 0
    for c in DURATION_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce").clip(lower=0)

    X["TotalPages"] = X["Administrative"] + X["Informational"] + X["ProductRelated"]
    X["TotalDuration"] = (
        X["Administrative_Duration"] + X["Informational_Duration"] + X["ProductRelated_Duration"]
    )
    X["AvgTimePerPage"] = (X["TotalDuration"] / X["TotalPages"].replace(0, np.nan)).fillna(0)
    X["HasPageValue"] = (X["PageValues"] > 0).astype(int)
    X["Weekend"] = X["Weekend"].astype(int)

    # heavy right tail -> log scale helps the linear model, harmless for trees
    for c in DURATION_COLS + ["TotalDuration", "AvgTimePerPage"]:
        X[c] = np.log1p(X[c])

    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


def recommend_action(proba: float, t_coupon: float, t_chat: float) -> str:
    """Map a purchase probability to a suggested on-site action."""
    if proba >= t_chat:
        return "chat"      # high intent: offer help to close the sale, no discount needed
    if proba >= t_coupon:
        return "coupon"    # on the fence: a coupon may tip the decision
    return "none"          # low intent: don't spend discount budget
