#!/usr/bin/env python3
"""
Enterprise Sales Forecasting Pipeline
=====================================

This script reads transactional sales data from CSV/TSV/TXT/Excel, detects the
required columns, creates monthly time series by State/Plant/Region, evaluates
seven forecasting models, selects the lowest-WAPE model for each entity, saves
CSV outputs, and generates forecast plots.

Install dependencies:
    pip install pandas numpy matplotlib statsmodels openpyxl

Run directly after updating INPUT_FILE and OUTPUT_FOLDER below.
"""

from __future__ import annotations

import logging
import re
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing


# =============================================================================
# USER CONFIGURATION
# =============================================================================

INPUT_FILE = Path(
    r"C:\Users\ritik\OneDrive\Desktop\Project\Enterprise Sales Forecasting Project\train.csv"
)

OUTPUT_FOLDER = Path(
    r"C:\Users\ritik\OneDrive\Desktop\Project\Enterprise Sales Forecasting Project\output"
)

MINIMUM_MONTHLY_OBSERVATIONS = 24
TEST_MONTHS = 6

warnings.filterwarnings("ignore")


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_HISTORICAL_AVERAGE = "Historical Average"
MODEL_LAST_VALUE = "Last Value (Naive)"
MODEL_THREE_MONTH_AVERAGE = "Three Month Moving Average"
MODEL_SIX_MONTH_AVERAGE = "Six Month Moving Average"
MODEL_SIMPLE_EXPONENTIAL_SMOOTHING = "Simple Exponential Smoothing"
MODEL_HOLT_LINEAR = "Holt Linear Trend"
MODEL_HOLT_DAMPED = "Holt Damped Trend"

MODEL_NAMES: Tuple[str, ...] = (
    MODEL_HISTORICAL_AVERAGE,
    MODEL_LAST_VALUE,
    MODEL_THREE_MONTH_AVERAGE,
    MODEL_SIX_MONTH_AVERAGE,
    MODEL_SIMPLE_EXPONENTIAL_SMOOTHING,
    MODEL_HOLT_LINEAR,
    MODEL_HOLT_DAMPED,
)

SUPPORTED_FILE_EXTENSIONS = {
    ".xlsx",
    ".xls",
    ".xlsm",
    ".xlsb",
    ".csv",
    ".tsv",
    ".txt",
}

DATE_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "Order Date",
    "Billing Date",
    "Invoice Date",
    "Sales Date",
    "Transaction Date",
    "Posting Date",
    "Document Date",
    "Ship Date",
    "Date",
)

ENTITY_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "State",
    "Plant",
    "Region",
    "Location",
    "Branch",
    "Site",
    "Territory",
    "Business Unit",
    "Warehouse",
    "Market",
)

SALES_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "Sales",
    "Net Sales",
    "Net Value",
    "Revenue",
    "Sales Value",
    "Invoice Value",
    "Amount",
    "Value",
    "Total Sales",
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class DetectedColumns:
    """Required columns detected in the input dataset."""

    date_column: str
    entity_column: str
    sales_column: str


@dataclass(frozen=True)
class MetricResult:
    """Forecast error metrics for one model."""

    mae: float
    rmse: float
    mape: float
    wape: float


@dataclass
class EntityForecastResult:
    """Complete forecasting result for one entity."""

    entity: str
    train_series: pd.Series
    test_series: pd.Series
    forecasts: Dict[str, np.ndarray]
    metrics: Dict[str, MetricResult]
    best_model: str
    best_forecast: np.ndarray
    reliability: str


# =============================================================================
# LOGGING AND GENERAL UTILITIES
# =============================================================================


def configure_logging(output_folder: Path) -> logging.Logger:
    """Configure console and file logging."""
    output_folder.mkdir(parents=True, exist_ok=True)
    log_file = output_folder / "forecasting_pipeline.log"

    logger = logging.getLogger("enterprise_sales_forecasting")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def normalize_column_name(column_name: object) -> str:
    """Normalize a column name for case-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", "", str(column_name).strip().lower())


def sanitize_filename(value: object, max_length: int = 150) -> str:
    """Convert an entity name into a safe filename."""
    text = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", str(value).strip())
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._ ")
    return (text or "unnamed_entity")[:max_length]


def round_metric(value: float, digits: int = 4) -> float:
    """Round finite metrics while preserving NaN and infinity."""
    if np.isnan(value) or np.isinf(value):
        return float(value)
    return round(float(value), digits)


def ensure_output_directories(output_folder: Path) -> Dict[str, Path]:
    """Create output directories."""
    directories = {
        "root": output_folder,
        "plots": output_folder / "plots",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


# =============================================================================
# INPUT READING
# =============================================================================


def read_delimited_file(file_path: Path) -> pd.DataFrame:
    """
    Read CSV, TSV, or TXT files with delimiter and encoding fallbacks.

    The Python engine is only used for automatic delimiter detection and is not
    passed low_memory, because that option is unsupported by the Python engine.
    """
    extension = file_path.suffix.lower()
    encodings = ("utf-8-sig", "utf-8", "latin-1", "cp1252")
    errors: List[str] = []

    for encoding in encodings:
        try:
            if extension == ".csv":
                return pd.read_csv(
                    file_path,
                    sep=",",
                    encoding=encoding,
                    low_memory=False,
                )

            if extension == ".tsv":
                return pd.read_csv(
                    file_path,
                    sep="\t",
                    encoding=encoding,
                    low_memory=False,
                )

            return pd.read_csv(
                file_path,
                sep=None,
                engine="python",
                encoding=encoding,
            )

        except Exception as exc:
            errors.append(f"Encoding {encoding}: {exc}")

    raise ValueError(
        "Unable to read the input delimited file.\n" + "\n".join(errors)
    )


def score_sheet_columns(columns: Iterable[object]) -> int:
    """Score an Excel sheet based on recognizable sales-data columns."""
    normalized_columns = {normalize_column_name(column) for column in columns}
    date_candidates = {normalize_column_name(c) for c in DATE_COLUMN_CANDIDATES}
    entity_candidates = {normalize_column_name(c) for c in ENTITY_COLUMN_CANDIDATES}
    sales_candidates = {normalize_column_name(c) for c in SALES_COLUMN_CANDIDATES}

    score = min(len(normalized_columns), 20)
    if normalized_columns.intersection(date_candidates):
        score += 10
    if normalized_columns.intersection(entity_candidates):
        score += 10
    if normalized_columns.intersection(sales_candidates):
        score += 10
    return score


def read_excel_with_automatic_sheet_detection(
    file_path: Path,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, str]:
    """Select and load the most likely transactional-data sheet."""
    try:
        excel_file = pd.ExcelFile(file_path)
    except Exception as exc:
        raise ValueError(f"Unable to open Excel file: {exc}") from exc

    if not excel_file.sheet_names:
        raise ValueError("The Excel workbook does not contain any sheets.")

    sheet_scores: List[Tuple[str, int]] = []
    for sheet_name in excel_file.sheet_names:
        try:
            sample = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=5)
            score = score_sheet_columns(sample.columns)
            if not sample.dropna(how="all").empty:
                score += 5
            sheet_scores.append((sheet_name, score))
            logger.debug("Sheet '%s' score: %s", sheet_name, score)
        except Exception as exc:
            logger.warning("Unable to inspect sheet '%s': %s", sheet_name, exc)

    if not sheet_scores:
        raise ValueError("No readable Excel sheets were found.")

    selected_sheet = max(sheet_scores, key=lambda item: item[1])[0]
    logger.info("Automatically selected Excel sheet: %s", selected_sheet)
    return pd.read_excel(excel_file, sheet_name=selected_sheet), selected_sheet


def load_input_data(
    input_file: Path,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, str]:
    """Load the source dataset."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    if not input_file.is_file():
        raise ValueError(f"Input path is not a file: {input_file}")

    suffix = input_file.suffix.lower()
    if suffix not in SUPPORTED_FILE_EXTENSIONS:
        raise ValueError(f"Unsupported input file type: {suffix}")

    logger.info("Loading input file: %s", input_file)

    if suffix in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
        data, source_name = read_excel_with_automatic_sheet_detection(
            input_file, logger
        )
    else:
        data = read_delimited_file(input_file)
        source_name = input_file.name

    if data.empty:
        raise ValueError("The input dataset is empty.")

    data.columns = [str(column).strip() for column in data.columns]
    unnamed_columns = [
        column
        for column in data.columns
        if normalize_column_name(column).startswith("unnamed")
    ]
    if unnamed_columns:
        data = data.drop(columns=unnamed_columns, errors="ignore")
        logger.info("Removed %s unnamed columns.", len(unnamed_columns))

    logger.info("Loaded %s rows and %s columns.", len(data), len(data.columns))
    return data, source_name


# =============================================================================
# COLUMN DETECTION
# =============================================================================


def find_exact_candidate_column(
    columns: Sequence[str],
    candidates: Sequence[str],
) -> Optional[str]:
    """Find an exact normalized candidate match."""
    lookup: Dict[str, str] = {}
    for column in columns:
        lookup.setdefault(normalize_column_name(column), column)
    for candidate in candidates:
        normalized = normalize_column_name(candidate)
        if normalized in lookup:
            return lookup[normalized]
    return None


def find_keyword_column(
    columns: Sequence[str],
    keywords: Sequence[str],
) -> Optional[str]:
    """Find a column containing one of the normalized keywords."""
    for keyword in keywords:
        normalized_keyword = normalize_column_name(keyword)
        for column in columns:
            if normalized_keyword in normalize_column_name(column):
                return column
    return None


def identify_date_column(
    data: pd.DataFrame,
    excluded_columns: Optional[Sequence[str]] = None,
) -> str:
    """Automatically identify the date column."""
    excluded = set(excluded_columns or [])
    columns = [column for column in data.columns if column not in excluded]

    match = find_exact_candidate_column(columns, DATE_COLUMN_CANDIDATES)
    if match:
        return match

    match = find_keyword_column(
        columns,
        (
            "orderdate",
            "billingdate",
            "invoicedate",
            "salesdate",
            "transactiondate",
            "shipdate",
            "date",
        ),
    )
    if match:
        return match

    best_column: Optional[str] = None
    best_ratio = 0.0
    for column in columns:
        sample = data[column].dropna().head(500)
        if sample.empty:
            continue
        if pd.api.types.is_datetime64_any_dtype(sample):
            return column
        parsed = pd.to_datetime(sample, errors="coerce")
        ratio = float(parsed.notna().mean())
        if ratio > best_ratio:
            best_column = column
            best_ratio = ratio

    if best_column and best_ratio >= 0.80:
        return best_column

    raise ValueError(
        "Unable to detect the date column. Available columns: "
        + ", ".join(map(str, data.columns))
    )


def identify_entity_column(
    data: pd.DataFrame,
    excluded_columns: Optional[Sequence[str]] = None,
) -> str:
    """Automatically identify the State, Plant, or Region column."""
    excluded = set(excluded_columns or [])
    columns = [column for column in data.columns if column not in excluded]

    match = find_exact_candidate_column(columns, ENTITY_COLUMN_CANDIDATES)
    if match:
        return match

    match = find_keyword_column(
        columns,
        ("state", "plant", "region", "location", "branch", "territory", "site", "market"),
    )
    if match:
        return match

    candidates: List[Tuple[str, float]] = []
    row_count = max(len(data), 1)
    for column in columns:
        series = data[column]
        non_null = series.dropna()
        if non_null.empty:
            continue
        is_text = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
        if not is_text:
            continue
        unique_count = non_null.nunique(dropna=True)
        unique_ratio = unique_count / row_count
        if 2 <= unique_count <= min(500, max(2, row_count // 2)):
            candidates.append((column, 1.0 - unique_ratio))

    if candidates:
        return max(candidates, key=lambda item: item[1])[0]

    raise ValueError(
        "Unable to detect the State, Plant, or Region column. Available columns: "
        + ", ".join(map(str, data.columns))
    )


def identify_sales_column(
    data: pd.DataFrame,
    excluded_columns: Optional[Sequence[str]] = None,
) -> str:
    """Automatically identify the numeric sales column."""
    excluded = set(excluded_columns or [])
    columns = [column for column in data.columns if column not in excluded]

    match = find_exact_candidate_column(columns, SALES_COLUMN_CANDIDATES)
    if match:
        return match

    match = find_keyword_column(
        columns,
        ("netsales", "salesvalue", "netvalue", "revenue", "sales", "invoicevalue", "amount"),
    )
    if match:
        return match

    candidates: List[Tuple[str, float]] = []
    for column in columns:
        converted = pd.to_numeric(
            data[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        numeric_ratio = float(converted.notna().mean())
        if numeric_ratio < 0.80:
            continue
        valid = converted.dropna()
        if valid.empty:
            continue
        variability = valid.nunique() / max(len(valid), 1)
        non_zero_ratio = float((valid != 0).mean())
        candidates.append((column, numeric_ratio + variability + non_zero_ratio))

    if candidates:
        return max(candidates, key=lambda item: item[1])[0]

    raise ValueError(
        "Unable to detect the sales column. Available columns: "
        + ", ".join(map(str, data.columns))
    )


def detect_required_columns(
    data: pd.DataFrame,
    logger: logging.Logger,
) -> DetectedColumns:
    """Detect date, entity, and sales columns."""
    date_column = identify_date_column(data)
    entity_column = identify_entity_column(data, [date_column])
    sales_column = identify_sales_column(data, [date_column, entity_column])

    logger.info("Detected date column: %s", date_column)
    logger.info("Detected entity column: %s", entity_column)
    logger.info("Detected sales column: %s", sales_column)

    return DetectedColumns(date_column, entity_column, sales_column)


# =============================================================================
# PREPROCESSING
# =============================================================================


def infer_day_first(date_series: pd.Series) -> bool:
    """Infer whether dates likely use DD-MM-YYYY ordering."""
    sample = date_series.dropna().astype(str).str.strip().head(1000)
    day_first_evidence = 0
    month_first_evidence = 0
    pattern = re.compile(r"^\s*(\d{1,4})[./-](\d{1,2})[./-](\d{1,4})")

    for value in sample:
        match = pattern.search(value)
        if not match:
            continue
        first = int(match.group(1))
        second = int(match.group(2))
        if 12 < first <= 31:
            day_first_evidence += 1
        elif 12 < second <= 31:
            month_first_evidence += 1

    return day_first_evidence > month_first_evidence


def parse_dates_robustly(
    date_series: pd.Series,
    logger: logging.Logger,
) -> pd.Series:
    """Parse dates with day-first fallback."""
    if pd.api.types.is_datetime64_any_dtype(date_series):
        return pd.to_datetime(date_series, errors="coerce")

    day_first = infer_day_first(date_series)

    try:
        parsed = pd.to_datetime(
            date_series,
            errors="coerce",
            dayfirst=day_first,
            format="mixed",
        )
    except TypeError:
        parsed = pd.to_datetime(date_series, errors="coerce", dayfirst=day_first)

    success_rate = float(parsed.notna().mean())

    try:
        alternative = pd.to_datetime(
            date_series,
            errors="coerce",
            dayfirst=not day_first,
            format="mixed",
        )
    except TypeError:
        alternative = pd.to_datetime(
            date_series,
            errors="coerce",
            dayfirst=not day_first,
        )

    alternative_rate = float(alternative.notna().mean())
    if alternative_rate > success_rate:
        parsed = alternative
        day_first = not day_first
        success_rate = alternative_rate

    logger.info(
        "Date parsing success rate: %.2f%% | dayfirst=%s",
        success_rate * 100,
        day_first,
    )
    return parsed


def clean_numeric_sales(series: pd.Series) -> pd.Series:
    """Convert sales values to numeric, including currency and accounting formats."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = series.astype(str).str.strip()
    accounting_negative = cleaned.str.match(r"^\(.*\)$", na=False)
    cleaned = cleaned.str.replace("(", "", regex=False)
    cleaned = cleaned.str.replace(")", "", regex=False)
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"[^\d.\-+eE]", "", regex=True)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    numeric.loc[accounting_negative & numeric.notna()] *= -1
    return numeric


def preprocess_transactional_data(
    raw_data: pd.DataFrame,
    detected_columns: DetectedColumns,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Clean transactional records and aggregate them to monthly sales."""
    data = raw_data[
        [
            detected_columns.date_column,
            detected_columns.entity_column,
            detected_columns.sales_column,
        ]
    ].copy()
    data.columns = ["Order Date", "Entity", "Sales"]

    initial_rows = len(data)
    data["Order Date"] = parse_dates_robustly(data["Order Date"], logger)
    data["Sales"] = clean_numeric_sales(data["Sales"])
    data["Entity"] = data["Entity"].astype("string").str.strip()

    invalid_entities = {"", "nan", "none", "null", "<na>", "n/a", "na"}
    valid_entity = ~data["Entity"].str.lower().isin(invalid_entities)
    data = data.loc[
        data["Order Date"].notna() & data["Sales"].notna() & valid_entity
    ].copy()

    logger.info("Removed %s invalid rows.", initial_rows - len(data))
    if data.empty:
        raise ValueError("No valid rows remain after preprocessing.")

    data["Month"] = data["Order Date"].dt.to_period("M").dt.to_timestamp()
    data["Year"] = data["Month"].dt.year
    data["Month Number"] = data["Month"].dt.month
    data["Month Name"] = data["Month"].dt.month_name()

    monthly_sales = (
        data.groupby(
            ["Entity", "Month", "Year", "Month Number", "Month Name"],
            as_index=False,
            observed=True,
        )["Sales"]
        .sum()
        .rename(columns={"Sales": "Monthly Sales"})
        .sort_values(["Entity", "Month"])
        .reset_index(drop=True)
    )

    logger.info("Generated %s monthly entity records.", len(monthly_sales))
    return monthly_sales


def build_complete_entity_series(
    monthly_sales: pd.DataFrame,
    logger: logging.Logger,
) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """Build continuous monthly series and skip entities with short history."""
    entity_series: Dict[str, pd.Series] = {}
    skipped_records: List[Dict[str, object]] = []

    for entity_value, entity_data in monthly_sales.groupby(
        "Entity", sort=True, observed=True
    ):
        entity = str(entity_value).strip()
        monthly = entity_data.groupby("Month")["Monthly Sales"].sum().sort_index()

        if monthly.empty:
            skipped_records.append(
                {
                    "State": entity,
                    "Available Months": 0,
                    "Required Months": MINIMUM_MONTHLY_OBSERVATIONS,
                    "Reason": "No monthly sales data",
                }
            )
            continue

        complete_index = pd.date_range(
            monthly.index.min(), monthly.index.max(), freq="MS"
        )
        complete_series = monthly.reindex(complete_index, fill_value=0.0).astype(float)
        complete_series.index.name = "Month"
        complete_series.name = "Monthly Sales"

        if len(complete_series) < MINIMUM_MONTHLY_OBSERVATIONS:
            skipped_records.append(
                {
                    "State": entity,
                    "Available Months": len(complete_series),
                    "Required Months": MINIMUM_MONTHLY_OBSERVATIONS,
                    "Reason": "Insufficient monthly history",
                }
            )
            logger.warning(
                "Skipping '%s': only %s monthly observations.",
                entity,
                len(complete_series),
            )
            continue

        entity_series[entity] = complete_series

    skipped = pd.DataFrame(
        skipped_records,
        columns=["State", "Available Months", "Required Months", "Reason"],
    )

    logger.info("Eligible entities: %s", len(entity_series))
    logger.info("Skipped entities: %s", len(skipped))

    if not entity_series:
        raise ValueError(
            f"No entity has at least {MINIMUM_MONTHLY_OBSERVATIONS} monthly observations."
        )

    return entity_series, skipped


# =============================================================================
# FORECASTING MODELS
# =============================================================================


def validate_training_values(training_values: np.ndarray) -> np.ndarray:
    """Validate training data."""
    values = np.asarray(training_values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Training data is empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Training data contains NaN or infinity.")
    return values


def historical_average_forecast(
    training_values: np.ndarray,
    horizon: int,
) -> np.ndarray:
    values = validate_training_values(training_values)
    return np.full(horizon, float(np.mean(values)), dtype=float)


def last_value_forecast(
    training_values: np.ndarray,
    horizon: int,
) -> np.ndarray:
    values = validate_training_values(training_values)
    return np.full(horizon, float(values[-1]), dtype=float)


def moving_average_forecast(
    training_values: np.ndarray,
    horizon: int,
    window: int,
) -> np.ndarray:
    values = validate_training_values(training_values)
    if len(values) < window:
        raise ValueError(
            f"Training history contains {len(values)} values, but {window} are required."
        )
    return np.full(horizon, float(np.mean(values[-window:])), dtype=float)


def simple_exponential_smoothing_forecast(
    training_values: np.ndarray,
    horizon: int,
) -> np.ndarray:
    values = validate_training_values(training_values)
    fitted = SimpleExpSmoothing(
        values, initialization_method="estimated"
    ).fit(optimized=True, remove_bias=False)
    return np.asarray(fitted.forecast(horizon), dtype=float)


def holt_linear_forecast(
    training_values: np.ndarray,
    horizon: int,
) -> np.ndarray:
    values = validate_training_values(training_values)
    fitted = Holt(
        values,
        damped_trend=False,
        initialization_method="estimated",
    ).fit(optimized=True, remove_bias=False)
    return np.asarray(fitted.forecast(horizon), dtype=float)


def holt_damped_forecast(
    training_values: np.ndarray,
    horizon: int,
) -> np.ndarray:
    values = validate_training_values(training_values)
    fitted = Holt(
        values,
        damped_trend=True,
        initialization_method="estimated",
    ).fit(optimized=True, remove_bias=False)
    return np.asarray(fitted.forecast(horizon), dtype=float)


def get_model_functions() -> Mapping[str, Callable[[np.ndarray, int], np.ndarray]]:
    """Return all model functions."""
    return {
        MODEL_HISTORICAL_AVERAGE: historical_average_forecast,
        MODEL_LAST_VALUE: last_value_forecast,
        MODEL_THREE_MONTH_AVERAGE: lambda values, horizon: moving_average_forecast(
            values, horizon, 3
        ),
        MODEL_SIX_MONTH_AVERAGE: lambda values, horizon: moving_average_forecast(
            values, horizon, 6
        ),
        MODEL_SIMPLE_EXPONENTIAL_SMOOTHING: simple_exponential_smoothing_forecast,
        MODEL_HOLT_LINEAR: holt_linear_forecast,
        MODEL_HOLT_DAMPED: holt_damped_forecast,
    }


def clean_forecast_values(
    forecast_values: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Validate forecast output and constrain negative forecasts to zero."""
    forecast = np.asarray(forecast_values, dtype=float).reshape(-1)
    if forecast.size != horizon:
        raise ValueError(
            f"Expected {horizon} forecast values, received {forecast.size}."
        )
    if not np.all(np.isfinite(forecast)):
        raise ValueError("Forecast contains NaN or infinity.")
    return np.maximum(forecast, 0.0)


def generate_all_model_forecasts(
    training_values: np.ndarray,
    horizon: int,
    entity: str,
    logger: logging.Logger,
) -> Dict[str, np.ndarray]:
    """Run all models for one entity."""
    functions = get_model_functions()
    forecasts: Dict[str, np.ndarray] = {}

    for model_name in MODEL_NAMES:
        try:
            forecasts[model_name] = clean_forecast_values(
                functions[model_name](training_values, horizon), horizon
            )
        except Exception as exc:
            logger.error(
                "Entity '%s' | Model '%s' failed: %s",
                entity,
                model_name,
                exc,
            )

    if not forecasts:
        raise RuntimeError(f"All forecasting models failed for entity '{entity}'.")
    return forecasts


# =============================================================================
# EVALUATION
# =============================================================================


def calculate_forecast_metrics(
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
) -> MetricResult:
    """Calculate MAE, RMSE, MAPE, and WAPE using NumPy arrays."""
    actual = np.asarray(actual_values, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_values, dtype=float).reshape(-1)

    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted arrays must have the same shape.")
    if actual.size == 0:
        raise ValueError("Cannot evaluate an empty forecast.")
    if not np.all(np.isfinite(actual)):
        raise ValueError("Actual values contain NaN or infinity.")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Predicted values contain NaN or infinity.")

    errors = actual - predicted
    absolute_errors = np.abs(errors)
    mae = float(np.mean(absolute_errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    non_zero = np.abs(actual) > np.finfo(float).eps
    if np.any(non_zero):
        mape = float(
            np.mean(absolute_errors[non_zero] / np.abs(actual[non_zero])) * 100.0
        )
    else:
        mape = float("nan")

    actual_total = float(np.sum(np.abs(actual)))
    error_total = float(np.sum(absolute_errors))
    if actual_total > np.finfo(float).eps:
        wape = float(error_total / actual_total * 100.0)
    elif error_total <= np.finfo(float).eps:
        wape = 0.0
    else:
        wape = float("inf")

    return MetricResult(mae, rmse, mape, wape)


def assign_forecast_reliability(wape: float) -> str:
    """Assign reliability from WAPE."""
    if np.isnan(wape) or np.isinf(wape):
        return "Very Low"
    if wape <= 10:
        return "High"
    if wape <= 20:
        return "Moderate"
    if wape <= 30:
        return "Low"
    return "Very Low"


def select_best_model(metrics: Mapping[str, MetricResult]) -> str:
    """Select the model with lowest WAPE, then RMSE, then MAE."""
    if not metrics:
        raise ValueError("No model metrics are available.")

    model_order = {name: index for index, name in enumerate(MODEL_NAMES)}

    def key(model_name: str) -> Tuple[float, float, float, int]:
        metric = metrics[model_name]
        return (
            metric.wape if np.isfinite(metric.wape) else float("inf"),
            metric.rmse if np.isfinite(metric.rmse) else float("inf"),
            metric.mae if np.isfinite(metric.mae) else float("inf"),
            model_order.get(model_name, len(MODEL_NAMES)),
        )

    return min(metrics.keys(), key=key)


# =============================================================================
# ENTITY FORECASTING
# =============================================================================


def forecast_single_entity(
    entity: str,
    complete_series: pd.Series,
    logger: logging.Logger,
) -> EntityForecastResult:
    """Train, test, evaluate, and select the best model for one entity."""
    if len(complete_series) < MINIMUM_MONTHLY_OBSERVATIONS:
        raise ValueError(f"Entity '{entity}' has insufficient history.")
    if len(complete_series) <= TEST_MONTHS:
        raise ValueError(
            f"Entity '{entity}' does not have enough observations for testing."
        )

    train_series = complete_series.iloc[:-TEST_MONTHS].copy()
    test_series = complete_series.iloc[-TEST_MONTHS:].copy()

    forecasts = generate_all_model_forecasts(
        train_series.to_numpy(dtype=float, copy=True),
        TEST_MONTHS,
        entity,
        logger,
    )

    actual = test_series.to_numpy(dtype=float, copy=True)
    metrics = {
        model_name: calculate_forecast_metrics(actual, forecast)
        for model_name, forecast in forecasts.items()
    }

    best_model = select_best_model(metrics)
    reliability = assign_forecast_reliability(metrics[best_model].wape)

    logger.info(
        "Entity: %s | Best Model: %s | WAPE: %.2f%% | Reliability: %s",
        entity,
        best_model,
        metrics[best_model].wape,
        reliability,
    )

    return EntityForecastResult(
        entity=entity,
        train_series=train_series,
        test_series=test_series,
        forecasts=forecasts,
        metrics=metrics,
        best_model=best_model,
        best_forecast=forecasts[best_model],
        reliability=reliability,
    )


def run_forecasting_for_all_entities(
    entity_series: Mapping[str, pd.Series],
    logger: logging.Logger,
) -> Tuple[List[EntityForecastResult], pd.DataFrame]:
    """Run forecasting for every eligible entity."""
    results: List[EntityForecastResult] = []
    failures: List[Dict[str, object]] = []

    total = len(entity_series)
    for position, (entity, series) in enumerate(
        sorted(entity_series.items(), key=lambda item: item[0]), start=1
    ):
        logger.info("Processing entity %s of %s: %s", position, total, entity)
        try:
            results.append(forecast_single_entity(entity, series, logger))
        except Exception as exc:
            logger.exception("Forecasting failed for entity '%s'.", entity)
            failures.append(
                {
                    "State": entity,
                    "Available Months": len(series),
                    "Reason": str(exc),
                }
            )

    failed_entities = pd.DataFrame(
        failures, columns=["State", "Available Months", "Reason"]
    )
    if not results:
        raise RuntimeError("Forecasting failed for every eligible entity.")
    return results, failed_entities


# =============================================================================
# OUTPUT TABLES
# =============================================================================


def build_forecast_results_table(
    results: Sequence[EntityForecastResult],
) -> pd.DataFrame:
    """Build actual-versus-predicted rows for the best model."""
    records: List[Dict[str, object]] = []

    for result in results:
        metric = result.metrics[result.best_model]
        for position, month in enumerate(result.test_series.index):
            timestamp = pd.Timestamp(month)
            actual = float(result.test_series.iloc[position])
            predicted = float(result.best_forecast[position])
            error = actual - predicted
            records.append(
                {
                    "State": result.entity,
                    "Month": timestamp.strftime("%Y-%m-%d"),
                    "Year": timestamp.year,
                    "Month Number": timestamp.month,
                    "Month Name": timestamp.month_name(),
                    "Actual Sales": actual,
                    "Predicted Sales": predicted,
                    "Error": error,
                    "Absolute Error": abs(error),
                    "Best Model": result.best_model,
                    "MAE": round_metric(metric.mae),
                    "RMSE": round_metric(metric.rmse),
                    "MAPE": round_metric(metric.mape),
                    "WAPE": round_metric(metric.wape),
                    "Forecast Reliability": result.reliability,
                }
            )

    output = pd.DataFrame(records)
    if not output.empty:
        output = output.sort_values(["State", "Month"]).reset_index(drop=True)
    return output


def build_model_comparison_table(
    results: Sequence[EntityForecastResult],
) -> pd.DataFrame:
    """Build metrics for every model and entity."""
    records: List[Dict[str, object]] = []

    for result in results:
        for model_name in MODEL_NAMES:
            if model_name not in result.metrics:
                continue
            metric = result.metrics[model_name]
            records.append(
                {
                    "State": result.entity,
                    "Model": model_name,
                    "MAE": round_metric(metric.mae),
                    "RMSE": round_metric(metric.rmse),
                    "MAPE": round_metric(metric.mape),
                    "WAPE": round_metric(metric.wape),
                    "Forecast Reliability": assign_forecast_reliability(metric.wape),
                    "Is Best Model": model_name == result.best_model,
                    "Training Months": len(result.train_series),
                    "Testing Months": len(result.test_series),
                    "Training Start": result.train_series.index.min().strftime("%Y-%m-%d"),
                    "Training End": result.train_series.index.max().strftime("%Y-%m-%d"),
                    "Testing Start": result.test_series.index.min().strftime("%Y-%m-%d"),
                    "Testing End": result.test_series.index.max().strftime("%Y-%m-%d"),
                }
            )

    output = pd.DataFrame(records)
    if not output.empty:
        order = {name: index for index, name in enumerate(MODEL_NAMES)}
        output["_order"] = output["Model"].map(order)
        output = (
            output.sort_values(["State", "_order"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )
    return output


def build_state_summary_table(
    results: Sequence[EntityForecastResult],
) -> pd.DataFrame:
    """Build one summary row per entity."""
    records: List[Dict[str, object]] = []

    for result in results:
        metric = result.metrics[result.best_model]
        records.append(
            {
                "State": result.entity,
                "Best Model": result.best_model,
                "MAE": round_metric(metric.mae),
                "RMSE": round_metric(metric.rmse),
                "MAPE": round_metric(metric.mape),
                "WAPE": round_metric(metric.wape),
                "Forecast Reliability": result.reliability,
                "Total Monthly Observations": len(result.train_series)
                + len(result.test_series),
                "Training Months": len(result.train_series),
                "Testing Months": len(result.test_series),
                "Training Start": result.train_series.index.min().strftime("%Y-%m-%d"),
                "Training End": result.train_series.index.max().strftime("%Y-%m-%d"),
                "Testing Start": result.test_series.index.min().strftime("%Y-%m-%d"),
                "Testing End": result.test_series.index.max().strftime("%Y-%m-%d"),
            }
        )

    output = pd.DataFrame(records)
    if not output.empty:
        output = output.sort_values(
            ["WAPE", "State"], ascending=[True, True], na_position="last"
        ).reset_index(drop=True)
    return output


def select_entity_for_best_model_export(
    results: Sequence[EntityForecastResult],
) -> EntityForecastResult:
    """Select the entity with the lowest best-model WAPE."""
    if not results:
        raise ValueError("No forecast result is available.")

    def key(result: EntityForecastResult) -> Tuple[float, float, float, str]:
        metric = result.metrics[result.best_model]
        return (
            metric.wape if np.isfinite(metric.wape) else float("inf"),
            metric.rmse if np.isfinite(metric.rmse) else float("inf"),
            metric.mae if np.isfinite(metric.mae) else float("inf"),
            result.entity,
        )

    return min(results, key=key)


def build_selected_best_model_table(
    result: EntityForecastResult,
) -> pd.DataFrame:
    """Build best_model.csv for the selected entity."""
    metric = result.metrics[result.best_model]
    records: List[Dict[str, object]] = []

    for position, month in enumerate(result.test_series.index):
        actual = float(result.test_series.iloc[position])
        predicted = float(result.best_forecast[position])
        error = actual - predicted
        records.append(
            {
                "Selected State": result.entity,
                "Best Model": result.best_model,
                "Month": pd.Timestamp(month).strftime("%Y-%m-%d"),
                "Actual Sales": actual,
                "Predicted Sales": predicted,
                "Error": error,
                "Absolute Error": abs(error),
                "MAE": round_metric(metric.mae),
                "RMSE": round_metric(metric.rmse),
                "MAPE": round_metric(metric.mape),
                "WAPE": round_metric(metric.wape),
                "Forecast Reliability": result.reliability,
            }
        )

    return pd.DataFrame(records)


def build_monthly_preprocessed_table(
    entity_series: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Build the completed monthly time-series audit table."""
    records: List[Dict[str, object]] = []

    for entity, series in sorted(entity_series.items(), key=lambda item: item[0]):
        for month, sales_value in series.items():
            timestamp = pd.Timestamp(month)
            records.append(
                {
                    "State": entity,
                    "Month": timestamp.strftime("%Y-%m-%d"),
                    "Year": timestamp.year,
                    "Month Number": timestamp.month,
                    "Month Name": timestamp.month_name(),
                    "Monthly Sales": float(sales_value),
                }
            )

    return pd.DataFrame(records)


# =============================================================================
# VISUALIZATION
# =============================================================================


def create_entity_forecast_plot(
    result: EntityForecastResult,
    plots_folder: Path,
    logger: logging.Logger,
) -> Path:
    """Save a training, test, and best-forecast plot for one entity."""
    figure, axis = plt.subplots(figsize=(14, 7))
    axis.plot(
        result.train_series.index,
        result.train_series.to_numpy(dtype=float),
        label="Training Data",
        linewidth=2,
    )
    axis.plot(
        result.test_series.index,
        result.test_series.to_numpy(dtype=float),
        label="Actual Test",
        linewidth=2,
        marker="o",
    )
    axis.plot(
        result.test_series.index,
        result.best_forecast,
        label=f"Best Model Forecast: {result.best_model}",
        linewidth=2,
        linestyle="--",
        marker="o",
    )

    metric = result.metrics[result.best_model]
    axis.set_title(
        f"Monthly Sales Forecast - {result.entity}\n"
        f"Best Model: {result.best_model} | WAPE: {metric.wape:.2f}% | "
        f"Reliability: {result.reliability}",
        fontsize=13,
    )
    axis.set_xlabel("Month")
    axis.set_ylabel("Monthly Sales")
    axis.grid(True, alpha=0.30)
    axis.legend()
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()

    output_path = plots_folder / f"{sanitize_filename(result.entity)}_forecast.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    logger.debug("Plot saved: %s", output_path)
    return output_path


def create_all_forecast_plots(
    results: Sequence[EntityForecastResult],
    plots_folder: Path,
    logger: logging.Logger,
) -> None:
    """Generate one plot per entity."""
    successful = 0
    failed = 0
    for result in results:
        try:
            create_entity_forecast_plot(result, plots_folder, logger)
            successful += 1
        except Exception:
            failed += 1
            logger.exception("Plot generation failed for '%s'.", result.entity)
    logger.info("Plots created: %s | Plot failures: %s", successful, failed)


# =============================================================================
# EXPORT
# =============================================================================


def export_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Export a dataframe to UTF-8 CSV."""
    try:
        dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Exported %s rows to %s", len(dataframe), output_path)
    except Exception as exc:
        raise IOError(f"Unable to export '{output_path}': {exc}") from exc


def export_all_outputs(
    results: Sequence[EntityForecastResult],
    entity_series: Mapping[str, pd.Series],
    skipped_entities: pd.DataFrame,
    failed_entities: pd.DataFrame,
    output_folder: Path,
    logger: logging.Logger,
) -> None:
    """Build and export every pipeline output."""
    export_dataframe(
        build_forecast_results_table(results),
        output_folder / "forecast_results.csv",
        logger,
    )
    export_dataframe(
        build_model_comparison_table(results),
        output_folder / "model_comparison.csv",
        logger,
    )
    export_dataframe(
        build_selected_best_model_table(select_entity_for_best_model_export(results)),
        output_folder / "best_model.csv",
        logger,
    )
    export_dataframe(
        build_state_summary_table(results),
        output_folder / "state_summary.csv",
        logger,
    )
    export_dataframe(
        build_monthly_preprocessed_table(entity_series),
        output_folder / "monthly_preprocessed_data.csv",
        logger,
    )
    export_dataframe(
        skipped_entities,
        output_folder / "skipped_entities.csv",
        logger,
    )
    export_dataframe(
        failed_entities,
        output_folder / "failed_entities.csv",
        logger,
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_pipeline(input_file: Path, output_folder: Path) -> None:
    """Run the complete forecasting pipeline."""
    directories = ensure_output_directories(output_folder)
    logger = configure_logging(directories["root"])

    logger.info("=" * 80)
    logger.info("ENTERPRISE SALES FORECASTING PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info("Input file: %s", input_file)
    logger.info("Output folder: %s", output_folder)
    logger.info("Minimum history: %s months", MINIMUM_MONTHLY_OBSERVATIONS)
    logger.info("Test period: %s months", TEST_MONTHS)

    try:
        raw_data, source_name = load_input_data(input_file, logger)
        logger.info("Input source selected: %s", source_name)

        detected_columns = detect_required_columns(raw_data, logger)
        monthly_sales = preprocess_transactional_data(
            raw_data, detected_columns, logger
        )
        entity_series, skipped_entities = build_complete_entity_series(
            monthly_sales, logger
        )
        results, failed_entities = run_forecasting_for_all_entities(
            entity_series, logger
        )

        export_all_outputs(
            results,
            entity_series,
            skipped_entities,
            failed_entities,
            directories["root"],
            logger,
        )
        create_all_forecast_plots(results, directories["plots"], logger)

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("Successfully forecast entities: %s", len(results))
        logger.info("Files saved to: %s", output_folder)
        logger.info("=" * 80)

        print("\nForecasting pipeline completed successfully.")
        print(f"Output folder: {output_folder}")
        print(f"Entities forecast: {len(results)}")

    except Exception:
        logger.critical("Pipeline execution failed.\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        run_pipeline(INPUT_FILE, OUTPUT_FOLDER)
    except KeyboardInterrupt:
        print("\nPipeline execution was interrupted by the user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)
