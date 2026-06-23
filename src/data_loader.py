"""Data loading and SQLite feature store management.

Loads the German Credit dataset (UCI via sklearn), inserts it into a SQLite
feature store, and exposes SQL-based feature extraction for downstream modelling.
"""

import os
import sqlite3
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import fetch_openml

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = PROJECT_ROOT / "sql"
DEFAULT_DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "credit_risk.db"))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_german_credit() -> pd.DataFrame:
    """Download and return the German Credit dataset as a clean DataFrame.

    Source: UCI German Credit Data via sklearn fetch_openml (no login required).
    Target encoding: default_flag = 1 (bad/default), 0 (good/non-default).

    Returns:
        pd.DataFrame: 1000-row DataFrame with 20 features + ``default_flag``.
    """
    logger.info("Fetching German Credit dataset from OpenML ...")
    credit = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df: pd.DataFrame = credit.frame.copy()

    # Normalise column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Encode target: bad = 1 (default), good = 0 (non-default)
    df["default_flag"] = (df["class"] == "bad").astype(int)
    df = df.drop(columns=["class"])

    # Cast numeric columns that sklearn sometimes returns as object/category
    numeric_cols = [
        "duration", "credit_amount", "installment_commitment",
        "residence_since", "age", "existing_credits", "num_dependents",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise categorical columns to plain string
    cat_cols = [c for c in df.columns if c not in numeric_cols + ["default_flag"]]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    logger.info("Dataset loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def _read_sql_file(filename: str) -> str:
    """Read a SQL file from the sql/ directory."""
    path = SQL_DIR / filename
    return path.read_text(encoding="utf-8")


def init_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create the SQLite database and all tables from create_tables.sql.

    Args:
        db_path: Filesystem path to the SQLite database file.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    ddl = _read_sql_file("create_tables.sql")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(ddl)
    logger.info("Database initialised at %s", db_path)


def insert_applicants(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> int:
    """Insert applicant records into the SQLite applicants table.

    Replaces any existing rows so the pipeline is idempotent.

    Args:
        df:      Clean DataFrame from ``load_german_credit()``.
        db_path: Path to the SQLite database.

    Returns:
        int: Number of rows inserted.
    """
    columns = [
        "checking_status", "duration", "credit_history", "purpose",
        "credit_amount", "savings_status", "employment",
        "installment_commitment", "personal_status", "other_parties",
        "residence_since", "property_magnitude", "age",
        "other_payment_plans", "housing", "existing_credits",
        "job", "num_dependents", "own_telephone", "foreign_worker",
        "default_flag",
    ]
    subset = df[columns].copy()

    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM applicants")
        subset.to_sql("applicants", conn, if_exists="append", index=False)
        count = conn.execute("SELECT COUNT(*) FROM applicants").fetchone()[0]

    logger.info("Inserted %d applicant records", count)
    return count


# ---------------------------------------------------------------------------
# Feature extraction via SQL
# ---------------------------------------------------------------------------

def extract_features(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Extract model features from SQLite using the CTE-based SQL query.

    Runs ``sql/feature_extraction.sql``, which derives ``debt_to_income_proxy``,
    ``ever_late_flag``, ``poor_checking_flag``, and ``age_per_month_credit``
    alongside all raw attributes.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        pd.DataFrame: Feature DataFrame including ``default_flag`` as target.
    """
    query = _read_sql_file("feature_extraction.sql")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    logger.info("Extracted %d rows with %d features", len(df), len(df.columns))
    return df


def run_vintage_analysis(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Run the window-function vintage analysis query.

    Returns default rate by credit duration bucket with LAG-based period-on-period
    change, demonstrating SQL window function proficiency.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        pd.DataFrame: Vintage analysis results with ``default_rate_change_pct``.
    """
    query = _read_sql_file("vintage_analysis.sql")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    return df


# ---------------------------------------------------------------------------
# Pipeline entrypoint
# ---------------------------------------------------------------------------

def run_data_pipeline(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Full data ingestion pipeline: download → store → extract.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        pd.DataFrame: Feature-engineered DataFrame ready for WoE encoding.
    """
    raw_df = load_german_credit()
    init_database(db_path)
    insert_applicants(raw_df, db_path)
    features_df = extract_features(db_path)

    vintage = run_vintage_analysis(db_path)
    logger.info("Vintage analysis:\n%s", vintage.to_string(index=False))

    return features_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = run_data_pipeline()
    print(df.head())
    print(f"\nDefault rate: {df['default_flag'].mean():.1%}")
