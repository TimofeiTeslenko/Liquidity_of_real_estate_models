# Скрипт обучает Cox proportional hazards model для оценки связи характеристик объявления
# с интенсивностью снятия объекта с рынка. Добавлен переключатель USE_DELAYED_ENTRY:
# если True, объявления до 01.10.2025 учитываются через delayed entry; если False,
# объявления до 01.10.2025 исключаются из модели.

import os
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy.stats import chi2

# =========================
# НАСТРОЙКИ
# =========================

INPUT_XLSX = r"C:\Users\Я\Documents\Учеба\Аспирантура - PhD\Кандидатская диссертация\3. Кодинг (Python)\Итоговый_файл + Хубер.xlsx"
OUTPUT_DIR = r"C:\Users\Я\Documents\Учеба\Аспирантура - PhD\Кандидатская диссертация\3. Кодинг (Python)\7. Модели\2. Cox - ексель"
OUTPUT_REPORT_XLSX = os.path.join(OUTPUT_DIR, "cox_report__v8.xlsx")

CUTOFF_DATE_STR = "30.06.2026"  # дата правого цензурирования: дд.мм.гггг
OBS_START_DATE_STR = "01.10.2025"  # дата начала фактического наблюдения: дд.мм.гггг

# Left truncation / delayed entry
# True  = старые объявления учитываются через entry_i. Для спецификации с delayed entry тест пропорциональности рисков на основе остатков не рассчитывается, поскольку используемая реализация lifelines не поддерживает вычисление residuals для Cox-моделей с entry_col.
# False = объявления до OBS_START_DATE_STR исключаются из модели
USE_DELAYED_ENTRY = False

# Cox regularization - смещение к 0
PENALIZER = 0

# No train/validation split - не нужно так как не предикт модель
USE_TRAIN_VALID_SPLIT = False

# Category controls
MIN_COUNT_PER_CATEGORY = 15
LOW_VAR_THRESHOLD = 1e-8

# Cap for Locality ID levels
MAX_LEVELS_LOC_ID = 500  # 0 = no cap

# DevHuber clipping - для учета Хубер цены
WINSOR_Q_LOW = 0.01
WINSOR_Q_HIGH = 0.99
HARD_CLIP_LOW = -0.99
HARD_CLIP_HIGH = 50.0

# =========================
# MODEL DESIGN SWITCHES
# =========================

INCLUDE_DEV_HUBER = False
INCLUDE_LOG_PRICE = True
INCLUDE_LOG_AREA = True
INCLUDE_OKS = True

INCLUDE_VRI_FE = True
INCLUDE_REGION_FE = True
INCLUDE_LOCALITY_TYPE_FE = True
INCLUDE_LOCALITY_ID_FE = False

# Включить M для левоценз
INCLUDE_TIME_FE = True
TIME_FE_FREQ = "M"  # "W" = week, "M" = month, "Q" = quarter

# Base categories
BASE_REGION = "58"
BASE_LOCALITY_TYPE = "3"
BASE_TIME_FE_BY_FREQ = {
    "M": "2025-10",
    "Q": "2025Q4",
}

# Block-drop tests
RUN_BLOCK_DROP_TESTS = True

BLOCK_DROP_ORDER = [
    "Time_FE",
    "Region_FE",
    "Locality_Type_FE",
    "Locality_ID_FE",
    "DevHuber",
    "log_Цена_чист",
    "Есть_объект",
    "ВРИ_группа",
    "log_Площадь_сот",
]

# =========================
# ИМЕНА КОЛОНОК
# =========================

COL_POSTED = "Дата создания"
COL_REMOVED = "Дата снятия объявления"
COL_DOM = "Дней на рынке"

COL_AREA = "Площадь_сот"
COL_PRICE = "Цена_чист"
COL_OKS = "Есть_объект"

COL_VRI = "ВРИ_группа"
COL_REGION = "Код региона"
COL_LOC_ID = "Locality ID"
COL_LOC_TYPE_ID = "Locality Type ID"

COL_DEV_HUBER_PCT = "Отклонение стоимости (Huber), %"

VAR_DEV_HUBER = "DevHuber"
VAR_LOG_AREA = "log_" + COL_AREA
VAR_LOG_PRICE = "log_" + COL_PRICE
VAR_TIME_FE = "Time_FE"

DURATION_COL = "T_i"
EVENT_COL = "delta_i"
ENTRY_COL = "entry_i"


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def ensure_cols_exist(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"В файле нет обязательных колонок: {missing}")


def parse_date_ru(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, dayfirst=True, format="mixed", errors="coerce")
    except TypeError:
        return pd.to_datetime(series, dayfirst=True, errors="coerce")


def to_numeric_ru(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace("\u00A0", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def to_category_str(series: pd.Series, force_integer: bool = False) -> pd.Series:
    if force_integer:
        x = to_numeric_ru(series)
        return x.astype("Int64").astype(str).replace({
            "<NA>": "__NA__",
            "nan": "__NA__",
            "None": "__NA__",
            "NaT": "__NA__",
        })

    s = series.astype(str).str.strip()
    s = s.replace({
        "": "__NA__",
        "nan": "__NA__",
        "None": "__NA__",
        "NaT": "__NA__",
        "<NA>": "__NA__",
    })

    numeric_candidate = pd.to_numeric(s.replace("__NA__", np.nan), errors="coerce")
    non_na_original = s.ne("__NA__").sum()
    non_na_numeric = numeric_candidate.notna().sum()

    if non_na_original > 0 and non_na_original == non_na_numeric:
        return numeric_candidate.astype("Int64").astype(str).replace({
            "<NA>": "__NA__",
            "nan": "__NA__",
            "None": "__NA__",
            "NaT": "__NA__",
        })

    return s


def collapse_rare_categories(
        s: pd.Series,
        min_count: int,
        rare_label: str = "__RARE__",
        protected_values: list[str] | None = None
) -> pd.Series:
    protected_values = protected_values or []
    vc = s.value_counts(dropna=False)
    rare = vc[vc < min_count].index
    rare = [x for x in rare if str(x) not in protected_values]
    return s.where(~s.isin(rare), other=rare_label)


def cap_category_levels(
        s: pd.Series,
        max_levels: int,
        protected_values: list[str] | None = None,
        other_label: str = "__OTHER_CAP__"
) -> pd.Series:
    if max_levels is None or max_levels <= 0:
        return s

    protected_values = protected_values or []
    vc = s.value_counts(dropna=False)

    keep = list(vc.head(max_levels).index)
    keep = list(set([str(x) for x in keep] + [str(x) for x in protected_values]))

    return s.where(s.astype(str).isin(keep), other=other_label)


def validate_design() -> None:
    if TIME_FE_FREQ not in ["W", "M", "Q"]:
        raise ValueError('TIME_FE_FREQ должен быть одним из значений: "W", "M", "Q".')

    if USE_DELAYED_ENTRY and pd.to_datetime(OBS_START_DATE_STR, dayfirst=True) >= pd.to_datetime(CUTOFF_DATE_STR,
                                                                                                 dayfirst=True):
        raise ValueError("OBS_START_DATE_STR должен быть раньше CUTOFF_DATE_STR.")


def drop_low_variance_columns(
        df: pd.DataFrame,
        exclude: list[str],
        thr: float
) -> tuple[pd.DataFrame, list[str]]:
    low_var_cols = []

    for c in df.columns:
        if c in exclude:
            continue

        v = df[c].var()

        if pd.notna(v) and v < thr:
            low_var_cols.append(c)

    if low_var_cols:
        df = df.drop(columns=low_var_cols)

    return df, low_var_cols


def make_dummies_with_base(
        df: pd.DataFrame,
        col: str,
        block_name: str,
        base_value: str | None,
        strict_base: bool,
        var_to_block: dict,
        var_to_base: dict,
        base_info: dict
) -> tuple[pd.DataFrame, list[str]]:
    categories = sorted(df[col].dropna().astype(str).unique().tolist())

    if not categories:
        base_info[col] = {
            "block": block_name,
            "base_category": None,
            "base_mode": "no_categories",
            "note": "No categories available."
        }
        return df.drop(columns=[col]), []

    if base_value is None:
        base_used = categories[0]
        base_mode = "automatic_first_by_order"
    else:
        base_used = str(base_value)
        base_mode = "manual"

    if base_used not in categories:
        if strict_base:
            raise ValueError(
                f"Базовая категория '{base_used}' отсутствует в колонке '{col}' "
                f"после фильтров и обработки редких категорий."
            )
        else:
            base_used = categories[0]
            base_mode = "fallback_to_first_by_order"

    dummies = pd.get_dummies(df[col].astype(str), prefix=col, dtype=float)

    base_dummy_col = f"{col}_{base_used}"

    if base_dummy_col not in dummies.columns:
        if strict_base:
            raise ValueError(f"Не найдена dummy-колонка для базовой категории: {base_dummy_col}")

        fallback_col = sorted(dummies.columns.tolist())[0]
        dummies = dummies.drop(columns=[fallback_col])
        base_used = fallback_col.replace(f"{col}_", "", 1)
        base_mode = "fallback_dummy_column"
    else:
        dummies = dummies.drop(columns=[base_dummy_col])

    dummy_cols = dummies.columns.tolist()

    for dc in dummy_cols:
        var_to_block[dc] = block_name
        var_to_base[dc] = f"{col}={base_used}"

    base_info[col] = {
        "block": block_name,
        "base_category": f"{col}={base_used}",
        "base_mode": base_mode,
        "note": "Base category is excluded from dummy variables."
    }

    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    return df, dummy_cols


def fit_cox(df: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=PENALIZER)

    if USE_DELAYED_ENTRY:
        cph.fit(
            df,
            duration_col=DURATION_COL,
            event_col=EVENT_COL,
            entry_col=ENTRY_COL
        )
    else:
        cph.fit(
            df,
            duration_col=DURATION_COL,
            event_col=EVENT_COL
        )

    return cph


def get_interpretation_unit(variable: str, block: str) -> str:
    if variable == VAR_DEV_HUBER:
        return "+1 percentage point"

    if variable == VAR_LOG_PRICE:
        return "+10% price"

    if variable == VAR_LOG_AREA:
        return "+10% area"

    if variable == COL_OKS:
        return "1 vs 0"

    if block in [
        "ВРИ_группа",
        "Region_FE",
        "Locality_Type_FE",
        "Locality_ID_FE",
        "Time_FE",
    ]:
        return "category vs base"

    return ""


def build_summary_full(
        cph: CoxPHFitter,
        var_to_block: dict,
        var_to_base: dict
) -> pd.DataFrame:
    summary = cph.summary.copy()
    summary.insert(0, "variable", summary.index.astype(str))

    summary["block"] = summary["variable"].map(var_to_block).fillna("__UNKNOWN__")
    summary["base_category"] = summary["variable"].map(var_to_base).fillna("")
    summary["interpretation_unit"] = summary.apply(
        lambda r: get_interpretation_unit(r["variable"], r["block"]),
        axis=1
    )

    summary["HR(+1pp)"] = np.nan
    summary["HR(+10% price)"] = np.nan
    summary["HR(+10% area)"] = np.nan
    summary["HR(1 vs 0)"] = np.nan

    if VAR_DEV_HUBER in summary["variable"].values:
        coef = float(summary.loc[summary["variable"] == VAR_DEV_HUBER, "coef"].iloc[0])
        summary.loc[summary["variable"] == VAR_DEV_HUBER, "HR(+1pp)"] = float(np.exp(coef * 0.01))

    if VAR_LOG_PRICE in summary["variable"].values:
        coef = float(summary.loc[summary["variable"] == VAR_LOG_PRICE, "coef"].iloc[0])
        summary.loc[summary["variable"] == VAR_LOG_PRICE, "HR(+10% price)"] = float(np.exp(coef * np.log(1.10)))

    if VAR_LOG_AREA in summary["variable"].values:
        coef = float(summary.loc[summary["variable"] == VAR_LOG_AREA, "coef"].iloc[0])
        summary.loc[summary["variable"] == VAR_LOG_AREA, "HR(+10% area)"] = float(np.exp(coef * np.log(1.10)))

    if COL_OKS in summary["variable"].values:
        coef = float(summary.loc[summary["variable"] == COL_OKS, "coef"].iloc[0])
        summary.loc[summary["variable"] == COL_OKS, "HR(1 vs 0)"] = float(np.exp(coef))

    return summary.reset_index(drop=True)


def build_ph_assumption(
        cph: CoxPHFitter,
        model_df: pd.DataFrame,
        var_to_block: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        ph = proportional_hazard_test(cph, model_df, time_transform="rank")
        ph_summary = ph.summary.reset_index().rename(columns={
            "index": "variable",
            "p": "p_value",
            "-log2(p)": "minus_log2_p",
        })

        if "test_statistic" not in ph_summary.columns:
            ph_summary["test_statistic"] = np.nan

        if "minus_log2_p" not in ph_summary.columns:
            ph_summary["minus_log2_p"] = np.nan

        ph_summary["variable"] = ph_summary["variable"].astype(str)
        ph_summary["block"] = ph_summary["variable"].map(var_to_block).fillna("__UNKNOWN__")

        ph_summary["PH_flag_5pct"] = (ph_summary["p_value"] < 0.05).astype(int)
        ph_summary["PH_flag_1pct"] = (ph_summary["p_value"] < 0.01).astype(int)

        ph_summary["interpretation"] = np.select(
            [
                ph_summary["p_value"] < 0.01,
                ph_summary["p_value"] < 0.05,
                ph_summary["p_value"] >= 0.05,
            ],
            [
                "Strong indication of PH violation",
                "Possible PH violation",
                "No clear PH violation",
            ],
            default="Unknown"
        )

        ph_blocks = (
            ph_summary
            .groupby("block", dropna=False)
            .agg(
                n_variables=("variable", "count"),
                n_PH_violations_5pct=("PH_flag_5pct", "sum"),
                n_PH_violations_1pct=("PH_flag_1pct", "sum"),
                min_p_value=("p_value", "min"),
                median_p_value=("p_value", "median"),
                max_test_statistic=("test_statistic", "max"),
            )
            .reset_index()
        )

        ph_blocks["share_PH_violations_5pct"] = (
                ph_blocks["n_PH_violations_5pct"] / ph_blocks["n_variables"]
        )

        ph_blocks["share_PH_violations_1pct"] = (
                ph_blocks["n_PH_violations_1pct"] / ph_blocks["n_variables"]
        )

        ph_blocks = ph_blocks[
            [
                "block",
                "n_variables",
                "n_PH_violations_5pct",
                "share_PH_violations_5pct",
                "n_PH_violations_1pct",
                "share_PH_violations_1pct",
                "min_p_value",
                "median_p_value",
                "max_test_statistic",
            ]
        ]

        return ph_summary, ph_blocks

    except Exception as e:
        ph_summary = pd.DataFrame([{
            "variable": "__ERROR__",
            "block": "__ERROR__",
            "test_statistic": np.nan,
            "p_value": np.nan,
            "minus_log2_p": np.nan,
            "PH_flag_5pct": np.nan,
            "PH_flag_1pct": np.nan,
            "interpretation": f"PH test failed: {str(e)}",
        }])

        ph_blocks = pd.DataFrame([{
            "block": "__ERROR__",
            "n_variables": np.nan,
            "n_PH_violations_5pct": np.nan,
            "share_PH_violations_5pct": np.nan,
            "n_PH_violations_1pct": np.nan,
            "share_PH_violations_1pct": np.nan,
            "min_p_value": np.nan,
            "median_p_value": np.nan,
            "max_test_statistic": np.nan,
        }])

        return ph_summary, ph_blocks


def build_block_drop_tests(
        full_cph: CoxPHFitter,
        model_df: pd.DataFrame,
        block_cols: dict
) -> pd.DataFrame:
    rows = []

    ll_full = float(full_cph.log_likelihood_)

    for order, block in enumerate(BLOCK_DROP_ORDER, start=1):
        cols_to_drop = [c for c in block_cols.get(block, []) if c in model_df.columns]
        n_drop = len(cols_to_drop)

        if n_drop == 0:
            rows.append({
                "block_order": order,
                "block": block,
                "status": "not_in_model",
                "n_vars_dropped": 0,
                "ll_full": ll_full,
                "ll_reduced": np.nan,
                "LR": np.nan,
                "df": 0,
                "p_value": np.nan,
                "note": "Block is not included in the current specification or was dropped by low-variance filter.",
            })
            continue

        reduced_df = model_df.drop(columns=cols_to_drop)

        try:
            cph_reduced = fit_cox(reduced_df)

            ll_reduced = float(cph_reduced.log_likelihood_)
            lr_stat = 2.0 * (ll_full - ll_reduced)
            p_value = float(chi2.sf(lr_stat, n_drop))

            rows.append({
                "block_order": order,
                "block": block,
                "status": "ok",
                "n_vars_dropped": n_drop,
                "ll_full": ll_full,
                "ll_reduced": ll_reduced,
                "LR": lr_stat,
                "df": n_drop,
                "p_value": p_value,
                "note": "Diagnostic LR-type block-drop test with penalized Cox model.",
            })

        except Exception as e:
            rows.append({
                "block_order": order,
                "block": block,
                "status": "error",
                "n_vars_dropped": n_drop,
                "ll_full": ll_full,
                "ll_reduced": np.nan,
                "LR": np.nan,
                "df": n_drop,
                "p_value": np.nan,
                "note": str(e),
            })

    return pd.DataFrame(rows)


def build_diagnostics(
        cph: CoxPHFitter,
        model_df: pd.DataFrame,
        raw_n: int,
        preprocess_meta: dict
) -> pd.DataFrame:
    n_total = int(len(model_df))
    n_events = int(model_df[EVENT_COL].sum())
    n_censored = int(n_total - n_events)
    event_rate = float(model_df[EVENT_COL].mean()) if n_total > 0 else np.nan

    ll = float(cph.log_likelihood_)
    k_params = int(len(cph.params_))

    aic = float(-2.0 * ll + 2.0 * k_params)
    bic = float(-2.0 * ll + np.log(max(n_total, 1)) * k_params)

    diagnostics = pd.DataFrame([{
        "model_name": "cox_report__v9",
        "n_raw": int(raw_n),
        "n_after_obs_start_filter": preprocess_meta.get("n_after_obs_start_filter"),
        "n_total": n_total,
        "n_events": n_events,
        "n_censored": n_censored,
        "event_rate": event_rate,
        "log_likelihood": ll,
        "k_params": k_params,
        "AIC": aic,
        "BIC": bic,
        "cindex_full": float(cph.concordance_index_),
        "penalizer": float(PENALIZER),
        "use_train_valid_split": bool(USE_TRAIN_VALID_SPLIT),
        "cutoff_date": preprocess_meta.get("cutoff_date"),
        "obs_start_date": preprocess_meta.get("obs_start_date"),
        "use_delayed_entry": preprocess_meta.get("use_delayed_entry"),
        "n_pre_obs_start_rows": preprocess_meta.get("n_pre_obs_start_rows"),
        "n_rows_with_positive_entry": preprocess_meta.get("n_rows_with_positive_entry"),
        "time_fe_freq": TIME_FE_FREQ if INCLUDE_TIME_FE else "",
        "min_count_per_category": int(MIN_COUNT_PER_CATEGORY),
        "low_var_threshold": float(LOW_VAR_THRESHOLD),
        "dropped_low_var_count": int(preprocess_meta.get("dropped_low_var_count", 0)),
        "dropped_low_var_columns": "; ".join(preprocess_meta.get("dropped_low_var_cols", [])),
        "winsor_q_low": WINSOR_Q_LOW,
        "winsor_q_high": WINSOR_Q_HIGH,
        "devhuber_q_low": preprocess_meta.get("devhuber_q_low"),
        "devhuber_q_high": preprocess_meta.get("devhuber_q_high"),
        "hard_clip_low": HARD_CLIP_LOW,
        "hard_clip_high": HARD_CLIP_HIGH,
    }])

    return diagnostics


def count_dropped_low_var_for_block(
        block: str,
        block_cols_before_low_var: dict,
        dropped_low_var: list[str]
) -> int:
    return len([c for c in dropped_low_var if c in block_cols_before_low_var.get(block, [])])


def build_model_blocks(
        block_cols: dict,
        block_cols_before_low_var: dict,
        base_info: dict,
        dropped_low_var: list[str]
) -> pd.DataFrame:
    rows = [
        {
            "block_order": 1,
            "block": "Time_FE",
            "included": INCLUDE_TIME_FE,
            "source_columns": COL_POSTED if INCLUDE_TIME_FE else "",
            "n_model_variables": len(block_cols.get("Time_FE", [])),
            "base_category": base_info.get(VAR_TIME_FE, {}).get("base_category", ""),
            "settings": f"TIME_FE_FREQ={TIME_FE_FREQ}; MIN_COUNT_PER_CATEGORY={MIN_COUNT_PER_CATEGORY}" if INCLUDE_TIME_FE else "",
            "dropped_low_var_count": count_dropped_low_var_for_block(
                "Time_FE", block_cols_before_low_var, dropped_low_var
            ),
        },
        {
            "block_order": 2,
            "block": "Region_FE",
            "included": INCLUDE_REGION_FE,
            "source_columns": COL_REGION if INCLUDE_REGION_FE else "",
            "n_model_variables": len(block_cols.get("Region_FE", [])),
            "base_category": base_info.get(COL_REGION, {}).get("base_category", ""),
            "settings": f"BASE_REGION={BASE_REGION}; MIN_COUNT_PER_CATEGORY={MIN_COUNT_PER_CATEGORY}" if INCLUDE_REGION_FE else "",
            "dropped_low_var_count": count_dropped_low_var_for_block(
                "Region_FE", block_cols_before_low_var, dropped_low_var
            ),
        },
        {
            "block_order": 3,
            "block": "Locality_Type_FE",
            "included": INCLUDE_LOCALITY_TYPE_FE,
            "source_columns": COL_LOC_TYPE_ID if INCLUDE_LOCALITY_TYPE_FE else "",
            "n_model_variables": len(block_cols.get("Locality_Type_FE", [])),
            "base_category": base_info.get(COL_LOC_TYPE_ID, {}).get("base_category", ""),
            "settings": f"BASE_LOCALITY_TYPE={BASE_LOCALITY_TYPE}; MIN_COUNT_PER_CATEGORY={MIN_COUNT_PER_CATEGORY}" if INCLUDE_LOCALITY_TYPE_FE else "",
            "dropped_low_var_count": count_dropped_low_var_for_block(
                "Locality_Type_FE", block_cols_before_low_var, dropped_low_var
            ),
        },
        {
            "block_order": 4,
            "block": "Locality_ID_FE",
            "included": INCLUDE_LOCALITY_ID_FE,
            "source_columns": COL_LOC_ID if INCLUDE_LOCALITY_ID_FE else "",
            "n_model_variables": len(block_cols.get("Locality_ID_FE", [])),
            "base_category": base_info.get(COL_LOC_ID, {}).get("base_category", ""),
            "settings": f"MAX_LEVELS_LOC_ID={MAX_LEVELS_LOC_ID}; MIN_COUNT_PER_CATEGORY={MIN_COUNT_PER_CATEGORY}" if INCLUDE_LOCALITY_ID_FE else "",
            "dropped_low_var_count": count_dropped_low_var_for_block(
                "Locality_ID_FE", block_cols_before_low_var, dropped_low_var
            ),
        },
        {
            "block_order": 5,
            "block": "DevHuber",
            "included": INCLUDE_DEV_HUBER,
            "source_columns": COL_DEV_HUBER_PCT if INCLUDE_DEV_HUBER else "",
            "n_model_variables": len(block_cols.get("DevHuber", [])),
            "base_category": "",
            "settings": "percentage points converted to shares" if INCLUDE_DEV_HUBER else "",
            "dropped_low_var_count": count_dropped_low_var_for_block(
                "DevHuber", block_cols_before_low_var, dropped_low_var
            ),
        },
        {
            "block_order": 6,
            "block": "log_Цена_чист",
            "included": INCLUDE_LOG_PRICE,
            "source_columns": COL_PRICE if INCLUDE_LOG_PRICE else "",
            "n_model_variables": len(block_cols.get("log_Цена_чист", [])),
            "base_category": "",
            "settings": "log(price)" if INCLUDE_LOG_PRICE else "",
            "dropped_low_var_count": count_dropped_low_var_for_block(
                "log_Цена_чист", block_cols_before_low_var, dropped_low_var
            ),
        },
        {
            "block_order": 7,
            "block": "Есть_объект",
            "included": INCLUDE_OKS,
            "source_columns": COL_OKS if INCLUDE_OKS else "",
            "n_model_variables": len(block_cols.get("Есть_объект", [])),
            "base_category": "0" if INCLUDE_OKS else "",
            "settings": "binary covariate" if INCLUDE_OKS else "",
            "dropped_low_var_count": count_dropped_low_var_for_block(
                "Есть_объект", block_cols_before_low_var, dropped_low_var
            ),
        },
        {
            "block_order": 8,
            "block": "ВРИ_группа",
            "included": INCLUDE_VRI_FE,
            "source_columns": COL_VRI if INCLUDE_VRI_FE else "",
            "n_model_variables": len(block_cols.get("ВРИ_группа", [])),
            "base_category": base_info.get(COL_VRI, {}).get("base_category", ""),
            "settings": f"MIN_COUNT_PER_CATEGORY={MIN_COUNT_PER_CATEGORY}; base = first category by order" if INCLUDE_VRI_FE else "",
            "dropped_low_var_count": count_dropped_low_var_for_block(
                "ВРИ_группа", block_cols_before_low_var, dropped_low_var
            ),
        },
        {
            "block_order": 9,
            "block": "log_Площадь_сот",
            "included": INCLUDE_LOG_AREA,
            "source_columns": COL_AREA if INCLUDE_LOG_AREA else "",
            "n_model_variables": len(block_cols.get("log_Площадь_сот", [])),
            "base_category": "",
            "settings": "log(area)" if INCLUDE_LOG_AREA else "",
            "dropped_low_var_count": count_dropped_low_var_for_block(
                "log_Площадь_сот", block_cols_before_low_var, dropped_low_var
            ),
        },
    ]

    return pd.DataFrame(rows)


# =========================
# ПОДГОТОВКА ДАННЫХ
# =========================

def prepare_model_df(input_xlsx: str):
    validate_design()

    df = pd.read_excel(input_xlsx)
    raw_n = len(df)

    required_cols = [
        COL_POSTED,
        COL_REMOVED,
        COL_DOM,
    ]

    if INCLUDE_LOG_AREA:
        required_cols.append(COL_AREA)

    if INCLUDE_LOG_PRICE:
        required_cols.append(COL_PRICE)

    if INCLUDE_OKS:
        required_cols.append(COL_OKS)

    if INCLUDE_DEV_HUBER:
        required_cols.append(COL_DEV_HUBER_PCT)

    if INCLUDE_VRI_FE:
        required_cols.append(COL_VRI)

    if INCLUDE_REGION_FE:
        required_cols.append(COL_REGION)

    if INCLUDE_LOCALITY_TYPE_FE:
        required_cols.append(COL_LOC_TYPE_ID)

    if INCLUDE_LOCALITY_ID_FE:
        required_cols.append(COL_LOC_ID)

    ensure_cols_exist(df, required_cols)

    df[COL_POSTED] = parse_date_ru(df[COL_POSTED])
    df[COL_REMOVED] = parse_date_ru(df[COL_REMOVED])

    cutoff_date = pd.to_datetime(CUTOFF_DATE_STR, dayfirst=True)
    obs_start_date = pd.to_datetime(OBS_START_DATE_STR, dayfirst=True)

    df[COL_DOM] = to_numeric_ru(df[COL_DOM])

    if INCLUDE_LOG_AREA:
        df[COL_AREA] = to_numeric_ru(df[COL_AREA])

    if INCLUDE_LOG_PRICE:
        df[COL_PRICE] = to_numeric_ru(df[COL_PRICE])

    if INCLUDE_OKS:
        df[COL_OKS] = to_numeric_ru(df[COL_OKS])

    if INCLUDE_DEV_HUBER:
        dev_pp = to_numeric_ru(df[COL_DEV_HUBER_PCT])
        df[VAR_DEV_HUBER] = dev_pp / 100.0

    df[EVENT_COL] = df[COL_REMOVED].notna().astype(int)

    df[DURATION_COL] = np.nan

    sold_mask = df[EVENT_COL] == 1
    active_mask = df[EVENT_COL] == 0

    df.loc[sold_mask, DURATION_COL] = df.loc[sold_mask, COL_DOM]
    df.loc[active_mask, DURATION_COL] = (
            cutoff_date - df.loc[active_mask, COL_POSTED]
    ).dt.days.astype(float)

    df[ENTRY_COL] = 0.0

    pre_obs_mask = df[COL_POSTED] < obs_start_date
    n_pre_obs_start_rows = int(pre_obs_mask.sum())

    if USE_DELAYED_ENTRY:
        df.loc[pre_obs_mask, ENTRY_COL] = (
                obs_start_date - df.loc[pre_obs_mask, COL_POSTED]
        ).dt.days.astype(float)

        df.loc[df[ENTRY_COL] < 0, ENTRY_COL] = 0.0

    else:
        df = df[df[COL_POSTED] >= obs_start_date].copy()
        df[ENTRY_COL] = 0.0

    n_after_obs_start_filter = int(len(df))
    n_rows_with_positive_entry = int((df[ENTRY_COL] > 0).sum())

    if INCLUDE_TIME_FE:
        periods = df[COL_POSTED].dt.to_period(TIME_FE_FREQ)
        df[VAR_TIME_FE] = periods.astype(str)
        df.loc[df[COL_POSTED].isna(), VAR_TIME_FE] = "__NA__"

    model_df = df.copy()
    model_df = model_df[model_df[DURATION_COL].notna()]
    model_df = model_df[model_df[DURATION_COL] > 0]

    if USE_DELAYED_ENTRY:
        model_df = model_df[model_df[ENTRY_COL].notna()]
        model_df = model_df[model_df[ENTRY_COL] >= 0]
        model_df = model_df[model_df[DURATION_COL] > model_df[ENTRY_COL]]

    keep_cols = [DURATION_COL, EVENT_COL]

    if USE_DELAYED_ENTRY:
        keep_cols.append(ENTRY_COL)

    if INCLUDE_DEV_HUBER:
        keep_cols.append(VAR_DEV_HUBER)

    if INCLUDE_LOG_AREA:
        keep_cols.append(COL_AREA)

    if INCLUDE_LOG_PRICE:
        keep_cols.append(COL_PRICE)

    if INCLUDE_OKS:
        keep_cols.append(COL_OKS)

    if INCLUDE_VRI_FE:
        keep_cols.append(COL_VRI)

    if INCLUDE_REGION_FE:
        keep_cols.append(COL_REGION)

    if INCLUDE_LOCALITY_TYPE_FE:
        keep_cols.append(COL_LOC_TYPE_ID)

    if INCLUDE_LOCALITY_ID_FE:
        keep_cols.append(COL_LOC_ID)

    if INCLUDE_TIME_FE:
        keep_cols.append(VAR_TIME_FE)

    model_df = model_df[keep_cols].copy()

    numeric_required = []

    if INCLUDE_DEV_HUBER:
        numeric_required.append(VAR_DEV_HUBER)

    if INCLUDE_LOG_AREA:
        numeric_required.append(COL_AREA)

    if INCLUDE_LOG_PRICE:
        numeric_required.append(COL_PRICE)

    if INCLUDE_OKS:
        numeric_required.append(COL_OKS)

    for c in numeric_required:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

    model_df = model_df.dropna(subset=numeric_required)

    devhuber_q_low = np.nan
    devhuber_q_high = np.nan

    if INCLUDE_DEV_HUBER:
        devhuber_q_low = float(model_df[VAR_DEV_HUBER].quantile(WINSOR_Q_LOW))
        devhuber_q_high = float(model_df[VAR_DEV_HUBER].quantile(WINSOR_Q_HIGH))

        model_df[VAR_DEV_HUBER] = model_df[VAR_DEV_HUBER].clip(
            lower=devhuber_q_low,
            upper=devhuber_q_high
        )

        model_df[VAR_DEV_HUBER] = model_df[VAR_DEV_HUBER].clip(
            lower=HARD_CLIP_LOW,
            upper=HARD_CLIP_HIGH
        )

    if INCLUDE_LOG_AREA:
        eps = 1e-9
        model_df[VAR_LOG_AREA] = np.log(np.clip(model_df[COL_AREA].values, eps, None))
        model_df = model_df.drop(columns=[COL_AREA])

    if INCLUDE_LOG_PRICE:
        eps = 1e-9
        model_df[VAR_LOG_PRICE] = np.log(np.clip(model_df[COL_PRICE].values, eps, None))
        model_df = model_df.drop(columns=[COL_PRICE])

    var_to_block = {}
    var_to_base = {}

    block_cols = {
        "Time_FE": [],
        "Region_FE": [],
        "Locality_Type_FE": [],
        "Locality_ID_FE": [],
        "DevHuber": [],
        "log_Цена_чист": [],
        "Есть_объект": [],
        "ВРИ_группа": [],
        "log_Площадь_сот": [],
    }

    base_info = {}

    if INCLUDE_DEV_HUBER:
        var_to_block[VAR_DEV_HUBER] = "DevHuber"
        var_to_base[VAR_DEV_HUBER] = ""
        block_cols["DevHuber"].append(VAR_DEV_HUBER)

    if INCLUDE_LOG_PRICE:
        var_to_block[VAR_LOG_PRICE] = "log_Цена_чист"
        var_to_base[VAR_LOG_PRICE] = ""
        block_cols["log_Цена_чист"].append(VAR_LOG_PRICE)

    if INCLUDE_LOG_AREA:
        var_to_block[VAR_LOG_AREA] = "log_Площадь_сот"
        var_to_base[VAR_LOG_AREA] = ""
        block_cols["log_Площадь_сот"].append(VAR_LOG_AREA)

    if INCLUDE_OKS:
        var_to_block[COL_OKS] = "Есть_объект"
        var_to_base[COL_OKS] = "0"
        block_cols["Есть_объект"].append(COL_OKS)

    if INCLUDE_VRI_FE:
        model_df[COL_VRI] = to_category_str(model_df[COL_VRI], force_integer=False)
        model_df[COL_VRI] = collapse_rare_categories(
            model_df[COL_VRI],
            min_count=MIN_COUNT_PER_CATEGORY,
            rare_label="__RARE__",
        )

        model_df, dummy_cols = make_dummies_with_base(
            df=model_df,
            col=COL_VRI,
            block_name="ВРИ_группа",
            base_value=None,
            strict_base=False,
            var_to_block=var_to_block,
            var_to_base=var_to_base,
            base_info=base_info,
        )

        block_cols["ВРИ_группа"].extend(dummy_cols)

    if INCLUDE_REGION_FE:
        model_df[COL_REGION] = to_category_str(model_df[COL_REGION], force_integer=True)
        model_df[COL_REGION] = collapse_rare_categories(
            model_df[COL_REGION],
            min_count=MIN_COUNT_PER_CATEGORY,
            rare_label="__RARE__",
            protected_values=[BASE_REGION],
        )

        model_df, dummy_cols = make_dummies_with_base(
            df=model_df,
            col=COL_REGION,
            block_name="Region_FE",
            base_value=BASE_REGION,
            strict_base=True,
            var_to_block=var_to_block,
            var_to_base=var_to_base,
            base_info=base_info,
        )

        block_cols["Region_FE"].extend(dummy_cols)

    if INCLUDE_LOCALITY_TYPE_FE:
        model_df[COL_LOC_TYPE_ID] = to_category_str(model_df[COL_LOC_TYPE_ID], force_integer=True)
        model_df[COL_LOC_TYPE_ID] = collapse_rare_categories(
            model_df[COL_LOC_TYPE_ID],
            min_count=MIN_COUNT_PER_CATEGORY,
            rare_label="__RARE__",
            protected_values=[BASE_LOCALITY_TYPE],
        )

        model_df, dummy_cols = make_dummies_with_base(
            df=model_df,
            col=COL_LOC_TYPE_ID,
            block_name="Locality_Type_FE",
            base_value=BASE_LOCALITY_TYPE,
            strict_base=True,
            var_to_block=var_to_block,
            var_to_base=var_to_base,
            base_info=base_info,
        )

        block_cols["Locality_Type_FE"].extend(dummy_cols)

    if INCLUDE_LOCALITY_ID_FE:
        model_df[COL_LOC_ID] = to_category_str(model_df[COL_LOC_ID], force_integer=True)
        model_df[COL_LOC_ID] = collapse_rare_categories(
            model_df[COL_LOC_ID],
            min_count=MIN_COUNT_PER_CATEGORY,
            rare_label="__RARE__",
        )

        model_df[COL_LOC_ID] = cap_category_levels(
            model_df[COL_LOC_ID],
            max_levels=MAX_LEVELS_LOC_ID,
            protected_values=[],
            other_label="__OTHER_CAP__",
        )

        model_df, dummy_cols = make_dummies_with_base(
            df=model_df,
            col=COL_LOC_ID,
            block_name="Locality_ID_FE",
            base_value=None,
            strict_base=False,
            var_to_block=var_to_block,
            var_to_base=var_to_base,
            base_info=base_info,
        )

        block_cols["Locality_ID_FE"].extend(dummy_cols)

    if INCLUDE_TIME_FE:
        model_df[VAR_TIME_FE] = model_df[VAR_TIME_FE].astype(str)
        model_df[VAR_TIME_FE] = model_df[VAR_TIME_FE].replace({
            "NaT": "__NA__",
            "nan": "__NA__",
            "None": "__NA__",
        })

        model_df[VAR_TIME_FE] = collapse_rare_categories(
            model_df[VAR_TIME_FE],
            min_count=MIN_COUNT_PER_CATEGORY,
            rare_label="__RARE__",
            protected_values=[BASE_TIME_FE_BY_FREQ[TIME_FE_FREQ]],
        )

        model_df, dummy_cols = make_dummies_with_base(
            df=model_df,
            col=VAR_TIME_FE,
            block_name="Time_FE",
            base_value=BASE_TIME_FE_BY_FREQ[TIME_FE_FREQ],
            strict_base=True,
            var_to_block=var_to_block,
            var_to_base=var_to_base,
            base_info=base_info,
        )

        block_cols["Time_FE"].extend(dummy_cols)

    block_cols_before_low_var = {
        block: cols.copy()
        for block, cols in block_cols.items()
    }

    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

    exclude_low_var = [DURATION_COL, EVENT_COL]

    if USE_DELAYED_ENTRY:
        exclude_low_var.append(ENTRY_COL)

    model_df, dropped_low_var = drop_low_variance_columns(
        model_df,
        exclude=exclude_low_var,
        thr=LOW_VAR_THRESHOLD
    )

    for block in block_cols:
        block_cols[block] = [c for c in block_cols[block] if c in model_df.columns]

    var_to_block = {k: v for k, v in var_to_block.items() if k in model_df.columns}
    var_to_base = {k: v for k, v in var_to_base.items() if k in model_df.columns}

    preprocess_meta = {
        "cutoff_date": str(cutoff_date.date()),
        "obs_start_date": str(obs_start_date.date()),
        "use_delayed_entry": bool(USE_DELAYED_ENTRY),
        "n_pre_obs_start_rows": int(n_pre_obs_start_rows),
        "n_after_obs_start_filter": int(n_after_obs_start_filter),
        "n_rows_with_positive_entry": int(n_rows_with_positive_entry),
        "devhuber_q_low": devhuber_q_low,
        "devhuber_q_high": devhuber_q_high,
        "dropped_low_var_count": int(len(dropped_low_var)),
        "dropped_low_var_cols": dropped_low_var,
    }

    return (
        model_df,
        raw_n,
        preprocess_meta,
        var_to_block,
        var_to_base,
        block_cols,
        block_cols_before_low_var,
        base_info,
        dropped_low_var,
    )


# =========================
# ОСНОВНОЙ КОД
# =========================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    (
        model_df,
        raw_n,
        preprocess_meta,
        var_to_block,
        var_to_base,
        block_cols,
        block_cols_before_low_var,
        base_info,
        dropped_low_var,
    ) = prepare_model_df(INPUT_XLSX)

    cph = fit_cox(model_df)

    summary_full = build_summary_full(
        cph=cph,
        var_to_block=var_to_block,
        var_to_base=var_to_base,
    )

    diagnostics = build_diagnostics(
        cph=cph,
        model_df=model_df,
        raw_n=raw_n,
        preprocess_meta=preprocess_meta,
    )

    ph_assumption, ph_assumption_blocks = build_ph_assumption(
        cph=cph,
        model_df=model_df,
        var_to_block=var_to_block,
    )

    model_blocks = build_model_blocks(
        block_cols=block_cols,
        block_cols_before_low_var=block_cols_before_low_var,
        base_info=base_info,
        dropped_low_var=dropped_low_var,
    )

    if RUN_BLOCK_DROP_TESTS:
        block_drop_tests = build_block_drop_tests(
            full_cph=cph,
            model_df=model_df,
            block_cols=block_cols,
        )
    else:
        block_drop_tests = pd.DataFrame([{
            "block_order": np.nan,
            "block": "",
            "status": "skipped",
            "n_vars_dropped": np.nan,
            "ll_full": np.nan,
            "ll_reduced": np.nan,
            "LR": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "note": "RUN_BLOCK_DROP_TESTS=False",
        }])

    with pd.ExcelWriter(OUTPUT_REPORT_XLSX, engine="openpyxl") as writer:
        summary_full.to_excel(writer, sheet_name="summary_full", index=False)
        diagnostics.to_excel(writer, sheet_name="diagnostics", index=False)
        ph_assumption.to_excel(writer, sheet_name="PH_assumption", index=False)
        ph_assumption_blocks.to_excel(writer, sheet_name="PH_assumption_blocks", index=False)
        model_blocks.to_excel(writer, sheet_name="model_blocks", index=False)
        block_drop_tests.to_excel(writer, sheet_name="block_drop_tests", index=False)

    print("Rows used:", len(model_df))
    print("Events:", int(model_df[EVENT_COL].sum()))
    print("Censored:", int(len(model_df) - model_df[EVENT_COL].sum()))
    print("Use delayed entry:", USE_DELAYED_ENTRY)

    if USE_DELAYED_ENTRY:
        print("Rows with positive entry_i:", int((model_df[ENTRY_COL] > 0).sum()))

    print("C-index:", cph.concordance_index_)
    print("Saved:", OUTPUT_REPORT_XLSX)


if __name__ == "__main__":
    main()
