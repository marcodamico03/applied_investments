from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Paths (the folder must be called "data" in lowercase, see CHANGE 9)
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

US_EXCHANGES = ["N", "A", "Q"]  # NYSE, AMEX, NASDAQ


# CHANGE 5: a filter is no longer skipped silently when a column is missing
def column_is_in(df: pd.DataFrame, col: str, allowed: list, label: str) -> pd.Series:
    """Return True/False per row: is df[col] one of the allowed values?
    If the column does not exist, print a warning and return all True."""
    if col not in df.columns:
        print(f"  !! WARNING: column '{col}' not found, filter '{label}' NOT applied")
        return pd.Series(True, index=df.index)
    return df[col].isin(allowed)


def filter_equal(df: pd.DataFrame, col: str, allowed, label: str) -> pd.DataFrame:
    """Keep only the rows where df[col] is one of the allowed values."""
    if not isinstance(allowed, (list, tuple, set)):
        allowed = [allowed]
    return df[column_is_in(df, col, list(allowed), label)]


# ==============================================================================
# 1. CRSP CLEANING
# ==============================================================================
def clean_crsp(
    input_file: Path = RAW_DIR / "crsp_monthly.csv",
    output_file: Path = INTERIM_DIR / "crsp_clean.csv",
) -> pd.DataFrame:
    print(f"\n[1/3] Loading CRSP Monthly Data from: {input_file}")
    if not input_file.exists():
        raise FileNotFoundError(f"Missing file: {input_file}")

    crsp = pd.read_csv(input_file, low_memory=False)
    print(f"  -> Raw records: {len(crsp):,}")

    # A. Date parsing and key integrity
    crsp["MthCalDt"] = pd.to_datetime(crsp["MthCalDt"], errors="coerce")
    crsp = crsp.dropna(subset=["MthCalDt", "PERMNO"])
    crsp["PERMNO"] = crsp["PERMNO"].astype(int)

    # B. Resolve duplicate PERMNO-months: keep the record with the dividend
    if "DisDivAmt" in crsp.columns:
        crsp = crsp.sort_values(by=["PERMNO", "MthCalDt", "DisDivAmt"], na_position="first")
    else:
        crsp = crsp.sort_values(by=["PERMNO", "MthCalDt"])
    crsp = crsp.drop_duplicates(subset=["PERMNO", "MthCalDt"], keep="last")
    print(f"  -> After deduplication on (PERMNO, MthCalDt): {len(crsp):,} rows")

    # C. Which stock-months are common US equity listed on NYSE/AMEX/NASDAQ?
    # We build a True/False column "eligible" instead of deleting rows directly.
    eligible = pd.Series(True, index=crsp.index)
    eligible &= column_is_in(crsp, "SecurityType", ["EQTY"], "equity only")
    eligible &= column_is_in(crsp, "SecuritySubType", ["COM"], "common stock only")
    eligible &= column_is_in(crsp, "ShareType", ["NS"], "no special share type")
    eligible &= column_is_in(crsp, "USIncFlg", ["Y"], "US incorporated")
    eligible &= column_is_in(crsp, "IssuerType", ["ACOR", "CORP"], "corporate issuer")  # CHANGE 10
    eligible &= column_is_in(crsp, "PrimaryExch", US_EXCHANGES, "NYSE/AMEX/NASDAQ")
    # Exclude mutual funds, closed-end funds (SIC 6720-6726) and REITs (SIC 6798)
    if "SICCD" in crsp.columns:
        eligible &= ~crsp["SICCD"].isin([6720, 6722, 6723, 6724, 6725, 6726, 6798])
    else:
        print("  !! WARNING: column 'SICCD' not found, fund/REIT filter NOT applied")

    # D. CHANGE 3: keep the delisting month.
    # In the month a stock delists, CRSP changes its descriptive columns
    # (exchange becomes "X", share type becomes blank, etc.), so the filters
    # above would delete that month. But in the new CRSP format that month's
    # MthRet contains the delisting return, which we must keep (otherwise we
    # get survivorship bias). Rule: keep a month if the stock is eligible
    # this month OR was eligible last month.
    eligible_last_month = eligible.groupby(crsp["PERMNO"]).shift(1, fill_value=False)
    crsp = crsp[eligible | eligible_last_month]
    print(f"  -> After common equity and exchange filters (delisting months kept): {len(crsp):,} rows")

    # E. Prices, returns, shares
    # CHANGE 1: the "price > 1000 -> divide by 1,000,000" rule was removed.
    # The raw CSV prices are correct; that rule only broke expensive stocks.
    # A negative price in CRSP means "no trade, bid/ask average", so we take abs().
    crsp["MthPrc"] = pd.to_numeric(crsp["MthPrc"], errors="coerce").abs()
    crsp["MthRet"] = pd.to_numeric(crsp["MthRet"], errors="coerce")
    crsp["ShrOut"] = pd.to_numeric(crsp["ShrOut"], errors="coerce")

    # Market capitalization in thousands of dollars:
    # price (dollars) * shares outstanding (thousands)
    crsp["mkt_cap"] = crsp["MthPrc"] * crsp["ShrOut"]

    # F. Fama-French annual alignment (July t to June t+1 gets formation_year t)
    crsp["formation_year"] = np.where(
        crsp["MthCalDt"].dt.month >= 7,
        crsp["MthCalDt"].dt.year,
        crsp["MthCalDt"].dt.year - 1,
    )

    keep_cols = [
        "PERMNO", "PERMCO", "Ticker", "PrimaryExch", "SICCD", "MthCalDt",
        "formation_year", "MthPrc", "MthRet", "MthRetx", "MthVol", "ShrOut",
        "mkt_cap", "DisDivAmt", "MthDelFlg", "DelActionType",
    ]
    crsp_clean = crsp[[c for c in keep_cols if c in crsp.columns]].copy()

    crsp_clean.to_csv(output_file, index=False)
    print(f"  -> Saved clean CRSP to: {output_file} {crsp_clean.shape}")
    return crsp_clean


# ==============================================================================
# 2. COMPUSTAT CLEANING
# ==============================================================================
def clean_compustat(
    input_file: Path = RAW_DIR / "ccm_funda.csv",
    output_file: Path = INTERIM_DIR / "compustat_clean.csv",
) -> pd.DataFrame:
    print(f"\n[2/3] Loading Compustat Fundamentals Annual from: {input_file}")
    if not input_file.exists():
        raise FileNotFoundError(f"Missing file: {input_file}")

    # Load only the columns we need out of the 988 available
    header_cols = pd.read_csv(input_file, nrows=1).columns.tolist()
    target_cols = [
        "GVKEY", "LINKPRIM", "LIID", "LINKTYPE", "LPERMNO", "LPERMCO", "LINKDT",
        "LINKENDDT", "datadate", "fyear", "indfmt", "consol", "popsrc", "datafmt",
        "tic", "cusip", "conm", "at", "sale", "cogs", "xsga", "xint", "ni", "ib",
        "ceq", "seq", "lt", "pstk", "pstkrv", "pstkl", "txditc", "oancf", "capx",
        "dltt", "dlc", "csho", "prcc_f", "xrd", "xad", "act", "lct", "che",
        "rect", "invt", "ap", "mib", "dv", "sich", "sic",
    ]
    missing = [c for c in target_cols if c not in header_cols]
    if missing:
        print(f"  !! WARNING: these columns are not in the file: {missing}")
    load_cols = [c for c in target_cols if c in header_cols]

    comp = pd.read_csv(input_file, usecols=load_cols, low_memory=False)
    print(f"  -> Loaded {len(load_cols)} columns across {len(comp):,} raw rows")

    # A. Link filters and standard Compustat formats
    comp = filter_equal(comp, "LINKPRIM", ["P", "C"], "primary links")
    comp = filter_equal(comp, "LINKTYPE", ["LU", "LC"], "reliable link types")
    comp = filter_equal(comp, "indfmt", "INDL", "industrial format")
    comp = filter_equal(comp, "datafmt", "STD", "standardized data")
    comp = filter_equal(comp, "popsrc", "D", "domestic")
    comp = filter_equal(comp, "consol", "C", "consolidated")

    # Drop records with missing or non-positive total assets
    comp["at"] = pd.to_numeric(comp["at"], errors="coerce")
    comp = comp[comp["at"] > 0]
    print(f"  -> After link and positive-asset filters: {len(comp):,} rows")

    # B. Link must be valid on the balance sheet date
    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
    comp["LINKDT"] = pd.to_datetime(comp["LINKDT"], errors="coerce")
    comp["LINKENDDT"] = pd.to_datetime(
        comp["LINKENDDT"].replace("E", "2099-12-31"), errors="coerce"
    ).fillna(pd.Timestamp("2099-12-31"))  # "E" or empty = link still active

    comp = comp[(comp["datadate"] >= comp["LINKDT"]) & (comp["datadate"] <= comp["LINKENDDT"])]

    comp["LPERMNO"] = pd.to_numeric(comp["LPERMNO"], errors="coerce")
    comp = comp.dropna(subset=["LPERMNO", "datadate"])
    comp["LPERMNO"] = comp["LPERMNO"].astype(int)

    # C. Deduplicate on (GVKEY, datadate).
    # CHANGE 7 (part 1): sort by LINKPRIM so that when there are two links,
    # the primary one ("P") is kept instead of an arbitrary one.
    comp = comp.sort_values(by=["GVKEY", "datadate", "LINKPRIM"])
    comp = comp.drop_duplicates(subset=["GVKEY", "datadate"], keep="last")

    # D. Convert accounting fields to numbers
    numeric_vars = [
        "sale", "cogs", "xsga", "xint", "ni", "ib", "ceq", "seq", "lt", "pstk",
        "pstkrv", "pstkl", "txditc", "oancf", "capx", "dltt", "dlc", "csho",
        "prcc_f", "xrd", "xad", "act", "lct", "che", "rect", "invt", "ap", "mib", "dv",
    ]
    for var in numeric_vars:
        if var in comp.columns:
            comp[var] = pd.to_numeric(comp[var], errors="coerce")
        else:
            comp[var] = np.nan

    # CHANGE 4: NO zero-filling of the original columns. Missing stays missing
    # (slide 10 of the course). Zeros are used only inside the book equity
    # formula below, which is the standard Fama-French convention.

    # E. Fama-French book equity (same recipe as Tidy Finance)
    # Preferred stock: redemption -> liquidating -> par value -> 0
    pref_stock = (
        comp["pstkrv"].combine_first(comp["pstkl"]).combine_first(comp["pstk"]).fillna(0.0)
    )
    # Stockholders' equity: seq -> ceq + pstk -> at - lt
    stockholders_equity = (
        comp["seq"]
        .combine_first(comp["ceq"] + comp["pstk"])
        .combine_first(comp["at"] - comp["lt"])
    )
    # BE = SE + deferred taxes - preferred stock (missing deferred taxes count as 0)
    comp["be"] = stockholders_equity + comp["txditc"].fillna(0.0) - pref_stock
    # CHANGE 6: negative or zero book equity is set to missing
    # (a negative book-to-market ratio has no economic meaning)
    comp.loc[comp["be"] <= 0, "be"] = np.nan

    # F. CHANGE 8: asset growth, computed BEFORE the merge.
    # Growth = at(this year) / at(previous year) - 1, only if the two
    # fiscal years are consecutive (no gap in the data).
    comp = comp.sort_values(by=["GVKEY", "fyear", "datadate"])
    comp = comp.drop_duplicates(subset=["GVKEY", "fyear"], keep="last")
    prev_at = comp.groupby("GVKEY")["at"].shift(1)
    prev_fyear = comp.groupby("GVKEY")["fyear"].shift(1)
    comp["at_lag"] = np.where(comp["fyear"] - prev_fyear == 1, prev_at, np.nan)
    comp["asset_growth"] = comp["at"] / comp["at_lag"] - 1

    # G. Formation year: the report for fiscal year ending in year t-1 is used
    # from July of year t
    comp["formation_year"] = comp["datadate"].dt.year + 1

    # CHANGE 7 (part 2): one accounting record per stock and formation year.
    # A firm that changes its fiscal year end can have two reports in the same
    # calendar year: we keep the most recent one.
    comp = comp.sort_values(by=["LPERMNO", "formation_year", "datadate"])
    comp = comp.drop_duplicates(subset=["LPERMNO", "formation_year"], keep="last")

    out_cols = [
        "GVKEY", "LPERMNO", "conm", "tic", "datadate", "fyear", "formation_year",
        "at", "at_lag", "asset_growth", "sale", "cogs", "xsga", "xint", "ni", "ib",
        "ceq", "be", "txditc", "oancf", "capx", "dltt", "dlc", "csho", "prcc_f",
        "xrd", "xad", "act", "lct", "che", "rect", "invt", "ap",
    ]
    comp_clean = comp[[c for c in out_cols if c in comp.columns]].copy()

    comp_clean.to_csv(output_file, index=False)
    print(f"  -> Saved clean Compustat to: {output_file} {comp_clean.shape}")
    return comp_clean


# ==============================================================================
# 3. MERGING INTO PROCESSED MASTER PANEL
# ==============================================================================
def merge_and_align_panel(
    crsp_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    output_file: Path = PROCESSED_DIR / "monthly_panel_clean.csv",
) -> pd.DataFrame:
    print(f"\n[3/3] Merging CRSP and Compustat onto: {output_file}")

    # Make sure dates are real dates (needed if the data was reloaded from CSV)
    crsp_df = crsp_df.copy()
    comp_df = comp_df.copy()
    crsp_df["MthCalDt"] = pd.to_datetime(crsp_df["MthCalDt"])
    comp_df["datadate"] = pd.to_datetime(comp_df["datadate"])

    # Left merge: every CRSP stock-month is kept, with or without accounting data
    panel = pd.merge(
        crsp_df,
        comp_df,
        left_on=["PERMNO", "formation_year"],
        right_on=["LPERMNO", "formation_year"],
        how="left",
    )

    # Months between the balance sheet date and the return month (diagnostic only)
    panel["lag_months"] = (
        (panel["MthCalDt"].dt.year - panel["datadate"].dt.year) * 12
        + (panel["MthCalDt"].dt.month - panel["datadate"].dt.month)
    )

    # CHANGE 2: the "drop rows with lag > 18 months" filter was removed.
    # The formation_year timing already guarantees a lag between 6 and 29
    # months, and dropping rows was deleting stock RETURNS, not just old data.
    # We only check that no accounting data is used too early.
    too_early = (panel["lag_months"] < 6).sum()
    print(f"  -> Rows using accounting data less than 6 months old (should be 0): {too_early}")

    # Safety check: one row per stock-month (guaranteed by CHANGE 7)
    dup_count = panel.duplicated(subset=["PERMNO", "MthCalDt"]).sum()
    print(f"  -> Duplicate (PERMNO, MthCalDt) rows (should be 0): {dup_count}")

    panel = panel.sort_values(by=["PERMNO", "MthCalDt"]).reset_index(drop=True)

    share_matched = panel["datadate"].notna().mean()
    print(f"  -> Share of stock-months with accounting data: {share_matched:.1%}")

    panel.to_csv(output_file, index=False)
    print(f"  -> Successfully generated master panel: {output_file}")
    print(f"  -> Final shape: {panel.shape[0]:,} rows, {panel.shape[1]} columns")
    return panel


if __name__ == "__main__":
    crsp_clean = clean_crsp()
    comp_clean = clean_compustat()
    merge_and_align_panel(crsp_clean, comp_clean)
