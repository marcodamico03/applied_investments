import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Define paths matching the project structure
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


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
    initial_rows = len(crsp)
    print(f"  -> Raw records: {initial_rows:,}")

    # A. Filter for common ordinary equities only (remove ETFs, ADRs, CEFs, REITs)
    if "SecurityType" in crsp.columns:
        crsp = crsp[crsp["SecurityType"] == "EQTY"]
    if "SecuritySubType" in crsp.columns:
        crsp = crsp[crsp["SecuritySubType"] == "COM"]
    if "ShareType" in crsp.columns:
        crsp = crsp[crsp["ShareType"] == "NS"]
    if "USIncFlg" in crsp.columns:
        crsp = crsp[crsp["USIncFlg"] == "Y"]
    if "PrimaryExch" in crsp.columns:
        crsp = crsp[crsp["PrimaryExch"].isin(["N", "A", "Q"])]  # NYSE, AMEX, NASDAQ

    # Exclude mutual funds, closed-end funds (SIC 6720-6726) and REITs (SIC 6798)
    if "SICCD" in crsp.columns:
        crsp = crsp[~crsp["SICCD"].isin([6720, 6722, 6723, 6724, 6725, 6726, 6798])]

    print(f"  -> After common equity & exchange filters: {len(crsp):,} rows")

    # B. Date parsing and key integrity
    crsp["MthCalDt"] = pd.to_datetime(crsp["MthCalDt"], errors="coerce")
    crsp = crsp.dropna(subset=["MthCalDt", "PERMNO"])
    crsp["PERMNO"] = crsp["PERMNO"].astype(int)

    # C. Resolve duplicate PERMNO-months: sort by DisDivAmt so dividend records are kept
    if "DisDivAmt" in crsp.columns:
        crsp = crsp.sort_values(
            by=["PERMNO", "MthCalDt", "DisDivAmt"], na_position="first"
        )
    else:
        crsp = crsp.sort_values(by=["PERMNO", "MthCalDt"])
    crsp = crsp.drop_duplicates(subset=["PERMNO", "MthCalDt"], keep="last")
    print(f"  -> After deduplication on (PERMNO, MthCalDt): {len(crsp):,} rows")

    # D. Price and Return standardizations
    crsp["MthPrc"] = pd.to_numeric(crsp["MthPrc"], errors="coerce").abs()
    # Normalize scaled WRDS prices (integer prices > 1,000 divided by 1e6)
    crsp["MthPrc"] = np.where(
        crsp["MthPrc"] > 1000.0, crsp["MthPrc"] / 1_000_000.0, crsp["MthPrc"]
    )

    crsp["MthRet"] = pd.to_numeric(crsp["MthRet"], errors="coerce")
    crsp["ShrOut"] = pd.to_numeric(crsp["ShrOut"], errors="coerce")

    # Market capitalization in thousands (\() = Price (\)) * ShrOut (thousands)
    crsp["mkt_cap"] = crsp["MthPrc"] * crsp["ShrOut"]

    # E. Fama-French annual alignment variable (July t to June t+1)
    # Months 1-6 align with accounting year t-2 (formation_year t-1)
    # Months 7-12 align with accounting year t-1 (formation_year t)
    crsp["formation_year"] = np.where(
        crsp["MthCalDt"].dt.month >= 7,
        crsp["MthCalDt"].dt.year,
        crsp["MthCalDt"].dt.year - 1,
    )

    # Select clean column subset
    keep_cols = [
        "PERMNO",
        "PERMCO",
        "Ticker",
        "PrimaryExch",
        "SICCD",
        "MthCalDt",
        "formation_year",
        "MthPrc",
        "MthRet",
        "MthRetx",
        "MthVol",
        "ShrOut",
        "mkt_cap",
        "DisDivAmt",
        "MthDelFlg",
        "DelActionType",
    ]
    crsp_clean = crsp[[c for c in keep_cols if c in crsp.columns]].copy()

    # Save to data/interim/
    crsp_clean.to_csv(output_file, index=False)
    print(f"  -> Saved clean CRSP to: {output_file} ({crsp_clean.shape})")
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

    # Inspect headers to load only essential columns out of the 988 available
    header_cols = pd.read_csv(input_file, nrows=1).columns.tolist()
    target_cols = [
        "GVKEY",
        "LINKPRIM",
        "LIID",
        "LINKTYPE",
        "LPERMNO",
        "LPERMCO",
        "LINKDT",
        "LINKENDDT",
        "datadate",
        "fyear",
        "indfmt",
        "consol",
        "popsrc",
        "datafmt",
        "tic",
        "cusip",
        "conm",
        "at",
        "sale",
        "cogs",
        "xsga",
        "xint",
        "ni",
        "ib",
        "ceq",
        "seq",
        "lt",
        "pstk",
        "pstkrv",
        "pstkl",
        "txditc",
        "oancf",
        "capx",
        "dltt",
        "dlc",
        "csho",
        "prcc_f",
        "xrd",
        "xad",
        "act",
        "lct",
        "che",
        "rect",
        "invt",
        "ap",
        "mib",
        "dv",
        "sich",
        "sic",
    ]
    load_cols = [c for c in target_cols if c in header_cols]

    comp = pd.read_csv(input_file, usecols=load_cols, low_memory=False)
    print(f"  -> Loaded {len(load_cols)} columns across {len(comp):,} raw rows")

    # A. Link filters & format standards
    if "LINKPRIM" in comp.columns:
        comp = comp[comp["LINKPRIM"].isin(["P", "C"])]
    if "LINKTYPE" in comp.columns:
        comp = comp[comp["LINKTYPE"].isin(["LU", "LC"])]
    for col, val in [
        ("indfmt", "INDL"),
        ("datafmt", "STD"),
        ("popsrc", "D"),
        ("consol", "C"),
    ]:
        if col in comp.columns:
            comp = comp[comp[col] == val]

    # Drop records with missing or non-positive assets
    comp["at"] = pd.to_numeric(comp["at"], errors="coerce")
    comp = comp[comp["at"] > 0]
    print(f"  -> After link & active asset filters: {len(comp):,} rows")

    # B. Link date validity check
    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
    comp["LINKDT"] = pd.to_datetime(comp["LINKDT"], errors="coerce")
    comp["LINKENDDT"] = pd.to_datetime(
        comp["LINKENDDT"].replace("E", "2099-12-31"), errors="coerce"
    ).fillna(pd.Timestamp("2099-12-31"))

    comp = comp[
        (comp["datadate"] >= comp["LINKDT"])
        & (comp["datadate"] <= comp["LINKENDDT"])
    ]

    comp["LPERMNO"] = pd.to_numeric(comp["LPERMNO"], errors="coerce")
    comp = comp.dropna(subset=["LPERMNO", "datadate"])
    comp["LPERMNO"] = comp["LPERMNO"].astype(int)

    # Deduplicate on (GVKEY, datadate)
    comp = comp.sort_values(by=["GVKEY", "datadate"]).drop_duplicates(
        subset=["GVKEY", "datadate"], keep="last"
    )

    # C. Convert accounting fields to numeric
    numeric_vars = [
        "sale",
        "cogs",
        "xsga",
        "xint",
        "ni",
        "ib",
        "ceq",
        "seq",
        "lt",
        "pstk",
        "pstkrv",
        "pstkl",
        "txditc",
        "oancf",
        "capx",
        "dltt",
        "dlc",
        "csho",
        "prcc_f",
        "xrd",
        "xad",
        "act",
        "lct",
        "che",
        "rect",
        "invt",
        "ap",
        "mib",
        "dv",
    ]
    for var in numeric_vars:
        if var in comp.columns:
            comp[var] = pd.to_numeric(comp[var], errors="coerce")
        else:
            comp[var] = np.nan

    # D. Missing = 0 Imputations (Tier 1)
    zero_fill_vars = ["xrd", "xad", "txditc", "pstk", "mib", "xint", "dv"]
    for var in zero_fill_vars:
        comp[var] = comp[var].fillna(0.0)

    # E. Fama-French Book Equity (BE) calculation (Tier 2)
    # Preferred stock priority: Redemption value -> Liquidating value -> Par value -> 0
    pref_stock = (
        comp["pstkrv"]
        .combine_first(comp["pstkl"])
        .combine_first(comp["pstk"])
        .fillna(0.0)
    )

    # Stockholders' equity priority: seq -> ceq + pstk -> at - lt
    stockholders_equity = (
        comp["seq"]
        .combine_first(comp["ceq"] + comp["pstk"])
        .combine_first(comp["at"] - comp["lt"])
    )

    # Book Equity = Stockholders' Equity + Deferred Taxes (txditc) - Preferred Stock
    comp["be"] = stockholders_equity + comp["txditc"] - pref_stock

    # Formation year for Fama-French matching: annual report for year t-1 matches year t
    comp["formation_year"] = comp["datadate"].dt.year + 1

    # Keep relevant accounting items
    out_cols = [
        "GVKEY",
        "LPERMNO",
        "conm",
        "tic",
        "datadate",
        "fyear",
        "formation_year",
        "at",
        "sale",
        "cogs",
        "xsga",
        "xint",
        "ni",
        "ib",
        "ceq",
        "be",
        "txditc",
        "oancf",
        "capx",
        "dltt",
        "dlc",
        "csho",
        "prcc_f",
        "xrd",
        "xad",
        "act",
        "lct",
        "che",
        "rect",
        "invt",
        "ap",
    ]
    comp_clean = comp[[c for c in out_cols if c in comp.columns]].copy()

    # Save to data/interim/
    comp_clean.to_csv(output_file, index=False)
    print(f"  -> Saved clean Compustat to: {output_file} ({comp_clean.shape})")
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

    # Merge on permanent security identifier and formation year
    panel = pd.merge(
        crsp_df,
        comp_df,
        left_on=["PERMNO", "formation_year"],
        right_on=["LPERMNO", "formation_year"],
        how="left",
    )

    # Validate timing lag: months elapsed between balance sheet datadate and stock month
    panel["lag_months"] = (
        (panel["MthCalDt"].dt.year - panel["datadate"].dt.year) * 12
        + (panel["MthCalDt"].dt.month - panel["datadate"].dt.month)
    )

    # Drop stale accounting data older than 18 months
    panel = panel[(panel["lag_months"].isna()) | (panel["lag_months"] <= 18)]

    # Deduplicate (PERMNO, MthCalDt): one CRSP row may match multiple Compustat
    # records when a firm has overlapping CCM link periods for different GVKEYs.
    # Keep the observation with the largest total assets (most complete match).
    if "at" in panel.columns:
        panel = panel.sort_values(
            by=["PERMNO", "MthCalDt", "at"], ascending=[True, True, False], na_position="last"
        )
    panel = panel.drop_duplicates(subset=["PERMNO", "MthCalDt"], keep="first")
    dup_count = panel.duplicated(subset=["PERMNO", "MthCalDt"]).sum()
    print(f"  -> Duplicates after post-merge deduplication: {dup_count}")

    # Sort deterministically
    panel = panel.sort_values(by=["PERMNO", "MthCalDt"]).reset_index(drop=True)

    panel.to_csv(output_file, index=False)
    print(f"  -> Successfully generated master panel: {output_file}")
    print(f"  -> Final shape: {panel.shape[0]:,} rows, {panel.shape[1]} columns")
    return panel


if __name__ == "__main__":
    crsp_clean = clean_crsp()
    comp_clean = clean_compustat()
    merge_and_align_panel(crsp_clean, comp_clean)