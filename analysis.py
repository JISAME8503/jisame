import pandas as pd
import numpy as np
import json
import os
import sqlite3
import streamlit as st
from scipy.stats import pearsonr
from config import MACRO_INDICATORS, SECTORS, ANALYSIS_CONFIG, DATA_DIR, CACHE_FILE, DB_FILE


def _load_csv(path: str) -> pd.DataFrame:
    """CSVを読み込んでDatetimeIndexに正規化する"""
    df = pd.read_csv(path, index_col=0)
    # yfinance由来のタプル文字列カラムを除去（正常なカラムが既に存在する）
    df = df[[c for c in df.columns if not str(c).startswith("(")]]
    # 日付文字列の先頭10文字（YYYY-MM-DD）だけ取り出してDatetimeIndexに変換
    df.index = pd.to_datetime(
        df.index.astype(str).str[:10], errors="coerce"
    )
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_macro() -> pd.DataFrame:
    path = f"{DATA_DIR}/macro.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = _load_csv(path)
    df = df.pct_change().dropna(how="all")
    return df


@st.cache_data(ttl=3600)
def load_sector(sector: str) -> pd.DataFrame:
    path = f"{DATA_DIR}/sector_{sector}.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = _load_csv(path)
    df = df.pct_change().dropna(how="all")
    return df


def lag_correlation(macro: pd.Series, stock: pd.Series, max_lag: int) -> dict:
    """ラグ相関を計算する。返り値: {lag: correlation}（lag=1〜max_lag、p>0.05は0扱い）"""
    results = {}
    aligned = pd.concat([macro, stock], axis=1).dropna()
    if len(aligned) < ANALYSIS_CONFIG["min_data_points"]:
        return results
    m = aligned.iloc[:, 0].values
    s = aligned.iloc[:, 1].values
    for lag in range(1, max_lag + 1):
        x, y = m[:-lag], s[lag:]
        corr, p_value = pearsonr(x, y)
        if np.isnan(corr):
            continue
        if p_value > 0.05:
            corr = 0.0
        results[lag] = round(corr, 4)
    return results


def best_lag(lag_corrs: dict) -> tuple[int, float]:
    """最も絶対値が大きいラグと相関係数を返す"""
    if not lag_corrs:
        return 0, 0.0
    best = max(lag_corrs.items(), key=lambda x: abs(x[1]))
    return best[0], best[1]


def moving_correlation(macro: pd.Series, stock: pd.Series, window: int = None) -> pd.Series:
    """移動相関を計算する"""
    if window is None:
        window = ANALYSIS_CONFIG["rolling_window"]
    aligned = pd.concat([macro, stock], axis=1).dropna()
    if len(aligned) < window:
        window = max(ANALYSIS_CONFIG["min_rolling_window"], len(aligned) // 2)
    return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])


def _period_lag_score(macro: pd.Series, stock: pd.Series, days: int, max_lag: int) -> float:
    """指定営業日数スライスでのlag絶対相関の最大値（p>0.05は0扱い）"""
    m = macro.iloc[-days:] if len(macro) > days else macro
    s = stock.iloc[-days:] if len(stock) > days else stock
    aligned = pd.concat([m, s], axis=1).dropna()
    min_pts = max(30, max_lag + 1)
    if len(aligned) < min_pts:
        return 0.0
    mv, sv = aligned.iloc[:, 0].values, aligned.iloc[:, 1].values
    scores = []
    for lag in range(1, max_lag + 1):
        x, y = mv[:-lag], sv[lag:]
        try:
            corr, p = pearsonr(x, y)
        except Exception:
            continue
        if np.isnan(corr) or p > 0.05:
            continue
        scores.append(abs(corr))
    return round(max(scores), 4) if scores else 0.0


def _classify_tag(s3m: float, s6m: float, s12m: float, threshold: float = 0.3) -> str:
    """3期間スコアから銘柄を3分類する（s3m/s12mで判定、s6mは互換のため残す）"""
    if s3m >= threshold and s12m >= threshold:
        return "構造的ラグ"
    if s3m >= threshold and s12m < threshold:
        return "新興ラグ"
    if s12m >= threshold and s3m < threshold:
        return "崩壊警戒"
    return ""


def _save_signals_db(rows: list[dict]) -> None:
    """全セクターの分析結果をSQLiteに保存する"""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            ticker            TEXT,
            sector            TEXT,
            macro_name        TEXT,
            best_lag          INTEGER,
            best_corr         REAL,
            score_3m          REAL,
            score_6m          REAL,
            score_12m         REAL,
            convergence_score REAL,
            tag               TEXT,
            updated_at        TEXT,
            PRIMARY KEY (ticker, sector, macro_name)
        )
    """)
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.executemany(
        """INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                r["ticker"], r["sector"], r["best_macro"],
                r["best_lag"], r["best_corr"],
                r.get("score_3m", 0.0), r.get("score_6m", 0.0), r.get("score_12m", 0.0),
                r.get("convergence_score", 0.0), r.get("tag", ""),
                ts,
            )
            for r in rows
        ],
    )
    conn.commit()
    conn.close()


def analyze_sector(sector: str) -> list[dict]:
    """セクター内の全銘柄を全マクロ指標と分析し、シグナルスコアでソートして返す"""
    macro_df = load_macro()
    stock_df = load_sector(sector)

    if macro_df.empty or stock_df.empty:
        return []

    results = []
    threshold = ANALYSIS_CONFIG["signal_threshold"]
    max_lag = ANALYSIS_CONFIG["max_lag"]

    for ticker in stock_df.columns:
        stock = stock_df[ticker].dropna()
        best_signals = []

        for macro_name, macro_ticker in MACRO_INDICATORS.items():
            if macro_ticker not in macro_df.columns:
                continue
            macro = macro_df[macro_ticker].dropna()
            lag_corrs = lag_correlation(macro, stock, max_lag)
            if not lag_corrs:
                continue
            lag, corr = best_lag(lag_corrs)
            if abs(corr) >= threshold:
                best_signals.append({
                    "macro_name": macro_name,
                    "macro_ticker": macro_ticker,
                    "lag": lag,
                    "correlation": corr,
                    "lag_corrs": lag_corrs,
                })

        if best_signals:
            best_signals.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            top = best_signals[0]
            top_macro = macro_df[top["macro_ticker"]].dropna()
            s3m  = _period_lag_score(top_macro, stock, 63,  max_lag)
            s6m  = _period_lag_score(top_macro, stock, 126, max_lag)
            s12m = _period_lag_score(top_macro, stock, 252, max_lag)
            scores = [s3m, s6m, s12m]
            conv  = round(min(scores) * (1 - float(np.std(scores))), 4)
            tag   = _classify_tag(s3m, s6m, s12m)
            results.append({
                "ticker": ticker,
                "sector": sector,
                "best_macro": top["macro_name"],
                "best_lag": top["lag"],
                "best_corr": top["correlation"],
                "score_3m": s3m,
                "score_6m": s6m,
                "score_12m": s12m,
                "convergence_score": conv,
                "tag": tag,
                "all_signals": best_signals[:5],
            })

    results.sort(key=lambda x: abs(x["best_corr"]), reverse=True)
    return results[:ANALYSIS_CONFIG["top_n"]]


def analyze_all() -> dict:
    """全セクターを分析してキャッシュとSQLiteに保存する"""
    all_results = {}
    all_rows = []
    for sector in SECTORS.keys():
        print(f"分析中: {sector}")
        sector_results = analyze_sector(sector)
        all_results[sector] = sector_results
        all_rows.extend(sector_results)

    _save_signals_db(all_rows)

    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            try:
                cache = json.load(f)
            except Exception:
                cache = {}

    cache["analysis"] = all_results
    cache["analysis_updated"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, default=str)

    print("分析完了 → キャッシュ保存済み")
    return all_results


def get_ticker_detail(ticker: str, sector: str) -> dict:
    """銘柄詳細ページ用のデータを返す"""
    macro_df = load_macro()
    stock_df = load_sector(sector)

    if macro_df.empty or stock_df.empty or ticker not in stock_df.columns:
        return {}

    stock = stock_df[ticker]
    max_lag = ANALYSIS_CONFIG["max_lag"]
    detail = {"ticker": ticker, "sector": sector, "signals": []}

    for macro_name, macro_ticker in MACRO_INDICATORS.items():
        if macro_ticker not in macro_df.columns:
            continue
        macro = macro_df[macro_ticker]
        lag_corrs = lag_correlation(macro, stock, max_lag)
        if not lag_corrs:
            continue
        lag, corr = best_lag(lag_corrs)
        mov_corr = moving_correlation(macro, stock)
        detail["signals"].append({
            "macro_name": macro_name,
            "macro_ticker": macro_ticker,
            "lag": lag,
            "correlation": corr,
            "lag_corrs": lag_corrs,
            "moving_corr": mov_corr.dropna().to_dict(),
        })

    detail["signals"].sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return detail


if __name__ == "__main__":
    analyze_all()
