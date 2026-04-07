"""
weekly_batch.py — GitHub Actions 週次バッチ

処理フロー:
  1. 月次チェック: 30日経過していれば出来高フィルタで銘柄リスト更新
  2. yfinance で対象銘柄 12ヶ月分を並列取得
  3. 騰落率変換 → lag=1〜3 / 3期間 / p値フィルタ で相関計算
  4. 収束スコア・3分類タグ計算
  5. SQLite + CSV に保存
  6. GitHub Actions ジョブサマリー出力
"""

import os
import json
import time
import sqlite3
import concurrent.futures
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr

from config import (
    MACRO_INDICATORS, SECTORS, ANALYSIS_CONFIG,
    DATA_DIR, CACHE_FILE, LOG_FILE, UNIVERSE_CACHE, DB_FILE,
)

# ── 定数 ──────────────────────────────────────────────────
JST = timezone(timedelta(hours=9))
NIKKEI_TICKERS = {"日経平均": "^N225", "TOPIX": "^TOPX"}

PERIOD_DAYS = {"3m": 63, "6m": 126, "12m": 252}

# 月次チェック: 前回ユニバース更新から何日で再フィルタするか
UNIVERSE_REFRESH_DAYS = 30

# 出来高フィルタ: 1日の売買金額（円） 500万円以上
VOLUME_FILTER_JPY = 5_000_000

# 並列ワーカー数（yfinance への過負荷を避けるため控えめに）
FETCH_WORKERS = 4

# バッチ間スリープ（秒）
FETCH_SLEEP = 1.2
BATCH_SLEEP = 10
BATCH_SIZE  = 40


# ── ログ ──────────────────────────────────────────────────

def _log(msg: str, level: str = "INFO"):
    ts = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {level}: {msg}")
    try:
        logs = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        logs.append({"timestamp": ts, "level": level, "message": msg})
        logs = logs[-1000:]
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ── Step 1: 月次チェック / 出来高フィルタ ───────────────

def _universe_needs_refresh() -> bool:
    if not os.path.exists(UNIVERSE_CACHE):
        return True
    with open(UNIVERSE_CACHE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    updated_str = cache.get("updated", "")
    if not updated_str:
        return True
    try:
        updated = datetime.fromisoformat(updated_str)
        return (datetime.now() - updated).days >= UNIVERSE_REFRESH_DAYS
    except Exception:
        return True


def _fetch_volume_jpy(ticker: str) -> float:
    """yfinance から直近の1日売買代金（円）を推定する"""
    try:
        info = yf.Ticker(ticker).fast_info
        vol   = getattr(info, "three_month_average_volume", None) or 0
        price = getattr(info, "last_price", None) or 0
        return float(vol) * float(price)
    except Exception:
        return 0.0


def refresh_universe() -> list[str]:
    """出来高フィルタを再適用して対象銘柄リストを更新する"""
    _log("月次チェック: ユニバース更新開始")

    # SECTORS に定義されている全銘柄を候補に
    candidates: list[str] = []
    for tickers in SECTORS.values():
        candidates.extend(tickers)
    candidates = list(dict.fromkeys(candidates))  # 重複除去・順序保持

    passed: list[str] = []
    failed_volume: list[str] = []

    for i, ticker in enumerate(candidates):
        if i > 0 and i % BATCH_SIZE == 0:
            _log(f"  出来高取得中... {i}/{len(candidates)}")
            time.sleep(BATCH_SLEEP)
        else:
            time.sleep(FETCH_SLEEP)

        jpy = _fetch_volume_jpy(ticker)
        if jpy >= VOLUME_FILTER_JPY:
            passed.append(ticker)
        else:
            failed_volume.append(ticker)
            _log(f"  除外 {ticker}: 売買代金 {jpy:,.0f}円/日", "WARN")

    _log(f"ユニバース更新完了: {len(passed)}/{len(candidates)} 銘柄が通過")

    cache = {
        "tickers": passed,
        "excluded": failed_volume,
        "count": len(passed),
        "volume_filter_jpy": VOLUME_FILTER_JPY,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(UNIVERSE_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    return passed


def load_universe() -> list[str]:
    if os.path.exists(UNIVERSE_CACHE):
        with open(UNIVERSE_CACHE, "r", encoding="utf-8") as f:
            return json.load(f).get("tickers", [])
    return [t for tickers in SECTORS.values() for t in tickers]


# ── Step 2: yfinance 並列取得 ────────────────────────────

def _download_one(ticker: str) -> tuple[str, pd.Series | None]:
    """1銘柄の Close を取得して返す"""
    try:
        time.sleep(FETCH_SLEEP)
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if df.empty:
            return ticker, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return ticker, df["Close"].rename(ticker)
    except Exception as e:
        _log(f"{ticker} 取得失敗: {e}", "ERROR")
        return ticker, None


def fetch_prices_parallel(tickers: list[str]) -> pd.DataFrame:
    """対象銘柄を並列取得して price DataFrame を返す"""
    _log(f"価格取得開始: {len(tickers)}銘柄 (workers={FETCH_WORKERS})")
    series_list: list[pd.Series] = []
    errors: list[str] = []

    # バッチに分割して並列実行
    for batch_start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[batch_start: batch_start + BATCH_SIZE]
        _log(f"  バッチ {batch_start // BATCH_SIZE + 1}: {len(batch)}銘柄")

        with concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
            futures = {ex.submit(_download_one, t): t for t in batch}
            for fut in concurrent.futures.as_completed(futures):
                ticker, s = fut.result()
                if s is not None:
                    series_list.append(s)
                else:
                    errors.append(ticker)

        if batch_start + BATCH_SIZE < len(tickers):
            _log(f"  バッチ間スリープ {BATCH_SLEEP}秒")
            time.sleep(BATCH_SLEEP)

    _log(f"価格取得完了: 成功={len(series_list)} 失敗={len(errors)}")
    if errors:
        _log(f"  失敗銘柄: {errors[:20]}", "WARN")

    if not series_list:
        return pd.DataFrame()

    price_df = pd.concat(series_list, axis=1)
    price_df.index = pd.to_datetime(price_df.index.astype(str).str[:10], errors="coerce")
    price_df = price_df[price_df.index.notna()]
    price_df = price_df[~price_df.index.duplicated(keep="last")]
    price_df.sort_index(inplace=True)
    return price_df


# ── Step 3: 相関計算ユーティリティ ──────────────────────

def _period_score(macro: np.ndarray, stock: np.ndarray, max_lag: int) -> float:
    """lag=1〜max_lag の絶対相関の最大値（p>0.05は除外）"""
    scores = []
    for lag in range(1, max_lag + 1):
        x, y = macro[:-lag], stock[lag:]
        if len(x) < 30:
            break
        try:
            corr, p = pearsonr(x, y)
        except Exception:
            continue
        if not np.isnan(corr) and p <= 0.05:
            scores.append(abs(corr))
    return round(max(scores), 4) if scores else 0.0


def _best_lag_corr(macro: np.ndarray, stock: np.ndarray, max_lag: int) -> tuple[int, float]:
    """最適ラグと相関係数を返す（p>0.05は0扱い）"""
    best_lag_v, best_corr = 1, 0.0
    for lag in range(1, max_lag + 1):
        x, y = macro[:-lag], stock[lag:]
        if len(x) < 30:
            break
        try:
            corr, p = pearsonr(x, y)
        except Exception:
            continue
        if np.isnan(corr):
            continue
        if p > 0.05:
            corr = 0.0
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag_v = lag
    return best_lag_v, round(best_corr, 4)


# ── Step 4: 収束スコア・タグ ────────────────────────────

def _convergence_score(s3m: float, s6m: float, s12m: float) -> float:
    scores = [s3m, s6m, s12m]
    return round(min(scores) * (1 - float(np.std(scores))), 4)


def _classify_tag(s3m: float, s6m: float, s12m: float, threshold: float = 0.3) -> str:
    if s3m >= threshold and s6m >= threshold and s12m >= threshold:
        return "構造的ラグ"
    if s3m >= threshold and s6m >= threshold and s12m < threshold:
        return "新興ラグ"
    if s12m >= threshold and s3m < threshold:
        return "崩壊警戒"
    return ""


# ── Step 5: 保存 ─────────────────────────────────────────

def _save_to_db(rows: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
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
    ts = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    conn.executemany(
        "INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                r["ticker"], r["sector"], r["macro_name"],
                r["best_lag"], r["best_corr"],
                r["score_3m"], r["score_6m"], r["score_12m"],
                r["convergence_score"], r["tag"], ts,
            )
            for r in rows
        ],
    )
    conn.commit()
    conn.close()
    _log(f"SQLite保存完了: {len(rows)}件 → {DB_FILE}")


def _save_to_csv(rows: list[dict]) -> None:
    if not rows:
        return
    path = os.path.join(DATA_DIR, "signals_weekly.csv")
    df = pd.DataFrame(rows)
    df["exported_at"] = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    _log(f"CSV保存完了: {len(rows)}件 → {path}")


# ── Step 6: ジョブサマリー ───────────────────────────────

def _write_job_summary(rows: list[dict], stats: dict) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "")
    lines: list[str] = []

    lines.append("# JISAME 週次バッチ 完了レポート\n")
    lines.append(f"実行日時: {datetime.now(JST).strftime('%Y/%m/%d %H:%M')} JST\n\n")

    lines.append("## 処理サマリー\n")
    lines.append("| 項目 | 値 |\n|---|---|\n")
    lines.append(f"| 処理銘柄数 | {stats['processed']} |\n")
    lines.append(f"| 取得エラー数 | {stats['fetch_errors']} |\n")
    lines.append(f"| シグナル件数 | {stats['signals']} |\n")
    lines.append(f"| ユニバース更新 | {'あり' if stats.get('universe_refreshed') else 'なし'} |\n")
    lines.append(f"| 所要時間 | {stats.get('duration_sec', 0)}秒 |\n\n")

    if rows:
        top10 = sorted(rows, key=lambda x: abs(x["convergence_score"]), reverse=True)[:10]
        lines.append("## 収束スコア TOP10\n")
        lines.append("| # | ティッカー | セクター | 先行指標 | ラグ | 相関 | 3m | 6m | 12m | 収束 | タグ |\n")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(top10, 1):
            lines.append(
                f"| {i} | {r['ticker']} | {r['sector']} | {r['macro_name']} "
                f"| +{r['best_lag']}日 | {r['best_corr']:+.3f} "
                f"| {r['score_3m']:.3f} | {r['score_6m']:.3f} | {r['score_12m']:.3f} "
                f"| {r['convergence_score']:.3f} | {r['tag'] or '—'} |\n"
            )

    summary_text = "".join(lines)
    print("\n" + "=" * 60)
    print(summary_text)
    print("=" * 60)

    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(summary_text)
        _log(f"ジョブサマリー書き込み完了: {summary_path}")


# ── メイン処理 ───────────────────────────────────────────

def run_weekly_batch() -> dict:
    start = datetime.now()
    _log("=" * 50)
    _log("JISAME 週次バッチ開始")
    os.makedirs(DATA_DIR, exist_ok=True)

    stats = {
        "processed": 0, "fetch_errors": 0, "signals": 0,
        "universe_refreshed": False, "duration_sec": 0,
    }

    # ── Step 1: 月次チェック ─────────────────────────────
    if _universe_needs_refresh():
        stock_tickers = refresh_universe()
        stats["universe_refreshed"] = True
    else:
        stock_tickers = load_universe()
        _log(f"ユニバースキャッシュ使用: {len(stock_tickers)}銘柄")

    # 先行指標（固定）とマクロ指標を取得対象に含める
    all_macro = {**MACRO_INDICATORS, **NIKKEI_TICKERS}
    macro_ticker_list = list(dict.fromkeys(all_macro.values()))

    # ── Step 2: 価格取得 ─────────────────────────────────
    _log("マクロ指標取得")
    macro_price_df = fetch_prices_parallel(macro_ticker_list)
    if macro_price_df.empty:
        _log("マクロ指標の取得に失敗しました", "ERROR")
        return stats

    _log("株式データ取得")
    stock_price_df = fetch_prices_parallel(stock_tickers)
    stats["processed"] = stock_price_df.shape[1] if not stock_price_df.empty else 0
    stats["fetch_errors"] = len(stock_tickers) - stats["processed"]

    if stock_price_df.empty:
        _log("株式データが空です", "ERROR")
        return stats

    # 既存 CSV に追記保存
    for sector, tickers in SECTORS.items():
        cols = [t for t in tickers if t in stock_price_df.columns]
        if not cols:
            continue
        path = os.path.join(DATA_DIR, f"sector_{sector}.csv")
        new_df = stock_price_df[cols]
        if os.path.exists(path):
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
            existing = existing[[c for c in existing.columns if not str(c).startswith("(")]]
            combined = pd.concat([existing, new_df])
            combined.index = pd.to_datetime(combined.index.astype(str).str[:10], errors="coerce")
            combined = combined[combined.index.notna()]
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            combined.to_csv(path)
        else:
            new_df.to_csv(path)

    macro_path = os.path.join(DATA_DIR, "macro.csv")
    macro_price_df.to_csv(macro_path)
    _log("CSV保存完了")

    # ── Step 3 & 4: 相関計算・収束スコア ────────────────
    _log("相関計算開始")
    max_lag = ANALYSIS_CONFIG["max_lag"]

    # 騰落率変換
    macro_ret  = macro_price_df.pct_change().dropna(how="all")
    stock_ret  = stock_price_df.pct_change().dropna(how="all")

    # インデックスを揃える
    common_idx = macro_ret.index.intersection(stock_ret.index)
    macro_ret  = macro_ret.loc[common_idx]
    stock_ret  = stock_ret.loc[common_idx]

    all_rows: list[dict] = []

    for sector, tickers in SECTORS.items():
        cols = [t for t in tickers if t in stock_ret.columns]
        for ticker in cols:
            stock_s = stock_ret[ticker].dropna()
            if len(stock_s) < ANALYSIS_CONFIG["min_data_points"]:
                continue

            best_row = None

            for macro_name, macro_ticker in all_macro.items():
                if macro_ticker not in macro_ret.columns:
                    continue
                macro_s = macro_ret[macro_ticker].dropna()

                aligned = pd.concat([macro_s, stock_s], axis=1).dropna()
                if len(aligned) < ANALYSIS_CONFIG["min_data_points"]:
                    continue

                m_all = aligned.iloc[:, 0].values
                s_all = aligned.iloc[:, 1].values

                lag_v, corr = _best_lag_corr(m_all, s_all, max_lag)
                if abs(corr) < ANALYSIS_CONFIG["signal_threshold"]:
                    continue

                # 3期間スコア
                s3m  = _period_score(m_all[-PERIOD_DAYS["3m"]:],  s_all[-PERIOD_DAYS["3m"]:],  max_lag)
                s6m  = _period_score(m_all[-PERIOD_DAYS["6m"]:],  s_all[-PERIOD_DAYS["6m"]:],  max_lag)
                s12m = _period_score(m_all[-PERIOD_DAYS["12m"]:], s_all[-PERIOD_DAYS["12m"]:], max_lag)

                conv = _convergence_score(s3m, s6m, s12m)
                tag  = _classify_tag(s3m, s6m, s12m)

                row = {
                    "ticker": ticker, "sector": sector, "macro_name": macro_name,
                    "best_lag": lag_v, "best_corr": corr,
                    "score_3m": s3m, "score_6m": s6m, "score_12m": s12m,
                    "convergence_score": conv, "tag": tag,
                }

                if best_row is None or abs(conv) > abs(best_row["convergence_score"]):
                    best_row = row

            if best_row:
                all_rows.append(best_row)

    stats["signals"] = len(all_rows)
    _log(f"相関計算完了: {len(all_rows)}件のシグナル")

    # ── Step 5: 保存 ─────────────────────────────────────
    _save_to_db(all_rows)
    _save_to_csv(all_rows)

    # correlation_cache.json を更新（Streamlit が参照するため）
    cache: dict = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    # analysis キーを再構築（セクター別TOP10）
    from collections import defaultdict
    by_sector: dict = defaultdict(list)
    for r in all_rows:
        by_sector[r["sector"]].append(r)

    analysis_result: dict = {}
    for sector, rows in by_sector.items():
        rows_sorted = sorted(rows, key=lambda x: abs(x["convergence_score"]), reverse=True)
        analysis_result[sector] = [
            {
                "ticker": r["ticker"], "sector": r["sector"],
                "best_macro": r["macro_name"],
                "best_lag": r["best_lag"], "best_corr": r["best_corr"],
                "score_3m": r["score_3m"], "score_6m": r["score_6m"],
                "score_12m": r["score_12m"],
                "convergence_score": r["convergence_score"], "tag": r["tag"],
                "all_signals": [],
            }
            for r in rows_sorted[:ANALYSIS_CONFIG["top_n"]]
        ]

    cache["analysis"] = analysis_result
    cache["analysis_updated"] = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    cache["last_updated"] = datetime.now(JST).strftime("%Y-%m-%d %H:%M")

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, default=str)
    _log(f"キャッシュ更新完了: {CACHE_FILE}")

    # ── Step 6: ジョブサマリー ────────────────────────────
    stats["duration_sec"] = (datetime.now() - start).seconds
    _write_job_summary(all_rows, stats)

    _log(f"週次バッチ完了 所要時間: {stats['duration_sec']}秒")
    _log("=" * 50)
    return stats


if __name__ == "__main__":
    run_weekly_batch()
