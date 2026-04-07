import yfinance as yf
import pandas as pd
import json
import os
import time
from datetime import datetime, date
from config import MACRO_INDICATORS, SECTORS, ANALYSIS_CONFIG, DATA_DIR, CACHE_FILE, LOG_FILE, UNIVERSE_CACHE


def log(message: str, level: str = "INFO"):
    """ログを記録する"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"timestamp": timestamp, "level": level, "message": message}
    print(f"[{timestamp}] {level}: {message}")

    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    logs.append(entry)
    # 最新1000件だけ保持
    logs = logs[-1000:]
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def fetch_ticker(ticker: str, period: str) -> pd.DataFrame | None:
    """1銘柄のデータを取得する"""
    try:
        time.sleep(ANALYSIS_CONFIG["fetch_interval"])
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        # MultiIndexカラムをフラット化
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df[["Close"]].copy()
        close.columns = [ticker]
        return close
    except Exception as e:
        log(f"{ticker} 取得失敗: {e}", "ERROR")
        return None


def load_existing(path: str) -> pd.DataFrame:
    """既存CSVを読み込む（なければ空のDataFrameを返す）"""
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    return pd.DataFrame()


def save_data(df: pd.DataFrame, path: str):
    """DataFrameをCSVに保存する"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def fetch_and_update(tickers: list[str], save_path: str, label: str) -> dict:
    """銘柄リストを取得して既存データに追記する"""
    existing = load_existing(save_path)

    # 初回か差分更新かを判断
    if existing.empty:
        period = ANALYSIS_CONFIG["initial_period"]
        log(f"{label}: 初回取得 {len(tickers)}銘柄 period={period}")
    else:
        period = ANALYSIS_CONFIG["update_period"]
        log(f"{label}: 差分更新 {len(tickers)}銘柄 period={period}")

    success, fail = 0, 0
    frames = []

    for ticker in tickers:
        df = fetch_ticker(ticker, period)
        if df is not None:
            frames.append(df)
            success += 1
        else:
            fail += 1

    if not frames:
        log(f"{label}: 取得できた銘柄が0件", "ERROR")
        return {"success": 0, "fail": len(tickers)}

    new_data = pd.concat(frames, axis=1)

    # MultiIndexカラムをフラット化
    if isinstance(new_data.columns, pd.MultiIndex):
        new_data.columns = [col[-1] for col in new_data.columns]
    else:
        # タプル文字列 "('JPY=X', 'JPY=X')" → "JPY=X" に変換
        new_data.columns = [
            c.split("'")[1] if c.startswith("(") else c
            for c in new_data.columns.astype(str)
        ]

    # 既存データと結合して重複行を除去
    if not existing.empty:
        combined = pd.concat([existing, new_data])
        combined.index = pd.to_datetime(combined.index, errors="coerce")
        combined = combined[combined.index.notna()]
        combined = combined[~combined.index.astype(str).str.contains("Price|Ticker", na=False)]
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
    else:
        new_data.index = pd.to_datetime(new_data.index, errors="coerce")
        new_data = new_data[new_data.index.notna()]
        combined = new_data

    save_data(combined, save_path)
    log(f"{label}: 完了 成功={success} 失敗={fail} 総行数={len(combined)}")
    return {"success": success, "fail": fail}


def fetch_tse_tickers(market: str = "prime") -> list[str]:
    """東証の銘柄リストをJPXの公開CSVから取得する"""
    try:
        import pandas as pd
        # JPXが公開しているCSV形式の上場銘柄一覧
        url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        df = pd.read_excel(url, header=0, engine="xlrd")

        if market == "prime":
            df = df[df["市場・商品区分"].str.contains("プライム", na=False)]

        tickers = [f"{int(code)}.T" for code in df["コード"].dropna() if str(code).isdigit()]

        cache = {
            "tickers": tickers,
            "market": market,
            "count": len(tickers),
            "updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(UNIVERSE_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        log(f"銘柄ユニバース取得完了: {market} {len(tickers)}銘柄")
        return tickers

    except Exception as e:
        log(f"銘柄リスト取得失敗: {e} → フォールバック使用", "WARN")
        if os.path.exists(UNIVERSE_CACHE):
            with open(UNIVERSE_CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            return cache.get("tickers", [])
        fallback = []
        for tickers in SECTORS.values():
            fallback.extend(tickers)
        return list(set(fallback))


def run_batch():
    """メインのバッチ処理"""
    start = datetime.now()
    log("=" * 50)
    log("SignalFlow バッチ開始")

    os.makedirs(DATA_DIR, exist_ok=True)
    results = {}

    # マクロ指標の取得
    macro_tickers = list(MACRO_INDICATORS.values())
    results["macro"] = fetch_and_update(
        macro_tickers,
        f"{DATA_DIR}/macro.csv",
        "マクロ指標"
    )

    # セクター別銘柄の取得
    for sector, tickers in SECTORS.items():
        results[sector] = fetch_and_update(
            tickers,
            f"{DATA_DIR}/sector_{sector}.csv",
            f"セクター:{sector}"
        )

    # 実行サマリーをキャッシュに保存
    summary = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "duration_sec": (datetime.now() - start).seconds,
    }
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"バッチ完了 所要時間: {summary['duration_sec']}秒")
    log("=" * 50)
    return summary


if __name__ == "__main__":
    run_batch()
