import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
import math

def _pearsonr(x, y):
    """scipy.stats.pearsonr の numpy/math 代替実装"""
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    r = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(r):
        return 0.0, 1.0
    if abs(r) >= 1.0:
        return r, 0.0
    t_stat = r * math.sqrt((n - 2) / (1 - r * r))
    df = n - 2
    try:  # Python 3.12+
        p = math.betainc(df / 2, 0.5, df / (df + t_stat * t_stat))
    except AttributeError:  # フォールバック: 正規分布近似
        p = 2 * (1 - (1 + math.erf(abs(t_stat) / math.sqrt(2))) / 2)
    return r, float(min(p, 1.0))
from config import MACRO_INDICATORS, SECTORS, DATA_DIR, CACHE_FILE, TICKER_NAMES
from analysis import get_ticker_detail, load_sector, load_macro, _period_lag_score, _classify_tag

st.set_page_config(page_title="JISAME", page_icon="🦈", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #181818 !important; }
[data-testid="stHeader"]           { background-color: #181818 !important; }
[data-testid="stSidebar"]          { background-color: #1e1e1e !important; }
.main .block-container { padding-top: 1.2rem !important; max-width: 900px !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }

/* セクション見出し */
h2, h3, h4 { color: #aaaaaa !important; font-size: 12px !important;
              font-weight: 600 !important; letter-spacing: 0.4px !important; }

/* ボタン全般 */
[data-testid="stButton"] > button {
    background: #282828 !important; border: 1px solid #333 !important;
    color: #cccccc !important; border-radius: 6px !important;
    font-size: 12px !important; padding: 4px 12px !important;
}
[data-testid="stButton"] > button:hover {
    background: #333 !important; border-color: #4d9fff !important; color: #fff !important;
}
/* セクター選択ボタン：primaryをグレー選択状態に上書き */
[data-testid="stBaseButton-primary"] {
    background: #444444 !important; border-color: #666666 !important;
    color: #ffffff !important; font-size: 12px !important; padding: 4px 12px !important;
}
[data-testid="stBaseButton-primary"]:hover {
    background: #555555 !important; border-color: #888888 !important; color: #ffffff !important;
}

/* セレクトボックス */
[data-testid="stSelectbox"] > div > div {
    background: #222 !important; border: 1px solid #333 !important;
    color: #ccc !important; font-size: 12px !important; border-radius: 6px !important;
}

/* ラジオ（セクタータブ） */
[data-testid="stRadio"] > label { display: none !important; }
[data-testid="stRadio"] > div {
    display: flex !important; flex-direction: row !important;
    gap: 6px !important; flex-wrap: wrap !important;
}
[data-testid="stRadio"] > div > label {
    display: inline-flex !important; align-items: center !important;
    background: #242424 !important; border: 1px solid #333 !important;
    color: #888 !important; border-radius: 20px !important;
    padding: 4px 14px !important; font-size: 12px !important;
    font-weight: 500 !important; cursor: pointer !important; margin: 0 !important;
}
[data-testid="stRadio"] > div > label:has(input:checked) {
    background: #1a3a5c !important; border-color: #4d9fff !important;
    color: #4d9fff !important; font-weight: 600 !important;
}
[data-testid="stRadio"] > div > label > div { display: none !important; }
[data-testid="stRadio"] > div > label > p {
    color: inherit !important; font-size: inherit !important;
    font-weight: inherit !important; margin: 0 !important;
}

/* divider */
hr { border-color: #2a2a2a !important; margin: 10px 0 !important; }

/* caption */
[data-testid="stCaptionContainer"] { color: #666 !important; font-size: 11px !important; }

/* expander 枠色 */
[data-testid="stExpander"] { border-color: #2a2a2a !important; }


/* ③コンテナ枠色を①②と統一（stVerticalBlock に直接 border が適用される） */
[data-testid="stVerticalBlock"] {
    border-color: #2a2a2a !important;
}

/* セクター選択ボタン：コンパクト */
div[data-testid="stHorizontalBlock"] button {
    padding: 4px 10px !important;
    font-size: 12px !important;
    min-height: 0px !important;
    height: auto !important;
}

/* スクロールバー */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #181818; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── ヘルパー関数 ─────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_macro() -> pd.DataFrame:
    return load_macro()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_sector(sector: str) -> pd.DataFrame:
    return load_sector(sector)


@st.cache_data(ttl=3600, show_spinner=False)
def get_heatmap(sector: str, macros: tuple, period_days: int = 0) -> tuple[list, list]:
    """ヒートマップ用：全銘柄×マクロの相関行列（1時間キャッシュ）"""
    macro_df = cached_load_macro()
    stock_df = cached_load_sector(sector)
    if macro_df.empty or stock_df.empty:
        return [], []

    if period_days > 0:
        macro_df = macro_df.iloc[-period_days:]
        stock_df = stock_df.iloc[-period_days:]

    y_labels, matrix = [], []
    for ticker in stock_df.columns:
        stock = stock_df[ticker].dropna()
        code = str(ticker).replace(".T", "")
        name = TICKER_NAMES.get(str(ticker), "")[:4]
        y_labels.append(f"{code} {name}" if name else f"{code} ")
        row = []
        for macro_name in macros:
            macro_ticker = MACRO_INDICATORS.get(macro_name)
            if macro_ticker and macro_ticker in macro_df.columns:
                aligned = pd.concat(
                    [macro_df[macro_ticker], stock], axis=1
                ).dropna()
                if len(aligned) >= 30:
                    corr, p_value = _pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    if p_value > 0.1:
                        corr = 0.0
                    row.append(round(corr, 3) if not pd.isna(corr) else None)
                else:
                    row.append(None)
            else:
                row.append(None)
        matrix.append(row)
    return y_labels, matrix


@st.cache_data(ttl=3600, show_spinner=False)
def get_top_heatmap(macros: tuple, period_days: int = 0, top_n: int = 25) -> tuple[list, list]:
    """全セクター横断で相関係数絶対値が高い上位N銘柄のヒートマップ"""
    macro_df = cached_load_macro()
    if period_days > 0:
        macro_df = macro_df.iloc[-period_days:]

    all_rows = []  # (max_abs_corr, y_label, corr_row)
    for sector_name in SECTORS:
        stock_df = cached_load_sector(sector_name)
        if stock_df.empty:
            continue
        s_df = stock_df.iloc[-period_days:] if period_days > 0 else stock_df
        for ticker in s_df.columns:
            stock = s_df[ticker].dropna()
            code = str(ticker).replace(".T", "")
            name = TICKER_NAMES.get(str(ticker), "")[:4]
            label = f"{code} {name}" if name else f"{code} "
            row = []
            max_abs = 0.0
            for macro_name in macros:
                macro_ticker = MACRO_INDICATORS.get(macro_name)
                if macro_ticker and macro_ticker in macro_df.columns:
                    aligned = pd.concat(
                        [macro_df[macro_ticker], stock], axis=1
                    ).dropna()
                    if len(aligned) >= 30:
                        corr, p = _pearsonr(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)
                        if p > 0.1:
                            corr = 0.0
                        val = round(corr, 3) if not pd.isna(corr) else None
                        row.append(val)
                        if val is not None:
                            max_abs = max(max_abs, abs(val))
                    else:
                        row.append(None)
                else:
                    row.append(None)
            all_rows.append((max_abs, label, row))

    all_rows.sort(key=lambda x: x[0], reverse=True)
    top = all_rows[:top_n]
    return [r[1] for r in top], [r[2] for r in top]


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_macro_latest() -> dict:
    path = f"{DATA_DIR}/macro.csv"
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, index_col=0)
    df.columns = [c.split("'")[1] if c.startswith("(") else c for c in df.columns.astype(str)]
    df.index = pd.to_datetime(df.index.astype(str).str[:10], errors="coerce")
    df = df[df.index.notna()]
    if df.empty or len(df) < 2:
        return {}
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all").sort_index()
    result = {}
    for name, ticker in MACRO_INDICATORS.items():
        if ticker not in df.columns:
            continue
        col = df[ticker].dropna()
        if len(col) < 2:
            continue
        val, p = col.iloc[-1], col.iloc[-2]
        if pd.isna(val) or pd.isna(p) or p == 0:
            continue
        result[name] = {
            "value": round(float(val), 4),
            "change": round(float((val - p) / p * 100), 2),
            "ticker": ticker,
        }
    return result

def ticker_name(t: str) -> str:
    return TICKER_NAMES.get(t, t)

TAG_BADGE_HTML = {
    "構造的ラグ": ('<span style="background:#1a3a2a;border:1px solid #26a69a55;color:#4caf84;'
                  'font-size:9px;padding:1px 7px;border-radius:8px;margin-left:6px;">構造的</span>'),
    "新興ラグ":   ('<span style="background:#2a2a1a;border:1px solid #ffca2855;color:#ffca28;'
                  'font-size:9px;padding:1px 7px;border-radius:8px;margin-left:6px;">新興</span>'),
    "崩壊警戒":   ('<span style="background:#2a1a1a;border:1px solid #ef535055;color:#ef9a9a;'
                  'font-size:9px;padding:1px 7px;border-radius:8px;margin-left:6px;">崩壊警戒</span>'),
}

def section_header(num: str, title: str, badge: str = "", badge_color: str = "#1a3a5c",
                   badge_text_color: str = "#4d9fff", badge_border: str = "#4d9fff44") -> str:
    badge_html = ""
    if badge:
        badge_html = (f'<span style="margin-left:auto;background:{badge_color};'
                      f'border:1px solid {badge_border};color:{badge_text_color};'
                      f'font-size:10px;padding:2px 9px;border-radius:10px;">{badge}</span>')
    return f"""
<div style="background:#222222;border:1px solid #2a2a2a;border-radius:10px;
     padding:14px 18px;margin-bottom:10px;">
  <div style="font-size:11px;font-weight:600;color:#888;letter-spacing:0.5px;
       margin-bottom:12px;display:flex;align-items:center;gap:8px;">
    <span style="background:#2a2a2a;border:1px solid #333;border-radius:50%;
         width:18px;height:18px;display:inline-flex;align-items:center;
         justify-content:center;font-size:10px;color:#aaa;">{num}</span>
    {title}{badge_html}
  </div>"""

# ── ページルーティング ────────────────────────────────────

params = st.query_params
page = params.get("page", "home")
ticker_param = params.get("ticker", "")
sector_param = params.get("sector", "")

# ══════════════════════════════════════════════════════════
# 銘柄詳細ページ
# ══════════════════════════════════════════════════════════
if page == "detail" and ticker_param:
    st.button("← ホームに戻る", on_click=lambda: st.query_params.clear())

    sector = sector_param or list(SECTORS.keys())[0]
    cache_key = f"detail_{ticker_param}"
    if cache_key not in st.session_state:
        with st.spinner("分析中..."):
            st.session_state[cache_key] = get_ticker_detail(ticker_param, sector)
    detail = st.session_state[cache_key]

    if not detail or not detail.get("signals"):
        st.warning(f"{ticker_param} のデータを分析中です。初回は30秒ほどかかる場合があります。")
        st.stop()

    # データ準備
    macro_df = cached_load_macro()
    stock_df = cached_load_sector(sector)
    top_sig = detail["signals"][0]
    top_macro_ticker = top_sig["macro_ticker"]
    top_macro_name = top_sig["macro_name"]
    best_lag_val = top_sig["lag"]
    best_corr = top_sig["correlation"]

    top_macro_series = macro_df[top_macro_ticker].dropna() if top_macro_ticker in macro_df.columns else pd.Series(dtype=float)
    stock_series = stock_df[ticker_param].dropna() if (not stock_df.empty and ticker_param in stock_df.columns) else pd.Series(dtype=float)

    max_lag = 3
    s3m  = _period_lag_score(top_macro_series, stock_series, 63,  max_lag) if not top_macro_series.empty else 0.0
    s6m  = _period_lag_score(top_macro_series, stock_series, 126, max_lag) if not top_macro_series.empty else 0.0
    s12m = _period_lag_score(top_macro_series, stock_series, 252, max_lag) if not top_macro_series.empty else 0.0
    tag  = _classify_tag(s3m, s6m, s12m)

    # ══ ① ヘッダー ══════════════════════════════════════════
    name = ticker_name(ticker_param)
    badge_html = TAG_BADGE_HTML.get(tag, "")
    corr_color = "#4caf84" if best_corr > 0 else "#ef9a9a"
    st.html(f"""
    <div style="padding:14px 0 10px 0;border-bottom:1px solid #2a2a2a;margin-bottom:12px;">
      <div style="font-size:22px;font-weight:700;color:#e0e0e0;line-height:1.3;">
        {ticker_param}&nbsp;<span style="font-size:15px;color:#888;">{name}</span>{badge_html}
      </div>
      <div style="font-size:12px;color:#666;margin-top:6px;display:flex;gap:18px;flex-wrap:wrap;">
        <span>セクター: <span style="color:#aaa;">{sector}</span></span>
        <span>先行指標: <span style="color:#aaa;">{top_macro_name}</span></span>
        <span>最適ラグ: <span style="color:#aaa;">{best_lag_val}日</span></span>
        <span>相関係数: <span style="color:{corr_color};font-weight:600;">{best_corr:+.3f}</span></span>
      </div>
    </div>
    """)

    # ══ ② 値動き比較チャート ═══════════════════════════════
    st.markdown("**値動き比較** — 先行指標（青）と銘柄（緑・ラグシフト済み）日次騰落率")
    period_map = {"3ヶ月": 63, "6ヶ月": 126, "12ヶ月": 252}
    period_sel = st.radio("期間", list(period_map.keys()), horizontal=True, index=0, key="detail_period")
    days = period_map[period_sel]

    if not top_macro_series.empty and not stock_series.empty and best_lag_val > 0:
        aligned = pd.concat([top_macro_series, stock_series], axis=1).dropna().iloc[-days:]
        if len(aligned) > best_lag_val + 5:
            m_slice = aligned.iloc[:, 0].iloc[:-best_lag_val]
            s_vals  = aligned.iloc[:, 1].values[best_lag_val:]
            m_pct = m_slice * 100
            s_pct = pd.Series(s_vals * 100, index=m_slice.index)

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(
                x=m_pct.index, y=m_pct.values, name=top_macro_name,
                line=dict(color="#4d9fff", width=2)
            ))
            fig_cmp.add_trace(go.Scatter(
                x=s_pct.index, y=s_pct.values, name=f"{ticker_param}（{best_lag_val}日シフト）",
                line=dict(color="#4caf84", width=2)
            ))
            fig_cmp.update_layout(
                height=300, margin=dict(l=0, r=0, t=8, b=0),
                plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                xaxis=dict(gridcolor="#2a2a2a", color="#666"),
                yaxis=dict(title="日次騰落率 (%)", gridcolor="#2a2a2a", color="#666"),
                legend=dict(bgcolor="#1e1e1e", font=dict(color="#aaa", size=11), orientation="h", y=1.08),
                hovermode="x unified",
            )
            fig_cmp.add_hline(y=0, line_color="#444", line_width=0.8)
            st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.caption("比較チャートを表示するにはデータが必要です。")

    # ══ ③ ラグ相関の推移グラフ ═════════════════════════════
    st.markdown("**ラグ相関の推移** — 過去12ヶ月（60日窓・4週ステップ）")
    if not top_macro_series.empty and not stock_series.empty:
        aligned_12m = pd.concat([top_macro_series, stock_series], axis=1).dropna().iloc[-252:]
        window, step = 60, 20
        trend_dates = []
        trend_corrs = {1: [], 2: [], 3: []}

        for end_idx in range(window, len(aligned_12m), step):
            chunk = aligned_12m.iloc[max(0, end_idx - window):end_idx]
            if len(chunk) < 30:
                continue
            trend_dates.append(aligned_12m.index[end_idx - 1])
            for lag in [1, 2, 3]:
                if lag >= len(chunk):
                    trend_corrs[lag].append(0.0)
                    continue
                x = chunk.iloc[:, 0].values[:-lag]
                y = chunk.iloc[:, 1].values[lag:]
                try:
                    corr, p = _pearsonr(x, y)
                    trend_corrs[lag].append(round(corr, 4) if p <= 0.05 and not np.isnan(corr) else 0.0)
                except Exception:
                    trend_corrs[lag].append(0.0)

        if trend_dates:
            lag_colors = {1: "#4d9fff", 2: "#4caf84", 3: "#a78bfa"}
            fig_trend = go.Figure()
            for lag in [1, 2, 3]:
                fig_trend.add_trace(go.Scatter(
                    x=trend_dates, y=trend_corrs[lag], name=f"lag={lag}日",
                    line=dict(color=lag_colors[lag], width=1.5),
                    mode="lines+markers", marker=dict(size=5)
                ))
            fig_trend.add_hline(y=0, line_color="#444", line_width=0.8)
            fig_trend.add_hline(y=0.3,  line_dash="dash", line_color="rgba(38,166,154,0.33)", line_width=1)
            fig_trend.add_hline(y=-0.3, line_dash="dash", line_color="rgba(239,83,80,0.33)", line_width=1)
            fig_trend.update_layout(
                height=260, margin=dict(l=0, r=0, t=8, b=0),
                plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                xaxis=dict(gridcolor="#2a2a2a", color="#666"),
                yaxis=dict(range=[-1, 1], title="相関係数", gridcolor="#2a2a2a", color="#666"),
                legend=dict(bgcolor="#1e1e1e", font=dict(color="#aaa", size=11), orientation="h", y=1.08),
                hovermode="x unified",
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    # ══ ④ 3期間スコアカード ════════════════════════════════
    st.markdown("**3期間スコアサマリー**")
    c1, c2, c3 = st.columns(3)
    for col, label, score in [(c1, "3ヶ月", s3m), (c2, "6ヶ月", s6m), (c3, "12ヶ月", s12m)]:
        score_color = "#4caf84" if score >= 0.3 else "#888"
        col.html(f"""
        <div style="background:#1e1e1e;border:1px solid #2a2a2a;border-radius:8px;
                    padding:14px 16px;text-align:center;margin:4px 0;">
          <div style="font-size:11px;color:#666;margin-bottom:6px;">{label}</div>
          <div style="font-size:30px;font-weight:700;color:{score_color};line-height:1;">{score:.3f}</div>
          <div style="font-size:10px;color:#555;margin-top:4px;">収束スコア</div>
        </div>
        """)

# ══════════════════════════════════════════════════════════
# ホーム画面
# ══════════════════════════════════════════════════════════
else:
    # Streamlit Cloud 対応：data/ が空なら自動でバッチ実行
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(f"{DATA_DIR}/macro.csv"):
        with st.spinner("初回データ取得中… 数分かかります"):
            from batch import run_batch
            run_batch()
            from analysis import analyze_all
            analyze_all()

    cache    = load_cache()
    analysis = cache.get("analysis", {})
    last_updated = cache.get("last_updated", "—")

    # ── 鮮度計算 ─────────────────────────────────────────
    _analysis_ts = cache.get("analysis_updated", "")
    _days_ago = None
    _date_str = "—"
    if _analysis_ts:
        try:
            _updated_dt = pd.Timestamp(_analysis_ts)
            _days_ago   = (pd.Timestamp.now() - _updated_dt).days
            _date_str   = _updated_dt.strftime("%Y/%m/%d")
        except Exception:
            pass

    # ── ヘッダー ─────────────────────────────────────────
    st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
     padding:6px 0 14px;border-bottom:1px solid #2a2a2a;margin-bottom:14px;">
  <div style="display:flex;align-items:center;gap:10px;">
    <svg width="28" height="28" viewBox="0 0 36 36" fill="none">
      <path d="M8 22 L14 8 L20 22" fill="#4d9fff" opacity="0.9"/>
      <path d="M4 24 Q18 18 32 24 Q28 30 18 29 Q8 30 4 24Z" fill="#4d9fff"/>
      <path d="M30 24 L36 18 L36 28Z" fill="#4d9fff" opacity="0.7"/>
      <circle cx="10" cy="24" r="1.2" fill="#181818"/>
    </svg>
    <span style="font-size:18px;font-weight:800;color:#e0e0e0;letter-spacing:-0.5px;">JISAME</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <span style="font-size:11px;color:#555;">最終更新</span>
    <span style="background:#1a2a3a;border:1px solid #2a4a6a;color:#4d9fff;
         font-size:11px;padding:3px 10px;border-radius:10px;">{last_updated}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── シグナル鮮度バナー ────────────────────────────────
    if _days_ago is not None:
        if _days_ago >= 7:
            st.markdown(f"""
<div style="background:#2a1818;border:1px solid #ef535066;border-radius:8px;
     padding:9px 16px;margin-bottom:12px;display:flex;
     justify-content:space-between;align-items:center;">
  <span style="font-size:12px;color:#ef9a9a;font-weight:600;">
    ⚠ 最終更新：{_date_str}（{_days_ago}日前）&emsp;シグナルが古い可能性があります
  </span>
  <span style="font-size:10px;color:#666;">このシグナルはlag=1〜3日を想定しています</span>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style="background:#1a1a24;border:1px solid #2a2a44;border-radius:8px;
     padding:9px 16px;margin-bottom:12px;display:flex;
     justify-content:space-between;align-items:center;">
  <span style="font-size:12px;color:#888;">
    最終更新：{_date_str}（{_days_ago}日前）
  </span>
  <span style="font-size:10px;color:#555;">このシグナルはlag=1〜3日を想定しています</span>
</div>""", unsafe_allow_html=True)

    # ── ① マクロ指標サマリー ─────────────────────────────
    macro_data = load_macro_latest()
    total_indicators = len(MACRO_INDICATORS)
    fetched = len(macro_data)

    html = section_header("①", "マクロ指標サマリー",
                          badge=f"{fetched}指標", badge_color="#1a2a3a",
                          badge_text_color="#4d9fff", badge_border="#2a4a6a")
    html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;">'

    for name, data in list(macro_data.items())[:8]:
        chg = data["change"]
        val = data["value"]
        val_color = "#26a69a" if chg > 0 else "#ef5350"
        chg_arrow = "▲" if chg > 0 else "▼"
        val_str = f"{val:,.1f}" if val < 1000 else f"{val:,.0f}"
        html += f"""
        <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:12px 14px;">
          <div style="font-size:11px;color:#666;margin-bottom:6px;">{name}</div>
          <div style="font-size:17px;font-weight:700;color:{val_color};margin-bottom:3px;">{val_str}</div>
          <div style="font-size:11px;font-weight:500;color:{val_color};">{chg_arrow}{abs(chg):.1f}%</div>
        </div>"""

    # データ品質カード
    html += f"""
        <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:12px 14px;">
          <div style="font-size:11px;color:#666;margin-bottom:6px;">データ品質</div>
          <div style="font-size:17px;font-weight:700;color:#e0e0e0;margin-bottom:3px;">{fetched}/{total_indicators}取得</div>
          <div style="font-size:11px;color:#888;">{'✓ 正常' if fetched >= total_indicators - 2 else '⚠ 一部欠損'}</div>
        </div>"""

    html += "</div></div>"
    st.html(html)

    # 更新ボタン
    col_l, col_r = st.columns([6, 1])
    with col_r:
        if st.button("🔄 更新", use_container_width=True):
            with st.spinner("取得中..."):
                from batch import run_batch
                run_batch()
            st.rerun()

    # analysisキーがなければ自動で分析実行
    if not analysis:
        with st.spinner("初回分析中... 数分かかります"):
            from analysis import analyze_all
            analyze_all()
        cache = load_cache()
        analysis = cache.get("analysis", {})

    # ── 期間セレクタ（②③共通） ───────────────────────────
    PERIOD_OPTIONS  = {"3ヶ月": 63, "6ヶ月": 126, "12ヶ月": 252}
    PERIOD_SCORE_KEY = {"3ヶ月": "score_3m", "6ヶ月": "score_6m", "12ヶ月": "score_12m"}
    selected_period = st.radio(
        "分析期間", list(PERIOD_OPTIONS.keys()),
        horizontal=True, key="signal_period",
        label_visibility="collapsed",
    )
    period_days = PERIOD_OPTIONS[selected_period]
    score_key   = PERIOD_SCORE_KEY[selected_period]

    # ── ② 本日の注目シグナル TOP10 ───────────────────────
    all_signals = []
    for sector, results in analysis.items():
        for r in results:
            all_signals.append(r | {"セクター": sector})
    # 期間スコアが存在する場合はそれで並べ替え、なければ best_corr にフォールバック
    all_signals.sort(
        key=lambda x: abs(x.get(score_key, x["best_corr"])),
        reverse=True,
    )

    threshold = 0.2
    html2_header = section_header("②", "本日の注目シグナル TOP10",
                                  badge=f"{selected_period} / 閾値 ±{threshold}以上",
                                  badge_color="#2a1a1a", badge_text_color="#ef9a9a",
                                  badge_border="#ef535044") + "</div>"
    st.html(html2_header)

    # ── 銘柄検索 ─────────────────────────────────────────
    search_q = st.text_input("🔍 銘柄検索", placeholder="ティッカー (例: 7203.T) または会社名",
                              label_visibility="collapsed")
    if search_q:
        q = search_q.strip().upper()
        matches = []
        seen = set()
        for sector_name, results in analysis.items():
            for r in results:
                t = r["ticker"]
                n = TICKER_NAMES.get(t, "")
                if q in t.upper() or q in n.upper():
                    if t not in seen:
                        matches.append((t, n, sector_name))
                        seen.add(t)
        if not matches:
            for t, n in TICKER_NAMES.items():
                if q in t.upper() or q in n.upper():
                    if t not in seen:
                        found_sector = ""
                        for s_name, s_tickers in SECTORS.items():
                            if t in s_tickers:
                                found_sector = s_name
                                break
                        matches.append((t, n, found_sector))
                        seen.add(t)
        if matches:
            result_html = ('<div style="background:#1e1e1e;border:1px solid #2a2a2a;'
                           'border-radius:8px;padding:12px 16px;margin-bottom:12px;">'
                           f'<div style="font-size:11px;color:#666;margin-bottom:8px;">'
                           f'{len(matches)}件ヒット</div>')
            for t, n, s in matches[:20]:
                href = f"?page=detail&ticker={t}&sector={s}" if s else f"?page=detail&ticker={t}"
                result_html += (
                    f'<div style="padding:6px 0;border-bottom:1px solid #2a2a2a;">'
                    f'<a href="{href}" target="_self" style="color:#4d9fff;font-size:13px;'
                    f'font-weight:700;text-decoration:none;">{t}</a>'
                    f'<span style="color:#aaa;font-size:12px;margin-left:6px;">{n}</span>'
                    f'<span style="color:#555;font-size:11px;margin-left:8px;">{s}</span></div>'
                )
            result_html += '</div>'
            st.html(result_html)

    html2 = '<div style="background:#222222;border:1px solid #2a2a2a;border-radius:10px;padding:0 18px 4px;">'
    if all_signals:
        for r in all_signals[:10]:
            period_corr = r.get(score_key)
            if period_corr is not None:
                # score_* は絶対値なので符号は best_corr から引き継ぐ
                sign = 1 if r["best_corr"] >= 0 else -1
                corr = sign * period_corr
            else:
                corr = r["best_corr"]
            corr_color = "#26a69a" if corr > 0 else "#ef5350"
            corr_str = f"+{corr:.2f}" if corr > 0 else f"{corr:.2f}"
            t = r["ticker"]
            n = ticker_name(t)
            lag = r["best_lag"]
            sec = r["セクター"]
            s3m_r  = r.get("score_3m",  0.0) or 0.0
            s12m_r = r.get("score_12m", 0.0) or 0.0
            tag = r.get("tag") or _classify_tag(s3m_r, 0.0, s12m_r)
            badge_html = TAG_BADGE_HTML.get(tag, "")
            conv = r.get("convergence_score")
            conv_str = f'<span style="font-size:12px;color:#888;margin-left:8px;">収束:{conv:.2f}</span>' if conv is not None else ""
            html2 += f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                 padding:10px 0;border-bottom:1px solid #2a2a2a;">
              <div>
                <a href="?page=detail&ticker={t}&sector={sec}" target="_self"
                   style="font-size:13px;font-weight:700;color:#4d9fff;text-decoration:none;">{t}</a>
                <span style="font-size:12px;color:#cccccc;margin-left:8px;">{n}</span>
                {badge_html}{conv_str}
                <div style="font-size:11px;color:#666;margin-top:3px;">
                  {r['best_macro']} → +{lag}営業日ラグ
                </div>
              </div>
              <div style="font-size:18px;font-weight:700;color:{corr_color};min-width:52px;text-align:right;">
                {corr_str}
              </div>
            </div>"""
        html2 += "</div>"
    else:
        html2 += '<div style="color:#666;font-size:12px;padding:8px 0;">分析データがありません</div></div>'

    st.html(html2)

    if all_signals:
        csv_rows = []
        for r in all_signals[:10]:
            csv_rows.append({
                "ticker": r["ticker"],
                "sector": r["セクター"],
                "best_macro": r["best_macro"],
                "best_lag": r["best_lag"],
                "best_corr": r["best_corr"],
                "score_3m": r.get("score_3m", ""),
                "score_6m": r.get("score_6m", ""),
                "score_12m": r.get("score_12m", ""),
                "convergence_score": r.get("convergence_score", ""),
                "tag": r.get("tag", ""),
            })
        csv_data = pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 TOP10 CSVダウンロード",
            data=csv_data,
            file_name="jisame_signals.csv",
            mime="text/csv",
        )

    if not all_signals:
        if st.button("▶ 分析実行"):
            with st.spinner("分析中..."):
                from analysis import analyze_all
                analyze_all()
            st.rerun()

    # ── ③ セクター別相関ヒートマップ ────────────────────
    with st.container(border=True):
        st.html("""
        <div style="font-size:11px;font-weight:600;color:#888;letter-spacing:0.5px;
             display:flex;align-items:center;gap:8px;margin-bottom:8px;">
          <span style="background:#2a2a2a;border:1px solid #333;border-radius:50%;
               width:18px;height:18px;display:inline-flex;align-items:center;
               justify-content:center;font-size:10px;color:#aaa;">③</span>
          セクター別相関ヒートマップ
        </div>
        """)

        TOP_SECTOR = "注目セクター"
        sector_list = [TOP_SECTOR] + list(SECTORS.keys())
        if "selected_sector" not in st.session_state:
            st.session_state["selected_sector"] = TOP_SECTOR

        selected_sector = st.session_state["selected_sector"]
        cols = st.columns([1] * len(sector_list))
        for i, (col, sector) in enumerate(zip(cols, sector_list)):
            is_selected = sector == selected_sector
            if col.button(sector, key=f"sector_{i}",
                          use_container_width=True,
                          type="primary" if is_selected else "secondary"):
                st.session_state["selected_sector"] = sector
                st.rerun()

        # ヒートマップ描画
        HEATMAP_MACROS = ("ドル円", "原油", "VIX", "銅先物", "フィラデルフィア半導体", "米10年金利")
        hm_macros = tuple(m for m in HEATMAP_MACROS if m in MACRO_INDICATORS)

        sector_results = [] if selected_sector == TOP_SECTOR else analysis.get(selected_sector, [])

        with st.spinner("計算中..."):
            if selected_sector == TOP_SECTOR:
                y_labels, matrix = get_top_heatmap(hm_macros, period_days)
            else:
                y_labels, matrix = get_heatmap(selected_sector, hm_macros, period_days)

        if matrix:
            fig_hm = go.Figure(go.Heatmap(
                z=matrix, x=list(hm_macros), y=y_labels,
                colorscale=[
                    [0, "#c62828"], [0.35, "#4a1a1a"],
                    [0.5, "#2d2d2d"], [0.65, "#1a3a2a"], [1, "#1b8040"],
                ],
                zmin=-1, zmax=1, showscale=True,
                xgap=2, ygap=2,
                colorbar=dict(thickness=10, len=0.8,
                    tickfont=dict(color="#888", size=10),
                    bgcolor="#1e1e1e", bordercolor="#2a2a2a"),
            ))
            hm_height = len(y_labels) * 44 + 120
            fig_hm.update_layout(
                autosize=False,
                height=hm_height,
                margin=dict(l=0, r=10, t=8, b=0),
                plot_bgcolor="#181818", paper_bgcolor="#1e1e1e",
                xaxis=dict(
                    side="top",
                    tickfont=dict(color="#cccccc", size=12),
                    tickangle=0,
                    showgrid=False,
                ),
                yaxis=dict(
                    tickfont=dict(color="#cccccc", size=12),
                    autorange="reversed",
                    showgrid=False,
                ),
            )
            st.plotly_chart(fig_hm, use_container_width=True, height=hm_height,
                            config={"displayModeBar": False})
        else:
            st.markdown('<div style="color:#555;font-size:12px;padding:8px 0;">データなし</div>',
                        unsafe_allow_html=True)

    # ── ④ 銘柄詳細ドリルダウン ──────────────────────────
    html4 = section_header("④", "銘柄詳細ドリルダウン",
                           badge="銘柄選択で展開",
                           badge_color="#1a2a1a", badge_text_color="#4caf84",
                           badge_border="#26a69a44")

    if "drill_ticker" in st.session_state and st.session_state["drill_ticker"]:
        dt = st.session_state["drill_ticker"]
        ds = st.session_state.get("drill_sector", selected_sector)
        html4 += f'<div style="font-size:12px;color:#4d9fff;font-weight:600;margin-bottom:8px;">{dt} {ticker_name(dt)}</div>'
        html4 += "</div>"
        st.html(html4)

        detail = get_ticker_detail(dt, ds)
        if detail and detail.get("signals"):
            signals = detail["signals"]
            sig = signals[0]
            lag_data = sig["lag_corrs"]
            lags = list(map(int, lag_data.keys()))
            corrs = list(lag_data.values())
            best_lag_v = sig["lag"]
            bar_colors = ["#4d9fff" if l == best_lag_v else "#1a3a5c" for l in lags]
            fig_d = go.Figure(go.Bar(x=lags, y=corrs, marker_color=bar_colors))
            fig_d.update_layout(
                xaxis_title="ラグ（営業日）", yaxis_title="相関係数",
                height=200, margin=dict(l=0, r=0, t=0, b=0),
                plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                xaxis=dict(gridcolor="#2a2a2a", color="#666"),
                yaxis=dict(range=[-1, 1], gridcolor="#2a2a2a", color="#666"),
            )
            fig_d.add_hline(y=0, line_color="#444", line_width=0.8)
            st.caption(f"ラグ相関: {sig['macro_name']}")
            st.plotly_chart(fig_d, use_container_width=True)
    else:
        html4 += """
        <div style="text-align:center;padding:24px 0;color:#555;">
          <div style="font-size:13px;margin-bottom:4px;">ヒートマップの銘柄をクリックすると</div>
          <div style="font-size:11px;">ラグ相関グラフ・移動相関・過去検証が展開されます</div>
        </div></div>"""
        st.html(html4)

    # ヒートマップクリックで詳細展開（銘柄リスト）
    if sector_results:
        with st.expander("銘柄リストから選択", expanded=False):
            for r in sector_results:
                t = r["ticker"]
                c1, c2, c3 = st.columns([3, 3, 1])
                with c1:
                    st.markdown(f'<a href="?page=detail&ticker={t}&sector={selected_sector}" target="_self"'
                                f' style="color:#4d9fff;font-size:12px;font-weight:600;text-decoration:none;">{t}</a>'
                                f'<span style="color:#aaa;font-size:11px;margin-left:6px;">{ticker_name(t)}</span>',
                                unsafe_allow_html=True)
                with c2:
                    corr = r["best_corr"]
                    cc = "#26a69a" if corr > 0 else "#ef5350"
                    st.markdown(f'<span style="font-size:11px;color:#666;">{r["best_macro"]}</span>'
                                f'<span style="font-size:12px;color:{cc};margin-left:8px;font-weight:600;">'
                                f'{corr:+.2f}</span>', unsafe_allow_html=True)
                with c3:
                    if st.button("展開", key=f"drill_{t}", use_container_width=True):
                        st.session_state["drill_ticker"] = t
                        st.session_state["drill_sector"] = selected_sector
                        st.rerun()
