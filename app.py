import streamlit as st
import pandas as pd
import requests
import json
import re
import os
import base64
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization

# Load environment variables
load_dotenv()

# --- CONFIG & CONSTANTS ---
CACHE_FILE = "price_history.json"

# --- KALSHI RSA SIGNER ---
class KalshiSigner:
    def __init__(self, key_id, private_key_str):
        self.key_id = key_id
        # Handle formatted/unformatted private key string
        if "-----BEGIN" not in private_key_str:
            # Try to wrap it if it looks like just the base64 part
            private_key_str = f"-----BEGIN RSA PRIVATE KEY-----\n{private_key_str}\n-----END RSA PRIVATE KEY-----"
        
        self.private_key = serialization.load_pem_private_key(
            private_key_str.encode(),
            password=None
        )

    def sign(self, method, path, timestamp):
        msg = f"{timestamp}{method}{path}"
        signature = self.private_key.sign(
            msg.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()

# --- KALSHI AUTH SESSION ---
def get_kalshi_session():
    session = requests.Session()
    
    # Priority 1: API Key/Secret (More robust)
    key_id = os.getenv("KALSHI_API_KEY_ID")
    key_secret = os.getenv("KALSHI_API_KEY")
    
    if key_id and key_secret:
        try:
            signer = KalshiSigner(key_id, key_secret)
            # Custom auth logic to inject headers on every request
            class KalshiAuth(requests.auth.AuthBase):
                def __init__(self, signer):
                    self.signer = signer
                def __call__(self, r):
                    timestamp = str(int(time.time() * 1000))
                    # Extract path for signing (without query params)
                    path = r.path_url.split('?')[0]
                    r.headers['KALSHI-ACCESS-KEY'] = self.signer.key_id
                    r.headers['KALSHI-ACCESS-SIGNATURE'] = self.signer.sign(r.method, path, timestamp)
                    r.headers['KALSHI-ACCESS-TIMESTAMP'] = timestamp
                    return r
            
            session.auth = KalshiAuth(signer)
            return session
        except Exception as e:
            st.sidebar.error(f"Signer Error: {e}")
    
    # Priority 2: Email/Pass (Simpler but session expires)
    email = os.getenv("KALSHI_EMAIL")
    password = os.getenv("KALSHI_PASSWORD")
    
    if email and password:
        try:
            login_url = "https://api.kalshi.com/trade-api/v2/login"
            resp = session.post(login_url, json={"email": email, "password": password}, timeout=10)
            if resp.status_code == 200:
                return session
        except:
            pass
    return session

# Persistent session for performance
if 'kalshi_session' not in st.session_state:
    st.session_state.kalshi_session = get_kalshi_session()

# --- STYLING ---
def apply_custom_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap');
        
        :root {
            --bb-bg: #000000;
            --bb-row-hover: #1a1a1a;
            --bb-header-bg: #1c1c1c;
            --bb-text: #ffffff;
            --bb-subtext: #cccccc;
            --bb-accent: #ff9900; /* Orange for tickers/titles */
            --bb-purple: #4a148c; /* Purple for sub-headers */
            --bb-yellow: #ffff00; /* Yellow for values */
            --bb-pos: #00ff66;
            --bb-neg: #ff3344;
            --bb-border: #333333;
        }

        .stApp {
            background-color: var(--bb-bg) !important;
            color: var(--bb-text) !important;
            font-family: 'Roboto Mono', monospace !important;
        }

        /* Bloomberg Section Headers */
        .bb-cat-header {
            background-color: #000000 !important;
            color: #ffff00 !important;
            padding: 4px 8px !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            border-bottom: 2px solid #ffff00 !important;
            margin-top: 10px !important;
            text-transform: uppercase !important;
        }
        
        .bb-subcat-header {
            background-color: #311b92 !important;
            color: #ffffff !important;
            padding: 2px 8px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
        }

        /* Table Row Styling */
        .bb-row {
            display: flex;
            align-items: center;
            border-bottom: 1px solid var(--bb-border);
            padding: 2px 0;
            transition: background 0.1s;
        }
        .bb-row:hover {
            background-color: var(--bb-row-hover) !important;
        }

        .bb-cell {
            padding: 0 4px;
            font-size: 11px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .bb-ticker { color: var(--bb-accent); font-weight: 600; }
        .bb-value { color: var(--bb-yellow); font-weight: 600; text-align: right; }
        .bb-pos { color: var(--bb-pos); text-align: right; }
        .bb-neg { color: var(--bb-neg); text-align: right; }
        .bb-vol { color: #ff00ff; text-align: right; } /* Pinkish for volume */
        .bb-source { color: #00ffff; font-size: 9px; }

        /* Expander/Dropdown Tweak */
        .stExpander {
            background-color: transparent !important;
            border: none !important;
            margin-bottom: 0px !important;
        }
        .stExpander summary {
            background-color: transparent !important;
            color: var(--bb-text) !important;
            padding: 2px 0 !important;
            border-bottom: 1px solid #1a1a1a !important;
        }
        .stExpander summary:hover {
            background-color: #111 !important;
        }
        .stExpander summary p {
            font-family: 'Roboto Mono', monospace !important;
            font-size: 11px !important;
            color: inherit !important;
            margin: 0 !important;
            white-space: pre-wrap !important; /* Allow wrapping */
            display: inline-block !important;
            width: 100% !important;
        }
        .stExpander summary svg {
            fill: #444 !important;
            margin-right: 10px !important;
        }
        
        .stExpander [data-testid="stExpanderDetails"] {
            border: none !important;
            padding: 10px 0 !important;
        }
        
        /* Remove default streamlit padding */
        [data-testid="stVerticalBlock"] > div {
            padding-top: 0px !important;
            padding-bottom: 0px !important;
        }
        
        .stMarkdown div p {
            margin-bottom: 0px !important;
        }

        /* Custom Arrow */
        .bb-arrow {
            display: inline-block;
            margin-right: 5px;
            font-size: 10px;
            color: #888;
        }

        hr { border-top: 1px solid #222; margin: 2px 0 !important; }
        </style>
    """, unsafe_allow_html=True)

# --- DATA PERSISTENCE ---

def load_history():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_history(history):
    with open(CACHE_FILE, "w") as f:
        json.dump(history, f)

def update_history(marker_id, price):
    history = load_history()
    today = datetime.now().strftime("%Y-%m-%d")
    
    if marker_id not in history:
        history[marker_id] = {}
    
    history[marker_id][today] = price
    
    # Keep only last 30 days
    cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    history[marker_id] = {d: p for d, p in history[marker_id].items() if d >= cutoff}
    
    save_history(history)

def get_changes(marker_id, current_price, history_dict=None):
    # Use provided history or fall back to local cache
    history = history_dict if history_dict else load_history().get(marker_id, {})
    if not history or len(history) < 2:
        return 0.0, 0.0, 0.0
    
    try:
        # Convert keys to datetimes and sort
        df = pd.DataFrame([{"Date": pd.to_datetime(t), "Price": p} for t, p in history.items()])
        df = df.sort_values("Date")
        
        now = datetime.now()
        
        def get_past_val(days):
            target = now - timedelta(days=days)
            # Filter for points before target
            past_points = df[df['Date'] <= target]
            if not past_points.empty:
                # Return the most recent point before target
                return past_points.iloc[-1]['Price']
            # Fallback for very new markets
            return None

        p1d = get_past_val(1)
        p7d = get_past_val(7)
        p30d = get_past_val(30)
        
        # If no data for a timeframe, use the oldest available point if it's "close enough" 
        # but for now let's just use first point as fallback for 1d if needed
        if p1d is None and len(df) > 0:
            p1d = df.iloc[0]['Price']
            
        c1d = current_price - p1d if p1d is not None else 0.0
        c7d = current_price - p7d if p7d is not None else 0.0
        c30d = current_price - p30d if p30d is not None else 0.0
        return c1d, c7d, c30d
    except:
        return 0.0, 0.0, 0.0

# --- DATA FETCHING ---

def parse_markets_file():
    items = []
    current_category = "GLOBAL MARKETS"
    current_subcategory = None
    
    if not os.path.exists("markets.txt"):
        return []

    with open("markets.txt", "r") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"): continue
        
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            is_followed_by_bracket = False
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if not next_line: continue
                if next_line.startswith("["):
                    is_followed_by_bracket = True
                break
            
            if is_followed_by_bracket:
                current_category = name
                current_subcategory = None
            else:
                current_subcategory = name
            continue
        
        platform = "Kalshi" if "kalshi.com" in line else "Polymarket" if "polymarket.com" in line else "Unknown"
        items.append({
            "category": current_category,
            "subcategory": current_subcategory,
            "url": line,
            "platform": platform
        })
    return items

def fetch_kalshi_data(url):
    session = st.session_state.kalshi_session
    try:
        parts = url.strip('/').split('/')
        if 'markets' not in parts: return []
        idx = parts.index('markets')
        if len(parts) <= idx + 1: return []
        
        event_ticker = parts[idx+1].upper()
        market_ticker = parts[-1].upper()
        
        # Try both domains and both /events/ and /markets/ endpoints
        data = None
        is_event = True
        target_domain = "api.elections.kalshi.com"
        
        for domain in ["api.elections.kalshi.com", "api.kalshi.com"]:
            # Try Event first
            try:
                resp = session.get(f"https://{domain}/trade-api/v2/events/{event_ticker}", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    is_event = True
                    target_domain = domain
                    break
            except: pass
            
            # Try Market direct
            try:
                resp = session.get(f"https://{domain}/trade-api/v2/markets/{market_ticker}", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    is_event = False
                    target_domain = domain
                    break
            except: pass
            
        if not data:
            return [{"id": f"err_{url}", "event_id": f"err_ev_{url}", "name": "Err", "contract": "Event Not Found", "value": 0, "change_1d": 0, "change_7d": 0, "change_30d": 0, "volume": 0, "source": "Error", "url": url}]
            
        series_ticker = data.get('event', {}).get('series_ticker') if is_event else data.get('market', {}).get('series_ticker')
        if not series_ticker: series_ticker = event_ticker # Fallback
        
        markets_data = data.get('markets', []) if is_event else [data.get('market', data)]
        
        # Better title extraction
        if is_event:
            event_title = data.get('event', {}).get('title') or data.get('event', {}).get('event_title') or 'Kalshi Event'
        else:
            m_obj = data.get('market', data)
            event_title = m_obj.get('event_title') or m_obj.get('title') or m_obj.get('subtitle') or 'Kalshi Market'

        results = []
        for m in markets_data:
            if not m or not isinstance(m, dict): continue
            
            val = m.get('last_price', 0)
            m_ticker = m.get('ticker')
            
            history = {}
            # Step 1: Fetch 1-Year Daily History
            try:
                now = int(time.time())
                start_1y = now - (365 * 24 * 3600)
                hist_url_daily = f"https://{target_domain}/trade-api/v2/series/{series_ticker}/markets/{m_ticker}/candlesticks?period_interval=1440&start_ts={start_1y}&end_ts={now}"
                h_resp = session.get(hist_url_daily, timeout=5)
                if h_resp.status_code == 200:
                    for c in h_resp.json().get('candlesticks', []):
                        ts = datetime.fromtimestamp(c['end_period_ts']).isoformat()
                        p = c.get('price', {}).get('close') or c.get('yes_ask', {}).get('close') or c.get('yes_bid', {}).get('close')
                        if p is not None: history[ts] = float(p)
            except: pass
            
            # Step 2: Fetch 14-Day Hourly History for better resolution
            try:
                start_14d = now - (14 * 24 * 3600)
                hist_url_hourly = f"https://{target_domain}/trade-api/v2/series/{series_ticker}/markets/{m_ticker}/candlesticks?period_interval=60&start_ts={start_14d}&end_ts={now}"
                h_resp = session.get(hist_url_hourly, timeout=5)
                if h_resp.status_code == 200:
                    for c in h_resp.json().get('candlesticks', []):
                        ts = datetime.fromtimestamp(c['end_period_ts']).isoformat()
                        p = c.get('price', {}).get('close') or c.get('yes_ask', {}).get('close') or c.get('yes_bid', {}).get('close')
                        if p is not None: history[ts] = float(p)
            except: pass
            
            marker_id = f"kalshi_{m_ticker}"
            update_history(marker_id, val)
            if not history:
                history = {datetime.now().isoformat(): val}
                
            c1d, c7d, c30d = get_changes(marker_id, val, history)
            
            # Volume filter
            volume = m.get('volume', 0)
            if volume < 2000: continue

            results.append({
                "id": marker_id,
                "event_id": f"kalshi_ev_{event_ticker}",
                "name": event_title,
                "contract": m.get('title', ''),
                "value": val,
                "change_1d": c1d,
                "change_7d": c7d,
                "change_30d": c30d,
                "low_30d": m.get('floor_price', 0),
                "high_30d": m.get('cap_price', 100),
                "volume": volume,
                "source": "Kalshi",
                "url": url,
                "history_data": history
            })
        return results
    except Exception as e:
        return [{"id": f"err_{url}", "event_id": f"err_ev_{url}", "name": "Error", "contract": str(e), "value": 0, "change_1d": 0, "change_7d": 0, "change_30d": 0, "volume": 0, "source": "Error", "url": url, "history_data": {}}]

def fetch_polymarket_data(url):
    try:
        slug_match = re.search(r'event/([^/]+)', url) or re.search(r'market/([^/]+)', url)
        if not slug_match: return []
        slug = slug_match.group(1)
        
        api_url = f"https://gamma-api.polymarket.com/events/slug/{slug}"
        resp = requests.get(api_url, timeout=10)
        if resp.status_code != 200:
             api_url = f"https://gamma-api.polymarket.com/markets/slug/{slug}"
             resp = requests.get(api_url, timeout=10)
        
        data = resp.json()
        if not data: return [{"id": f"err_{url}", "event_id": f"err_ev_{url}", "name": "Err", "contract": "Polymarket Not Found", "value": 0, "change_1d": 0, "change_7d": 0, "change_30d": 0, "volume": 0, "source": "Error", "url": url}]
        
        markets_data = data.get('markets', [data] if 'question' in data else [])
        event_title = data.get('title', data.get('question', 'Polymarket'))
        
        results = []
        for m in markets_data:
            if not isinstance(m, dict) or 'question' not in m: continue
            
            prices_raw = m.get('outcomePrices', ['0', '0'])
            if isinstance(prices_raw, str):
                try: prices = json.loads(prices_raw)
                except: prices = ['0', '0']
            else:
                prices = prices_raw
                
            try: val = float(prices[0]) * 100 if prices and len(prices) > 0 else 0
            except: val = 0
            
            history = {}
            try:
                clob_ids = json.loads(m.get('clobTokenIds', '[]')) if isinstance(m.get('clobTokenIds'), str) else m.get('clobTokenIds', [])
                if clob_ids:
                    clob_id = clob_ids[0]
                    # Fetch max history with 1-hour resolution
                    hist_url = f"https://clob.polymarket.com/prices-history?market={clob_id}&interval=max&fidelity=60"
                    h_resp = requests.get(hist_url, timeout=5)
                    if h_resp.status_code == 200:
                        for p in h_resp.json().get('history', []):
                            ts = datetime.fromtimestamp(p['t']).isoformat()
                            history[ts] = float(p['p']) * 100
            except: pass

            m_id = m.get('conditionId', m.get('id', slug))
            marker_id = f"poly_{m_id}"
            update_history(marker_id, val)
            if not history:
                history = {datetime.now().isoformat(): val}
                
            c1d, c7d, c30d = get_changes(marker_id, val, history)
            
            # Volume filter
            volume = float(m.get('volume', 0))
            if volume < 2000: continue

            results.append({
                "id": marker_id,
                "event_id": f"poly_ev_{slug}",
                "name": event_title,
                "contract": m.get('groupItemTitle', m.get('question', '')),
                "value": val,
                "change_1d": c1d,
                "change_7d": c7d,
                "change_30d": c30d,
                "low_30d": 0,
                "high_30d": 100,
                "volume": volume,
                "source": "PolyM",
                "url": url,
                "history_data": history
            })
        return results
    except Exception as e:
        return [{"id": f"err_{url}", "event_id": f"err_ev_{url}", "name": "Error", "contract": str(e), "value": 0, "change_1d": 0, "change_7d": 0, "change_30d": 0, "volume": 0, "source": "Error", "url": url, "history_data": {}}]

# --- COMPONENTS ---

def render_plotly_chart(marker_id, name, history_data=None):
    history = history_data if history_data else {}
    if not history or len(history) < 2:
        st.info("📈 Chart will populate once more data points are collected.")
        return
    
    # Sort dates and prepare dataframe
    df = pd.DataFrame([{"Date": pd.to_datetime(t), "Price": p} for t, p in history.items()])
    df = df.sort_values("Date")
    
    # Define Plotly Rendering Function
    def render_fig(data, title_suffix):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['Price'],
            fill='tozeroy',
            line=dict(color='#ff9900', width=2), # Bloomberg Orange
            fillcolor='rgba(255, 153, 0, 0.1)',
            hovertemplate='Price: %{y:.1f}%<br>Date: %{x}<extra></extra>'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            height=200,
            template='plotly_dark',
            xaxis=dict(showgrid=False, title="", tickfont=dict(color='#888', size=10)),
            yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', title="", range=[0, 101], tickfont=dict(color='#888', size=10)),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"chart_{marker_id}_{title_suffix}")

    # Tabs for different timeframes
    t1, t2, t3, t4 = st.tabs(["1D", "1W", "1M", "ALL"])
    now = datetime.now()
    
    with t1:
        d1 = df[df['Date'] >= (now - timedelta(days=1))]
        if not d1.empty: render_fig(d1, "1d")
        else: st.info("No data for 1D")
        
    with t2:
        dw = df[df['Date'] >= (now - timedelta(weeks=1))]
        if not dw.empty: render_fig(dw, "1w")
        else: st.info("No data for 1W")
        
    with t3:
        dm = df[df['Date'] >= (now - timedelta(days=30))]
        if not dm.empty: render_fig(dm, "1m")
        else: st.info("No data for 1M")
        
    with t4:
        render_fig(df, "all")

# --- MAIN APP ---

def main():
    st.set_page_config(page_title="SA Predict Dashboard", layout="wide", initial_sidebar_state="collapsed")
    apply_custom_style()
    
    # Top Bar
    st.markdown("<div style='color: #ffcc00; font-size: 24px; font-weight: bold; margin-bottom: 20px;'>SA <span style='color: #fff; font-weight: normal;'>Predict</span></div>", unsafe_allow_html=True)

    market_configs = parse_markets_file()
    if not market_configs:
        st.info("Add URLs to markets.txt")
        return

    # Data Fetching
    all_items = []
    with st.spinner("UPDATING..."):
        for config in market_configs:
            if config['platform'] == "Kalshi":
                markets = fetch_kalshi_data(config['url'])
            else:
                markets = fetch_polymarket_data(config['url'])
                
            for m in markets:
                m['category'] = config['category']
                m['subcategory'] = config['subcategory']
                all_items.append(m)

    if not all_items:
        st.warning("No data found matching criteria.")
        return

    # Grouping by Category -> Subcategory -> Event
    structured_data = {}
    for item in all_items:
        cat = item['category']
        sub = item['subcategory'] or "GENERAL"
        ev_id = item['event_id']
        
        if cat not in structured_data: structured_data[cat] = {}
        if sub not in structured_data[cat]: structured_data[cat][sub] = {}
        if ev_id not in structured_data[cat][sub]: structured_data[cat][sub][ev_id] = []
        
        structured_data[cat][sub][ev_id].append(item)

    def format_row(m, indent=False, is_header=False):
        # Layout: Name (100), Value (8), d/d (8), w/w (8), m/m (8), Vol (10), Src (10), Upd (8)
        col_w = 100
        if is_header:
            name = f"{'Concept / Market':<{col_w}}"
            v = f"{'Value':>8}"
            c1 = f"{'d/d':>8}"
            c7 = f"{'w/w':>8}"
            c30 = f"{'m/m':>8}"
            vol = f"{'Volume':>10}"
            upd = f"{'Update':>8}"
            src = f"{'Src':>10}"
            return f"  {name} {v} {c1} {c7} {c30} {vol} {upd} {src}"

        name = m['name'] if not indent else m['contract']
        if indent: 
            name = f"  {name}"
        
        # Avoid hard truncation if possible, just pad
        name_str = f"{name:<{col_w}}"
        
        v = f"{m['value']:>7.1f}%"
        c1 = f"{m['change_1d']:>+7.1f}"
        c7 = f"{m['change_7d']:>+7.1f}"
        c30 = f"{m['change_30d']:>+7.1f}"
        
        vol = m['volume']
        vol_str = f"{vol/1e6:5.1f}M" if vol >= 1e6 else f"{vol/1e3:5.0f}k" if vol >= 1e3 else f"{int(vol):>6}"
        vol_str = f"{vol_str:>10}"
        
        upd = datetime.now().strftime("%H:%M")
        upd_str = f"{upd:>8}"
        
        src = f"{m['source']:>10}"
        return f"{name_str} {v} {c1} {c7} {c30} {vol_str} {upd_str} {src}"

    # Draw Global Header
    st.markdown(f"<div style='font-family: monospace; font-size: 11px; color: #888; padding: 4px 0; border-bottom: 2px solid #333; white-space: pre;'>{format_row(None, is_header=True)}</div>", unsafe_allow_html=True)

    # Render Content
    for cat_name, subcats in structured_data.items():
        st.markdown(f"<div class='bb-cat-header'>{cat_name}</div>", unsafe_allow_html=True)
        
        for sub_name, events in subcats.items():
            st.markdown(f"<div class='bb-subcat-header'>{sub_name}</div>", unsafe_allow_html=True)
            
            for ev_id, markets in events.items():
                # If multiple markets, the "Event" is a header
                if len(markets) > 1:
                    first = markets[0]
                    # Use an expander as the header to allow collapsing the whole group
                    with st.expander(first['name'], expanded=False):
                        for m in markets:
                            with st.expander(format_row(m, indent=True)):
                                render_plotly_chart(m['id'], m['name'], m.get('history_data'))
                else:
                    m = markets[0]
                    with st.expander(format_row(m)):
                        render_plotly_chart(m['id'], m['name'], m.get('history_data'))

if __name__ == "__main__":
    main()
