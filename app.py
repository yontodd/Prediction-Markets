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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Theme Variables */
        :root {
            --sa-bg: #fff;
            --sa-text: #1e293b;
            --sa-subtext: #475569;
            --sa-border: #e2e8f0;
            --sa-header-bg: linear-gradient(90deg, #f8fafc 0%, #f1f5f9 100%);
            --sa-accent: #d97706;
            --sa-source: #0284c7;
            --sa-price: #d97706;
            --sa-tab-active: #d97706;
            --sa-tab-text: #475569;
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --sa-bg: #0e1117;
                --sa-text: #f8fafc;
                --sa-subtext: #94a3b8;
                --sa-border: #1e293b;
                --sa-header-bg: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
                --sa-accent: #fbbf24;
                --sa-source: #38bdf8;
                --sa-price: #fbbf24;
                --sa-tab-active: #fbbf24;
                --sa-tab-text: #cbd5e1;
            }
        }

        .stApp {
            font-family: 'Inter', sans-serif !important;
        }

        /* Modern Section Headers */
        .bb-header {
            background: var(--sa-header-bg) !important;
            color: var(--sa-accent) !important;
            padding: 10px 16px !important;
            font-weight: 700 !important;
            font-size: 15px !important;
            margin-top: 32px !important;
            margin-bottom: 8px !important;
            border-left: 4px solid var(--sa-accent) !important;
            border-radius: 4px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            display: block !important;
            width: 100% !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        }
        
        .bb-subheader {
            background-color: transparent !important;
            color: var(--sa-subtext) !important;
            padding: 6px 0 !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            border-bottom: 1px solid var(--sa-border) !important;
            margin-bottom: 12px !important;
        }

        /* Chart Tabs Visibility Fix */
        button[data-baseweb="tab"] {
            padding: 10px 20px !important;
        }
        
        button[data-baseweb="tab"] p {
            font-weight: 700 !important;
            font-size: 13px !important;
            color: var(--sa-tab-text) !important;
        }
        
        button[aria-selected="true"] p {
            color: var(--sa-tab-active) !important;
        }

        /* Value Colors */
        .bb-value-pos { color: #10b981 !important; font-weight: 600 !important; }
        .bb-value-neg { color: #ef4444 !important; font-weight: 600 !important; }
        .bb-value-neutral { color: var(--sa-subtext) !important; }
        
        .market-name { color: var(--sa-text) !important; font-weight: 600 !important; font-size: 14px !important; margin-bottom: 2px !important; }
        .contract-name { color: var(--sa-subtext) !important; font-size: 11px !important; font-weight: 400 !important; }
        
        .source-tag { 
            font-size: 10px !important; 
            color: var(--sa-source) !important; 
            font-weight: 700 !important; 
            border: 1px solid var(--sa-source) !important; 
            padding: 2px 6px !important; 
            border-radius: 4px !important; 
            background: transparent !important;
        }
        
        .price-val { font-weight: 700 !important; font-size: 17px !important; color: var(--sa-price) !important; }

        /* Expander Improvements */
        .stExpander {
            border: none !important;
            border-bottom: 1px solid var(--sa-border) !important;
        }
        
        [data-testid="stExpanderDetails"] {
            border-radius: 0 0 8px 8px !important;
            padding: 20px !important;
        }

        /* Links */
        .source-url-link {
            color: var(--sa-source) !important;
            text-decoration: none !important;
            font-weight: 600 !important;
        }
        .source-url-link:hover {
            text-decoration: underline !important;
        }

        /* Modern Input/Buttons */
        .stTextInput input {
            border: 1px solid var(--sa-border) !important;
            border-radius: 8px !important;
        }
        
        .stButton button {
            background: var(--sa-header-bg) !important;
            color: var(--sa-accent) !important;
            border: 1px solid var(--sa-border) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s !important;
        }
        
        .stButton button:hover {
            border-color: var(--sa-accent) !important;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.05) !important;
        }

        /* Hide streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        
        hr {
            border-top: 1px solid var(--sa-border) !important;
            margin: 10px 0 !important;
        }
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
        return 0.0, 0.0
    
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
            # Fallback for very new markets: return the first data point if it's older than 1hr
            elif not df.empty and (now - df.iloc[0]['Date']).total_seconds() > 3600:
                return df.iloc[0]['Price']
            return None

        p1d = get_past_val(1)
        p5d = get_past_val(5)
        
        if p1d is None and len(df) > 1:
            p1d = df.iloc[0]['Price']
            
        c1d = current_price - p1d if p1d is not None else 0.0
        c5d = current_price - p5d if p5d is not None else 0.0
        return c1d, c5d
    except:
        return 0.0, 0.0

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
            return [{"id": f"err_{url}", "name": "Err", "contract": "Event Not Found", "value": 0, "source": "Error", "url": url}]
            
        series_ticker = data.get('event', {}).get('series_ticker') if is_event else data.get('market', {}).get('series_ticker')
        if not series_ticker: series_ticker = event_ticker # Fallback
        
        markets_data = data.get('markets', []) if is_event else [data.get('market', data)]
        event_title = data.get('event', {}).get('title', 'Kalshi Event') if is_event else data.get('market', {}).get('event_title', 'Kalshi Market')

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
                
            c1d, c5d = get_changes(marker_id, val, history)
            
            results.append({
                "id": marker_id,
                "name": event_title,
                "contract": m.get('title', ''),
                "value": val,
                "change_1d": c1d,
                "change_5d": c5d,
                "low_30d": m.get('floor_price', 0),
                "high_30d": m.get('cap_price', 100),
                "volume": m.get('volume', 0),
                "source": "Kalshi",
                "url": url,
                "history_data": history
            })
        return results
    except Exception as e:
        return [{"id": f"err_{url}", "name": "Error", "contract": str(e), "value": 0, "source": "Error", "url": url, "history_data": {}}]

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
        if not data: return []
        
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
                
            c1d, c5d = get_changes(marker_id, val, history)
            
            results.append({
                "id": marker_id,
                "name": event_title,
                "contract": m.get('groupItemTitle', m.get('question', '')),
                "value": val,
                "change_1d": c1d,
                "change_5d": c5d,
                "low_30d": 0,
                "high_30d": 100,
                "volume": float(m.get('volume', 0)),
                "source": "PolyM",
                "url": url,
                "history_data": history
            })
        return results
    except Exception as e:
        return [{"id": f"err_{url}", "name": "Error", "contract": str(e), "value": 0, "source": "Error", "url": url, "history_data": {}}]

# --- COMPONENTS ---

def render_range_bar(low, high, current):
    safe_current = max(low, min(high, current))
    total_range = high - low if high > low else 100
    pos = ((safe_current - low) / total_range) * 100
    
    return f"""
    <div style="display: flex; align-items: center; justify-content: space-between; font-size: 10px; color: #555; min-width: 120px;">
        <span style="color: #ff3344;">{low:.0f}</span>
        <div style="flex-grow: 1; height: 3px; background: #222; margin: 0 6px; position: relative; border-radius: 2px;">
            <div style="position: absolute; left: 0; width: 100%; height: 100%; border-left: 1px solid #444; border-right: 1px solid #444;"></div>
            <div style="position: absolute; left: {pos}%; top: -4px; width: 4px; height: 11px; background: #55aaff; border-radius: 1px; box-shadow: 0 0 5px #55aaff88;"></div>
        </div>
        <span style="color: #00ff66;">{high:.0f}</span>
    </div>
    """

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
            line=dict(color='#0ea5e9', width=2.5), # Vibrant Sky Blue
            fillcolor='rgba(14, 165, 233, 0.1)',
            hovertemplate='Price: %{y:.1f}%<br>Date: %{x}<extra></extra>'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            height=220,
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.1)', title="", range=[0, 101]),
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
    st.set_page_config(page_title="SA Prediction Markets Dashboard", layout="wide", initial_sidebar_state="collapsed")
    apply_custom_style()
    
    # Top Bar
    t1, t2, t3 = st.columns([2, 2, 1])
    with t1:
        st.markdown("<div style='font-size: 24px; font-weight: bold; color: #ffcc00; margin-top: 5px;'>SA PREDICT <span style='color: #888; font-weight: normal; font-size: 14px; margin-left: 10px;'>Dashboard</span></div>", unsafe_allow_html=True)
    with t2:
        search = st.text_input("Search Tickers...", placeholder="e.g. Fed, Trump, Rates", label_visibility="collapsed")
    with t3:
        if st.button("↻ REFRESH", use_container_width=True):
            st.rerun()

    market_configs = parse_markets_file()
    if not market_configs:
        st.info("💡 Add market URLs to `markets.txt` to get started.")
        return

    grouped_data = {}
    with st.spinner("FETCHING REAL-TIME PRICES..."):
        for config in market_configs:
            cat = config['category']
            if cat not in grouped_data: grouped_data[cat] = []
            
            if config['platform'] == "Kalshi":
                markets = fetch_kalshi_data(config['url'])
            else:
                markets = fetch_polymarket_data(config['url'])
                
            for m in markets:
                if search and search.lower() not in m['name'].lower() and search.lower() not in m['contract'].lower():
                    continue
                m['subcategory'] = config['subcategory']
                grouped_data[cat].append(m)

    # Render Categories
    for cat_name, items in grouped_data.items():
        if not items: 
            continue
        
        st.markdown(f"<div class='bb-header'>{cat_name}</div>", unsafe_allow_html=True)
        
        # Column Headers: Concept, Price, 1D, 5D, Range, Vol, Update, Src
        h_cols = st.columns([0.35, 0.1, 0.08, 0.08, 0.15, 0.1, 0.1, 0.04])
        headers = ["CONCEPT / EVENT", "PRICE", "1D", "5D", "RANGE", "VOL", "UPDATE", "SRC"]
        for col, text in zip(h_cols, headers):
            col.markdown(f"<div style='color: #888; font-size: 11px; font-weight: bold; padding: 10px 0;'>{text}</div>", unsafe_allow_html=True)
        
        current_sub = None
        for item in items:
            if item.get('subcategory') != current_sub:
                current_sub = item.get('subcategory')
                if current_sub:
                    st.markdown(f"<div class='bb-subheader'>{current_sub}</div>", unsafe_allow_html=True)
            
            # Prepare data with safe defaults
            val = item.get('value', 0)
            c1d = item.get('change_1d', 0)
            c5d = item.get('change_5d', 0)
            low_30 = item.get('low_30d', 0)
            high_30 = item.get('high_30d', 100)
            vol = item.get('volume', 0)
            source = item.get('source', 'Unknown')
            m_url = item.get('url', '#')
            m_id = item.get('id', 'unknown')
            m_name = item.get('name', 'Unknown')
            m_contract = item.get('contract', '')
            
            c1d_cls = "bb-value-pos" if c1d > 0 else "bb-value-neg" if c1d < 0 else "bb-value-neutral"
            c5d_cls = "bb-value-pos" if c5d > 0 else "bb-value-neg" if c5d < 0 else "bb-value-neutral"
            vol_str = f"{vol/1e6:.1f}M" if vol >= 1e6 else f"{vol/1e3:.0f}k" if vol >= 1e3 else str(int(vol))
            time_str = datetime.now().strftime("%H:%M")
            source_color = "#ff3344" if source == "Error" else "#55aaff"

            # Create a clean data row using st.columns
            r_cols = st.columns([0.35, 0.1, 0.08, 0.08, 0.15, 0.1, 0.1, 0.04])
            
            with r_cols[0]:
                st.markdown(f"<div class='market-name' style='line-height:1.2; font-size:13px;'>{m_name}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='contract-name' style='font-size:10px;'>{m_contract}</div>", unsafe_allow_html=True)
            
            r_cols[1].markdown(f"<div style='text-align: right; color: #fbbf24; font-weight: 700; font-size:16px; padding-top:4px;'>{val:.1f}%</div>", unsafe_allow_html=True)
            r_cols[2].markdown(f"<div style='text-align: right; padding-top:7px;' class='{c1d_cls}'>{c1d:+.1f}</div>", unsafe_allow_html=True)
            r_cols[3].markdown(f"<div style='text-align: right; padding-top:7px;' class='{c5d_cls}'>{c5d:+.1f}</div>", unsafe_allow_html=True)
            
            with r_cols[4]:
                st.markdown("<div style='padding-top:7px;'></div>", unsafe_allow_html=True)
                st.markdown(render_range_bar(low_30, high_30, val), unsafe_allow_html=True)
                
            r_cols[5].markdown(f"<div style='text-align: right; color: #64748b; padding-top:7px; font-size: 12px;'>{vol_str}</div>", unsafe_allow_html=True)
            r_cols[6].markdown(f"<div style='text-align: right; color: #475569; font-size: 11px; padding-top:7px;'>{time_str}</div>", unsafe_allow_html=True)
            r_cols[7].markdown(f"<div style='text-align: right; padding-top:7px;'><span class='source-tag'>{source}</span></div>", unsafe_allow_html=True)
            
            # Expander for chart below the row
            with st.expander(f"📊 DATA / CHART - {m_name}", expanded=False):
                if source != "Error":
                    render_plotly_chart(m_id, m_name, item.get('history_data'))
                st.markdown(f"<div style='margin-top:15px; font-size:11px; color:var(--sa-subtext);'><b>Source URL:</b> <a href='{m_url}' target='_blank' class='source-url-link'>{m_url}</a></div>", unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
