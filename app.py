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
def apply_bloomberg_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap');
        
        /* Dark Theme Overrides */
        .stApp {
            background-color: #000000 !important;
            color: #ffffff !important;
            font-family: 'Roboto Mono', monospace !important;
        }

        [data-testid="stHeader"] {
            background-color: #000000 !important;
        }

        /* Bloomberg Table Headers */
        .bb-header {
            background-color: #333333 !important;
            color: #ffcc00 !important;
            padding: 6px 12px !important;
            font-weight: bold !important;
            font-size: 14px !important;
            margin-top: 24px !important;
            border-bottom: 2px solid #555555 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            display: block !important;
            width: 100% !important;
        }
        
        .bb-subheader {
            background-color: #1a1a2e !important;
            color: #e08aff !important;
            padding: 4px 12px !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            border-bottom: 1px solid #333344 !important;
        }

        /* Table Grid */
        .bb-table {
            width: 100% !important;
            border-collapse: collapse !important;
            margin-bottom: 20px !important;
        }
        
        .bb-table th {
            text-align: left !important;
            color: #aaaaaa !important;
            border-bottom: 1px solid #333333 !important;
            padding: 10px 12px !important;
            font-weight: normal !important;
            font-size: 12px !important;
            text-transform: uppercase !important;
        }
        
        .bb-table td {
            padding: 8px 12px !important;
            border-bottom: 1px solid #1a1a1a !important;
            vertical-align: middle !important;
            background-color: transparent !important;
        }

        .bb-value-pos { color: #00ff66 !important; }
        .bb-value-neg { color: #ff3344 !important; }
        .bb-value-neutral { color: #888888 !important; }
        
        .market-name { color: #ff9900 !important; font-weight: bold !important; font-size: 14px !important; margin-bottom: 2px !important; }
        .contract-name { color: #cccccc !important; font-size: 12px !important; }
        .source-tag { font-size: 11px !important; color: #55aaff !important; font-weight: bold !important; border: 1px solid #55aaff33 !important; padding: 1px 4px !important; border-radius: 2px !important; }
        
        .price-val { font-weight: 700 !important; font-size: 16px !important; color: #ffcc00 !important; }

        /* Sidebar/Widgets */
        .stTextInput input {
            background-color: #111 !important;
            color: #fff !important;
            border: 1px solid #333 !important;
        }
        
        .stButton button {
            background-color: #222 !important;
            color: #ffcc00 !important;
            border: 1px solid #444 !important;
        }

        /* Hide streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
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

def get_changes(marker_id, current_price):
    history = load_history().get(marker_id, {})
    
    def get_past_price(days_ago):
        target_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        dates = sorted([d for d in history.keys() if d <= target_date], reverse=True)
        return history[dates[0]] if dates else None

    p1d = get_past_price(1)
    p5d = get_past_price(5)
    
    c1d = current_price - p1d if p1d is not None else 0.0
    c5d = current_price - p5d if p5d is not None else 0.0
    
    return c1d, c5d

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
        
        event_ticker = parts[idx+1]
        market_ticker = parts[-1] if len(parts) > idx + 3 else event_ticker
        
        attempts = [
            (f"https://api.elections.kalshi.com/trade-api/v2/events/{event_ticker}", "event"),
            (f"https://api.kalshi.com/trade-api/v2/events/{event_ticker}", "event"),
            (f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}", "market"),
            (f"https://api.kalshi.com/trade-api/v2/markets/{market_ticker}", "market")
        ]
        
        data = None
        for api_url, mode in attempts:
            try:
                resp = session.get(api_url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    is_event = (mode == "event")
                    break
            except:
                continue
        
        if not data:
            return [{"id": f"err_{url}", "name": "Err", "contract": "API Fail", "value": 0, "source": "Error", "url": url}]
        
        markets_data = data.get('markets', [data.get('market', data)]) if is_event else [data.get('market', data)]
        event_title = data.get('event', {}).get('title', 'Kalshi Event') if is_event else data.get('market', {}).get('event_title', 'Kalshi Market')

        results = []
        for m in markets_data:
            if not m or not isinstance(m, dict): continue
            if 'last_price' not in m and 'title' not in m: continue
            
            val = m.get('last_price', 0)
            m_ticker = m.get('ticker', market_ticker)
            
            # Fetch Historical Candles for Chart
            history = {}
            try:
                hist_url = f"https://api.kalshi.com/trade-api/v2/markets/{m_ticker}/candles?limit=30&period_interval=1440"
                h_resp = session.get(hist_url, timeout=5)
                if h_resp.status_code == 200:
                    candles = h_resp.json().get('candles', [])
                    for c in candles:
                        ts = datetime.fromtimestamp(c['time']).strftime("%Y-%m-%d")
                        history[ts] = float(c['close'])
            except:
                pass
            
            marker_id = f"kalshi_{m_ticker}"
            update_history(marker_id, val)
            if not history: # Fallback to local history
                history = load_history().get(marker_id, {datetime.now().strftime("%Y-%m-%d"): val})
                
            c1d, c5d = get_changes(marker_id, val)
            
            results.append({
                "id": marker_id,
                "name": m.get('event_title', event_title),
                "contract": m.get('title', ''),
                "value": val,
                "buy_price": m.get('yes_ask', m.get('last_price', 0)),
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
        return []

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
            
            m_id = m.get('conditionId', m.get('id', slug))
            
            # Fetch Historical Prices for Chart
            history = {}
            try:
                hist_url = f"https://gamma-api.polymarket.com/pricehistory?market={m_id}"
                h_resp = requests.get(hist_url, timeout=5)
                if h_resp.status_code == 200:
                    points = h_resp.json()
                    for p in points:
                        ts = datetime.fromtimestamp(p['t']).strftime("%Y-%m-%d")
                        history[ts] = float(p['p']) * 100
            except:
                pass

            marker_id = f"poly_{m_id}"
            update_history(marker_id, val)
            if not history:
                history = load_history().get(marker_id, {datetime.now().strftime("%Y-%m-%d"): val})
                
            c1d, c5d = get_changes(marker_id, val)
            
            results.append({
                "id": marker_id,
                "name": event_title,
                "contract": m.get('groupItemTitle', m.get('question', '')),
                "value": val,
                "buy_price": val, # Polymarket Price is the mid/last
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
        return []

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
    history = history_data if history_data else load_history().get(marker_id, {})
    if not history or len(history) < 2:
        st.info("📈 Chart will populate once more data points are collected.")
        return
    
    # Sort dates and prepare data
    dates = sorted(history.keys())
    prices = [history[d] for d in dates]
    
    df = pd.DataFrame({"Date": dates, "Price": prices})
    df['Date'] = pd.to_datetime(df['Date'])
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Add Area Chart
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Price'],
        fill='tozeroy',
        line=dict(color='#55aaff', width=3),
        fillcolor='rgba(85, 170, 255, 0.1)',
        hovertemplate='Price: %{y:.1f}%<br>Date: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=250,
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=True, gridcolor='#222', title="Prob (%)", range=[0, 100]),
        title=dict(text=f"{name} - Price History", font=dict(color='#ffcc00', size=14))
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- MAIN APP ---

def main():
    st.set_page_config(page_title="Bloomberg | PREDICT", layout="wide", initial_sidebar_state="collapsed")
    apply_bloomberg_style()
    
    # Top Bar
    t1, t2, t3 = st.columns([2, 2, 1])
    with t1:
        st.markdown("<div style='font-size: 24px; font-weight: bold; color: #ffcc00; margin-top: 5px;'>PREDICT <span style='color: #888; font-weight: normal; font-size: 14px; margin-left: 10px;'>Prediction Markets</span></div>", unsafe_allow_html=True)
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
        
        # Column Headers [Concept, BUY, LAST, 1D, 5D, Range, Vol, Update, Src]
        h_cols = st.columns([0.3, 0.08, 0.08, 0.08, 0.08, 0.15, 0.08, 0.1, 0.05])
        headers = ["CONCEPT / EVENT", "BUY", "LAST", "1D", "5D", "RANGE", "VOL", "UPDATE", "SRC"]
        for col, text in zip(h_cols, headers):
            col.markdown(f"<div style='color: #888; font-size: 11px; font-weight: bold; padding: 10px 0;'>{text}</div>", unsafe_allow_html=True)
        
        current_sub = None
        for item in items:
            if item['subcategory'] != current_sub:
                current_sub = item['subcategory']
                if current_sub:
                    st.markdown(f"<div class='bb-subheader'>{current_sub}</div>", unsafe_allow_html=True)
            
            # Prepare data
            val = item['value']
            buy = item.get('buy_price', val)
            c1d, c5d = item['change_1d'], item['change_5d']
            c1d_cls = "bb-value-pos" if c1d > 0 else "bb-value-neg" if c1d < 0 else "bb-value-neutral"
            c5d_cls = "bb-value-pos" if c5d > 0 else "bb-value-neg" if c5d < 0 else "bb-value-neutral"
            vol = item['volume']
            vol_str = f"{vol/1e6:.1f}M" if vol >= 1e6 else f"{vol/1e3:.0f}k" if vol >= 1e3 else str(int(vol))
            time_str = datetime.now().strftime("%H:%M")
            source_color = "#ff3344" if item['source'] == "Error" else "#55aaff"

            # Create a clean data row using st.columns
            r_cols = st.columns([0.3, 0.08, 0.08, 0.08, 0.08, 0.15, 0.08, 0.1, 0.05])
            
            with r_cols[0]:
                st.markdown(f"<div class='market-name' style='line-height:1.2; font-size:13px;'>{item['name']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='contract-name' style='font-size:10px;'>{item['contract']}</div>", unsafe_allow_html=True)
            
            r_cols[1].markdown(f"<div style='text-align: right; color: #00ff66; font-weight: bold; font-size:15px; padding-top:5px;'>{buy:.1f}%</div>", unsafe_allow_html=True)
            r_cols[2].markdown(f"<div style='text-align: right; color: #ffcc00; font-weight: bold; font-size:15px; padding-top:5px;'>{val:.1f}%</div>", unsafe_allow_html=True)
            r_cols[3].markdown(f"<div style='text-align: right; padding-top:8px;' class='{c1d_cls}'>{c1d:+.1f}</div>", unsafe_allow_html=True)
            r_cols[4].markdown(f"<div style='text-align: right; padding-top:8px;' class='{c5d_cls}'>{c5d:+.1f}</div>", unsafe_allow_html=True)
            
            with r_cols[5]:
                st.markdown("<div style='padding-top:8px;'></div>", unsafe_allow_html=True)
                st.markdown(render_range_bar(item['low_30d'], item['high_30d'], val), unsafe_allow_html=True)
                
            r_cols[6].markdown(f"<div style='text-align: right; color: #ccc; padding-top:8px;'>{vol_str}</div>", unsafe_allow_html=True)
            r_cols[7].markdown(f"<div style='text-align: right; color: #666; font-size: 11px; padding-top:8px;'>{time_str}</div>", unsafe_allow_html=True)
            r_cols[8].markdown(f"<div style='text-align: right; padding-top:8px;'><span class='source-tag' style='color: {source_color}; border-color: {source_color}33;'>{item['source']}</span></div>", unsafe_allow_html=True)
            
            # Expander for chart below the row
            with st.expander(f"📊 DATA / CHART - {item['name']}", expanded=False):
                if item['source'] != "Error":
                    render_plotly_chart(item['id'], item['name'], item.get('history_data'))
                st.markdown(f"**Source URL:** [{item['url']}]({item['url']})")
            
            st.markdown("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #111;'>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
