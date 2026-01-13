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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    current_tab = "General"
    current_category = "General"
    
    # Define known "Top Level" tabs to distinguish from subcategories
    # Or simply: Any header that is followed by another header is a Tab?
    # Better: explicit list based on user input
    known_tabs = ["Finance/Economics", "Business", "News", "Military/Geopolitics", "Sports/Entertainment/Culture/Misc."]
    
    if not os.path.exists("markets.txt"):
        return []

    with open("markets.txt", "r") as f:
        lines = f.readlines()
        
    order_counter = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"): continue
        
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            if name in known_tabs:
                current_tab = name
                current_category = "General" # Reset category when new tab starts
            else:
                current_category = name
            continue
            
        url = line
        platform = "Kalshi" if "kalshi.com" in url else "Polymarket"
        items.append({
            "platform": platform, 
            "url": url, 
            "tab": current_tab,
            "category": current_category,
            "order": order_counter
        })
        order_counter += 1
    return items

@st.cache_data(ttl=600)
def fetch_kalshi_data(url):
    try:
        session = get_kalshi_session()
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        
        # Clean URL of query params and trailing slashes
        clean_url = url.split('?')[0].rstrip('/')
        parts = [p.upper() for p in clean_url.split('/') if p]
        
        if 'MARKETS' not in parts: return []
        idx = parts.index('MARKETS')
        # Ticker candidates: everything after 'markets'
        candidates = parts[idx+1:]
        if not candidates: return []
        
        data = None
        is_event = True
        target_domain = None
        event_ticker = None # Initialize event_ticker for later use
        
        # Try every candidate slug as either an event or a market ticker
        # Priority: last slug as event, then last slug as market, then first slug as event
        search_order = []
        # Add last candidate first for higher priority
        if candidates:
            last_candidate = candidates[-1]
            search_order.append((last_candidate, True))  # Try as Event
            search_order.append((last_candidate, False)) # Try as Market
        
        # Add all other candidates
        for c in candidates:
            if (c, True) not in search_order: search_order.append((c, True))
            if (c, False) not in search_order: search_order.append((c, False))

        for domain in ["api.kalshi.com", "api.elections.kalshi.com"]:
            for ticker, try_event in search_order:
                endpoint = "events" if try_event else "markets"
                try:
                    resp = session.get(f"https://{domain}/trade-api/v2/{endpoint}/{ticker}", headers=headers, timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        is_event = try_event
                        target_domain = domain
                        event_ticker = ticker # For reporting
                        break
                except: pass
            if data: break
            
        if not data:
            return [{"id": f"err_{url}", "event_id": f"err_ev_{url}", "name": "Err", "contract": f"Event/Market Not Found ({candidates[-1]})", "value": 0, "change_1d": 0, "change_7d": 0, "change_30d": 0, "volume": 0, "source": "Error", "url": url}]
        
        m_list = data.get('markets', []) if is_event else [data.get('market', data)]
        
        # Robust title extraction
        e_obj = data.get('event', {}) if is_event else data.get('market', data)
        base_title = e_obj.get('title') or e_obj.get('event_title') or 'Kalshi Event'
        subtitle = e_obj.get('subtitle') or e_obj.get('event_subtitle') or ''
        
        if subtitle and subtitle != base_title:
            event_title = f"{base_title}: {subtitle}"
        else:
            event_title = base_title
        
        # Extra check: if it's a single market and title is generic
        if not is_event and (event_title == 'Kalshi Event' or event_title == base_title):
            ticker_title = e_obj.get('ticker')
            if ticker_title:
                event_title = f"{event_title} ({ticker_title})"

        results = []
        now_ts = datetime.utcnow().timestamp()
        
        for m in m_list:
            if not m or not isinstance(m, dict): continue
            
            # Filter: Only open/active markets
            status = m.get('status', 'open')
            if status not in ['active', 'open']: continue
            
            # Filter: Check close time
            close_time_str = m.get('close_time')
            if close_time_str:
                try:
                    # Fix for potential isoformat errors with variable precision
                    if '.' in close_time_str:
                        # Truncate micros/nanos if necessary or just handle Z
                        base_ts = close_time_str.split('Z')[0] 
                        # Python < 3.11 fromisoformat is picky, might need simple strptime if standard format
                        # Attempt standard replacement
                        close_ts = datetime.fromisoformat(base_ts).timestamp()
                    else:
                        close_ts = datetime.fromisoformat(close_time_str.replace('Z', '+00:00')).timestamp()
                        
                    if close_ts < now_ts: continue
                except: pass

            m_ticker = m.get('ticker')
            if not m_ticker: continue

            # Determine Contract Name
            # Priority 1: Subtitle (contains the specific differentiation like "Before 2025")
            contract_name = m.get('subtitle')
            
            # Priority 2: Yes/No Subtitle (MVP/Player markets often hide names here)
            if not contract_name:
                contract_name = m.get('yes_sub_title') or m.get('no_sub_title')
            
            # Priority 3: Title (if all above missing)
            if not contract_name:
                contract_name = m.get('title')
            
            # Priority 4: Ticker Parsing (if name is generic/redundant)
            # Check if name is same as event title or contains generic "Winner?" text
            is_generic = (contract_name == base_title) or ('Winner?' in str(contract_name)) or (not contract_name)
            
            if is_generic:
                # Try to parse from ticker: KXNFLNFCCHAMP-25-SEA -> SEA
                # Split by dash, take last part if it looks like an acronym or name
                parts = m_ticker.split('-')
                if len(parts) > 1:
                    suffix = parts[-1]
                    # Map common suffixes to names if possible, or just use suffix
                    contract_name = suffix
            
            # Additional Ticker mapping for known NFL codes (optional but helpful)
            nfl_map = {
                'SEA': 'Seattle Seahawks', 'SF': 'San Francisco 49ers', 
                'CHI': 'Chicago Bears', 'LA': 'Los Angeles Rams', 
                'GB': 'Green Bay Packers', 'MIN': 'Minnesota Vikings',
                'DET': 'Detroit Lions', 'PHI': 'Philadelphia Eagles',
                'DAL': 'Dallas Cowboys', 'NYG': 'New York Giants',
                'WAS': 'Washington Commanders'
            }
            if contract_name in nfl_map:
                contract_name = nfl_map[contract_name]

            # Kalshi provides price in cents (1-99)
            val_raw = m.get('last_price', 0)
            val = float(val_raw)
            
            # Use series_ticker if available, else fallback
            series_ticker = m.get('series_ticker') or data.get('event', {}).get('series_ticker') or m.get('event_ticker') or event_ticker
            
            history = {}
            # Fetch History (pruned for performance)
            try:
                now = int(time.time())
                start_1y = now - (365 * 24 * 3600)
                hist_url = f"https://{target_domain}/trade-api/v2/series/{series_ticker}/markets/{m_ticker}/candlesticks?period_interval=1440&start_ts={start_1y}&end_ts={now}"
                h_resp = session.get(hist_url, headers=headers, timeout=5)
                if h_resp.status_code == 200:
                    for c in h_resp.json().get('candlesticks', []):
                        ts = datetime.fromtimestamp(c['end_period_ts']).isoformat()
                        p = c.get('price', {}).get('close') or c.get('yes_ask', {}).get('close') or c.get('yes_bid', {}).get('close')
                        
                        if p is not None:
                            history[ts] = float(p)
            except: pass
            
            volume = float(m.get('volume', 0))
            # Fallback for 24h volume aliases
            vol24 = float(m.get('volume_24h', m.get('volume24h', 0)))
            
            marker_id = f"kalshi_{m_ticker}"
            update_history(marker_id, val)
            if not history: history = {datetime.now().isoformat(): val}
                
            c1d, c7d, c30d = get_changes(marker_id, val, history)
            
            # Increase volume filter to prioritize "highest volume"
            if volume < 500: continue

            results.append({
                "id": marker_id,
                "event_id": f"kalshi_ev_{event_ticker}",
                "name": event_title,
                "contract": contract_name,
                "value": val,
                "change_1d": c1d,
                "change_7d": c7d,
                "change_30d": c30d,
                "volume": volume,
                "volume24h": vol24,
                "source": "Kalshi",
                "url": url,
                "history_data": history
            })
        return results
    except Exception as e:
        return [{"id": f"err_{url}", "event_id": f"err_ev_{url}", "name": "Error", "contract": f"Kalshi Logic Error for {url}: {str(e)}", "value": 0, "change_1d": 0, "change_7d": 0, "change_30d": 0, "volume": 0, "source": "Error", "url": url, "history_data": {}}]

@st.cache_data(ttl=600)
def fetch_polymarket_data(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        # Improved regex to exclude query params
        slug_match = re.search(r'event/([^/?#]+)', url) or re.search(r'market/([^/?#]+)', url)
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
        
        # Volume filter
        volume_main = float(data.get('volume', 0))
        volume24h_main = float(data.get('volume24h', data.get('volume_24h', 0)))
        
        results = []
        for m in markets_data:
            if not isinstance(m, dict) or 'question' not in m: continue
            
            # Filter: Only active markets
            if not m.get('active') or m.get('closed'): continue
            
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
            # Fallback to event-level 24h volume if market-level is missing/0
            volume24h = float(m.get('volume24h', m.get('volume_24h', 0)))
            if volume24h == 0:
                volume24h = volume24h_main

            # Heuristic for active markets: either 24h volume is present or total volume is significant
            if volume < 10000 and volume24h < 100: continue

            results.append({
                "id": marker_id,
                "event_id": f"poly_ev_{slug}",
                "name": event_title,
                "contract": m.get('groupItemTitle', m.get('question', '')),
                "value": val,
                "change_1d": c1d,
                "change_7d": c7d,
                "change_30d": c30d,
                "volume": volume,
                "volume24h": volume24h,
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
    # Just render ALL for simplicitly in the new layout, or keep tabs? 
    # User said "pull up the contract chart (as currently displayed)" so keep functionality.
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
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("<div style='color: #ffcc00; font-size: 24px; font-weight: bold; margin-bottom: 20px;'>SA <span style='color: #fff; font-weight: normal;'>Predict</span></div>", unsafe_allow_html=True)
    with c2:
        # Right aligned refresh components
        rc1, rc2 = st.columns([1, 2])
        with rc1:
            if st.button("Refresh"):
                st.rerun()
        with rc2:
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    market_configs = parse_markets_file()
    if not market_configs:
        st.info("Add URLs to markets.txt")
        return

    # Data Fetching (Parallel)
    all_items = []
    with st.spinner("Fetching Data..."):
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_config = {
                executor.submit(
                    fetch_kalshi_data if config['platform'] == "Kalshi" else fetch_polymarket_data, 
                    config['url']
                ): config for config in market_configs
            }
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    markets = future.result()
                    for m in markets:
                        m['tab'] = config['tab']
                        m['category'] = config['category']
                        m['order'] = config.get('order', 0)
                        all_items.append(m)
                except Exception as e:
                    st.error(f"Error fetching {config['url']}: {e}")

    # FILTERING LOGIC: If a contract (Event) has 6 or more sub-contracts, limit to just the top 3 by price
    # Group by event_id to apply filter
    event_groups = {}
    for item in all_items:
        eid = item['event_id']
        if eid not in event_groups: event_groups[eid] = []
        event_groups[eid].append(item)
    
    filtered_items = []
    for eid, markets in event_groups.items():
        if len(markets) >= 6:
            # Sort by value descending and take top 3
            markets.sort(key=lambda x: -x.get('value', 0))
            filtered_items.extend(markets[:3])
        else:
            filtered_items.extend(markets)
    
    # Replace all_items with filtered version
    all_items = filtered_items

    # Sort all_items by original order from markets.txt -> then by Price (Value) descending
    all_items.sort(key=lambda x: (x.get('order', 0), -x.get('value', 0)))

    # 1. TOP SUMMARY TABLE
    # Convert all items to a nice DataFrame
    df_data = []
    for m in all_items:
        df_data.append({
            "#": m.get('order', 0) + 1,
            "Ticker": m['id'].split('_')[-1][:10].upper(),
            "Contract Title": m['name'],
            "Details": m['contract'],
            "Value": m['value'],
            "1d Change": m['change_1d'],
            "7d Change": m['change_7d'],
            "30d Change": m['change_30d'],
            "24h Vol": m.get('volume24h', 0),
            "Total Vol": m['volume'],
            "Source": m['source']
        })
    
    df = pd.DataFrame(df_data)
    # Sort initially by the original file order
    df = df.sort_values("#")
    
    # Styling function for reds/greens
    def color_changes(val):
        color = '#00ff66' if val > 0 else '#ff3344' if val < 0 else '#888'
        return f'color: {color}; font-weight: bold'

    st.markdown("### Market Summary")
    
    # Display the dataframe with advanced UI components
    st.dataframe(
        df.style.map(color_changes, subset=['1d Change', '7d Change', '30d Change']),
        column_config={
            "#": st.column_config.NumberColumn("#", help="Original order from markets.txt", format="%d"),
            "Ticker": st.column_config.TextColumn("Ticker", help="Market Identifier"),
            "Value": st.column_config.NumberColumn("Price", format="%.1f%%", help="Current probability"),
            "1d Change": st.column_config.NumberColumn("1d Δ", format="%+1.1f%%"),
            "7d Change": st.column_config.NumberColumn("7d Δ", format="%+1.1f%%"),
            "30d Change": st.column_config.NumberColumn("30d Δ", format="%+1.1f%%"),
            "24h Vol": st.column_config.ProgressColumn("24h Vol", format="$%.0f", min_value=0, max_value=float(max(df['24h Vol'].max(), 1) if not df.empty else 100)),
            "Total Vol": st.column_config.NumberColumn("Total Vol", format="$%.0f"),
        },
        height=500,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.markdown("### Detailed Active Markets")

    # 2. BOTTOM DETAILED VIEW (List of Expanders)
    # Grouping by Tab -> Category -> Event
    structured_data = {}
    for item in all_items:
        tab = item.get('tab', 'General')
        cat = item['category']
        ev_id = item['event_id']
        
        if tab not in structured_data: structured_data[tab] = {}
        if cat not in structured_data[tab]: structured_data[tab][cat] = {}
        if ev_id not in structured_data[tab][cat]: structured_data[tab][cat][ev_id] = []
        
        structured_data[tab][cat][ev_id].append(item)

    tab_names = list(structured_data.keys())
    
    def format_details_row(m):
        col_w = 40
        name = m['contract']
        name_str = f"{name:<{col_w}}"
        v = f"{m['value']:>6.1f}%"
        src = f"{m['source']:>8}"
        return f"{name_str} {v} {src}"

    for tab_name in tab_names:
        st.markdown(f"<div style='font-size: 16px; font-weight: bold; color: #ff9900; margin-top: 20px; border-bottom: 1px solid #444;'>{tab_name}</div>", unsafe_allow_html=True)
        
        categories = structured_data[tab_name]
        for cat_name, events in categories.items():
            if cat_name != "General":
                st.markdown(f"<div class='bb-header' style='margin-top: 10px;'>{cat_name}</div>", unsafe_allow_html=True)
            
            for ev_id, markets in events.items():
                # We want a header for the Event if multiple markets, or just the market
                # User wants: "just the contract name, details, and a down arrow"
                # If there are multiple markets for one event, we should probably group them visually
                # But creating a clean list:
                
                # If event has multiple markets
                if len(markets) > 1:
                    st.markdown(f"<div style='margin-top:5px; color:#ccc; font-size:12px;'>{markets[0]['name']}</div>", unsafe_allow_html=True)
                    for m in markets:
                        # Anchor for potential linking
                        st.markdown(f"<div id='{m['id']}'></div>", unsafe_allow_html=True)
                        with st.expander(format_details_row(m)):
                            st.write(f"**{m['name']}** - {m['contract']}")
                            render_plotly_chart(m['id'], m['name'], m.get('history_data'))
                else:
                    # Single market
                    m = markets[0]
                    st.markdown(f"<div id='{m['id']}'></div>", unsafe_allow_html=True)
                    # Title is Event Name, Expander text is "Contract Details" (which mimics user request)
                    # Or just: Name [Details]
                    display_name = f"{m['name']} - {m['contract']}"
                    
                    with st.expander(display_name + f"   ({m['value']:.1f}%)"):
                         render_plotly_chart(m['id'], m['name'], m.get('history_data'))

if __name__ == "__main__":
    main()
