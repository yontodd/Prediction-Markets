import requests

def test_poly_clob_id(slug):
    url = f"https://gamma-api.polymarket.com/events/slug/{slug}"
    print(f"Testing Gamma for {slug}: {url}")
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            markets = data.get('markets', [])
            if markets:
                m = markets[0]
                clob_id = m.get('clobTokenIds')
                print(f"Market: {m.get('question')}")
                print(f"CLOB Token IDs: {clob_id}")
                if clob_id and isinstance(clob_id, list):
                    # Usually it's a list, the first one is 'Yes' probably.
                    # Wait, Polymarket uses token IDs for Yes and No.
                    # We want the Yes one.
                    return clob_id[0]
    except Exception as e:
        print(f"Error: {e}")
    return None

if __name__ == "__main__":
    cid = test_poly_clob_id("will-trump-acquire-greenland-before-2027")
    if cid:
        # Now test historical for this ID
        url = f"https://clob.polymarket.com/prices-history?market={cid}&interval=1d&fidelity=1440"
        print(f"Testing CLOB History: {url}")
        resp = requests.get(url)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            history = resp.json().get('history', [])
            print(f"History: {len(history)}")
            if history: print(f"Sample: {history[0]}")
