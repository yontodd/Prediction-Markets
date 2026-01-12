# Prediction Market Dashboard (Bloomberg Style)

A premium, Streamlit-based dashboard for tracking prediction markets from Polymarket and Kalshi, inspired by the Bloomberg PREDICT function.

## Features
- **Bloomberg Aesthetic**: High-contrast dark mode with signature yellow/purple headers and monospace typography.
- **Multi-Platform Support**: Fetch real-time data from Polymarket and Kalshi.
- **Dynamic Tracking**: Input URLs of specific contracts or events in a simple text file.
- **Category Grouping**: Group markets into logical categories and subcategories using bracket notation.
- **Real-time Price History**: Local persistence of prices to calculate 1D and 5D net changes.
- **Range Bars**: Visualized low/high ranges for market sentiment.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Markets**:
   Edit `markets.txt` to add your own categories and URLs. Use the following format:
   ```text
   [Main Category]
   [Subcategory]
   https://polymarket.com/event/slug
   https://kalshi.com/markets/ticker
   ```

3. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## Kalshi API Keys
If you encounter connection issues or hit rate limits with Kalshi, you can provide your API keys by creating a `.env` file:
```env
KALSHI_API_KEY_ID=your_id
KALSHI_API_KEY=your_key
```
The application will automatically use these if available.

## Project Structure
- `app.py`: Main Streamlit application and data fetching logic.
- `markets.txt`: User-configurable list of markets to track.
- `price_history.json`: Local cache for price changes (auto-generated).
- `requirements.txt`: Python dependencies.
