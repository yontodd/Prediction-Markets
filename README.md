# Prediction Markets Dashboard Documentation

## Overview
This application aggregates live contract data from **Kalshi** and **Polymarket** into a unified, high-performance dashboard. It features a sortable market summary table, active market filtering, and detailed historical price charts.

## Usage
Add URLs to `markets.txt` to track specific markets. The app automatically determines the platform and event details.
Grouping is determined by `[Headers]` in the text file.

## Data Methodology

### Status & Filtering
*   **Kalshi**: Retains markets with status `active` or `open`. Expired markets (where `close_time` < Now) are automatically filtered out.
*   **Polymarket**: Retains markets where `active` is True and `closed` is False.
*   **Volume Thresholds**:
    *   **Kalshi**: Minimum total volume > 500.
    *   **Polymarket**: Minimum total volume > 10,000 OR 24h volume > 100.

### Price Change Calculations (1d, 7d, 30d)
Changes are calculated as a simple difference: `Current Price - Reference Price`.

*   **Reference Time**: Calculations use a rolling window relative to the current moment (Now - 24 hours, Now - 7 days, etc.).
*   **Reference Price**: The last available price point *recorded before* that rolling timestamp.
    *   **Polymarket**: Uses **Hourly Resolution**. The change is a true **Rolling 24h Delta** (Current vs Price ~24h ago).
    *   **Kalshi**: Uses **Hourly Resolution**. The change is a true **Rolling 24h Delta** (Current vs Price ~24h ago), derived from a 45-day history window to ensure granular data availability.

### Data Normalization
*   **Prices**: All prices are normalized to a **0-100%** probability scale. Kalshi prices (often 1-99 cents) are treated as percentages.
*   **Titles**:
    *   **Kalshi**: Prioritizes the `subtitle` (e.g., "Before 2025") for distinctness. Falls back to parsing the `ticker` suffix (e.g., "SEA" from "...-SEA") if titles are generic.
    *   **Polymarket**: Uses the market question or group item title.

## Running the App
```bash
streamlit run app.py
```
