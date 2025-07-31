# Trade Reporting System

A comprehensive trade analysis and reporting system for the ORB (Opening Range Breakout) trading bot. This system captures detailed trade information during backtests and generates professional reports with all relevant trading data.

## Features

### Comprehensive Trade Tracking
- **Complete Trade Details**: Entry/exit times, prices, direction, size, P&L, commission
- **ORB-Specific Data**: Opening range high/low, range size, breakout direction
- **Risk Management**: Take profit, stop loss levels, R-multiples
- **ML Integration**: Model probabilities when ML filter is enabled
- **Exit Analysis**: Automatic classification of exit reasons (TP, SL, OTHER)

### Professional Reports
- **Console Output**: Formatted table with all trade details
- **CSV Export**: For spreadsheet analysis and further processing
- **HTML Export**: Professional web-based reports with styling
- **JSON Metadata**: Complete backtest configuration and summary statistics

### Summary Statistics
- Win rate, total P&L, best/worst trades
- Long vs short trade analysis
- Average trade duration and R-multiples
- Profit factor and commission totals
- Strategy-specific metrics

## Installation

Install required dependencies:

```bash
python install_requirements.py
```

Or manually install:
```bash
pip install pandas tabulate pyyaml
```

## Quick Start

The trade reporting system is automatically integrated into the backtester. When you run a backtest, it will:

1. **Automatically capture** all trade data during the backtest
2. **Display a detailed report** in the console after completion
3. **Export files** with timestamps for later analysis

### Basic Usage

```python
from trading.backtester import Backtester
from common.config import load_config

# Load configuration
config = load_config('config.yaml')

# Run backtest (trade reporter is automatically integrated)
backtester = Backtester(config)
trade_reporter = backtester.run(test_data)

# The report is automatically generated and exported
```

### Manual Usage

You can also use the trade reporter independently:

```python
from reports.trade_reporter import create_trade_reporter_from_config

# Create reporter from config
reporter = create_trade_reporter_from_config('config.yaml')

# Add trade manually
trade_data = {
    'entry_time': datetime.now(),
    'exit_time': datetime.now() + timedelta(hours=1),
    'entry_price': 500.0,
    'exit_price': 505.0,
    'direction': 1,  # Long
    'size': 10,
    'pnl': 50.0,
    'commission': 2.50,
    'orb_high': 502.0,
    'orb_low': 498.0,
    'orb_size': 4.0,
    'breakout_direction': 'BULLISH',
    'take_profit': 504.0,
    'stop_loss': 496.0,
    'exit_reason': 'TP',
    'ml_probability': 0.75
}

reporter.add_trade(trade_data)

# Generate reports
reporter.print_detailed_report()
reporter.export_to_csv()
reporter.export_to_html()
```

## Report Structure

### Console Report

```
================================================================================
TRADING BACKTEST REPORT - ml_open_range_breakout
================================================================================

BACKTEST METADATA:
--------------------------------------------------
strategy: ml_open_range_breakout
start_date: 2025-01-01
end_date: 2025-06-30
total_bars: 78000
initial_cash: 100000
stake_size: 10
commission_spread_points: 0.8

STRATEGY CONFIGURATION:
--------------------------------------------------
RANGE:
  start: 09:30:00
  end: 09:45:00

RISK:
  take_profit_multiplier: 1.0
  stop_loss_multiplier: 1.1

MODEL:
  use_ml_filter: false
  probability_threshold: 0.55

SUMMARY STATISTICS:
--------------------------------------------------
Total Trades: 25
Winning Trades: 15
Losing Trades: 10
Win Rate (%): 60.00
Total PnL: $1,250.50
Average PnL: $50.02
Best Trade: $125.75
Worst Trade: -$87.25
Average R-Multiple: 0.85
Total Commission: $62.50
Long Trades: 13
Short Trades: 12
Average Trade Duration (min): 45.25
Profit Factor: 2.15

DETAILED TRADES (showing first 25 trades):
----------------------------------------------------------------------------------------------------
| Entry_Date | Entry_Time | Direction | entry_price | Exit_Date | Exit_Time | exit_price | ...
----------------------------------------------------------------------------------------------------
| 2025-01-02 | 09:45:15   | LONG     | 500.25      | 2025-01-02| 10:30:22  | 504.75     | ...
| 2025-01-03 | 09:52:31   | SHORT    | 498.80      | 2025-01-03| 11:15:45  | 495.30     | ...
```

### Exported Files

**CSV File**: `trade_report_ml_open_range_breakout_YYYYMMDD_HHMMSS.csv`
- Spreadsheet-compatible format
- All trade details and calculated metrics
- Easy to import into Excel, Google Sheets, etc.

**HTML File**: `trade_report_ml_open_range_breakout_YYYYMMDD_HHMMSS.html`
- Professional web-based report
- Summary statistics and detailed trade table
- Color-coded profit/loss indicators
- Responsive design for viewing on any device

**JSON Metadata**: `backtest_metadata_ml_open_range_breakout_YYYYMMDD_HHMMSS.json`
- Complete configuration used for the backtest
- Summary statistics in machine-readable format
- Timestamp and version information

## Customization

### Adding Custom Fields

You can extend the trade data structure by adding fields to the `additional_info` dictionary:

```python
trade_data = {
    # ... standard fields ...
    'additional_info': {
        'market_volatility': 0.25,
        'news_event': 'FOMC',
        'custom_indicator': 1.5
    }
}
```

### Custom Export Formats

The `TradeReporter` class can be extended to support additional export formats:

```python
class CustomTradeReporter(TradeReporter):
    def export_to_excel(self, filename=None):
        """Export to Excel with multiple sheets."""
        # Implementation here
        pass
```

## Integration with ORB Strategy

The system is tightly integrated with the ORB strategy to capture:

- **Opening Range Data**: Automatically calculated and stored
- **Breakout Signals**: Direction and timing of breakouts
- **ML Probabilities**: When the ML filter is enabled
- **Risk Levels**: Dynamic TP/SL based on range size
- **Exit Classification**: Automatic detection of TP/SL hits

## Best Practices

1. **Review Reports Regularly**: Use the detailed tables to identify patterns
2. **Export for Analysis**: Use CSV exports for deeper statistical analysis
3. **Track Configuration**: JSON metadata helps reproduce successful backtests
4. **Monitor Performance**: Use summary statistics to track strategy improvements
5. **Compare Periods**: Run reports on different date ranges to assess consistency

## Troubleshooting

### Missing Dependencies
If you see import errors, run:
```bash
python install_requirements.py
```

### No Trades Captured
Ensure the strategy inherits from `BaseStrategy` and that the trade reporter is properly set:
```python
strategy.set_trade_reporter(reporter)
```

### File Export Errors
Check write permissions in the current directory and ensure sufficient disk space.

## Example Output

Run the demonstration script to see the system in action:

```bash
python example_trade_report.py
```

This will show you exactly what the reports look like with sample data.

## Configuration Compatibility

The system automatically reads your strategy configuration from `config.yaml` and includes:

- Strategy parameters (range times, multipliers, etc.)
- ML model settings (if enabled)
- Risk management parameters
- Backtest date ranges and broker settings

This ensures complete traceability of your backtest results.
