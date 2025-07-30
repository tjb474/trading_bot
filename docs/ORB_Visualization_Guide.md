# Enhanced ORB Backtest Visualization

This guide explains how to use the new enhanced Opening Range Breakout (ORB) visualization that shows ORB rectangles with trade entry/exit points.

## Features

The enhanced ORB visualization provides:

1. **ORB Rectangles**: Purple semi-transparent boxes showing the opening range for each trading day
2. **Range Size Labels**: Dollar amount of each opening range displayed in the center of rectangles
3. **Breakout Signals**: Blue triangles (↑) for bullish breakouts, red triangles (↓) for bearish breakouts
4. **Trade Entry Points**: Green circles (●) for long entries, orange circles (●) for short entries
5. **Trade Exit Points**: Green X marks for profitable exits, red X marks for loss exits
6. **Comprehensive Legend**: Clear identification of all visual elements

## Automatic Integration

When you run a backtest with the `ml_open_range_breakout` strategy, the enhanced visualization will automatically be created:

```bash
python main.py backtest ml_open_range_breakout
```

The system will:
- Detect that you're using an ORB strategy
- Automatically capture trade entry/exit data during the backtest
- Generate the enhanced visualization instead of the standard backtrader plot
- Save the chart as a PNG file with the naming pattern: `orb_backtest_results_[strategy]_[start_date]_to_[end_date].png`

## Manual Visualization

You can also create ORB visualizations manually using the `create_backtest_orb_chart` function:

```python
from viz.backtest_orb_visualization import create_backtest_orb_chart

# Your OHLC data with features
price_data = your_dataframe_with_features

# Your trades data (list of dictionaries)
trades_data = [
    {
        'entry_time': '2024-06-10 10:32:00',
        'entry_price': 445.20,
        'direction': 1,  # 1 for long, -1 for short
        'exit_time': '2024-06-10 12:15:00',
        'exit_price': 447.80,
        'pnl': 260.0,
        'commission': 4.50
    },
    # ... more trades
]

# Create the visualization
create_backtest_orb_chart(
    price_data=price_data,
    trades_data=trades_data,
    range_start='09:30:00',
    range_end='10:30:00', 
    start_date='2024-06-10',
    end_date='2024-06-13',
    timeframe='5min',
    save_path='my_orb_analysis.png'
)
```

## Configuration

The ORB visualization uses your strategy configuration from `config.yaml`:

```yaml
ml_open_range_breakout:
  range:
    start: "09:30:00"  # Opening range start time
    end: "10:30:00"    # Opening range end time
```

## Chart Elements Explained

### ORB Rectangles
- **Purple semi-transparent boxes** that span from range start to range end time
- **Height** represents the price range (high to low) during the opening period
- **Width** represents the time duration of the opening range
- **Label** in the center shows the dollar size of the range

### Breakout Signals
- **Blue triangles pointing up (↑)**: Bullish breakouts when price exceeds the opening range high
- **Red triangles pointing down (↓)**: Bearish breakouts when price falls below the opening range low

### Trade Markers
- **Green circles (●)**: Long trade entries
- **Orange circles (●)**: Short trade entries  
- **Green X marks**: Profitable trade exits
- **Red X marks**: Losing trade exits

## Output Information

The visualization displays:

1. **Chart Title**: Shows date range and key parameters
2. **Legend**: Identifies all visual elements
3. **Console Summary**: Statistics including:
   - Total trading days analyzed
   - Number of breakout signals detected
   - Number of trades executed
   - Win rate percentage
   - Average opening range size

## Timeframe Options

You can view the data in different timeframes:
- `'1min'`: 1-minute bars (detailed view)
- `'5min'`: 5-minute bars (recommended for most analysis)
- `'15min'`: 15-minute bars (broader view)
- `'H'`: Hourly bars
- `'D'`: Daily bars

## Data Requirements

For the visualization to work properly, your data should include:
- OHLC price data with datetime index
- `breakout_direction` feature (automatically created by the feature engineering system)
- Eastern timezone timestamps (handled automatically by the data manager)

## Troubleshooting

If you encounter issues:

1. **Missing dependencies**: Install required packages:
   ```bash
   pip install mplfinance matplotlib pandas numpy
   ```

2. **No trades showing**: Ensure your strategy is actually generating trades and the trade tracking is working

3. **Empty chart**: Check that your date range contains data and that the opening range times are valid

4. **Timezone issues**: The system automatically handles timezone conversion, but ensure your data timestamps are consistent

## Example Output

A typical ORB visualization will show:
- 5-20 ORB rectangles (depending on date range)
- Multiple breakout signals per day
- Entry/exit points clearly marked
- Performance statistics in the console output

This enhanced visualization makes it much easier to:
- Validate your ORB strategy logic
- Identify patterns in breakout behavior
- Analyze trade timing and execution
- Optimize strategy parameters
- Present results to stakeholders
