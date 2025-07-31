# reports/trade_reporter.py
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import os
from tabulate import tabulate


class TradeReporter:
    """
    Comprehensive trade reporting system for backtest analysis.
    Captures detailed trade information and generates formatted reports.
    """
    
    def __init__(self, strategy_name: str = None, config_data: Dict = None):
        """
        Initialize the trade reporter.
        
        Args:
            strategy_name: Name of the trading strategy
            config_data: Configuration data from config.yaml
        """
        self.strategy_name = strategy_name or "Unknown Strategy"
        self.config_data = config_data or {}
        self.trades_data = []
        self.backtest_metadata = {}
        
    def set_backtest_metadata(self, metadata: Dict):
        """Set metadata for the backtest run."""
        self.backtest_metadata = metadata
        
    def add_trade(self, trade_data: Dict):
        """
        Add a completed trade to the report.
        
        Expected trade_data format:
        {
            'entry_time': datetime,
            'exit_time': datetime,
            'entry_price': float,
            'exit_price': float,
            'direction': int (1 for long, -1 for short),
            'size': float,
            'pnl': float,
            'commission': float,
            'orb_high': float,
            'orb_low': float,
            'orb_size': float,
            'breakout_direction': str,
            'take_profit': float,
            'stop_loss': float,
            'exit_reason': str,  # 'TP', 'SL', 'EOD', etc.
            'ml_probability': float,  # ML model probability if used
            'additional_info': dict  # Any other strategy-specific data
        }
        """
        self.trades_data.append(trade_data)
        
    def get_trade_summary_df(self) -> pd.DataFrame:
        """Generate a comprehensive DataFrame with all trade information."""
        if not self.trades_data:
            return pd.DataFrame()
            
        # Convert trades data to DataFrame
        df = pd.DataFrame(self.trades_data)
        
        # Add calculated columns
        df['Duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60  # minutes
        df['Direction_Label'] = df['direction'].map({1: 'LONG', -1: 'SHORT'})
        df['R_Multiple'] = self._calculate_r_multiple(df)
        df['Trade_Return_Pct'] = (df['pnl'] / (df['entry_price'] * df['size'])) * 100
        df['Cumulative_PnL'] = df['pnl'].cumsum()
        
        # Format datetime columns
        df['Entry_Date'] = df['entry_time'].dt.strftime('%Y-%m-%d')
        df['Entry_Time'] = df['entry_time'].dt.strftime('%H:%M:%S')
        df['Exit_Date'] = df['exit_time'].dt.strftime('%Y-%m-%d')
        df['Exit_Time'] = df['exit_time'].dt.strftime('%H:%M:%S')
        
        return df
        
    def _calculate_r_multiple(self, df: pd.DataFrame) -> pd.Series:
        """Calculate R-multiple for each trade."""
        r_multiples = []
        for _, row in df.iterrows():
            if row['direction'] == 1:  # Long
                risk = row['entry_price'] - row['stop_loss']
            else:  # Short
                risk = row['stop_loss'] - row['entry_price']
                
            if risk > 0:
                r_multiple = row['pnl'] / (risk * row['size'])
            else:
                r_multiple = 0
                
            r_multiples.append(r_multiple)
            
        return pd.Series(r_multiples)
        
    def generate_summary_stats(self) -> Dict:
        """Generate summary statistics for the backtest."""
        if not self.trades_data:
            return {}
            
        df = self.get_trade_summary_df()
        
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        # Calculate win rate by exit type (TP vs SL)
        tp_exits = len(df[df['exit_reason'] == 'TP'])
        sl_exits = len(df[df['exit_reason'] == 'SL'])
        other_exits = len(df[~df['exit_reason'].isin(['TP', 'SL'])])
        
        stats = {
            'Total Trades': total_trades,
            'Winning Trades (by PnL)': winning_trades,
            'Losing Trades (by PnL)': losing_trades,
            'Win Rate by PnL (%)': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'TP Exits': tp_exits,
            'SL Exits': sl_exits,
            'Other Exits': other_exits,
            'Win Rate by Exit Type (%)': (tp_exits / total_trades * 100) if total_trades > 0 else 0,
            'Total PnL': df['pnl'].sum(),
            'Average PnL': df['pnl'].mean(),
            'Best Trade': df['pnl'].max(),
            'Worst Trade': df['pnl'].min(),
            'Average R-Multiple': df['R_Multiple'].mean(),
            'Total Commission': df['commission'].sum(),
            'Long Trades': len(df[df['direction'] == 1]),
            'Short Trades': len(df[df['direction'] == -1]),
            'Average Trade Duration (min)': df['Duration'].mean(),
            'Profit Factor': abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()) if len(df[df['pnl'] < 0]) > 0 else float('inf')
        }
        
        return stats
        
    def print_detailed_report(self, max_trades: int = 50):
        """Print a detailed trade report to console."""
        print("=" * 100)
        print(f"TRADING BACKTEST REPORT - {self.strategy_name}")
        print("=" * 100)
        
        # Print backtest metadata
        if self.backtest_metadata:
            print("\nBACKTEST METADATA:")
            print("-" * 50)
            for key, value in self.backtest_metadata.items():
                print(f"{key}: {value}")
        
        # Print strategy configuration
        if self.config_data:
            print(f"\nSTRATEGY CONFIGURATION:")
            print("-" * 50)
            
            # Trading parameters
            trading_params = self.config_data.get('trading', {})
            if trading_params:
                print("TRADING PARAMETERS:")
                for key, value in trading_params.items():
                    print(f"  {key}: {value}")
                print()
            
            # Strategy-specific configuration
            strategy_config = self.config_data.get('strategies', {}).get(self.strategy_name, {})
            for section, params in strategy_config.items():
                print(f"{section.upper()}:")
                if isinstance(params, dict):
                    for key, value in params.items():
                        if isinstance(value, list):
                            print(f"  {key}: {', '.join(map(str, value))}")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"  {params}")
                print()
        
        # Print summary statistics
        stats = self.generate_summary_stats()
        if stats:
            print("\nSUMMARY STATISTICS:")
            print("-" * 50)
            
            # Basic trade counts
            print(f"Total Trades: {stats['Total Trades']}")
            print(f"Long Trades: {stats['Long Trades']}")
            print(f"Short Trades: {stats['Short Trades']}")
            print()
            
            # Win rates - make the distinction clear
            print("WIN RATE ANALYSIS:")
            print(f"Win Rate by PnL (actual profit): {stats['Win Rate by PnL (%)']:.2f}%")
            print(f"  └─ Winning Trades (by PnL): {stats['Winning Trades (by PnL)']}")
            print(f"  └─ Losing Trades (by PnL): {stats['Losing Trades (by PnL)']}")
            print()
            print(f"Win Rate by Exit Type (hitting TP): {stats['Win Rate by Exit Type (%)']:.2f}%")
            print(f"  └─ TP Exits: {stats['TP Exits']}")
            print(f"  └─ SL Exits: {stats['SL Exits']}")
            print(f"  └─ Other Exits: {stats['Other Exits']}")
            print()
            
            # Performance metrics
            print("PERFORMANCE:")
            print(f"Total PnL: ${stats['Total PnL']:.2f}")
            print(f"Average PnL: ${stats['Average PnL']:.2f}")
            print(f"Best Trade: ${stats['Best Trade']:.2f}")
            print(f"Worst Trade: ${stats['Worst Trade']:.2f}")
            print(f"Average R-Multiple: {stats['Average R-Multiple']:.2f}")
            print(f"Total Commission: {stats['Total Commission']:.2f}")
            print(f"Average Trade Duration (min): ${stats['Average Trade Duration (min)']:.2f}")
            print(f"Profit Factor: {stats['Profit Factor']:.2f}")
            print()
        
        # Print detailed trades table
        df = self.get_trade_summary_df()
        if not df.empty:
            print(f"\nDETAILED TRADES (showing first {min(max_trades, len(df))} trades):")
            print("-" * 100)
            
            # Select columns for display
            display_columns = [
                'Entry_Date', 'Entry_Time', 'Direction_Label', 'entry_price',
                'Exit_Date', 'Exit_Time', 'exit_price', 'orb_high', 'orb_low',
                'orb_size', 'breakout_direction', 'take_profit', 'stop_loss',
                'pnl', 'commission', 'R_Multiple', 'exit_reason'
            ]
            
            # Filter columns that exist in the DataFrame
            available_columns = [col for col in display_columns if col in df.columns]
            display_df = df[available_columns].head(max_trades)
            
            # Format numeric columns
            numeric_columns = ['entry_price', 'exit_price', 'orb_high', 'orb_low', 
                             'orb_size', 'take_profit', 'stop_loss', 'pnl', 
                             'commission', 'R_Multiple']
            
            for col in numeric_columns:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(2)
            
            print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
            
            if len(df) > max_trades:
                print(f"\n... showing {max_trades} of {len(df)} total trades")
        
        print("=" * 100)
        
    def export_to_csv(self, filename: str = None):
        """Export detailed trade data to CSV."""
        # Create reports directory if it doesn't exist
        reports_dir = "reports/exports"
        os.makedirs(reports_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{reports_dir}/trade_report_{self.strategy_name}_{timestamp}.csv"
        else:
            filename = f"{reports_dir}/{filename}"
            
        df = self.get_trade_summary_df()
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"Trade report exported to: {filename}")
        else:
            print("No trades data to export.")
            
    def export_to_html(self, filename: str = None):
        """Export detailed trade report to HTML."""
        # Create reports directory if it doesn't exist
        reports_dir = "reports/exports"
        os.makedirs(reports_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{reports_dir}/trade_report_{self.strategy_name}_{timestamp}.html"
        else:
            filename = f"{reports_dir}/{filename}"
            
        df = self.get_trade_summary_df()
        stats = self.generate_summary_stats()
        
        if df.empty:
            print("No trades data to export.")
            return
            
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Backtest Report - {self.strategy_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .stats-table {{ width: 100%; }}
                .config-table {{ width: 100%; }}
                .side-by-side {{ display: flex; justify-content: space-between; gap: 20px; }}
                .side-by-side > div {{ flex: 1; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .section-header {{ background-color: #e8f4fd; font-weight: bold; }}
                .config-header {{ background-color: #f0f8ff; font-weight: bold; }}
                h1, h2 {{ color: #333; }}
                h3 {{ color: #555; margin-top: 30px; }}
                .container {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>Trading Backtest Report</h1>
            <h2>Strategy: {self.strategy_name}</h2>
            
            <div class="side-by-side">
                <div>
                    <h3>Summary Statistics</h3>
                    <table class="stats-table">
        """
        
        # Add basic trade counts
        html_content += '<tr class="section-header"><td colspan="2">TRADE COUNTS</td></tr>'
        for key in ['Total Trades', 'Long Trades', 'Short Trades']:
            if key in stats:
                html_content += f'<tr><td><strong>{key}</strong></td><td>{stats[key]}</td></tr>'
        
        # Add win rate analysis
        html_content += '<tr class="section-header"><td colspan="2">WIN RATE ANALYSIS</td></tr>'
        win_rate_keys = [
            ('Win Rate by PnL (%)', 'Actual profit after commissions'),
            ('Winning Trades (by PnL)', ''),
            ('Losing Trades (by PnL)', ''),
            ('Win Rate by Exit Type (%)', 'Percentage hitting take profit'),
            ('TP Exits', ''),
            ('SL Exits', ''),
            ('Other Exits', '')
        ]
        
        for key, description in win_rate_keys:
            if key in stats:
                value = stats[key]
                if isinstance(value, float):
                    value_str = f"{value:.2f}%"
                else:
                    value_str = str(value)
                
                key_display = f"{key}"
                if description:
                    key_display += f" <em>({description})</em>"
                    
                html_content += f'<tr><td><strong>{key_display}</strong></td><td>{value_str}</td></tr>'
        
        # Add performance metrics
        html_content += '<tr class="section-header"><td colspan="2">PERFORMANCE METRICS</td></tr>'
        performance_keys = [
            'Total PnL', 'Average PnL', 'Best Trade', 'Worst Trade', 
            'Average R-Multiple', 'Total Commission', 'Average Trade Duration (min)', 'Profit Factor'
        ]
        
        for key in performance_keys:
            if key in stats:
                value = stats[key]
                if isinstance(value, float):
                    if 'PnL' in key or 'Trade' in key:
                        value_str = f"${value:.2f}"
                        css_class = "positive" if value > 0 else "negative" if value < 0 else ""
                    else:
                        value_str = f"{value:.2f}"
                        css_class = ""
                else:
                    value_str = str(value)
                    css_class = ""
                    
                html_content += f'<tr><td><strong>{key}</strong></td><td class="{css_class}">{value_str}</td></tr>'
            
        html_content += """
                    </table>
                </div>
                
                <div>
                    <h3>Configuration Metadata</h3>
                    <table class="config-table">
        """
        
        # Add backtest metadata
        if self.backtest_metadata:
            html_content += '<tr class="config-header"><td colspan="2">BACKTEST SETTINGS</td></tr>'
            for key, value in self.backtest_metadata.items():
                key_display = key.replace('_', ' ').title()
                html_content += f'<tr><td><strong>{key_display}</strong></td><td>{value}</td></tr>'
        
        # Add strategy configuration from config.yaml
        if self.config_data:
            strategy_config = self.config_data.get('strategies', {}).get(self.strategy_name, {})
            
            # Trading parameters
            trading_params = self.config_data.get('trading', {})
            if trading_params:
                html_content += '<tr class="config-header"><td colspan="2">TRADING PARAMETERS</td></tr>'
                for key, value in trading_params.items():
                    if key not in ['backtest_start_date', 'backtest_end_date']:  # These are in backtest metadata
                        key_display = key.replace('_', ' ').title()
                        html_content += f'<tr><td><strong>{key_display}</strong></td><td>{value}</td></tr>'
            
            # Strategy-specific configuration
            for section_name, section_data in strategy_config.items():
                if isinstance(section_data, dict):
                    section_title = section_name.replace('_', ' ').upper()
                    html_content += f'<tr class="config-header"><td colspan="2">{section_title}</td></tr>'
                    
                    for key, value in section_data.items():
                        key_display = key.replace('_', ' ').title()
                        # Handle list values (like feature_list)
                        if isinstance(value, list):
                            value_str = ', '.join(map(str, value))
                        else:
                            value_str = str(value)
                        html_content += f'<tr><td><strong>{key_display}</strong></td><td>{value_str}</td></tr>'
                else:
                    key_display = section_name.replace('_', ' ').title()
                    html_content += f'<tr><td><strong>{key_display}</strong></td><td>{section_data}</td></tr>'
        
        html_content += """
                    </table>
                </div>
            </div>
            
            <h3>Detailed Trades</h3>
        """
        
        # Add trades table
        html_content += df.to_html(index=False, table_id="trades-table")
        
        html_content += """
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
            
        print(f"HTML trade report exported to: {filename}")
        
    def save_metadata(self, filename: str = None):
        """Save backtest metadata and configuration to JSON."""
        # Create reports directory if it doesn't exist
        reports_dir = "reports/exports"
        os.makedirs(reports_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{reports_dir}/backtest_metadata_{self.strategy_name}_{timestamp}.json"
        else:
            filename = f"{reports_dir}/{filename}"
            
        metadata = {
            'strategy_name': self.strategy_name,
            'backtest_metadata': self.backtest_metadata,
            'config_data': self.config_data,
            'summary_stats': self.generate_summary_stats(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        print(f"Backtest metadata saved to: {filename}")


def create_trade_reporter_from_config(config_path: str, strategy_name: str = None) -> TradeReporter:
    """
    Create a TradeReporter instance from a config file.
    
    Args:
        config_path: Path to the config.yaml file
        strategy_name: Name of the strategy (will use active_strategy from config if not provided)
    
    Returns:
        Configured TradeReporter instance
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    if not strategy_name:
        strategy_name = config_data.get('active_strategy', 'unknown_strategy')
    
    return TradeReporter(strategy_name=strategy_name, config_data=config_data)
