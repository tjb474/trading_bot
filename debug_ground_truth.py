"""
Ground Truth Debug System for Opening Range Calculations

This script creates a comprehensive debugging framework to identify the source
of discrepancies between backtest and notebook Opening Range calculations.
"""

import pandas as pd
import databento as db
import pytz
from datetime import datetime, time
import json
import os
import sys

class ORGroundTruth:
    def __init__(self):
        self.debug_data = {}
        self.target_date = '2025-06-18'
        self.eastern = pytz.timezone('US/Eastern')
        
    def load_dbn_data(self):
        """Load and process DBN data exactly as notebook does"""
        print("📊 Loading DBN data...")
        
        # Load DBN file
        dbn_path = 'spy_ohlcv_20190102_20250710.dbn'
        if not os.path.exists(dbn_path):
            print(f"❌ DBN file not found: {dbn_path}")
            return None
            
        spy_data = db.DBNStore.from_file(dbn_path)
        df = spy_data.to_df()
        
        # Apply timezone conversion
        if df.index.tz is not None:
            df.index = df.index.tz_convert(self.eastern)
        
        self.debug_data['dbn_source'] = {
            'file_path': dbn_path,
            'total_rows': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'timezone': str(df.index.tz),
            'columns': list(df.columns)
        }
        
        return df
    
    def load_csv_data(self):
        """Load CSV data if available for comparison"""
        print("📈 Checking for CSV data...")
        
        csv_path = 'SPY_1min_full.csv'
        if not os.path.exists(csv_path):
            print(f"⚠️  CSV file not found: {csv_path}")
            return None
            
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            
            # Debug: Check what columns we actually have
            print(f"📋 CSV columns found: {list(df.columns)}")
            print(f"📋 Sample data:")
            print(df.head(2))
            
            # Standardize column names - handle different naming conventions
            column_mapping = {}
            
            # Map common column name variations to standard names
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['open', 'o']:
                    column_mapping[col] = 'open'
                elif col_lower in ['high', 'h']:
                    column_mapping[col] = 'high' 
                elif col_lower in ['low', 'l']:
                    column_mapping[col] = 'low'
                elif col_lower in ['close', 'c']:
                    column_mapping[col] = 'close'
                elif col_lower in ['volume', 'v', 'vol']:
                    column_mapping[col] = 'volume'
            
            if len(column_mapping) < 4:  # Need at least OHLC
                print(f"❌ CSV file doesn't have required OHLC columns")
                print(f"   Found mappable columns: {column_mapping}")
                return None
                
            # Rename columns to standard format
            df = df.rename(columns=column_mapping)
            
            # Add volume column if missing
            if 'volume' not in df.columns:
                df['volume'] = 0  # Default volume
                print("⚠️  Added default volume column (all zeros)")
            
            # Apply timezone if needed
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert(self.eastern)
            elif df.index.tz != self.eastern:
                df.index = df.index.tz_convert(self.eastern)
                
            self.debug_data['csv_source'] = {
                'file_path': csv_path,
                'total_rows': len(df),
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'timezone': str(df.index.tz),
                'original_columns': list(pd.read_csv(csv_path, nrows=1).columns),
                'mapped_columns': list(df.columns),
                'column_mapping': column_mapping
            }
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading CSV file: {e}")
            return None
    
    def calculate_or_multiple_methods(self, df, date_str):
        """Calculate OR using multiple methods to identify differences"""
        print(f"🔍 Calculating OR for {date_str} using multiple methods...")
        
        try:
            day_data = df.loc[date_str]
        except KeyError:
            print(f"❌ No data found for date {date_str}")
            return {'error': f'No data for {date_str}'}
        
        if len(day_data) == 0:
            print(f"❌ Empty dataset for date {date_str}")
            return {'error': f'Empty data for {date_str}'}
            
        results = {}
        
        # Verify we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in day_data.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return {'error': f'Missing columns: {missing_cols}'}
        
        # Method 1: pd.Timestamp slice (notebook method)
        or_start = pd.Timestamp(f"{date_str} 09:30:00", tz=self.eastern)
        or_end = pd.Timestamp(f"{date_str} 10:15:00", tz=self.eastern)
        
        try:
            method1_data = day_data.loc[or_start:or_end]
        except Exception as e:
            print(f"⚠️  Method 1 failed: {e}")
            method1_data = pd.DataFrame()
        
        results['method1_timestamp_slice'] = {
            'name': 'pd.Timestamp slice (notebook)',
            'or_high': method1_data['high'].max() if len(method1_data) > 0 else None,
            'or_low': method1_data['low'].min() if len(method1_data) > 0 else None,
            'bar_count': len(method1_data),
            'start_time': str(or_start),
            'end_time': str(or_end),
            'first_bar': str(method1_data.index[0]) if len(method1_data) > 0 else None,
            'last_bar': str(method1_data.index[-1]) if len(method1_data) > 0 else None
        }
        
        # Method 2: between_time (backtest method)
        or_start_time = time(9, 30)
        or_end_time = time(10, 15)
        
        try:
            method2_data = day_data.between_time(or_start_time, or_end_time, inclusive='left')
        except Exception as e:
            print(f"⚠️  Method 2 failed: {e}")
            method2_data = pd.DataFrame()
        
        results['method2_between_time'] = {
            'name': 'between_time (backtest)',
            'or_high': method2_data['high'].max() if len(method2_data) > 0 else None,
            'or_low': method2_data['low'].min() if len(method2_data) > 0 else None,
            'bar_count': len(method2_data),
            'start_time': f"{date_str} 09:30:00",
            'end_time': f"{date_str} 10:15:00",
            'first_bar': str(method2_data.index[0]) if len(method2_data) > 0 else None,
            'last_bar': str(method2_data.index[-1]) if len(method2_data) > 0 else None
        }
        
        # Method 3: String-based filtering
        try:
            method3_data = day_data[
                (day_data.index.time >= or_start_time) & 
                (day_data.index.time <= or_end_time)
            ]
        except Exception as e:
            print(f"⚠️  Method 3 failed: {e}")
            method3_data = pd.DataFrame()
        
        results['method3_time_filter'] = {
            'name': 'time filter',
            'or_high': method3_data['high'].max() if len(method3_data) > 0 else None,
            'or_low': method3_data['low'].min() if len(method3_data) > 0 else None,
            'bar_count': len(method3_data),
            'start_time': f"{date_str} 09:30:00",
            'end_time': f"{date_str} 10:15:00",
            'first_bar': str(method3_data.index[0]) if len(method3_data) > 0 else None,
            'last_bar': str(method3_data.index[-1]) if len(method3_data) > 0 else None
        }
        
        # Store the raw data for each method for detailed comparison
        results['raw_data'] = {
            'method1_bars': method1_data.to_dict('index') if len(method1_data) <= 100 else f"Too many bars ({len(method1_data)})",
            'method2_bars': method2_data.to_dict('index') if len(method2_data) <= 100 else f"Too many bars ({len(method2_data)})",
            'method3_bars': method3_data.to_dict('index') if len(method3_data) <= 100 else f"Too many bars ({len(method3_data)})"
        }
        
        return results
    
    def search_for_backtest_values(self, df, target_high=599.74, target_low=598.76):
        """Search entire dataset for the specific values backtest is reporting"""
        print(f"🎯 Searching for backtest values: High={target_high}, Low={target_low}")
        
        # Search for exact matches
        high_matches = df[df['high'] == target_high]
        low_matches = df[df['low'] == target_low]
        
        # Search for bars that have both values (in case OR spans multiple bars)
        combined_search = df[
            (df['high'] >= target_high - 0.01) & (df['high'] <= target_high + 0.01) &
            (df['low'] >= target_low - 0.01) & (df['low'] <= target_low + 0.01)
        ]
        
        results = {
            'target_high': target_high,
            'target_low': target_low,
            'high_exact_matches': len(high_matches),
            'low_exact_matches': len(low_matches),
            'combined_matches': len(combined_search),
            'high_match_dates': high_matches.index.strftime('%Y-%m-%d %H:%M:%S').tolist() if len(high_matches) <= 20 else f"Too many ({len(high_matches)})",
            'low_match_dates': low_matches.index.strftime('%Y-%m-%d %H:%M:%S').tolist() if len(low_matches) <= 20 else f"Too many ({len(low_matches)})",
            'combined_match_dates': combined_search.index.strftime('%Y-%m-%d %H:%M:%S').tolist() if len(combined_search) <= 20 else f"Too many ({len(combined_search)})"
        }
        
        # Check if any matches are in our target date
        target_date_high = high_matches[high_matches.index.date == pd.Timestamp(self.target_date).date()]
        target_date_low = low_matches[low_matches.index.date == pd.Timestamp(self.target_date).date()]
        
        results['target_date_analysis'] = {
            'date': self.target_date,
            'high_matches_on_date': len(target_date_high),
            'low_matches_on_date': len(target_date_low),
            'high_times_on_date': target_date_high.index.strftime('%H:%M:%S').tolist(),
            'low_times_on_date': target_date_low.index.strftime('%H:%M:%S').tolist()
        }
        
        return results
    
    def analyze_data_integrity(self, df, date_str):
        """Check for data quality issues"""
        print(f"🔎 Analyzing data integrity for {date_str}...")
        
        day_data = df.loc[date_str]
        
        # Check for missing minutes
        expected_trading_minutes = []
        start_time = pd.Timestamp(f"{date_str} 09:30:00", tz=self.eastern)
        end_time = pd.Timestamp(f"{date_str} 16:00:00", tz=self.eastern)
        
        current = start_time
        while current <= end_time:
            expected_trading_minutes.append(current)
            current += pd.Timedelta(minutes=1)
        
        missing_minutes = []
        for expected in expected_trading_minutes:
            if expected not in day_data.index:
                missing_minutes.append(expected)
        
        # Check for data anomalies
        zero_volume = day_data[day_data['volume'] == 0]
        identical_ohlc = day_data[
            (day_data['open'] == day_data['high']) & 
            (day_data['high'] == day_data['low']) & 
            (day_data['low'] == day_data['close'])
        ]
        
        integrity_report = {
            'total_bars': len(day_data),
            'expected_bars': len(expected_trading_minutes),
            'missing_bars': len(missing_minutes),
            'missing_times': [str(t) for t in missing_minutes[:10]],  # First 10 only
            'zero_volume_bars': len(zero_volume),
            'identical_ohlc_bars': len(identical_ohlc),
            'price_range': {
                'daily_high': day_data['high'].max(),
                'daily_low': day_data['low'].min(),
                'daily_open': day_data.iloc[0]['open'],
                'daily_close': day_data.iloc[-1]['close']
            }
        }
        
        return integrity_report
    
    def generate_ground_truth_report(self):
        """Generate comprehensive ground truth report"""
        print("🏗️  Generating Ground Truth Report...")
        print("=" * 60)
        
        # Load data sources
        dbn_df = self.load_dbn_data()
        csv_df = self.load_csv_data()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'target_date': self.target_date,
            'data_sources': self.debug_data
        }
        
        # Analyze DBN data
        if dbn_df is not None:
            print(f"\n📊 ANALYZING DBN DATA SOURCE")
            report['dbn_analysis'] = {
                'or_calculations': self.calculate_or_multiple_methods(dbn_df, self.target_date),
                'backtest_value_search': self.search_for_backtest_values(dbn_df),
                'data_integrity': self.analyze_data_integrity(dbn_df, self.target_date)
            }
        
        # Analyze CSV data if available
        if csv_df is not None:
            print(f"\n📈 ANALYZING CSV DATA SOURCE")
            try:
                report['csv_analysis'] = {
                    'or_calculations': self.calculate_or_multiple_methods(csv_df, self.target_date),
                    'backtest_value_search': self.search_for_backtest_values(csv_df),
                    'data_integrity': self.analyze_data_integrity(csv_df, self.target_date)
                }
            except Exception as e:
                print(f"❌ Error analyzing CSV data: {e}")
                report['csv_analysis'] = {'error': str(e)}
        
        # Cross-reference analysis
        if dbn_df is not None and csv_df is not None:
            print(f"\n🔄 CROSS-REFERENCE ANALYSIS")
            report['cross_reference'] = self.compare_data_sources(dbn_df, csv_df)
        
        # Save report
        report_file = f'ground_truth_report_{self.target_date.replace("-", "")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Report saved to: {report_file}")
        
        # Print summary
        self.print_summary(report)
        
        return report
    
    def compare_data_sources(self, dbn_df, csv_df):
        """Compare DBN and CSV data sources"""
        date_str = self.target_date
        
        # Get day data from both sources
        dbn_day = dbn_df.loc[date_str]
        csv_day = csv_df.loc[date_str]
        
        comparison = {
            'dbn_bar_count': len(dbn_day),
            'csv_bar_count': len(csv_day),
            'bar_count_diff': len(dbn_day) - len(csv_day),
            'price_differences': {},
            'sample_comparison': {}
        }
        
        # Compare overlapping timestamps
        common_times = dbn_day.index.intersection(csv_day.index)
        
        if len(common_times) > 0:
            dbn_common = dbn_day.loc[common_times]
            csv_common = csv_day.loc[common_times]
            
            price_diffs = {
                'open_diff_max': abs(dbn_common['open'] - csv_common['open']).max(),
                'high_diff_max': abs(dbn_common['high'] - csv_common['high']).max(),
                'low_diff_max': abs(dbn_common['low'] - csv_common['low']).max(),
                'close_diff_max': abs(dbn_common['close'] - csv_common['close']).max(),
            }
            comparison['price_differences'] = price_diffs
            
            # Sample comparison for first few bars
            sample_times = common_times[:5]
            sample_comp = {}
            for t in sample_times:
                dbn_bar = dbn_day.loc[t]
                csv_bar = csv_day.loc[t]
                sample_comp[str(t)] = {
                    'dbn': dbn_bar.to_dict(),
                    'csv': csv_bar.to_dict(),
                    'differences': {
                        'open': abs(dbn_bar['open'] - csv_bar['open']),
                        'high': abs(dbn_bar['high'] - csv_bar['high']),
                        'low': abs(dbn_bar['low'] - csv_bar['low']),
                        'close': abs(dbn_bar['close'] - csv_bar['close'])
                    }
                }
            comparison['sample_comparison'] = sample_comp
        
        return comparison
    
    def print_summary(self, report):
        """Print a human-readable summary of the ground truth report"""
        print("\n" + "="*60)
        print("📋 GROUND TRUTH SUMMARY")
        print("="*60)
        
        # DBN Analysis Summary
        if 'dbn_analysis' in report:
            dbn = report['dbn_analysis']
            print(f"\n🎯 DBN DATA SOURCE RESULTS:")
            
            or_calc = dbn['or_calculations']
            for method_key, method_data in or_calc.items():
                if method_key != 'raw_data':
                    high = method_data['or_high']
                    low = method_data['or_low']
                    bars = method_data['bar_count']
                    print(f"   {method_data['name']}: High=${high:.2f}, Low=${low:.2f}, Bars={bars}")
            
            # Backtest value search
            search = dbn['backtest_value_search']
            print(f"\n🔍 BACKTEST VALUE SEARCH (599.74/598.76):")
            print(f"   High matches: {search['high_exact_matches']}")
            print(f"   Low matches: {search['low_exact_matches']}")
            print(f"   On target date: {search['target_date_analysis']['high_matches_on_date']} high, {search['target_date_analysis']['low_matches_on_date']} low")
        
        # CSV Analysis Summary (if available)
        if 'csv_analysis' in report:
            print(f"\n📈 CSV DATA SOURCE: Available for comparison")
        else:
            print(f"\n📈 CSV DATA SOURCE: Not available")
        
        # Conclusions
        print(f"\n🎯 KEY FINDINGS:")
        if 'dbn_analysis' in report:
            dbn_methods = report['dbn_analysis']['or_calculations']
            method1 = dbn_methods['method1_timestamp_slice']
            method2 = dbn_methods['method2_between_time']
            
            if abs(method1['or_high'] - method2['or_high']) < 0.01:
                print(f"   ✅ OR calculation methods are consistent")
                print(f"   📊 Notebook/Backtest should show: High=${method1['or_high']:.2f}, Low=${method1['or_low']:.2f}")
                print(f"   ⚠️  If backtest shows 599.74/598.76, the issue is NOT calculation method")
                print(f"   🔍 Check: Different data file, different date, or timezone issues")
            else:
                print(f"   ❌ OR calculation methods show differences!")
                print(f"   📊 This explains the discrepancy")

if __name__ == "__main__":
    # Change to data directory
    os.chdir('c:\\Users\\Tom\\workspace\\trading_bot\\data')
    
    # Generate ground truth report
    gt = ORGroundTruth()
    report = gt.generate_ground_truth_report()
