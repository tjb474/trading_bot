# To-Do List

## High Priority
- [X] Much more data - model needs more anyway
- [X] Only taking long trades - enable short trades too (direction of breakout feature will support this)
- [X] Backtesting without meta labeling, to test the base strategy
- [X] **Fix incorrect trade direction bug (June 11th case)** - Added validation and enhanced logging
- [X] **Double check entry logic: close on or above the OR threshold** - Added breakout validation
- [ ] Close the trade if TP or SL not hit by 4:55pm
- [ ] Check that the entry is realistic i.e. on close of first bar outside OR  
- [ ] TP and SL based on ATR
- [ ] Improve core strategy - see chat with gemini (regime)
- [ ] Update KPI to an R:R or % gain, or something to do with ATR
- [ ] Ensemble model
- [ ] Feature Selection

## Medium Priority
- [ ] Feature - Size of range relative to previous day/week/month - might stop situ of large OR, then fake break, massive SL
- [ ] Feature - Day of week
- [ ] Feature - Time of year
- [ ] Feature - Session Gap?
- [ ] Feature - ISM PMI - lagged N months, days since PMI release
- [ ] Feature - market regime
- [ ] Feature - is_nr4, is_nr7, is_nr20
- [ ] Feature - CBOE S&P500 volatility index - short breakouts when it's low 
- [ ] Feature - Macroeconomic from FRED
- [ ] Feature - range_rank_7_day: A continuous feature (e.g., a value from 0.0 to 1.0 representing the rank of today's range vs the last 7).
- [ ] Feature - narrowed day in X days? Was it the narrowest in 7 days, or the narrowest in 70? This "degree of narrowness" is a powerful piece of information
- [ ] Feature - to get more training data, use more tickers, and add the ticker as a feature, and trade that ticker

## Low Priority
- [ ] Class imbalance?
- [ ] Task 2
- [ ] More tickers - trade only the most likely?
- [ ] Test - Exit - After 3pm if entry price reached, close trade

## Notes
- Add any additional notes or context here.

## to check

- Jun 5th - take profit is smaller than the OR
- Jun 6th - long is triggered but price didn't rise above OR
- Jun 11th - a short is triggered but price never closed below the OR