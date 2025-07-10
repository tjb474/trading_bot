This is a reddit thread.

I’ve commented briefly on some other posts mentioning this approach, and there usually seems to be some interest so I figured it would be good to make a full post.

There is a lot of misunderstanding and misconceptions about how to use machine learning for algo trading, and unrealistic expectations for what it’s capable of.

I see many people asking about using machine learning to predict price, find a strategy, etc. However, this is almost always bound to fail - machine learning is NOT good at creating its own edge out of nowhere (especially LLM’s, I see that a lot too. They’ll just tell you what it thinks you want to hear. They’re an amazing tool, but not for that purpose.)

ML will not find patterns by itself from candlesticks or indicators or whatever else you just throw at it (too much noise, it can't generalize well).

A much better approach for using machine learning is to have an underlying strategy that has an existing edge, and train a model on the results of that strategy so it learns to filter out low quality trades. The labels you train on could be either the win / loss outcomes of each trade (binary classification, usually the easiest), the pl distribution, or any metric you want, but this means it’s a supervised learning problem instead of unsupervised, which is MUCH easier, especially when the use case is trading. The goal is for the model to AMPLIFY your strategies existing edge.

Finding an edge -> ml bad

Improving an existing edge -> ml good

Introduction
Meta labeling was made popular by Marco Lopez de Prado (head of Abu Dhabi Investment fund). I highly recommend his book “Advances in Financial Machine Learning” where he introduces the method. It is used by many funds / individuals and has been proven to be effective, unlike many other ml applications in trading.

With meta labeling, instead of trying to forecast raw market movements, you run a primary strategy first — one that you’ve backtested and know already has at least a small edge and a positive expectancy. The core idea is that you separate the signal generation and the signal filtering. The primary signal is from your base strategy — for example, a simple trend-following or mean-reversion rule that generates all potential trade entry and exit times. The meta label is a machine learning model that predicts whether each individual signal should be taken or skipped based on features available at the time.

Example: your primary strategy takes every breakout, but many breakouts fail. The meta model learns to spot conditions where breakouts tend to fail — like low volatility or no volume expansion — and tells you to skip those. This keeps you aligned with your strategy’s logic while cutting out the worst trades. In my experience, my win rate improves anywhere from 1-3% (modest but absolutely worth it - don’t get your hopes up for a perfect strategy). This has the biggest impact on drawdowns, allowing me to withstand downturns better. This small % improvement can be the difference between losing money with the strategy or never needing to work again.

Basic Workflow
1.	Run Your Primary Strategy
Generate trade signals as usual. Log each signal with entry time, exit time, and resulting label you will assign to the trade (i.e. win or loss). IMPORTANT - for this dataset, you want to record EVERY signal, even if you’re already in a trade at the time. This is crucial because the ML filter may skip many trades, so you don’t know whether you would have really been in a trade at that time or not. I would recommend having AT LEAST 1000 trades for this. The models need enough data to learn from. The more data the better, but 5000+ is where I start to feel more comfortable.

2.	Label the Signals
Assign a binary label to each signal: 1 if the trade was profitable above a certain threshold, 0 if not. This becomes your target for the meta model to learn / predict. (It is possible to label based on pnl distribution or other metrics, but I’d highly recommend starting with binary classification. Definitely easiest to implement to get started and works great.) A trick I like to use is to label a trade as a loser also if it took too long to play out (> n bars for example). This emphasizes the signals that followed through quickly to the model.

3.	Gather Features for Each Signal
For every signal, collect features that were available at the time of entry. (Must be EXACTLY at entry time to ensure no data leakage!) These might include indicators, price action stats, volatility measures, or order book features.

4.	Train the Meta Model
Use these features and labels to train a classifier that predicts whether a new signal will be a win or loss (1 or 0). (More about this below)

5.	Deploy
In live trading, the primary strategy generates signals as usual, but each signal is passed through the trained meta model filter, along with the features the model uses. Only signals predicted with over a certain confidence level are executed.

Feature Engineering Tips:
•	Use diverse feature types: combine price-based, volume-based, volatility-based, order book, and time-based features to capture different market dimensions. Models will learn better this way.

•	Prioritize features that stay relevant over time; markets change, so test for non-stationarity and avoid features that decay fast.

•	Track regime shifts: include features that hint at different market states (trend vs. chop, high vs. low volatility).

•	Use proper feature selection: methods like RFECV, mutual information, or embedded model importance help drop useless or redundant features.

•	Always verify that features are available at signal time — no future data leaks.
Modeling Approaches:
It’s important to balance the classes in the models. I would look up how to do this if your labels are not close to 50-50, there is plenty of information out there on this as it’s not unique to meta labeling.

Don’t rely on just one ML model. Train several different types — like XGBoost, Random Forest, SVM, or plain Logistic Regression — because each picks up different patterns in your features. Use different feature sets and tune hyperparameters for each base model to avoid all of them making the same mistakes.

Once you have these base models, you can use their individual predictions (should be probabilities from 0-1) to train an ensemble method to make the final prediction. A simple Logistic Regression works well here: it takes each base model’s probability as input and learns how to weight them together.

Calibrate each base model’s output first (with Platt scaling or isotonic regression) so their probabilities actually reflect real-world hit rates. The final ensemble probability gives you a more reliable confidence score for each signal — which you can use to filter trades or size positions more effectively.

I’d recommend making a calibration plot (image 2) to see if your ensemble is accurate (always on out-of-fold test sets of course). If it is, you can choose the confidence threshold required to take a trade when you go live. If it’s not, it can still work, but you may not be able to pick a specific threshold (would just pick > 0.5 instead).

Backtesting Considerations + Common Mistakes
When testing, always compare the meta-labeled strategy to the raw strategy. Look for improvements in average trade return, higher Sharpe, reduced drawdown, and more stable equity curves. Check if you’re filtering out too many good trades — too aggressive filtering can destroy your edge. Plotting the equity and drawdown curves on the same plot can help visualize the improvement (image 1). This is done by making one out of sample (discussed later) prediction for every trade, and using those predictions on each trade to reconstruct your backtest results (this removes trades that the model said to skip from your backtest results).

An important metric that I would try to optimize for is the precision model. This is the percentage of trades the model predicted as winners that were actually winners.

Now to the common mistakes that can completely ruin this whole process, and make your results unreliable and unusable. You need to be 100% sure that you prevent/check for these issues in your code before you can be confident in and trust the results.

Overfitting: This happens when your model learns patterns that aren’t real — just noise in your data. It shows perfect results on your training set and maybe even on a single test split, but fails live because it can’t generalize.

To prevent this, use a robust cross validation technique. If your trades are IID (look this up to see if it applies to you), use nested cross-validation. It works like this:

•	You split your data into several folds.

•	The outer loop holds out one fold as a true test set — this part never sees any model training or tuning.

•	The inner loop splits the remaining folds again to tune hyperparameters and train the model.

•	After tuning, you test the tuned model on the untouched outer fold. The only thing you use the current outer fold for is these predictions!
This way, your final test results come from data the model has never seen in any form — no leakage. This is repeated n times for n folds, and if your results are consistent across all test folds, you can be much more confident it is not overfit (never can be positive though until forward testing).

If your trades are not IID, use combinatorial purged cross-validation instead. It’s stricter: it removes overlapping data points between training and testing folds that could leak future info backward. This keeps the model from “peeking” at data it wouldn’t have in real time.

The result: you get a realistic sense of how your meta model will perform live when you combine the results from each outer fold — not just how well it fits past noise.

Data Leakage: This happens when your model accidentally uses information it wouldn’t have in real time. Leakage destroys your backtest because the model looks smarter than it is.

Classic examples: using future price data to build features, using labels that peek ahead, or failing to time-align indicators properly.

To prevent it:

•	Double-check that every feature comes only from information available at the exact moment your signal fires. (Labels are the only thing that is from later). 

•	Lag your features if needed — for example, don’t use the current candle’s close if you couldn’t have known it yet.

•	Use strict walk-forward or combinatorial purged cross-validation to catch hidden leaks where training and test sets overlap in time.
A leaked model might show perfect backtest results but will break down instantly in live trading because it’s solving an impossible problem with information you won’t have.

These two will be specific to your unique set ups, just make sure to be careful and keep them in mind.

Those are the two most important, but here’s some others:

•	Unstable Features: Features that change historically break your model. Test features for consistent distributions over time. 

•	Redundant Features: Too many similar features confuse the model and add noise. Use feature selection to drop what doesn’t help. It may seem like the more features you throw at it the better, but this is not true.

•	Too Small Sample Size: Too few trades means model can’t learn, and you won’t have enough data for accurate cross validation.

•	Ignoring Costs: Always include slippage, fees, and real fills. (Should go without saying)
Closing Thoughts:

Meta labeling doesn’t create an edge from nothing — it sharpens an edge you already have. If your base strategy is random, filtering it won’t save you. But if you have a real signal, a well-built meta model can boost your risk-adjusted returns, smooth your equity curve, and cut drawdowns. Keep it simple, test honestly, and treat it like a risk filter, not a crystal ball.

Images explained: I am away from my computer right now so sorry the images are the clearest, they’re what I had available. Let me try to explain them.

This shows the equity curve and drawdown as a % of final value for each backtest. The original strategy with no meta labeling applied is blue, and the ensemble model is green. You can see the ensemble ended with a similar profit as the original model, but its drawdowns were far lower. You could leverage higher each trade while staying within the same risk to increase profits, or just keep the lower risk.

This plot shows the change in average trade values (expected per trade) on the y-axis, and the win rate on the x-axis. Each point is a result from an outer test fold, each using different seeds to randomize shuffling, training splits, etc. This lets you estimate the confidence interval that the true improvement from the meta labeling model lies in. In this case, you can see it is 95% confident the average trade improvement is within the green shaded area (average of $12.03 higher per trade), and the win rate (since I used wins/losses as my labels!) increase is within the yellow shaded area (average of 2.94% more accurate).

Example of how a calibration plot may look for the ensemble model. Top horizontal dashed line is the original win rate of the primary models strategy. Lower dashed line is the win rate from the filtered labels based on win/loss and time threshold I used (must have won quicker than n bars…). You can see the win rate for the ensemble model in the green and blue lines, choosing a threshold over either dashed line signifies a win % improvement at that confidence level!

-----

Q&A:

Q1:
Read your post with great attention, very valuable, thanks! If I had awards, I'd give you one.

I've been experimenting with ML for some time now - and am currently deep down the rabbithole. But, I am still pretty new to this so I have some follow up questions:

- On average, how many features are you training your models with?
- Do you use just OHLCV data or do you complement with ticker/orderbook or other datasets?
- Do you have any ALGO's in production, what do returns (& drawdowns ;-)) look like (feel free not to answer) ;-)
- If you have any in production, how? Personally I am considering running them on my homelab Kubernetes cluster for now - ideally colocated at some point. I ask because perhaps there is a way smarter approach
- How do you design your pipeline, any tips? I find myself starting completely from scratch for every strat I come up with which isnt very effective
- Where do you source reliable historic data (for cheap ;-))? Particularly tick and orderbook data seems pretty expensive if you need a bunch of it
- I notice you do not mention deep learning at all, what is your experience with that? LSTM/GRU etc? Or are metalabeled sets usually too small to consider deep learning?
- In your experience, how much performance loss might you expect if you compare bench results (backtesting on unseen test data or walkforward) compared to production?
- Can you give any guidance on realistic slippage settings on ES and NQ?

Any feedback is greatly appreciated man!

A1:
I use six base models, and the features used range from 20-70 for each. Some with the lower range, some with higher range.

I use mbo data (level 3) from Databento, so I have some orderbook features, as well as some macro based features from FRED.

just two in production right now, one for NQ one for ES. Both around a 1.15 profit factor and 2.5 calmar ratio.

I have them coded in Ninjatrader. Definitely better solutions out there but it works for what I need.

modulate the parts are typically repeated across strategies, then just reuse those parts for the new ones.

I use Databento, it’s probably the best source but a little pricey. Kibot has pretty good quality and much cheaper - good alternative.

personally I haven’t tried deep learning other than some basic ANN’s. I feel like it’s overkill for many applications, and trading is one of them. YMMV though.

ideally, if you set everything up right, you will see 0 performance loss compared to your bench results. But you need to give it enough time to play out as well (be sure you did everything 100% correct though, otherwise you’ll be waiting forever).

For ES, 1 tick for entries and exits is usually more than enough. For NQ, 4 ticks is more reasonable. However I use the mbo data, so I can simulate slippage in real time based on the order book, which I would highly recommend.

Q2:

Amazing, thanks a lot!

If you don't mind, a couple more :-)

- You mentioned applying ML only to capitalize on an existing Edge ... but .. what are your minimum requirements before even considering applying ML?
- How often do you retrain your models in production? Is this time based or do you monitor some metric?
- I am considering using non-binary labels, so as opposed to winner/loser: predict a profit/loss quantile or some risk adjusted metric ... do you have any experience with this?
- The six models you describe, do they all have the same purpose or do you create specialized models for different purposes?
- How are you simulating slippage? Are you replaying the mbo data tick by tick?
- How are you getting your predictions (Python I assume) into Ninjatrader?

A2:

No problem! Also this first one is a really good question. I typically look for a strategy that was profitable in over ~80% of the years I backtested on, and always do at least five years. Mainly looking for a profit factor of at least 1.07, over ~2500 total trades (enough to be statistically significant), and a max drawdown that’s no higher than the average yearly profit. I’m confident in the results from my backtest engine since it models slippage and fees accurately, so these would each be a bit higher otherwise.

For the rest:

I retrain weekly. Technically every day there’s a new signal would be ideal, but if your dataset already has thousands of trades, weekly or even monthly is fine.

if you use non-binary, what you said would be a multi-class classification problem. Certainly should work fine! Just may be a bit harder to implement. The other approach would be predicting a continuous variable, but I personally hate doing that and find it way harder. Binary or multi-class are both great though!

I guess slightly different purposes since each has their strengths and weaknesses, and then the ensemble attempts to detect these and take advantage of where each is strongest.

yeah exactly, pretty much iterating through every single event. Then at trade times, you know the bid/ask prices and their sizes so you know where your fill would be / if you would get filled at all if using a limit order. If you want to do this there’s plenty of good tutorials online. I think it’s worth the effort so you can actually trust the results.

I run a python script with a socket that opens a port on my pc, then for every trade signal on Ninjatrader I send the features needed to the script, it makes its prediction, then sends whatever action it decided back to Ninjatrader for it to handle.

Q3: 

Are you also running your trades / strategy through Ninjatrader ? Been in the process of basically doing what you have described with running all the data and features in Ninjatrader. Or are you running a separate dashboard outside of Ninjatrader to tell you when to take trades and you execute in Ninjatrader ?



A3: 

I did integrate the meta labeling output into my Ninjatrader strategy - I was actually pretty surprised it’s able to do this. You can use a socket to open a port on your pc, and run a python script separately to listen on that port. Then at each of the base strategies signals on Ninjatrader, it sends an array of all the feature values at that time to your Python script, then uses the trained model to make a prediction on that array, and sends the output back to Ninjatrader. Then do whatever on Ninjatrader based on that result.

This is nice because it lets you do any preprocessing or that kind of stuff all on Python which in my opinion is much more flexible / easy to use than ninjascript, and then handle all the strategy based logic and entries / exits on Ninjatrader. The latency can be a bit annoying depending on what your doing in the Python script, but it should still be well under 1 second still total which has been fine for me.

Q4: 

I'm currently working on a 6 stack ensemble of XGB, lightGBM, GBRT, HGB, CatBoost, and ADA - with a lightGBM meta trainer. When it comes to your pipeline, there are so many viable variations. Comes down to feature engineering, Optuna hyper-parameter tuning, train time, and how well you make sure not to have any look-ahead bias, data leakage, etc. I enjoyed your post, thanks.


A4:

Thank you, and those are good choices! I really like HGB and CatBoost too, they complement the other models well.

Q5:

Great post thanks for this detailed write up! A question for you, how many assets are you trading at once and at what frequency? You mention it is good to have 5000+ trades to feel comfortable so just wondering how you generally hit that.

Also do you train a new ensemble for each asset or generalize it across assets with a specific strategy?

A5:

Thank you! And for my use case, I use it for my day trading strategies on ES and NQ, a separate model for each though. They generate about 3-6 signals a day, so over 5+ years of backtests I get to around that 5000 number.

You could use the same primary base strategy on multiple instruments/assets to increase the number if your strategy doesn’t take as many trades, but for this you would always want to include a categorical feature that represents the symbol the signal is for.

Q6:

Thanks for sharing, Existing edge could mean literally annotating times when a trade would've given 5 or 1 tick gain ? And using that as label


A6:

By existing edge I mean more like a backtest that over a large number of individual trades, there is a consistent trend upwards in net profit. That could be a mix of different gains, but over time it shows repeatable results. But the underlying strategy should have at least 1000+ trades to have some confidence that the results are actually statistically significant.

Unless you’re simulating slippage perfectly, I’d aim to use it on strategies that have at least 2-3 tick wins at a minimum. Every single trade gets its own individual label based on the metric you’re using, even the losing trades - they’re just as important for the model to learn what to avoid!

When deciding how to label each, you could label the winners (1) as trades with > 1 tick profit or > 5 tick profit based on your example. It’s up to your discretion for what you want the model to interpret as a “good trade” that it should be looking to take.