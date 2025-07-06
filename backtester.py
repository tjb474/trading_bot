import backtrader as bt
import pandas as pd
import joblib
import config
from data_handler import get_ig_historical_data
from ml_model import create_features

class MLStrategy(bt.Strategy):
    params = (
        ('short_window', config.SHORT_WINDOW),
        ('long_window', config.LONG_WINDOW),
        ('model_path', config.MODEL_PATH),
        ('prob_threshold', config.PROBABILITY_THRESHOLD),
    )
    def __init__(self):
        self.short_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.short_window)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.long_window)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)
        self.model = joblib.load(self.p.model_path)
        self.df = pd.DataFrame({
            'Open': self.data.open.get(size=len(self.data)),
            'High': self.data.high.get(size=len(self.data)),
            'Low': self.data.low.get(size=len(self.data)),
            'Close': self.data.close.get(size=len(self.data)),
            'Volume': self.data.volume.get(size=len(self.data)),
        }, index=self.data.datetime.get(size=len(self.data)))
        self.features_df = create_features(self.df)
    def next(self):
        if not self.position:
            if self.crossover > 0:
                current_date = self.data.datetime.date(0)
                try:
                    live_features = self.features_df.loc[current_date.strftime('%Y-%m-%d')][['returns', 'volatility']].values.reshape(1, -1)
                    profit_prob = self.model.predict_proba(live_features)[0][1]
                    if profit_prob > self.p.prob_threshold:
                        self.buy()
                except KeyError:
                    pass
        elif self.crossover < 0:
            self.close()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MLStrategy)
    data = get_ig_historical_data(config.EPIC, '2023-01-01', '2023-12-31')
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    strat = results[0]
    print('Sharpe Ratio:', strat.analyzers.sharpe_ratio.get_analysis()['sharperatio'])
    print('Max Drawdown:', strat.analyzers.drawdown.get_analysis()['max']['drawdown'])
    cerebro.plot()
