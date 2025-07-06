import time
import joblib
from trading_ig import IGService
import config
from data_handler import get_ig_historical_data
from trading_strategy import generate_signal
from ml_model import create_features, get_prediction

def run_trader():
    ig_service = IGService(
        config.IG_USERNAME, 
        config.IG_PASSWORD, 
        config.IG_API_KEY, 
        config.IG_ACC_TYPE
    )
    ig_service.create_session()
    model = joblib.load(config.MODEL_PATH)
    print("Trader connected to IG. Awaiting signals...")
    while True:
        open_positions = ig_service.fetch_open_positions()
        position_for_epic = open_positions[open_positions['epic'] == config.EPIC]
        in_position = not position_for_epic.empty
        data = get_ig_historical_data(ig_service, config.EPIC, resolution='D', num_points=config.LONG_WINDOW + 2)
        technical_signal = generate_signal(data, config.SHORT_WINDOW, config.LONG_WINDOW)
        print(f"Technical Signal: {technical_signal}")
        if technical_signal == 'BUY' and not in_position:
            print("Buy signal detected. Consulting ML model...")
            live_features_df = create_features(data.tail(21))
            live_features = live_features_df[['returns', 'volatility']].iloc[[-1]]
            profit_prob = get_prediction(model, live_features)
            print(f"ML Model Profit Probability: {profit_prob:.2f}")
            if profit_prob > config.PROBABILITY_THRESHOLD:
                print("Placing long position.")
                ig_service.create_open_position(
                    epic=config.EPIC,
                    direction='BUY',
                    currency_code='GBP',
                    order_type='MARKET',
                    size=config.STAKE_SIZE,
                    force_open=True,
                    guaranteed_stop=False,
                    stop_distance=config.STOP_DISTANCE,
                    limit_distance=config.LIMIT_DISTANCE
                )
        elif technical_signal == 'SELL' and in_position:
            if position_for_epic.iloc[0]['direction'] == 'BUY':
                print("Sell signal detected. Closing position.")
                deal_id = position_for_epic.iloc[0]['dealId']
                ig_service.close_position(deal_id=deal_id, direction='SELL', order_type='MARKET', size=config.STAKE_SIZE)
        else:
            print("Holding or no signal.")
        time.sleep(60 * 15)

if __name__ == '__main__':
    run_trader()
