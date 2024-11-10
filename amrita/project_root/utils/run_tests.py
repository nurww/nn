import asyncio
from test_redis_mysql import test_redis_connection, test_mysql_connection
from test_models import test_orderbook_model
from test_rewards import test_reward_system
from test_commands import test_trial_trade
from test_logging import test_logging

async def main():
    print("Testing Redis and MySQL connections...")
    test_redis_connection()
    test_mysql_connection()
    
    print("\nTesting OrderBook Model...")
    test_orderbook_model()
    
    print("\nTesting Reward System...")
    test_reward_system()
    
    print("\nTesting Trial Trade Command...")
    await test_trial_trade()
    
    print("\nTesting Logging...")
    test_logging()
    
if __name__ == "__main__":
    asyncio.run(main())
