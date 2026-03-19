import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.financial_analyzer import financial_analyzer

async def test_news():
    print("Testing get_stock_news...")
    result = await financial_analyzer.get_stock_news('AAPL')
    print("Result:", result)
    print("Type:", type(result))
    
    if result:
        print("Number of articles:", len(result))
        if len(result) > 0:
            print("First article:", result[0])
    
    print("Testing get_stock_sentiment...")
    result = await financial_analyzer.get_stock_sentiment('AAPL')
    print("Result:", result)

if __name__ == "__main__":
    asyncio.run(test_news())