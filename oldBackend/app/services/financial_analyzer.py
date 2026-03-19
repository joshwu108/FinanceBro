"""
Financial Analysis Service
Interprets ML predictions and provides actionable financial advice
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import openai
from dataclasses import dataclass
import yfinance as yf
from fastapi import HTTPException
from app.services.sentiment_analyzer import sentiment_analyzer
import time
import streamlit as st
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from bs4 import BeautifulSoup
from openai import OpenAI
from langchain_community.llms import OpenAI as LangChainOpenAI
from langchain_core.tools import tool, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import initialize_agent, AgentType, create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from app.services.ml_models import ml_manager
from app.services.data_collector import data_collector
from app.services.feature_engineering import feature_engineer
from langchain_openai import ChatOpenAI
import os
import torch
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
# Set up Together AI API key for DeepSeek models
if not os.getenv("DEEPSEEK_API_KEY"):
    logger.warning("DEEPSEEK_API_KEY (Together AI key) not found in environment variables")


@tool
def extract_article_text(article_url: str):
    """Extract the text of the article using an agent"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(article_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return article_text

    except Exception as e:
        logger.error(f"Error extracting article text: {str(e)}")
        return None

@tool
def analyze_sentiment(text: str) -> List[Dict]:
    """Analyze sentiment of financial texts"""
    model_configs = {
        'general': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'financial': 'ProsusAI/finbert'
    }
    model_name = model_configs.get('financial')
    model = pipeline('sentiment-analysis', model=model_name, device=0 if torch.cuda.is_available() else -1)
    try:
        if not model:
            return []
            
        if not text or len(text.strip()) == 0:
            return {'text': text, 'sentiment': 'neutral', 'score': 0.0}
                
        # Truncate text if too long
        if len(text) > 500:
            text = text[:500]
            
        result = model(text)
        return {
            'text': text[:100],
            'sentiment': result[0]['label'],
            'score': result[0]['score']
        }
            
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return []
    
@tool
def get_market_data_sync(query: str) -> str:
    """Synchronous version of market data retrieval"""
    try:
        parts = query.split()
        symbol = parts[0] if parts else "AAPL"
        import yfinance as yf
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="5d")
            
        if hist.empty:
            return f"No market data available for {symbol}"
            
        # Extract key metrics
        current_price = hist['Close'].iloc[-1]
        previous_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        price_change = current_price - previous_price
        price_change_percent = (price_change / previous_price) * 100 if previous_price != 0 else 0
            
        volume = hist['Volume'].iloc[-1]
        avg_volume = hist['Volume'].mean()
            
        # Get additional info
        market_cap = info.get('marketCap', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        dividend_yield = info.get('dividendYield', 'N/A')
            
        # Format the response
        market_cap_str = f"${market_cap:,}" if isinstance(market_cap, (int, float)) else 'N/A'
        pe_ratio_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else 'N/A'
        dividend_yield_str = f"{dividend_yield:.2%}" if isinstance(dividend_yield, (int, float)) else 'N/A'
            
        market_data = f"""
        Market Data for {symbol}:
            
        Price Information:
        - Current Price: ${current_price:.2f}
        - Price Change: ${price_change:.2f} ({price_change_percent:+.2f}%)
        - Previous Close: ${previous_price:.2f}
            
        Volume:
        - Current Volume: {volume:,}
        - Average Volume: {avg_volume:,.0f}
            
        Key Metrics:
        - Market Cap: {market_cap_str}
        - P/E Ratio: {pe_ratio_str}
        - Dividend Yield: {dividend_yield_str}
            
        Data Source: Yahoo Finance
        """
            
        return market_data
            
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {str(e)}")
        return f"Error retrieving market data for {symbol}: {str(e)}"
    
@tool
def analyze_news_sync(query: str) -> str:
    """Synchronous version of news analysis"""
    try:
        parts = query.split()
        symbol = parts[0] if parts else "AAPL"
        stock = yf.Ticker(symbol)
        
        try:
            news = stock.news
            if news:
                news_info = []
                for article in news[:5]:
                    news_info.append({
                        'title': article.get('title', ''),
                        'summary': article.get('summary', ''),
                        'url': article.get('link', ''),
                    })
                return news_info
            else:
                return f"No recent news available for {symbol}"
        except:
            return f"Could not fetch news for {symbol}"
            
    except Exception as e:
        return f"Error analyzing news for {symbol}: {str(e)}"

@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    trend: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    strength: str  # "strong", "moderate", "weak"
    duration: str  # "short-term", "medium-term", "long-term"
    key_levels: List[float]  # Support/resistance levels
    reasoning: str  # Explanation


@dataclass
class TradingAdvice:
    """Trading advice result"""
    action: str  # "buy", "sell", "hold", "wait"
    confidence: float  # 0.0 to 1.0
    risk_level: str  # "low", "medium", "high"
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""
    timeframe: str = "1-3 days"


class FinancialAnalyzer:
    """Analyzes financial data and provides trading advice"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.sentiment_analyzer = sentiment_analyzer
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.agent_prompt = f"""
        You are a financial analyst agent. Your task is to analyze online investor sentiment about a provided stock.

        Focus on identifying whether overall sentiment is positive, negative, or neutral, and extract any common themes or catalysts (e.g., earnings reports, product news, rumors, macroeconomic changes).

        Respond with a brief summary in this format:
            •	Ticker/Symbol: [e.g., TSLA]
            •	Overall Sentiment: [Positive / Negative / Neutral]
            •	Key Drivers:
            •	[e.g., “Bullish comments on Q3 earnings”]
            •	[e.g., “Rumors of upcoming partnership with Apple”]
            •	Social Volume Spike: [Yes/No, and platform if applicable]
            •	Sample Quote: “Insert a representative quote or headline here”
        Keep your summary concise, actionable, and professional.

        """
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You're a helpful financial analyst assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}"),
        ])
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        #self.memory = MemorySaver()
        self.llm = ChatOpenAI(
            #model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            model="lgai/exaone-3-5-32b-instruct",
            openai_api_base="https://api.together.xyz/v1",
            openai_api_key=os.getenv("TOGETHER_API_KEY"),  
            temperature=0.7
        )
        self.tools = [
            extract_article_text,
            analyze_sentiment,
            get_market_data_sync,
            analyze_news_sync,
        ]
        #self.agent = create_tool_calling_agent(tools=self.tools, llm=self.llm, prompt=self.prompt)
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate",
        )
        try:
            self.analyst_agent = pipeline(
                "text-generation", 
                model="gpt2-medium",  
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Could not load text generation model: {e}")
            self.analyst_agent = None
            
        try:
            self.qa_agent = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float32,
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Could not load QA model: {e}")
            self.qa_agent = None
            
        if openai_api_key:
            openai.api_key = openai_api_key
    
    async def get_stock_news(self, symbol: str, news_count: int = 10):
        """Get news for a stock with better error handling"""
        try:
            news = yf.Search(symbol, news_count).news
            processed_news = []
            for article in news[:news_count]:
                try:
                    processed_article = {
                        'title': article.get('title', ''),
                        'content': article.get('summary', ''),
                        'url': article.get('link', ''),
                        'published_date': article.get('providerPublishTime', ''),
                        'source': article.get('publisher', 'Yahoo Finance')
                    }
                    processed_news.append(processed_article)
                except Exception as e:
                    logger.warning(f"Error processing article: {e}")
                    continue
            
            return processed_news
            
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {str(e)}")
            return []  # Return empty list instead of raising exception
            
    def process_article(self, article: Dict):
        """Process the individual sentiment of the article relating to a stock"""
        article_title = article['title']
        article_url = article['url']
        article_summary = article['content']
        headline_text = f"Title: {article_title}\nSummary: {article_summary}"
        headline_sentiment = self.sentiment_analyzer.analyze_sentiment(headline_text, 'financial')

        article_full_text = ""
        if article_url:
            article_full_text = self.extract_article_text(article_url)
            time.sleep(1)
        
        article_sentiment = None
        if article_full_text:
            article_sentiment = self.sentiment_analyzer.analyze_sentiment(article_full_text, 'financial')

        return headline_sentiment, article_sentiment

    async def get_stock_sentiment(self, symbol:str, news_count: int = 5):
        """Get the sentiment of the stock"""
        news = await self.get_stock_news(symbol, news_count)
        headline_sentiment = None
        article_sentiment = None
        collective_sentiment = []
        for article in news:
            headline_sentiment, article_sentiment = self.process_article(article)
            collective_sentiment.append({
                'title': article['title'],
                'headline_sentiment': headline_sentiment['sentiment'],
                'article_sentiment': article_sentiment['sentiment'],
                'headline_score': headline_sentiment['score'],
                'article_score': article_sentiment['score']
            })
        return collective_sentiment
    
    async def get_stock_analysis(self, symbol: str):
        """Financial analysit agent using scraped data from yfinance"""
        try:
            # Use the headmaster agent for analysis
            if self.agent_executor is None:
                logger.warning("Headmaster agent not available, returning basic analysis")
                return f"Basic analysis for {symbol}: Sentiment analysis not available due to model loading issues."
            
            # Create a prompt that forces tool usage
            analysis_prompt = f"""
            You are a financial analyst. You MUST use the available tools to analyze {symbol}.
            
            Follow these steps:
            1. Use get_market_data_sync to get current market data for {symbol}
            2. Use analyze_news_sync to get recent news for {symbol}
            3. Use analyze_sentiment to analyze the sentiment of the news
            4. Use extract_article_text to extract the text of the article
            5. Based on the tool results, provide a brief analysis
            
            NEVER generate analysis that is hypothetical and not grounded on real information.
            DO NOT generate analysis without using the tools first.
            """
            
            try:
                # Try agent first
                response = self.agent_executor.run(analysis_prompt)
                logger.info(f"Successfully generated stock analysis for {symbol} using agent")
                return response
            except Exception as agent_error:
                logger.warning(f"Agent failed, using direct LLM: {agent_error}")
                # Fallback to direct LLM call with simpler prompt
                fallback_prompt = f"""
                Provide a brief financial analysis for {symbol} stock.
                Include: sentiment, key factors, and outlook.
                Keep it concise and professional.
                """
                direct_response = self.llm.invoke(fallback_prompt)
                logger.info(f"Successfully generated stock analysis for {symbol} using direct LLM")
                return direct_response.content
        except Exception as e:
            logger.error(f"Error generating stock analysis: {str(e)}")
            return f"Error generating stock analysis for {symbol}: {str(e)}"
        
    
    def extract_article_text(self, article_url: str):
        """Extract the text of the article using an agent"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(article_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            for script_or_style in soup(['script', 'style']):
                script_or_style.extract()

            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text() for p in paragraphs])
            return article_text

        except Exception as e:
            logger.error(f"Error extracting article text: {str(e)}")
            return None

@tool
def analyze_trend(
    stock_data: pd.DataFrame, 
    technical_indicators: pd.DataFrame,
    ml_prediction: float,
    symbol: str
) -> TrendAnalysis:
    """
    Analyze market trend using technical indicators and ML predictions
        
    Args:
        stock_data: OHLCV data
        technical_indicators: Calculated technical indicators
        ml_prediction: ML model prediction (0-1 probability)
        symbol: Stock symbol
        
    Returns:
        TrendAnalysis object
    """
    try:
        # Calculate trend indicators
        current_price = stock_data['close'].iloc[-1]
        sma_20 = technical_indicators['sma_20'].iloc[-1]
        sma_50 = technical_indicators['sma_50'].iloc[-1]
        rsi = technical_indicators['rsi'].iloc[-1]
        macd = technical_indicators['macd'].iloc[-1]
        macd_signal = technical_indicators['macd_signal'].iloc[-1]
            
        # Price vs moving averages
        price_above_sma20 = current_price > sma_20
        price_above_sma50 = current_price > sma_50
        sma20_above_sma50 = sma_20 > sma_50
            
        # RSI analysis
        rsi_bullish = rsi > 50
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
            
        # MACD analysis
        macd_bullish = macd > macd_signal
        macd_positive = macd > 0
            
        # Trend determination
        bullish_signals = 0
        bearish_signals = 0
            
        if price_above_sma20:
            bullish_signals += 1
        else:
            bearish_signals += 1
                
        if price_above_sma50:
            bullish_signals += 1
        else:
            bearish_signals += 1
                
        if sma20_above_sma50:
            bullish_signals += 1
        else:
            bearish_signals += 1
                
        if rsi_bullish:
            bullish_signals += 1
        else:
            bearish_signals += 1
                
        if macd_bullish:
            bullish_signals += 1
        else:
            bearish_signals += 1
                
        if macd_positive:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        # ML prediction influence
        if ml_prediction > 0.6:
            bullish_signals += 2
        elif ml_prediction < 0.4:
            bearish_signals += 2
            
        # Determine trend
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            trend = "neutral"
            confidence = 0.5
        elif bullish_signals > bearish_signals:
            trend = "bullish"
            confidence = bullish_signals / total_signals
        else:
            trend = "bearish"
            confidence = bearish_signals / total_signals
            
        # Determine strength
        if confidence > 0.7:
            strength = "strong"
        elif confidence > 0.5:
            strength = "moderate"
        else:
            strength = "weak"
            
        # Determine duration based on moving averages
        if price_above_sma50 and sma20_above_sma50:
            duration = "long-term"
        elif price_above_sma20:
            duration = "medium-term"
        else:
            duration = "short-term"
            
        # Calculate key levels
        key_levels = calculate_key_levels(stock_data, technical_indicators)
            
        # Generate reasoning
        reasoning = generate_trend_reasoning(
            trend, strength, duration, current_price, 
            sma_20, sma_50, rsi, macd, ml_prediction
        )
            
        return TrendAnalysis(
            trend=trend,
            confidence=confidence,
            strength=strength,
            duration=duration,
            key_levels=key_levels,
            reasoning=reasoning
        )
            
    except Exception as e:
        logger.error(f"Error analyzing trend: {str(e)}")
        return TrendAnalysis(
            trend="neutral",
            confidence=0.5,
            strength="weak",
            duration="short-term",
            key_levels=[],
            reasoning=f"Error in analysis: {str(e)}"
        )
    
@tool
def generate_trading_advice(
    trend_analysis: TrendAnalysis,
    stock_data: pd.DataFrame,
    technical_indicators: pd.DataFrame,
    ml_prediction: float,
    symbol: str
) -> TradingAdvice:
    """
    Generate trading advice based on trend analysis
        
    Args:
        trend_analysis: Trend analysis result
        stock_data: OHLCV data
        technical_indicators: Technical indicators
        ml_prediction: ML prediction
        symbol: Stock symbol
        
    Returns:
        TradingAdvice object
    """
    try:
        current_price = stock_data['close'].iloc[-1]
            
        # Determine action based on trend and confidence
        if trend_analysis.trend == "bullish" and trend_analysis.confidence > 0.6:
            action = "buy"
            confidence = trend_analysis.confidence
        elif trend_analysis.trend == "bearish" and trend_analysis.confidence > 0.6:
            action = "sell"
            confidence = trend_analysis.confidence
        elif trend_analysis.confidence < 0.4:
            action = "wait"
            confidence = 1 - trend_analysis.confidence
        else:
            action = "hold"
            confidence = 0.5
            
        # Determine risk level
        if trend_analysis.strength == "strong" and trend_analysis.confidence > 0.7:
            risk_level = "low"
        elif trend_analysis.strength == "moderate":
            risk_level = "medium"
        else:
            risk_level = "high"
            
        # Calculate target price and stop loss
        target_price = None
        stop_loss = None
            
        if action == "buy":
            # Target: 5-10% above current price
            target_price = current_price * 1.07
            # Stop loss: 3-5% below current price
            stop_loss = current_price * 0.96
        elif action == "sell":
            # Target: 5-10% below current price
            target_price = current_price * 0.93
            # Stop loss: 3-5% above current price
            stop_loss = current_price * 1.04
        else:
            target_price = None
            stop_loss = None
            
        # Generate reasoning
        reasoning = generate_advice_reasoning(
            action, trend_analysis, current_price, ml_prediction
        )
            
        return TradingAdvice(
            action=action,
            confidence=confidence,
            risk_level=risk_level,
            target_price=target_price,
            stop_loss=stop_loss,
            reasoning=reasoning,
            timeframe="1-3 days"
        )
            
    except Exception as e:
        logger.error(f"Error generating trading advice: {str(e)}")
        return TradingAdvice(
            action="hold",
            confidence=0.5,
            risk_level="high",
            reasoning=f"Error generating advice: {str(e)}"
        )
    
@tool
def calculate_key_levels(
    stock_data: pd.DataFrame, 
    technical_indicators: pd.DataFrame
) -> List[float]:
    """Calculate key support and resistance levels"""
    try:
        current_price = stock_data['close'].iloc[-1]
        levels = []
            
        # Add moving averages as key levels
        sma_20 = technical_indicators['sma_20'].iloc[-1]
        sma_50 = technical_indicators['sma_50'].iloc[-1]
        sma_200 = technical_indicators['sma_200'].iloc[-1]
            
        if not pd.isna(sma_20):
            levels.append(round(sma_20, 2))
        if not pd.isna(sma_50):
            levels.append(round(sma_50, 2))
        if not pd.isna(sma_200):
            levels.append(round(sma_200, 2))
            
        # Add recent highs and lows
        recent_high = stock_data['high'].tail(20).max()
        recent_low = stock_data['low'].tail(20).min()
            
        levels.append(round(recent_high, 2))
        levels.append(round(recent_low, 2))
            
        # Remove duplicates and sort
        levels = sorted(list(set(levels)))
            
        return levels
            
    except Exception as e:
        logger.error(f"Error calculating key levels: {str(e)}")
        return []

@tool
def generate_trend_reasoning(
    trend: str, 
    strength: str, 
    duration: str,
    current_price: float,
    sma_20: float,
    sma_50: float,
    rsi: float,
    macd: float,
    ml_prediction: float
) -> str:
    """Generate human-readable trend reasoning"""
    
    reasoning_parts = []
    
    # Price vs moving averages
    if current_price > sma_20:
        reasoning_parts.append("Price is above 20-day moving average")
    else:
        reasoning_parts.append("Price is below 20-day moving average")
        
    if current_price > sma_50:
        reasoning_parts.append("Price is above 50-day moving average")
    else:
        reasoning_parts.append("Price is below 50-day moving average")
    
    # RSI analysis
    if rsi > 70:
        reasoning_parts.append("RSI indicates overbought conditions")
    elif rsi < 30:
        reasoning_parts.append("RSI indicates oversold conditions")
    else:
        reasoning_parts.append(f"RSI at {rsi:.1f} indicates neutral momentum")
    
    # MACD analysis
    if macd > 0:
        reasoning_parts.append("MACD is positive, indicating upward momentum")
    else:
        reasoning_parts.append("MACD is negative, indicating downward momentum")
    
    # ML prediction
    if ml_prediction > 0.6:
        reasoning_parts.append("ML model predicts bullish movement")
    elif ml_prediction < 0.4:
        reasoning_parts.append("ML model predicts bearish movement")
    else:
        reasoning_parts.append("ML model shows neutral prediction")
    
    # Overall trend
    reasoning_parts.append(f"Overall trend is {trend} with {strength} {duration} momentum")
    
    return ". ".join(reasoning_parts) + "."


@tool
def generate_advice_reasoning(
    action: str,
    trend_analysis: dict,
    current_price: float,
    ml_prediction: float
) -> str:
    """Generate trading advice reasoning"""
    
    if action == "buy":
        return f"Strong {trend_analysis['trend']} signals with {trend_analysis['confidence']:.1%} confidence. ML model predicts {ml_prediction:.1%} probability of upward movement. Consider buying with proper risk management."
    elif action == "sell":
        return f"Strong {trend_analysis['trend']} signals with {trend_analysis['confidence']:.1%} confidence. ML model predicts {ml_prediction:.1%} probability of upward movement. Consider selling or reducing position."
    elif action == "hold":
        return f"Mixed signals with {trend_analysis['confidence']:.1%} confidence. ML model shows {ml_prediction:.1%} probability. Best to hold current position and monitor for clearer signals."
    else:  # wait
        return f"Low confidence signals ({trend_analysis['confidence']:.1%}). ML model shows {ml_prediction:.1%} probability. Wait for clearer market direction before taking action."


@tool
def calculate_key_levels(
    stock_data: pd.DataFrame, 
    technical_indicators: pd.DataFrame
) -> List[float]:
    """Calculate key support and resistance levels"""
    try:
        current_price = stock_data['close'].iloc[-1]
        levels = []
        
        # Add moving averages as key levels
        sma_20 = technical_indicators['sma_20'].iloc[-1]
        sma_50 = technical_indicators['sma_50'].iloc[-1]
        sma_200 = technical_indicators['sma_200'].iloc[-1]
        
        if not pd.isna(sma_20):
            levels.append(round(sma_20, 2))
        if not pd.isna(sma_50):
            levels.append(round(sma_50, 2))
        if not pd.isna(sma_200):
            levels.append(round(sma_200, 2))
        
        # Add recent highs and lows
        recent_high = stock_data['high'].tail(20).max()
        recent_low = stock_data['low'].tail(20).min()
        
        levels.append(round(recent_high, 2))
        levels.append(round(recent_low, 2))
        
        # Remove duplicates and sort
        levels = sorted(list(set(levels)))
        
        return levels
        
    except Exception as e:
        logger.error(f"Error calculating key levels: {str(e)}")
        return []


class TrendAnalyzer:
    """Separate agent for analyzing hard financial data and trends"""
    
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        #self.memory = MemorySaver()
        self.llm = ChatOpenAI(
            #model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            model="lgai/exaone-3-5-32b-instruct",
            openai_api_base="https://api.together.xyz/v1",
            openai_api_key=os.getenv("TOGETHER_API_KEY"),  
            temperature=0.7
        )
        
        # Tools for trend analysis
        self.tools = [
            generate_trend_reasoning,
            generate_advice_reasoning,
            calculate_key_levels,
            analyze_trend,
            generate_trading_advice,
            get_market_data_sync,
        ]
        
        # Initialize agent
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def analyze_trend(self, symbol: str):
        """Analyze trend using technical indicators and ML predictions"""
        try:
            # Get real market data
            stock_data = data_collector.get_stock_data_yahoo(symbol)
            if stock_data is None or stock_data.empty:
                return f"Error: No data available for {symbol}"
            
            # Calculate technical indicators
            technical_indicators = feature_engineer.calculate_technical_indicators(stock_data)
            
            # Get ML prediction
            ml_prediction = ml_manager.predict(stock_data)
            
            # Create a comprehensive analysis prompt
            agent_prompt = f"""
            You are an expert financial analyst specializing in technical analysis and trend identification.
            
            TASK: Analyze the stock {symbol} using real market data and provide a comprehensive trend analysis.
            
            AVAILABLE DATA:
            - Current stock data: {stock_data['close'].iloc[-1]:.2f} (latest close price)
            - Technical indicators: RSI, MACD, Moving Averages (20, 50, 200 day)
            - ML prediction: {ml_prediction:.3f} (probability of upward movement)
            
            REQUIRED ANALYSIS STEPS:
            1. Use analyze_trend tool to determine the overall trend direction (bullish/bearish/neutral)
            2. Use calculate_key_levels tool to identify key support and resistance levels
            3. Use generate_trend_reasoning tool to explain the technical analysis
            4. Use generate_trading_advice tool to provide actionable trading recommendations
            5. Use generate_advice_reasoning tool to explain the trading advice
            
            EXPECTED OUTPUT FORMAT:
            - Trend Analysis: [bullish/bearish/neutral] with confidence level
            - Key Levels: Support and resistance prices
            - Technical Reasoning: Clear explanation of indicators
            - Trading Advice: [buy/sell/hold/wait] with risk assessment
            - Reasoning: Detailed explanation of the recommendation
            
            IMPORTANT: Use the tools in sequence to build a comprehensive analysis. Do not skip any steps.
            Be specific about price levels, confidence percentages, and risk factors.
            """
            
            result = self.agent_executor.run(agent_prompt)
            return result
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return f"Error analyzing trend: {str(e)}"

trend_analyzer = TrendAnalyzer()
financial_analyzer = FinancialAnalyzer() 