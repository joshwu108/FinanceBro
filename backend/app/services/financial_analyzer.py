"""
Financial Analysis Service
Interprets ML predictions and provides actionable financial advice
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from datetime import datetime, timedelta
import openai
from dataclasses import dataclass
import yfinance as yf
from fastapi import HTTPException
from app.services.sentiment_analyzer import sentiment_analyzer
import time
import re
import streamlit as st
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import torch
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


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
        self.openai_api_key = openai_api_key
        self.sentiment_analyzer = sentiment_analyzer
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        try:
            self.analyst_agent = pipeline(
                "text-generation", 
                model="gpt2-xl",  
                torch_dtype=torch.float16,
                device_map=self.device,
            )
        except Exception as e:
            logger.warning(f"Could not load text generation model: {e}")
            self.analyst_agent = None
            
        try:
            self.qa_agent = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device_map=self.device,
                torch_dtype=torch.float32,
            )
        except Exception as e:
            logger.warning(f"Could not load QA model: {e}")
            self.qa_agent = None
            
        if openai_api_key:
            openai.api_key = openai_api_key
    
    def analyze_trend(
        self, 
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
            key_levels = self._calculate_key_levels(stock_data, technical_indicators)
            
            # Generate reasoning
            reasoning = self._generate_trend_reasoning(
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
    
    def generate_trading_advice(
        self, 
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
            
            # Generate reasoning
            reasoning = self._generate_advice_reasoning(
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
    
    def _calculate_key_levels(
        self, 
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
    
    def _generate_trend_reasoning(
        self, 
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
    
    def _generate_advice_reasoning(
        self,
        action: str,
        trend_analysis: TrendAnalysis,
        current_price: float,
        ml_prediction: float
    ) -> str:
        """Generate trading advice reasoning"""
        
        if action == "buy":
            return f"Strong {trend_analysis.trend} signals with {trend_analysis.confidence:.1%} confidence. ML model predicts {ml_prediction:.1%} probability of upward movement. Consider buying with proper risk management."
        elif action == "sell":
            return f"Strong {trend_analysis.trend} signals with {trend_analysis.confidence:.1%} confidence. ML model predicts {ml_prediction:.1%} probability of upward movement. Consider selling or reducing position."
        elif action == "hold":
            return f"Mixed signals with {trend_analysis.confidence:.1%} confidence. ML model shows {ml_prediction:.1%} probability. Best to hold current position and monitor for clearer signals."
        else:  # wait
            return f"Low confidence signals ({trend_analysis.confidence:.1%}). ML model shows {ml_prediction:.1%} probability. Wait for clearer market direction before taking action."
    
    def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get overall market sentiment for a stock"""
        # This could integrate with news APIs, social media sentiment, etc.
        return {
            "symbol": symbol,
            "sentiment": "neutral",
            "confidence": 0.5,
            "factors": ["technical_analysis", "ml_prediction"],
            "last_updated": datetime.now().isoformat()
        }
    
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
        agent_prompt = f"""
        You are a financial analyst agent. Your task is to analyze online investor sentiment about {symbol} stock.

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
        articles = await self.get_stock_news(symbol, 5)

        context = ""
        for article in articles:
            title = str(article.get('title', ''))
            url = str(article.get('url', ''))
            article_text = self.extract_article_text(url)
            context += f"Title: {title}\nSummary: {url}\nText:\n{article_text}\n\n"

        agent_prompt += f"""
        Context: {context}
        """
        logger.info(f"successfully generated agent prompt {agent_prompt}")
        try:
            # Check if analyst_agent is available
            if self.analyst_agent is None:
                logger.warning("Analyst agent not available, returning basic analysis")
                return f"Basic analysis for {symbol}: Sentiment analysis not available due to model loading issues."
            
            messages = [{'role': 'user', 'content': agent_prompt}]
            response = self.analyst_agent(messages, max_new_tokens=256)
            logger.info(f"successfully generated stock analysis {response}")
            return response[0]['generated_text'][-1]
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


# Global instance
financial_analyzer = FinancialAnalyzer() 