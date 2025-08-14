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
from langchain_community.llms import OpenAI as LangChainOpenAI
from langchain_core.tools import tool, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import initialize_agent, AgentType, create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import asyncio
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


# Global instance
financial_analyzer = FinancialAnalyzer() 