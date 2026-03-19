import torch
from transformers import pipeline
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:

    def __init__(self):
        self.models={}

        #current free models for use in sentiment analysis
        self.model_configs = {
            'general': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'financial': 'ProsusAI/finbert'
        }

    def load_model(self, model_type: str):
        if model_type in self.models:
            return self.models[model_type]

        try:
            model_name = self.model_configs.get(model_type)
            if not model_name:
                logger.error(f"No model configuration found for {model_type}")
                return None
            
            model = pipeline('sentiment-analysis', model=model_name, device=0 if torch.cuda.is_available() else -1)

            self.models[model_type] = model
            logger.info(f"Loaded {model_type} model")
            return model
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {str(e)}")

    def analyze_sentiment(self, text: str, model_type: str = 'financial') -> List[Dict]:
        """Analyze sentiment of financial texts"""
        try:
            model = self.load_model(model_type)
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

    def analyze_news_sentiment(self, news_articles: List[Dict]) -> Dict:
        """Analyze sentiment of news articles"""
        try:
            # Extract texts from news articles
            texts = []
            for article in news_articles:
                if 'title' in article:
                    texts.append(article['title'])
                if 'content' in article:
                    texts.append('Content: ' + article['content'])
                elif 'summary' in article:
                    texts.append('Summary: ' + article['summary'])
            
            if not texts:
                return {'overall_sentiment': 'neutral', 'confidence': 0.0, 'details': []}
            
            # Analyze with financial sentiment model
            results = []
            for text in texts:
                results.append(self.analyze_sentiment(text, 'financial'))
            
            # Calculate overall sentiment
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            total_score = 0.0
            
            for result in results:
                if result['sentiment'].lower() in ['positive', 'pos']:
                    positive_count += 1
                    total_score += result['score']
                elif result['sentiment'].lower() in ['negative', 'neg']:
                    negative_count += 1
                    total_score -= result['score']
                else:
                    neutral_count += 1
            
            # Determine overall sentiment
            total_articles = len(results)
            if total_articles == 0:
                return {'overall_sentiment': 'neutral', 'confidence': 0.0, 'details': []}
            
            if positive_count > negative_count:
                overall_sentiment = 'positive'
                confidence = positive_count / total_articles
            elif negative_count > positive_count:
                overall_sentiment = 'negative'
                confidence = negative_count / total_articles
            else:
                overall_sentiment = 'neutral'
                confidence = neutral_count / total_articles
            
            return {
                'overall_sentiment': overall_sentiment,
                'confidence': confidence,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'total_articles': total_articles,
                'average_score': total_score / total_articles if total_articles > 0 else 0.0,
                'details': results
            }
            
        except Exception as e:
            logger.error(f"Error in news sentiment analysis: {str(e)}")
            return {'overall_sentiment': 'neutral', 'confidence': 0.0, 'details': []}
    
sentiment_analyzer = SentimentAnalyzer()