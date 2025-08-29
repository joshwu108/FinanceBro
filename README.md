# FinanceBro - AI-Powered Financial Agent

A comprehensive financial analysis platform that monitors stock trends, creates alerts, and makes data-driven investment decisions using machine learning.

## Features

- **Real-time Stock Monitoring**: Live price tracking and trend analysis
- **ML-Powered Predictions**: Advanced models for stock price forecasting
- **Smart Alerts**: Customizable notifications based on market conditions
- **Portfolio Management**: Track and analyze investment performance
- **Interactive Dashboard**: Real-time visualization of market data
- **API Integration**: Multiple data sources for comprehensive analysis

## Tech Stack

### Backend
- **FastAPI**: High-performance web framework
- **SQLAlchemy**: Database ORM
- **PostgreSQL**: Primary database
- **Redis**: Caching and real-time data
- **Celery**: Background task processing
- **Scikit-learn/PyTorch**: Machine learning models

### Frontend
- **React 18**: UI framework
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling
- **Chart.js**: Data visualization
- **Socket.io**: Real-time updates

### Data & ML
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Yahoo Finance API**: Stock data
- **Alpha Vantage API**: Market data
- **Streamlit**: ML model development

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- PostgreSQL
- Redis

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

