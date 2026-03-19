const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

//Interfaces
export interface StockData {
  symbol: string;
  data_points: number;
  columns: string[];
  latest_price: number | null;
  data_source: string;
}

export interface StockPrediction {
  symbol: string;
  model: string;
  prediction: number | null;
  confidence: string;
  model_trained: boolean;
  model_name: string;
}

export interface StockAnalysis {
  symbol: string;
  current_price: number;
  analysis_date: string;
  ml_prediction: {
    model: string;
    probability: number;
    confidence: string;
  };
  trend_analysis: {
    trend: string;
    confidence: string;
    strength: string;
    duration: string;
    key_levels: number[];
    reasoning: string;
  };
  trading_advice: {
    action: string;
    confidence: string;
    risk_level: string;
    target_price: number | null;
    stop_loss: number | null;
    timeframe: string;
    reasoning: string;
  };
  market_sentiment: string;
  technical_indicators: {
    rsi: number | null;
    macd: number | null;
    sma_20: number | null;
    sma_50: number | null;
    volume: number | null;
  };
}

export interface Alert {
  id: number;
  stock_symbol: string;
  alert_type: string;
  status: string;
  target_price: number | null;
  message: string;
  created_at: string | null;
  triggered_at: string | null;
}

export interface PortfolioData {
  id: number;
  name: string;
  description: string | null;
  total_value: number;
  total_return: number;
  total_return_percent: number;
  created_at: string | null;
  updated_at: string | null;
}

export interface PortfolioHolding {
  id: number;
  stock_symbol: string;
  shares: number;
  average_price: number;
  current_price: number | null;
  current_value: number | null;
  total_return: number;
  total_return_percent: number;
  purchase_date: string | null;
}

export interface PortfolioDetails {
  portfolio: PortfolioData;
  holdings: PortfolioHolding[];
  total_holdings: number;
}

export interface ChartDataPoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ChartData {
  symbol: string;
  period: string;
  interval: string;
  data_points: number;
  data: ChartDataPoint[];
  latest_price: number;
  data_source: string;
}

export interface RealTimeData {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  timestamp: string;
}

export interface ChartConfig{
  symbol: string;
  timeframe: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w';
  period: '1d' | '5d' | '1mo' | '3mo' | '6mo' | '1y';
}

export interface RealTimeUpdate {
  type: 'price_update';
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  timestamp: string;
}

class ApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const headers : Record<string, string> = {};
    if (options.body) {
      headers['Content-Type'] = 'application/json';
    }

    const response = await fetch(url, {
      headers: {
        ...headers,
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    return response.json();
  }

  // Stock endpoints
  async getStocks(): Promise<{ stocks: string[]; message: string }> {
    return this.request<{ stocks: string[]; message: string }>('/stocks/');
  }

  async getStockData(symbol: string): Promise<StockData> {
    return this.request<StockData>(`/stocks/${symbol}`);
  }

  async getChartData(
    symbol: string,
    period: string = "1mo",
    interval: string = "1d"
  ): Promise<ChartData> {
    try {
      const params = new URLSearchParams();
      params.append('period', period);
      params.append('interval', interval);
      return this.request<ChartData>(`/stocks/${symbol}/chart?${params}`);
    } catch (error) {
      console.error('Error fetching chart data:', error);
      throw error;
    }
  }

  async getRealTimeData(symbol: string): Promise<RealTimeData> {
    return this.request<RealTimeData>(`/stocks/${symbol}/realtime`);
  }

  async getPopularStocks(): Promise<{ symbol: string; name: string; price: number; change: string; volume: string; marketCap: string }[]> {
    return this.request<{ symbol: string; name: string; price: number; change: string; volume: string; marketCap: string }[]>('/stocks/popular');
  }

  async searchStocks(query: string): Promise<{ symbol: string; name: string; exchange: string }[]> {
    const params = new URLSearchParams();
    params.append('q', query);
    return this.request<{ symbol: string; name: string; exchange: string }[]>(`/stocks/search?${params}`);
  }

  async getStockPrediction(
    symbol: string,
    model: string = 'random_forest'
  ): Promise<StockPrediction> {
    return this.request<StockPrediction>(`/stocks/${symbol}/predict?model=${model}`);
  }

  async trainStockModels(symbol: string): Promise<{
    symbol: string;
    training_completed: boolean;
    models_trained: string[];
    message: string;
  }> {
    return this.request(`/stocks/${symbol}/train`, {
      method: 'POST',
    });
  }

  // Alert endpoints
  async getAlerts(userId: number = 1, status?: string): Promise<{
    alerts: Alert[];
    total: number;
  }> {
    const params = new URLSearchParams();
    params.append('user_id', userId.toString());
    if (status) params.append('status', status);

    return this.request<{ alerts: Alert[]; total: number }>(`/alerts/?${params}`);
  }

  async createAlert(
    stockSymbol: string,
    alertType: string,
    targetPrice?: number,
    targetPercent?: number,
    message?: string,
    userId: number = 1
  ): Promise<{
    id: number;
    stock_symbol: string;
    alert_type: string;
    status: string;
    message: string;
  }> {
    const params = new URLSearchParams();
    params.append('stock_symbol', stockSymbol);
    params.append('alert_type', alertType);
    params.append('user_id', userId.toString());
    if (targetPrice) params.append('target_price', targetPrice.toString());
    if (targetPercent) params.append('target_percent', targetPercent.toString());
    if (message) params.append('message', message);

    return this.request(`/alerts/?${params}`, {
      method: 'POST',
    });
  }

  async updateAlert(
    alertId: number,
    updates: {
      alert_type?: string;
      target_price?: number;
      target_percent?: number;
      message?: string;
      status?: string;
    }
  ): Promise<{
    id: number;
    stock_symbol: string;
    alert_type: string;
    status: string;
    message: string;
  }> {
    const params = new URLSearchParams();
    Object.entries(updates).forEach(([key, value]) => {
      if (value !== undefined) params.append(key, value.toString());
    });

    return this.request(`/alerts/${alertId}?${params}`, {
      method: 'PUT',
    });
  }

  async deleteAlert(alertId: number): Promise<{ message: string }> {
    return this.request(`/alerts/${alertId}`, {
      method: 'DELETE',
    });
  }

  async checkAlerts(userId: number = 1): Promise<{
    checked_alerts: number;
    triggered_alerts: any[];
    message: string;
  }> {
    const params = new URLSearchParams();
    params.append('user_id', userId.toString());

    return this.request(`/alerts/check?${params}`, {
      method: 'POST',
    });
  }

  // Portfolio endpoints
  async getPortfolios(userId: number = 1): Promise<{
    portfolios: PortfolioData[];
    total: number;
  }> {
    const params = new URLSearchParams();
    params.append('user_id', userId.toString());

    return this.request<{ portfolios: PortfolioData[]; total: number }>(`/portfolio/?${params}`);
  }

  async createPortfolio(name: string, description?: string, userId: number = 1): Promise<{
    id: number;
    name: string;
    description: string | null;
    message: string;
  }> {
    const params = new URLSearchParams();
    params.append('name', name);
    params.append('user_id', userId.toString());
    if (description) params.append('description', description);

    return this.request(`/portfolio/?${params}`, {
      method: 'POST',
    });
  }

  async deletePortfolio(portfolioId: number): Promise<{ message: string }> {
    return this.request(`/portfolio/${portfolioId}`, {
      method: 'DELETE',
    });
  };

  async getPortfolioDetails(portfolioId: number): Promise<PortfolioDetails> {
    return this.request<PortfolioDetails>(`/portfolio/${portfolioId}`);
  }

  async addHolding(portfolioId: number, stockSymbol: string, shares: number, averagePrice: number, purchaseDate?: string): Promise<{
    id: number;
    stock_symbol: string;
    shares: number;
    average_price: number;
    message: string;
  }> {
    const params = new URLSearchParams();
    params.append('stock_symbol', stockSymbol);
    params.append('shares', shares.toString());
    params.append('average_price', averagePrice.toString());
    if (purchaseDate) params.append('purchase_date', purchaseDate);

    return this.request(`/portfolio/${portfolioId}/holdings?${params}`, {
      method: 'POST',
    });
  }

  async updateHolding(portfolioId: number, holdingId: number, shares?: number, averagePrice?: number): Promise<{
    id: number;
    stock_symbol: string;
    shares: number;
    average_price: number;
    message: string;
  }> {
    const params = new URLSearchParams();
    if (shares !== undefined) params.append('shares', shares.toString());
    if (averagePrice !== undefined) params.append('average_price', averagePrice.toString());

    return this.request(`/portfolio/${portfolioId}/holdings/${holdingId}?${params}`, {
      method: 'PUT',
    });
  }

  async deleteHolding(portfolioId: number, holdingId: number): Promise<{ message: string }> {
    return this.request(`/portfolio/${portfolioId}/holdings/${holdingId}`, {
      method: 'DELETE',
    });
  }

  async getStockAnalysis(symbol:string): Promise<string> {
    return this.request<string>(`/stocks/${symbol}/analysis`);
  }

}

export const apiService = new ApiService();
export default apiService; 