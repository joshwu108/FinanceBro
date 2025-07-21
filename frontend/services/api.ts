const API_BASE_URL = 'http://localhost:8000/api';

// api endpoint calls
export const api = {
    getStocks: () => fetch(`${API_BASE_URL}/stocks`),
    getStockData: (symbol: string) => fetch(`${API_BASE_URL}/stocks/${symbol}/`),
    getStockPrediction: (symbol: string) => fetch(`${API_BASE_URL}/predictions/${symbol}/`),
    getStockPortfolio: async () => {
        const response = await fetch(`${API_BASE_URL}/portfolio/`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({})
        })
    }
}
