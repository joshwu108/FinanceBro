import React, { useState, useEffect } from 'react';
import apiService, { PortfolioData, PortfolioDetails } from '../services/api';
import websocketService from '../services/websocket';

const Portfolio = () => {
    const [portfolios, setPortfolios] = useState<PortfolioData[]>([]);
    const [selectedPortfolio, setSelectedPortfolio] = useState<PortfolioDetails | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [holding, setHolding] = useState<string | null>(null);
    const [showAddHolding, setShowAddHolding] = useState(false);

    useEffect(() => {
        loadPortfolios();
    }, []);

    useEffect(() => {
        if (selectedPortfolio) {
            websocketService.connect('/ws/portfolio', (data: any) => {
                console.log('Received portfolio update:', data);
                if (data.type === 'portfolio_update' && data.portfolio_id === selectedPortfolio?.portfolio.id) {
                    setSelectedPortfolio(prev => {
                        if (prev) {
                            return {
                                ...prev,
                                holdings: data.holdings,
                            };
                        }
                        return prev;
                    })
                }
                if (data.type === 'price_update') {
                    setSelectedPortfolio(prev => {
                        if (prev) {
                            const updatedHoldings = prev.holdings.map(holding => {
                                if (holding.stock_symbol === data.symbol) {
                                    return {
                                        ...holding,
                                        current_price: data.price,
                                    };
                                }
                                return holding;
                            });

                            return {
                                ...prev,
                                holdings: updatedHoldings,
                            };
                        }
                        return prev;
                    });
                }
            });

            return () => {
                websocketService.disconnect();
            };
        }
    }, [selectedPortfolio]);

    const loadPortfolios = async () => {
        try {
            setLoading(true);
            const response = await apiService.getPortfolios();
            setPortfolios(response.portfolios);
        } catch (error) {
            setError(error instanceof Error ? error.message : 'An error occurred while loading portfolios');
        } finally {
            setLoading(false);
        }
    };

    const loadPortfolioDetails = async (Id: number) => {
        try {
            setLoading(true);
            const details = await apiService.getPortfolioDetails(Id);
            setSelectedPortfolio(details);
        } catch (error) {
            setError(error instanceof Error ? error.message : 'An error occurred while loading portfolio details');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const createPortfolio = async (name: string, description? : string, userId?: number) => {
        try {
            setLoading(true);
            await apiService.createPortfolio(name, description, userId);
            await loadPortfolios();
        } catch (error) {
            setError(error instanceof Error ? error.message : 'An error occurred while creating portfolio');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const addHolding = async (portfolioId: number, stockSymbol: string, shares: number, averagePrice: number, purchaseDate?: string) => {
        try {
            setLoading(true);
            await apiService.addHolding(portfolioId, stockSymbol, shares, averagePrice, purchaseDate);
            await loadPortfolioDetails(portfolioId);
        } catch (error) {
            setError(error instanceof Error ? error.message : 'An error occurred while adding holding');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);

    // Add this to your portfolio component
    useEffect(() => {
        if (selectedPortfolio) {
        const interval = setInterval(() => {
            loadPortfolioDetails(selectedPortfolio.portfolio.id);
        }, 30000); // Refresh every 30 seconds
        
        return () => clearInterval(interval);
        }
    }, [selectedPortfolio]);

    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error}</div>;

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6">
            <div className="max-w-7xl mx-auto">
                <h1 className="text-4xl font-bold text-gray-800 mb-8">Portfolio Management</h1>
            
                {/* Portfolio List */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                    {portfolios.map((portfolio) => (
                        <div key={portfolio.id} className="bg-white p-6 rounded-lg shadow-md">
                            <h3 className="text-xl font-semibold text-gray-800 mb-2">{portfolio.name}</h3>
                            <p className="text-gray-600 mb-4">{portfolio.description}</p>
                            <div className="space-y-2">
                                <p className="text-2xl font-bold text-green-600">${portfolio.total_value.toLocaleString()}</p>
                                <p className={`text-sm ${portfolio.total_return_percent >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                    {portfolio.total_return_percent >= 0 ? '+' : ''}{portfolio.total_return_percent.toFixed(2)}%
                                </p>
                            </div>
                            <button
                                onClick={() => loadPortfolioDetails(portfolio.id)}
                                className="mt-4 w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
                            >
                                View Details
                            </button>
                        </div>
                    ))}
                </div>
                {/* Create Portfolio Form */}
                <div className="bg-white p-6 rounded-lg shadow-md mb-8">
                    <h2 className="text-2xl font-bold text-gray-800 mb-4">Create New Portfolio</h2>
                    <form onSubmit={(e) => {
                        e.preventDefault();
                        const formData = new FormData(e.currentTarget);
                        createPortfolio(
                            formData.get('name') as string,
                            formData.get('description') as string
                        );
                    }}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <input
                            name="name"
                            type="text"
                            placeholder="Portfolio Name"
                            className="p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            required
                        />
                        <input
                            name="description"
                            type="text"
                            placeholder="Description (optional)"
                            className="p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                    </div>
                    <button
                        type="submit"
                        className="mt-4 bg-green-600 text-white py-2 px-6 rounded-lg hover:bg-green-700 transition-colors"
                    >
                        Create Portfolio
                    </button>
                    </form>
                </div>

                {/* Portfolio Details */}
                {selectedPortfolio && (
                    <div className="bg-white p-6 rounded-lg shadow-md">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="text-2xl font-bold text-gray-800">
                                {selectedPortfolio.portfolio.name} - Details
                            </h2>
                            <button
                                onClick={() => setShowAddHolding(true)}
                                className="bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors"
                            >
                                Add Holding
                            </button>
                        </div>
            
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                            <div className="text-center">
                                <p className="text-sm text-gray-500">Total Value</p>
                                <p className="text-2xl font-bold text-green-600">
                                    ${selectedPortfolio.portfolio.total_value.toLocaleString()}
                                </p>
                            </div>
                        <div className="text-center">
                        <p className="text-sm text-gray-500">Total Return</p>
                        <p className={`text-2xl font-bold ${selectedPortfolio.portfolio.total_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            ${selectedPortfolio.portfolio.total_return.toLocaleString()}
                        </p>
                    </div>
                    <div className="text-center">
                        <p className="text-sm text-gray-500">Return %</p>
                        <p className={`text-2xl font-bold ${selectedPortfolio.portfolio.total_return_percent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {selectedPortfolio.portfolio.total_return_percent.toFixed(2)}%
                        </p>
                    </div>
                </div>

                {/* Holdings Table */}
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="border-b border-gray-200">
                                <th className="text-left py-3 px-4">Symbol</th>
                                <th className="text-right py-3 px-4">Shares</th>
                                <th className="text-right py-3 px-4">Avg Price</th>
                                <th className="text-right py-3 px-4">Current Price</th>
                                <th className="text-right py-3 px-4">Current Value</th>
                                <th className="text-right py-3 px-4">Return</th>
                                <th className="text-right py-3 px-4">Return %</th>
                            </tr>
                        </thead>
                        <tbody>
                            {selectedPortfolio.holdings.map((holding) => (
                                <tr key={holding.id} className="border-b border-gray-100">
                                    <td className="py-3 px-4 font-semibold">{holding.stock_symbol}</td>
                                    <td className="text-right py-3 px-4">{holding.shares.toLocaleString()}</td>
                                    <td className="text-right py-3 px-4">${holding.average_price.toFixed(2)}</td>
                                    <td className="text-right py-3 px-4">
                                        ${holding.current_price?.toFixed(2) || 'N/A'}
                                    </td>
                                    <td className="text-right py-3 px-4">
                                        ${holding.current_value?.toLocaleString() || 'N/A'}
                                    </td>
                                    <td className={`text-right py-3 px-4 ${holding.total_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                        ${holding.total_return.toLocaleString()}
                                    </td>
                                    <td className={`text-right py-3 px-4 ${holding.total_return_percent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                        {holding.total_return_percent.toFixed(2)}%
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
            )}
            
            {/* Add Holding Modal for Portfolio Details */}
            {showAddHolding && selectedPortfolio && (
                <div className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-md bg-black bg-opacity-50">
                    <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-md mx-4">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-xl font-bold text-gray-800">Add New Holding to {selectedPortfolio.portfolio.name}</h3>
                            <button
                                onClick={() => setShowAddHolding(false)}
                                className="text-gray-400 hover:text-gray-600 text-2xl font-bold"
                            >
                                ×
                            </button>
                        </div>
                        
                        <form onSubmit={(e) => {
                            e.preventDefault();
                            const formData = new FormData(e.currentTarget);
                            addHolding(
                                selectedPortfolio.portfolio.id,
                                formData.get('stockSymbol') as string,
                                Number(formData.get('shares')),
                                Number(formData.get('averagePrice')),
                                formData.get('purchaseDate') as string
                            );
                            setShowAddHolding(false);
                        }}>
                            <div className="space-y-4">
                                <div>
                                    <label htmlFor="stockSymbol" className="block text-sm font-medium text-gray-700 mb-1">
                                        Stock Symbol
                                    </label>
                                    <input
                                        id="stockSymbol"
                                        name="stockSymbol"
                                        type="text"
                                        placeholder="e.g., AAPL"
                                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        required
                                    />
                                </div>
                                
                                <div>
                                    <label htmlFor="shares" className="block text-sm font-medium text-gray-700 mb-1">
                                        Number of Shares
                                    </label>
                                    <input
                                        id="shares"
                                        name="shares"
                                        type="number"
                                        step="0.01"
                                        min="0"
                                        placeholder="e.g., 10"
                                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        required
                                    />
                                </div>
                                
                                <div>
                                    <label htmlFor="averagePrice" className="block text-sm font-medium text-gray-700 mb-1">
                                        Average Price per Share
                                    </label>
                                    <input
                                        id="averagePrice"
                                        name="averagePrice"
                                        type="number"
                                        step="0.01"
                                        min="0"
                                        placeholder="e.g., 150.00"
                                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        required
                                    />
                                </div>
                                
                                <div>
                                    <label htmlFor="purchaseDate" className="block text-sm font-medium text-gray-700 mb-1">
                                        Purchase Date
                                    </label>
                                    <input
                                        id="purchaseDate"
                                        name="purchaseDate"
                                        type="date"
                                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                    />
                                </div>
                            </div>
                            
                            <div className="flex space-x-3 mt-6">
                                <button
                                    type="button"
                                    onClick={() => setShowAddHolding(false)}
                                    className="flex-1 bg-gray-300 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-400 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
                                >
                                    Add Holding
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    </div>  
    );
};

export default Portfolio;