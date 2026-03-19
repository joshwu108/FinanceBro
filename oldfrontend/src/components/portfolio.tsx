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
            console.log('Loading portfolios...');
            const response = await apiService.getPortfolios();
            console.log('Portfolios response:', response);
            setPortfolios(response.portfolios || []);
            console.log('Portfolios set:', response.portfolios || []);
        } catch (error) {
            console.error('Error loading portfolios:', error);
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
            if (selectedPortfolio) {
                await loadPortfolioDetails(selectedPortfolio.portfolio.id);
            }
        } catch (error) {
            setError(error instanceof Error ? error.message : 'An error occurred while adding holding');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const removeHolding = async (portfolioId: number, holdingId: number) => {
        try {
            setLoading(true);
            await apiService.deleteHolding(portfolioId, holdingId);
            if (selectedPortfolio) {
                await loadPortfolioDetails(selectedPortfolio.portfolio.id);
            }
        } catch (error) {
            setError(error instanceof Error ? error.message : 'An error occurred while removing holding');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const calculateTotalValue = (holdings: any[]) => {
        return holdings.reduce((total, holding) => {
            return total + (holding.current_price * holding.shares);
        }, 0);
    };

    const calculateTotalReturn = (holdings: any[]) => {
        return holdings.reduce((total, holding) => {
            const currentValue = holding.current_price * holding.shares;
            const costBasis = holding.average_price * holding.shares;
            return total + (currentValue - costBasis);
        }, 0);
    };

    useEffect(() => {
        if (selectedPortfolio) {
            const interval = setInterval(() => {
                loadPortfolioDetails(selectedPortfolio.portfolio.id);
            }, 30000); // Refresh every 30 seconds
            
            return () => clearInterval(interval);
        }
    }, [selectedPortfolio]);

    if (loading) return <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading...</p>
        </div>
    </div>;

    if (error) return <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
            <p className="text-red-600">Error: {error}</p>
        </div>
    </div>;

    return (
        <div className="min-h-screen bg-gray-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* Header */}
                <div className='mb-8'>
                    <h1 className='text-4xl font-bold text-gray-900 mb-2'>Portfolio Management</h1>
                    <p className='text-gray-600'>Manage your investment portfolios and track performance</p>
                </div>

                {error && (
                    <div className='mb-6 p-4 bg-red-100 border border-red-300 rounded-lg'>
                        <p className='text-red-800'>{error}</p>
                    </div>
                )}



                <div className='grid grid-cols-1 lg:grid-cols-3 gap-8'>
                    {/* Portfolio List */}
                    <div className='lg:col-span-1'>
                        <div className='bg-white rounded-xl shadow-sm border border-gray-200 p-6'>
                            <div className='flex items-center justify-between mb-6'>
                                <h2 className='text-2xl font-bold text-gray-900'>Portfolios</h2>
                                <button
                                    onClick={() => setShowAddHolding(true)}
                                    className='bg-gradient-to-r from-gray-800 to-gray-700 text-white px-4 py-2 rounded-lg font-medium hover:from-gray-900 hover:to-gray-800 transition-all duration-200 shadow-sm hover:shadow-md'
                                >
                                    Add Portfolio
                                </button>
                            </div>

                            {loading && (
                                <div className='flex justify-center items-center py-8'>
                                    <div className='animate-spin rounded-full h-8 w-8 border-b-2 border-gray-600'></div>
                                </div>
                            )}

                            <div className='space-y-3'>
                                {portfolios.length === 0 ? (
                                    <div className='text-center py-8'>
                                        <div className='w-12 h-12 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4'>
                                            <svg className='w-6 h-6 text-gray-600' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                                <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' />
                                            </svg>
                                        </div>
                                        <h3 className='text-lg font-semibold text-gray-900 mb-2'>No Portfolios Yet</h3>
                                        <p className='text-gray-600 mb-4'>Create your first portfolio to get started</p>
                                        <button
                                            onClick={() => setShowAddHolding(true)}
                                            className='bg-gradient-to-r from-gray-800 to-gray-700 text-white px-4 py-2 rounded-lg font-medium hover:from-gray-900 hover:to-gray-800 transition-all duration-200'
                                        >
                                            Create Portfolio
                                        </button>
                                    </div>
                                ) : (
                                    portfolios.map((portfolio) => (
                                        <div
                                            key={portfolio.id}
                                            onClick={() => loadPortfolioDetails(portfolio.id)}
                                            className={`p-4 rounded-lg border cursor-pointer transition-all duration-200 hover:shadow-md ${
                                                selectedPortfolio?.portfolio.id === portfolio.id
                                                    ? 'bg-gray-100 border-gray-400'
                                                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                                            }`}
                                        >
                                            <div className='flex items-center justify-between'>
                                                <div>
                                                    <h3 className='font-semibold text-gray-900'>{portfolio.name}</h3>
                                                    <p className='text-sm text-gray-600'>{portfolio.description}</p>
                                                </div>
                                                <div className='w-8 h-8 bg-gradient-to-br from-gray-500 to-gray-600 rounded-lg flex items-center justify-center text-white font-bold text-sm shadow-sm'>
                                                    {portfolio.name.charAt(0)}
                                                </div>
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Portfolio Details */}
                    <div className='lg:col-span-2'>
                        {selectedPortfolio ? (
                            <div className='bg-white rounded-xl shadow-sm border border-gray-200 p-6'>
                                <div className='flex items-center justify-between mb-6'>
                                    <div>
                                        <h2 className='text-2xl font-bold text-gray-900'>{selectedPortfolio.portfolio.name}</h2>
                                        <p className='text-gray-600'>{selectedPortfolio.portfolio.description}</p>
                                    </div>
                                    <button
                                        onClick={() => setShowAddHolding(true)}
                                        className='bg-gradient-to-r from-gray-800 to-gray-700 text-white px-4 py-2 rounded-lg font-medium hover:from-gray-900 hover:to-gray-800 transition-all duration-200 shadow-sm hover:shadow-md'
                                    >
                                        Add Holding
                                    </button>
                                </div>

                                {/* Portfolio Summary */}
                                <div className='grid grid-cols-1 md:grid-cols-3 gap-4 mb-6'>
                                    <div className='bg-gray-50 p-4 rounded-lg border border-gray-200'>
                                        <p className='text-sm font-medium text-gray-500 mb-1'>Total Value</p>
                                        <p className='text-2xl font-bold text-gray-900'>
                                            ${calculateTotalValue(selectedPortfolio.holdings).toLocaleString()}
                                        </p>
                                    </div>
                                    <div className='bg-gray-50 p-4 rounded-lg border border-gray-200'>
                                        <p className='text-sm font-medium text-gray-500 mb-1'>Total Return</p>
                                        <p className={`text-2xl font-bold ${
                                            calculateTotalReturn(selectedPortfolio.holdings) >= 0 ? 'text-green-600' : 'text-red-600'
                                        }`}>
                                            ${calculateTotalReturn(selectedPortfolio.holdings).toLocaleString()}
                                        </p>
                                    </div>
                                    <div className='bg-gray-50 p-4 rounded-lg border border-gray-200'>
                                        <p className='text-sm font-medium text-gray-500 mb-1'>Holdings</p>
                                        <p className='text-2xl font-bold text-gray-900'>{selectedPortfolio.holdings.length}</p>
                                    </div>
                                </div>

                                {/* Holdings Table */}
                                <div className='overflow-x-auto'>
                                    <table className='min-w-full divide-y divide-gray-200'>
                                        <thead className='bg-gray-50'>
                                            <tr>
                                                <th className='px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>Symbol</th>
                                                <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Shares</th>
                                                <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Avg Price</th>
                                                <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Current Price</th>
                                                <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Total Value</th>
                                                <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Return</th>
                                                <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody className='bg-white divide-y divide-gray-200'>
                                            {selectedPortfolio.holdings.map((holding) => (
                                                <tr key={holding.id} className='hover:bg-gray-50'>
                                                    <td className='px-6 py-4 whitespace-nowrap'>
                                                        <div className='flex items-center'>
                                                            <div className='w-8 h-8 bg-gradient-to-br from-gray-500 to-gray-600 rounded-lg flex items-center justify-center text-white font-bold text-sm shadow-sm'>
                                                                {holding.stock_symbol.charAt(0)}
                                                            </div>
                                                            <div className='ml-3'>
                                                                <div className='text-sm font-medium text-gray-900'>{holding.stock_symbol}</div>
                                                            </div>
                                                        </div>
                                                    </td>
                                                    <td className='px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900'>
                                                        {holding.shares.toLocaleString()}
                                                    </td>
                                                    <td className='px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900'>
                                                        ${holding.average_price.toFixed(2)}
                                                    </td>
                                                    <td className='px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900'>
                                                        ${holding.current_price?.toFixed(2) || 'N/A'}
                                                    </td>
                                                    <td className='px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900'>
                                                        ${holding.current_price ? (holding.current_price * holding.shares).toLocaleString() : 'N/A'}
                                                    </td>
                                                    <td className='px-6 py-4 whitespace-nowrap text-right'>
                                                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                                            holding.current_price && (holding.current_price - holding.average_price) >= 0
                                                                ? 'bg-green-100 text-green-800'
                                                                : 'bg-red-100 text-red-800'
                                                        }`}>
                                                            {holding.current_price ? ((holding.current_price - holding.average_price) * holding.shares).toLocaleString() : 'N/A'}
                                                        </span>
                                                    </td>
                                                    <td className='px-6 py-4 whitespace-nowrap text-right text-sm font-medium'>
                                                        <button
                                                            onClick={() => removeHolding(selectedPortfolio.portfolio.id, holding.id)}
                                                            className='text-red-600 hover:text-red-800'
                                                        >
                                                            Remove
                                                        </button>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        ) : (
                            <div className='bg-white rounded-xl shadow-sm border border-gray-200 p-6 text-center'>
                                <div className='w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4'>
                                    <svg className='w-8 h-8 text-gray-600' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                        <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' />
                                    </svg>
                                </div>
                                <h3 className='text-lg font-semibold text-gray-900 mb-2'>Select a Portfolio</h3>
                                <p className='text-gray-600'>Choose a portfolio from the list to view its details and holdings.</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Create Portfolio Modal */}
                {showAddHolding && !selectedPortfolio && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-md bg-black bg-opacity-50">
                        <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-md mx-4">
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="text-xl font-bold text-gray-900">Create New Portfolio</h3>
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
                                createPortfolio(
                                    formData.get('name') as string,
                                    formData.get('description') as string
                                );
                                setShowAddHolding(false);
                            }}>
                                <div className="space-y-4">
                                    <div>
                                        <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                                            Portfolio Name
                                        </label>
                                        <input
                                            id="name"
                                            name="name"
                                            type="text"
                                            placeholder="e.g., Growth Portfolio"
                                            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent"
                                            required
                                        />
                                    </div>
                                    
                                    <div>
                                        <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-1">
                                            Description (Optional)
                                        </label>
                                        <input
                                            id="description"
                                            name="description"
                                            type="text"
                                            placeholder="e.g., Long-term growth stocks"
                                            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent"
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
                                        className="flex-1 bg-gray-800 text-white py-2 px-4 rounded-lg hover:bg-gray-700 transition-colors"
                                    >
                                        Create Portfolio
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                )}

                {/* Add Holding Modal for Portfolio Details */}
                {showAddHolding && selectedPortfolio && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-md bg-black bg-opacity-50">
                        <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-md mx-4">
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="text-xl font-bold text-gray-900">Add New Holding to {selectedPortfolio.portfolio.name}</h3>
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
                                            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent"
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
                                            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent"
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
                                            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent"
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
                                            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent"
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
                                        className="flex-1 bg-gray-800 text-white py-2 px-4 rounded-lg hover:bg-gray-700 transition-colors"
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