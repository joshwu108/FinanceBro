import React, { useState, useEffect } from 'react';
import api from '../services/api';

const Stocks = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [loadingAnalysis, setLoadingAnalysis] = useState(false);
    const [aiAnalysis, setAiAnalysis] = useState<string>('');
    const [stocks, setStocks] = useState<{ symbol: string; name: string; price: number; change: string; volume: string; marketCap: string }[]>([]);
    const [loadingStocks, setLoadingStocks] = useState(true);
    const [searchResults, setSearchResults] = useState<{ symbol: string; name: string; exchange: string }[]>([]);
    const [showSearchResults, setShowSearchResults] = useState(false);

    useEffect(() => {
        const loadPopularStocks = async () => {
            try {
                setLoadingStocks(true);
                const popularStocks = await api.getPopularStocks();
                setStocks(popularStocks);
            } catch (error) {
                console.error('Error loading popular stocks:', error);
            } finally {
                setLoadingStocks(false);
            }
        };
        
        loadPopularStocks();
    }, []);

    useEffect(() => {
        const searchStocks = async () => {
            if (searchTerm.length >= 2) {
                try {
                    const results = await api.searchStocks(searchTerm);
                    setSearchResults(results);
                    setShowSearchResults(true);
                } catch (error) {
                    console.error('Error searching stocks:', error);
                    setSearchResults([]);
                }
            } else {
                setSearchResults([]);
                setShowSearchResults(false);
            }
        };

        const timeoutId = setTimeout(searchStocks, 300); // Debounce search
        return () => clearTimeout(timeoutId);
    }, [searchTerm]);

    const filteredStocks = stocks.filter(stock =>
        stock.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
        stock.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const handleStockSelect = (symbol: string) => {
        setSearchTerm(symbol);
        setShowSearchResults(false);
    };

    const handleClickOutside = () => {
        setShowSearchResults(false);
    };

    const getAIAnalysis = async () => {
        if (!searchTerm) {
            alert('Please enter a stock symbol');
            return;
        }
        setLoadingAnalysis(true);
        try {
            const response = await api.getStockAnalysis(searchTerm);
            setAiAnalysis(response);
        } catch (error) {
            console.error('Error fetching AI analysis:', error);
        } finally {
            setLoadingAnalysis(false);
        }
    }

    return (
        <div className='min-h-screen bg-gray-50'>
            <div className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8'>
                {/* Header */}
                <div className='mb-8'>
                    <h1 className='text-4xl font-bold text-gray-900 mb-2'>Stocks</h1>
                    <p className='text-gray-600'>Search and analyze stocks with AI-powered insights</p>
                </div>

                {/* Search and Filters */}
                <div className='bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6'>
                    <div className='flex flex-col md:flex-row gap-4'>
                        <div className='flex-1 relative'>
                            <input
                                type='text'
                                placeholder='Search stocks by symbol or company name...'
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className='w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent bg-white text-gray-900 placeholder-gray-500'
                            />
                            {/* Search Results Dropdown */}
                            {showSearchResults && searchResults.length > 0 && (
                                <div className='absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto'>
                                    {searchResults.map((result, index) => (
                                        <div
                                            key={index}
                                            onClick={() => handleStockSelect(result.symbol)}
                                            className='px-4 py-3 hover:bg-gray-100 cursor-pointer border-b border-gray-200 last:border-b-0'
                                        >
                                            <div className='flex justify-between items-center'>
                                                <div>
                                                    <div className='font-medium text-gray-900'>{result.symbol}</div>
                                                    <div className='text-sm text-gray-600'>{result.name}</div>
                                                </div>
                                                <div className='text-xs text-gray-500'>{result.exchange}</div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                        <button 
                            onClick={getAIAnalysis}
                            className={`bg-gradient-to-r from-gray-800 to-gray-700 text-white px-6 py-3 rounded-lg font-medium hover:from-gray-900 hover:to-gray-800 transition-all duration-200 shadow-sm hover:shadow-md ${loadingAnalysis ? 'opacity-50 cursor-not-allowed' : ''}`}
                            disabled={loadingAnalysis}
                        >
                            {loadingAnalysis ? (
                                <div className="flex items-center">
                                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Analyzing...
                                </div>
                            ) : (
                                'Run AI Analysis'
                            )}
                        </button>
                    </div>
                    {aiAnalysis && (
                        <div className='mt-6 p-4 bg-gray-100 rounded-lg border border-gray-300'>
                            <h3 className='font-semibold text-gray-900 mb-2'>AI Analysis</h3>
                            <p className='text-gray-800'>{aiAnalysis}</p>
                        </div>
                    )}
                </div>

                {/* Market Overview */}
                <div className='grid grid-cols-1 md:grid-cols-3 gap-6 mb-8'>
                    <div className='bg-white p-6 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200'>
                        <div className='flex items-center justify-between'>
                            <div>
                                <h3 className='text-lg font-semibold text-gray-900 mb-2'>S&P 500</h3>
                                <p className='text-2xl font-bold text-gray-900'>4,567.89</p>
                                <p className='text-sm text-green-500 flex items-center mt-1'>
                                    <svg className='w-4 h-4 mr-1' fill='currentColor' viewBox='0 0 20 20'>
                                        <path fillRule='evenodd' d='M12 7a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0V8.414l-4.293 4.293a1 1 0 01-1.414 0L8 10.414l-4.293 4.293a1 1 0 01-1.414-1.414l5-5a1 1 0 011.414 0L12 10.586z' clipRule='evenodd' />
                                    </svg>
                                    +0.8% today
                                </p>
                            </div>
                            <div className='w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center'>
                                <svg className='w-6 h-6 text-green-600' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                    <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M13 7h8m0 0v8m0-8l-8 8-4-4-6 6' />
                                </svg>
                            </div>
                        </div>
                    </div>

                    <div className='bg-white p-6 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200'>
                        <div className='flex items-center justify-between'>
                            <div>
                                <h3 className='text-lg font-semibold text-gray-900 mb-2'>NASDAQ</h3>
                                <p className='text-2xl font-bold text-gray-900'>14,234.56</p>
                                <p className='text-sm text-green-500 flex items-center mt-1'>
                                    <svg className='w-4 h-4 mr-1' fill='currentColor' viewBox='0 0 20 20'>
                                        <path fillRule='evenodd' d='M12 7a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0V8.414l-4.293 4.293a1 1 0 01-1.414 0L8 10.414l-4.293 4.293a1 1 0 01-1.414-1.414l5-5a1 1 0 011.414 0L12 10.586z' clipRule='evenodd' />
                                    </svg>
                                    +1.2% today
                                </p>
                            </div>
                            <div className='w-12 h-12 bg-gray-200 rounded-lg flex items-center justify-center'>
                                <svg className='w-6 h-6 text-gray-700' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                    <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M13 7h8m0 0v8m0-8l-8 8-4-4-6 6' />
                                </svg>
                            </div>
                        </div>
                    </div>

                    <div className='bg-white p-6 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200'>
                        <div className='flex items-center justify-between'>
                            <div>
                                <h3 className='text-lg font-semibold text-gray-900 mb-2'>DOW JONES</h3>
                                <p className='text-2xl font-bold text-gray-900'>34,567.12</p>
                                <p className='text-sm text-green-500 flex items-center mt-1'>
                                    <svg className='w-4 h-4 mr-1' fill='currentColor' viewBox='0 0 20 20'>
                                        <path fillRule='evenodd' d='M12 7a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0V8.414l-4.293 4.293a1 1 0 01-1.414 0L8 10.414l-4.293 4.293a1 1 0 01-1.414-1.414l5-5a1 1 0 011.414 0L12 10.586z' clipRule='evenodd' />
                                    </svg>
                                    +0.5% today
                                </p>
                            </div>
                            <div className='w-12 h-12 bg-gray-200 rounded-lg flex items-center justify-center'>
                                <svg className='w-6 h-6 text-gray-700' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                    <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M13 7h8m0 0v8m0-8l-8 8-4-4-6 6' />
                                </svg>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Stocks Table */}
                <div className='bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden'>
                    <div className='px-6 py-4 border-b border-gray-200'>
                        <h2 className='text-2xl font-bold text-gray-900'>Popular Stocks</h2>
                    </div>
                    {loadingStocks ? (
                        <div className='p-8 text-center'>
                            <div className='inline-flex items-center'>
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Loading popular stocks...
                            </div>
                        </div>
                    ) : (
                        <div className='overflow-x-auto'>
                            <table className='min-w-full divide-y divide-gray-200'>
                            <thead className='bg-gray-50'>
                                <tr>
                                    <th className='px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>Symbol</th>
                                    <th className='px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>Company</th>
                                    <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Price</th>
                                    <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Change</th>
                                    <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Volume</th>
                                    <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Market Cap</th>
                                </tr>
                            </thead>
                            <tbody className='bg-white divide-y divide-gray-200'>
                                {filteredStocks.map((stock, index) => (
                                    <tr key={index} className='hover:bg-gray-50 transition-colors duration-200'>
                                        <td className='px-6 py-4 whitespace-nowrap'>
                                            <div className='flex items-center'>
                                                <div className='w-8 h-8 bg-gradient-to-br from-gray-500 to-gray-600 rounded-lg flex items-center justify-center text-white font-bold text-sm shadow-sm'>
                                                    {stock.symbol.charAt(0)}
                                                </div>
                                                <div className='text-sm font-medium text-gray-900 ml-3'>{stock.symbol}</div>
                                            </div>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap'>
                                            <div className='text-sm text-gray-900'>{stock.name}</div>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-gray-900'>
                                            ${stock.price}
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-right'>
                                            <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                                stock.change.startsWith('+') 
                                                    ? 'bg-green-100 text-green-800' 
                                                    : 'bg-red-100 text-red-800'
                                            }`}>
                                                {stock.change}
                                            </span>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900'>
                                            {stock.volume}
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900'>
                                            {stock.marketCap}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Stocks; 