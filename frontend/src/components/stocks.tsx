import React, { useState } from 'react';
import api from '../services/api';

const Stocks = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [loadingAnalysis, setLoadingAnalysis] = useState(false);
    const [aiAnalysis, setAiAnalysis] = useState<string>('');
    
    const stocks = [
        { symbol: 'AAPL', name: 'Apple Inc.', price: 175.43, change: '+2.1%', volume: '45.2M', marketCap: '2.7T' },
        { symbol: 'TSLA', name: 'Tesla Inc.', price: 242.12, change: '-1.3%', volume: '32.1M', marketCap: '770B' },
        { symbol: 'MSFT', name: 'Microsoft Corp.', price: 312.67, change: '+0.8%', volume: '28.9M', marketCap: '2.3T' },
        { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 142.56, change: '+1.2%', volume: '22.4M', marketCap: '1.8T' },
        { symbol: 'AMZN', name: 'Amazon.com Inc.', price: 134.89, change: '-0.5%', volume: '35.7M', marketCap: '1.4T' },
        { symbol: 'NVDA', name: 'NVIDIA Corp.', price: 485.23, change: '+3.2%', volume: '18.3M', marketCap: '1.2T' },
    ];

    const filteredStocks = stocks.filter(stock =>
        stock.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
        stock.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

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
                        <div className='flex-1'>
                            <input
                                type='text'
                                placeholder='Search stocks by symbol or company name...'
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className='w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent bg-white text-gray-900 placeholder-gray-500'
                            />
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
                </div>
            </div>
        </div>
    );
};

export default Stocks; 