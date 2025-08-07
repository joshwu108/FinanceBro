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
        <div className='min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6'>
            <div className='max-w-7xl mx-auto'>
                {/* Header */}
                <div className='mb-8'>
                    <h1 className='text-4xl font-bold text-gray-800 mb-2'>Stocks</h1>
                    <p className='text-gray-600'>Search and analyze stocks with AI-powered insights</p>
                </div>

                {/* Search and Filters */}
                <div className='bg-white rounded-lg shadow-md p-6 mb-6'>
                    <div className='flex flex-col md:flex-row gap-4'>
                        <div className='flex-1'>
                            <input
                                type='text'
                                placeholder='Search stocks by symbol or company name...'
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className='w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent'
                            />
                        </div>
                        <button 
                            onClick={getAIAnalysis}
                            className={`bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors ${loadingAnalysis ? 'opacity-50 cursor-not-allowed' : ''}`}
                            disabled={loadingAnalysis}
                        >
                            Run AI Analysis
                        </button>
                    </div>
                    <div className='mt-4'>
                        <p className='text-gray-600'>{aiAnalysis}</p>
                    </div>
                </div>

                {/* Market Overview */}
                <div className='grid grid-cols-1 md:grid-cols-3 gap-6 mb-8'>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-lg font-semibold text-gray-800 mb-2'>S&P 500</h3>
                        <p className='text-2xl font-bold text-green-600'>4,567.89</p>
                        <p className='text-sm text-green-500'>+0.8% today</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-lg font-semibold text-gray-800 mb-2'>NASDAQ</h3>
                        <p className='text-2xl font-bold text-blue-600'>14,234.56</p>
                        <p className='text-sm text-blue-500'>+1.2% today</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-lg font-semibold text-gray-800 mb-2'>DOW JONES</h3>
                        <p className='text-2xl font-bold text-purple-600'>34,567.12</p>
                        <p className='text-sm text-purple-500'>+0.5% today</p>
                    </div>
                </div>

                {/* Stocks Table */}
                <div className='bg-white rounded-lg shadow-md overflow-hidden'>
                    <div className='px-6 py-4 border-b border-gray-200'>
                        <h2 className='text-2xl font-bold text-gray-800'>Popular Stocks</h2>
                    </div>
                    <div className='overflow-x-auto'>
                        <table className='w-full'>
                            <thead className='bg-gray-50'>
                                <tr>
                                    <th className='px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>Symbol</th>
                                    <th className='px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>Company</th>
                                    <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Price</th>
                                    <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Change</th>
                                    <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Volume</th>
                                    <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Market Cap</th>
                                    <th className='px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider'>Actions</th>
                                </tr>
                            </thead>
                            <tbody className='bg-white divide-y divide-gray-200'>
                                {filteredStocks.map((stock, index) => (
                                    <tr key={index} className='hover:bg-gray-50'>
                                        <td className='px-6 py-4 whitespace-nowrap'>
                                            <div className='text-sm font-bold text-gray-900'>{stock.symbol}</div>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap'>
                                            <div className='text-sm text-gray-900'>{stock.name}</div>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-right'>
                                            <div className='text-sm font-semibold text-gray-900'>${stock.price}</div>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-right'>
                                            <span className={`text-sm font-medium ${
                                                stock.change.startsWith('+') ? 'text-green-600' : 'text-red-600'
                                            }`}>
                                                {stock.change}
                                            </span>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-right'>
                                            <div className='text-sm text-gray-900'>{stock.volume}</div>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-right'>
                                            <div className='text-sm text-gray-900'>{stock.marketCap}</div>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-center'>
                                            <div className='flex justify-center space-x-2'>
                                                <button className='bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700 transition-colors'>
                                                    Analyze
                                                </button>
                                                <button className='bg-green-600 text-white px-3 py-1 rounded text-sm hover:bg-green-700 transition-colors'>
                                                    Add
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* AI Insights */}
                <div className='mt-8 bg-white rounded-lg shadow-md p-6'>
                    <h2 className='text-2xl font-bold text-gray-800 mb-4'>AI Insights</h2>
                    <div className='grid grid-cols-1 md:grid-cols-2 gap-6'>
                        <div className='bg-gradient-to-r from-blue-50 to-blue-100 p-6 rounded-lg'>
                            <h3 className='text-lg font-semibold text-blue-800 mb-2'>Bullish Signals</h3>
                            <ul className='space-y-2 text-blue-700'>
                                <li>• AAPL showing strong momentum</li>
                                <li>• NVDA breaking resistance levels</li>
                                <li>• MSFT volume increasing</li>
                            </ul>
                        </div>
                        <div className='bg-gradient-to-r from-red-50 to-red-100 p-6 rounded-lg'>
                            <h3 className='text-lg font-semibold text-red-800 mb-2'>Bearish Signals</h3>
                            <ul className='space-y-2 text-red-700'>
                                <li>• TSLA facing resistance at $250</li>
                                <li>• AMZN showing weakness</li>
                                <li>• GOOGL consolidating</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Stocks; 