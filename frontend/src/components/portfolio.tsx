import React from 'react';

const Portfolio = () => {
    const holdings = [
        { symbol: 'AAPL', name: 'Apple Inc.', shares: 50, avgPrice: 165.20, currentPrice: 175.43, totalValue: 8771.50, gainLoss: 511.50, gainLossPercent: 6.2 },
        { symbol: 'TSLA', name: 'Tesla Inc.', shares: 25, avgPrice: 235.80, currentPrice: 242.12, totalValue: 6053.00, gainLoss: 158.00, gainLossPercent: 2.7 },
        { symbol: 'MSFT', name: 'Microsoft Corp.', shares: 30, avgPrice: 298.50, currentPrice: 312.67, totalValue: 9380.10, gainLoss: 425.10, gainLossPercent: 4.7 },
        { symbol: 'GOOGL', name: 'Alphabet Inc.', shares: 15, avgPrice: 138.90, currentPrice: 142.56, totalValue: 2138.40, gainLoss: 54.90, gainLossPercent: 2.6 },
        { symbol: 'NVDA', name: 'NVIDIA Corp.', shares: 10, avgPrice: 470.00, currentPrice: 485.23, totalValue: 4852.30, gainLoss: 152.30, gainLossPercent: 3.2 },
    ];

    const totalValue = holdings.reduce((sum, holding) => sum + holding.totalValue, 0);
    const totalGainLoss = holdings.reduce((sum, holding) => sum + holding.gainLoss, 0);
    const totalGainLossPercent = (totalGainLoss / (totalValue - totalGainLoss)) * 100;

    return (
        <div className='min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6'>
            <div className='max-w-7xl mx-auto'>
                {/* Header */}
                <div className='mb-8'>
                    <h1 className='text-4xl font-bold text-gray-800 mb-2'>Portfolio</h1>
                    <p className='text-gray-600'>Manage your investments and track performance</p>
                </div>

                {/* Portfolio Summary */}
                <div className='grid grid-cols-1 md:grid-cols-4 gap-6 mb-8'>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Total Value</h3>
                        <p className='text-3xl font-bold text-gray-800'>${totalValue.toLocaleString()}</p>
                        <p className={`text-sm ${totalGainLoss >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                            {totalGainLoss >= 0 ? '+' : ''}${totalGainLoss.toLocaleString()} ({totalGainLossPercent.toFixed(1)}%)
                        </p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Total Cost</h3>
                        <p className='text-3xl font-bold text-gray-800'>${(totalValue - totalGainLoss).toLocaleString()}</p>
                        <p className='text-sm text-gray-500'>Average cost basis</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Holdings</h3>
                        <p className='text-3xl font-bold text-gray-800'>{holdings.length}</p>
                        <p className='text-sm text-gray-500'>Different stocks</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Best Performer</h3>
                        <p className='text-3xl font-bold text-green-600'>AAPL</p>
                        <p className='text-sm text-green-500'>+6.2%</p>
                    </div>
                </div>

                {/* Portfolio Chart Placeholder */}
                <div className='bg-white rounded-lg shadow-md p-6 mb-8'>
                    <h2 className='text-2xl font-bold text-gray-800 mb-4'>Performance Chart</h2>
                    <div className='h-64 bg-gray-100 rounded-lg flex items-center justify-center'>
                        <p className='text-gray-500'>Chart component will be added here</p>
                    </div>
                </div>

                {/* Holdings Table */}
                <div className='bg-white rounded-lg shadow-md overflow-hidden'>
                    <div className='px-6 py-4 border-b border-gray-200 flex justify-between items-center'>
                        <h2 className='text-2xl font-bold text-gray-800'>Holdings</h2>
                        <button className='bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors'>
                            Add Stock
                        </button>
                    </div>
                    <div className='overflow-x-auto'>
                        <table className='w-full'>
                            <thead className='bg-gray-50'>
                                <tr>
                                    <th className='px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>Stock</th>
                                    <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Shares</th>
                                    <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Avg Price</th>
                                    <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Current Price</th>
                                    <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Total Value</th>
                                    <th className='px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider'>Gain/Loss</th>
                                    <th className='px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider'>Actions</th>
                                </tr>
                            </thead>
                            <tbody className='bg-white divide-y divide-gray-200'>
                                {holdings.map((holding, index) => (
                                    <tr key={index} className='hover:bg-gray-50'>
                                        <td className='px-6 py-4 whitespace-nowrap'>
                                            <div className='flex items-center'>
                                                <div className='w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold'>
                                                    {holding.symbol.charAt(0)}
                                                </div>
                                                <div className='ml-3'>
                                                    <div className='text-sm font-bold text-gray-900'>{holding.symbol}</div>
                                                    <div className='text-sm text-gray-500'>{holding.name}</div>
                                                </div>
                                            </div>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-right'>
                                            <div className='text-sm text-gray-900'>{holding.shares}</div>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-right'>
                                            <div className='text-sm text-gray-900'>${holding.avgPrice}</div>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-right'>
                                            <div className='text-sm font-semibold text-gray-900'>${holding.currentPrice}</div>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-right'>
                                            <div className='text-sm font-semibold text-gray-900'>${holding.totalValue.toLocaleString()}</div>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-right'>
                                            <div className={`text-sm font-medium ${
                                                holding.gainLoss >= 0 ? 'text-green-600' : 'text-red-600'
                                            }`}>
                                                {holding.gainLoss >= 0 ? '+' : ''}${holding.gainLoss.toFixed(2)} ({holding.gainLossPercent.toFixed(1)}%)
                                            </div>
                                        </td>
                                        <td className='px-6 py-4 whitespace-nowrap text-center'>
                                            <div className='flex justify-center space-x-2'>
                                                <button className='bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700 transition-colors'>
                                                    Buy
                                                </button>
                                                <button className='bg-red-600 text-white px-3 py-1 rounded text-sm hover:bg-red-700 transition-colors'>
                                                    Sell
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Portfolio Allocation */}
                <div className='mt-8 grid grid-cols-1 lg:grid-cols-2 gap-8'>
                    <div className='bg-white rounded-lg shadow-md p-6'>
                        <h2 className='text-2xl font-bold text-gray-800 mb-4'>Sector Allocation</h2>
                        <div className='space-y-4'>
                            <div className='flex justify-between items-center'>
                                <span className='text-sm text-gray-600'>Technology</span>
                                <span className='text-sm font-semibold text-gray-800'>65%</span>
                            </div>
                            <div className='w-full bg-gray-200 rounded-full h-2'>
                                <div className='bg-blue-600 h-2 rounded-full' style={{width: '65%'}}></div>
                            </div>
                            <div className='flex justify-between items-center'>
                                <span className='text-sm text-gray-600'>Automotive</span>
                                <span className='text-sm font-semibold text-gray-800'>15%</span>
                            </div>
                            <div className='w-full bg-gray-200 rounded-full h-2'>
                                <div className='bg-green-600 h-2 rounded-full' style={{width: '15%'}}></div>
                            </div>
                            <div className='flex justify-between items-center'>
                                <span className='text-sm text-gray-600'>Other</span>
                                <span className='text-sm font-semibold text-gray-800'>20%</span>
                            </div>
                            <div className='w-full bg-gray-200 rounded-full h-2'>
                                <div className='bg-purple-600 h-2 rounded-full' style={{width: '20%'}}></div>
                            </div>
                        </div>
                    </div>

                    <div className='bg-white rounded-lg shadow-md p-6'>
                        <h2 className='text-2xl font-bold text-gray-800 mb-4'>AI Recommendations</h2>
                        <div className='space-y-4'>
                            <div className='bg-green-50 p-4 rounded-lg border-l-4 border-green-500'>
                                <h3 className='font-semibold text-green-800 mb-1'>Buy More AAPL</h3>
                                <p className='text-sm text-green-700'>Strong momentum and positive technical indicators</p>
                            </div>
                            <div className='bg-yellow-50 p-4 rounded-lg border-l-4 border-yellow-500'>
                                <h3 className='font-semibold text-yellow-800 mb-1'>Hold TSLA</h3>
                                <p className='text-sm text-yellow-700'>Wait for better entry point</p>
                            </div>
                            <div className='bg-blue-50 p-4 rounded-lg border-l-4 border-blue-500'>
                                <h3 className='font-semibold text-blue-800 mb-1'>Consider NVDA</h3>
                                <p className='text-sm text-blue-700'>AI sector growth potential</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Portfolio; 