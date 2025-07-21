import React, { useState } from 'react';

const Alerts = () => {
    const [activeTab, setActiveTab] = useState('active');
    
    const activeAlerts = [
        { id: 1, symbol: 'AAPL', type: 'price', condition: 'above', value: 175.00, status: 'active', created: '2024-01-15', triggered: null },
        { id: 2, symbol: 'TSLA', type: 'price', condition: 'below', value: 240.00, status: 'active', created: '2024-01-14', triggered: null },
        { id: 3, symbol: 'MSFT', type: 'volume', condition: 'above', value: 30000000, status: 'active', created: '2024-01-13', triggered: null },
        { id: 4, symbol: 'GOOGL', type: 'price', condition: 'above', value: 145.00, status: 'triggered', created: '2024-01-12', triggered: '2024-01-15 10:30 AM' },
    ];

    const alertHistory = [
        { id: 5, symbol: 'NVDA', type: 'price', condition: 'above', value: 480.00, status: 'triggered', created: '2024-01-10', triggered: '2024-01-15 09:15 AM' },
        { id: 6, symbol: 'AMZN', type: 'price', condition: 'below', value: 130.00, status: 'triggered', created: '2024-01-08', triggered: '2024-01-14 02:45 PM' },
        { id: 7, symbol: 'AAPL', type: 'volume', condition: 'above', value: 50000000, status: 'triggered', created: '2024-01-05', triggered: '2024-01-12 11:20 AM' },
    ];

    const allAlerts = [...activeAlerts, ...alertHistory];

    return (
        <div className='min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6'>
            <div className='max-w-7xl mx-auto'>
                {/* Header */}
                <div className='mb-8'>
                    <h1 className='text-4xl font-bold text-gray-800 mb-2'>Alerts</h1>
                    <p className='text-gray-600'>Set up price and volume alerts with AI-powered insights</p>
                </div>

                {/* Alert Stats */}
                <div className='grid grid-cols-1 md:grid-cols-4 gap-6 mb-8'>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Active Alerts</h3>
                        <p className='text-3xl font-bold text-blue-600'>{activeAlerts.filter(a => a.status === 'active').length}</p>
                        <p className='text-sm text-blue-500'>Currently monitoring</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Triggered Today</h3>
                        <p className='text-3xl font-bold text-green-600'>{allAlerts.filter(a => a.triggered && a.triggered.includes('2024-01-15')).length}</p>
                        <p className='text-sm text-green-500'>Alerts fired</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Success Rate</h3>
                        <p className='text-3xl font-bold text-purple-600'>92%</p>
                        <p className='text-sm text-purple-500'>Accurate predictions</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>AI Suggestions</h3>
                        <p className='text-3xl font-bold text-orange-600'>5</p>
                        <p className='text-sm text-orange-500'>New alert opportunities</p>
                    </div>
                </div>

                {/* Create New Alert */}
                <div className='bg-white rounded-lg shadow-md p-6 mb-8'>
                    <h2 className='text-2xl font-bold text-gray-800 mb-4'>Create New Alert</h2>
                    <div className='grid grid-cols-1 md:grid-cols-4 gap-4'>
                        <div>
                            <label className='block text-sm font-medium text-gray-700 mb-2'>Stock Symbol</label>
                            <input type='text' placeholder='AAPL' className='w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent' />
                        </div>
                        <div>
                            <label className='block text-sm font-medium text-gray-700 mb-2'>Alert Type</label>
                            <select className='w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent'>
                                <option>Price</option>
                                <option>Volume</option>
                                <option>Percentage Change</option>
                            </select>
                        </div>
                        <div>
                            <label className='block text-sm font-medium text-gray-700 mb-2'>Condition</label>
                            <select className='w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent'>
                                <option>Above</option>
                                <option>Below</option>
                                <option>Equals</option>
                            </select>
                        </div>
                        <div>
                            <label className='block text-sm font-medium text-gray-700 mb-2'>Value</label>
                            <input type='number' placeholder='175.00' className='w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent' />
                        </div>
                    </div>
                    <div className='mt-4 flex gap-4'>
                        <button className='bg-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors'>
                            Create Alert
                        </button>
                        <button className='bg-green-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-green-700 transition-colors'>
                            AI Suggest
                        </button>
                    </div>
                </div>

                {/* Alert Tabs */}
                <div className='bg-white rounded-lg shadow-md overflow-hidden'>
                    <div className='border-b border-gray-200'>
                        <nav className='flex space-x-8 px-6'>
                            <button
                                onClick={() => setActiveTab('active')}
                                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                                    activeTab === 'active'
                                        ? 'border-blue-500 text-blue-600'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                }`}
                            >
                                Active Alerts ({activeAlerts.filter(a => a.status === 'active').length})
                            </button>
                            <button
                                onClick={() => setActiveTab('history')}
                                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                                    activeTab === 'history'
                                        ? 'border-blue-500 text-blue-600'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                }`}
                            >
                                Alert History ({alertHistory.length})
                            </button>
                        </nav>
                    </div>

                    <div className='p-6'>
                        {activeTab === 'active' && (
                            <div className='space-y-4'>
                                {activeAlerts.filter(alert => alert.status === 'active').map((alert) => (
                                    <div key={alert.id} className='flex items-center justify-between p-4 bg-gray-50 rounded-lg'>
                                        <div className='flex items-center'>
                                            <div className='w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold'>
                                                {alert.symbol.charAt(0)}
                                            </div>
                                            <div className='ml-4'>
                                                <h3 className='font-semibold text-gray-800'>{alert.symbol}</h3>
                                                <p className='text-sm text-gray-600'>
                                                    {alert.type} {alert.condition} ${alert.value}
                                                </p>
                                            </div>
                                        </div>
                                        <div className='flex items-center space-x-4'>
                                            <span className='inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800'>
                                                Active
                                            </span>
                                            <button className='text-red-600 hover:text-red-800 text-sm font-medium'>
                                                Delete
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {activeTab === 'history' && (
                            <div className='space-y-4'>
                                {alertHistory.map((alert) => (
                                    <div key={alert.id} className='flex items-center justify-between p-4 bg-gray-50 rounded-lg'>
                                        <div className='flex items-center'>
                                            <div className='w-10 h-10 bg-green-500 rounded-full flex items-center justify-center text-white font-bold'>
                                                {alert.symbol.charAt(0)}
                                            </div>
                                            <div className='ml-4'>
                                                <h3 className='font-semibold text-gray-800'>{alert.symbol}</h3>
                                                <p className='text-sm text-gray-600'>
                                                    {alert.type} {alert.condition} ${alert.value}
                                                </p>
                                                <p className='text-xs text-gray-500'>Triggered: {alert.triggered}</p>
                                            </div>
                                        </div>
                                        <div className='flex items-center space-x-4'>
                                            <span className='inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800'>
                                                Triggered
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {/* AI Alert Suggestions */}
                <div className='mt-8 bg-white rounded-lg shadow-md p-6'>
                    <h2 className='text-2xl font-bold text-gray-800 mb-4'>AI Alert Suggestions</h2>
                    <div className='grid grid-cols-1 md:grid-cols-2 gap-6'>
                        <div className='bg-gradient-to-r from-blue-50 to-blue-100 p-6 rounded-lg'>
                            <h3 className='text-lg font-semibold text-blue-800 mb-2'>Price Breakout Alert</h3>
                            <p className='text-blue-700 mb-4'>AAPL is approaching resistance at $180. Set alert for breakout.</p>
                            <button className='bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700 transition-colors'>
                                Create Alert
                            </button>
                        </div>
                        <div className='bg-gradient-to-r from-green-50 to-green-100 p-6 rounded-lg'>
                            <h3 className='text-lg font-semibold text-green-800 mb-2'>Volume Spike Alert</h3>
                            <p className='text-green-700 mb-4'>TSLA showing unusual volume activity. Monitor for price movement.</p>
                            <button className='bg-green-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-green-700 transition-colors'>
                                Create Alert
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Alerts; 