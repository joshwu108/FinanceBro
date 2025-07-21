import React from 'react';

const Dashboard = () => {
    return (
        <div className='min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6'>
            <div className='max-w-7xl mx-auto'>
                {/* Header */}
                <div className='mb-8'>
                    <h1 className='text-4xl font-bold text-gray-800 mb-2'>Dashboard</h1>
                    <p className='text-gray-600'>Welcome back! Here's your financial overview.</p>
                </div>

                {/* Key Metrics */}
                <div className='grid grid-cols-1 md:grid-cols-4 gap-6 mb-8'>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Portfolio Value</h3>
                        <p className='text-3xl font-bold text-green-600'>$125,430</p>
                        <p className='text-sm text-green-500'>+2.4% today</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Total Return</h3>
                        <p className='text-3xl font-bold text-blue-600'>+$12,450</p>
                        <p className='text-sm text-blue-500'>+11.2% YTD</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Active Alerts</h3>
                        <p className='text-3xl font-bold text-orange-600'>8</p>
                        <p className='text-sm text-orange-500'>3 triggered today</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>AI Predictions</h3>
                        <p className='text-3xl font-bold text-purple-600'>85%</p>
                        <p className='text-sm text-purple-500'>Accuracy rate</p>
                    </div>
                </div>

                {/* Main Content Grid */}
                <div className='grid grid-cols-1 lg:grid-cols-3 gap-8'>
                    {/* Portfolio Overview */}
                    <div className='lg:col-span-2 bg-white rounded-lg shadow-md p-6'>
                        <h2 className='text-2xl font-bold text-gray-800 mb-4'>Portfolio Overview</h2>
                        <div className='space-y-4'>
                            <div className='flex justify-between items-center p-4 bg-gray-50 rounded-lg'>
                                <div className='flex items-center'>
                                    <div className='w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold'>A</div>
                                    <div className='ml-3'>
                                        <h3 className='font-semibold text-gray-800'>AAPL</h3>
                                        <p className='text-sm text-gray-600'>Apple Inc.</p>
                                    </div>
                                </div>
                                <div className='text-right'>
                                    <p className='font-semibold text-gray-800'>$175.43</p>
                                    <p className='text-sm text-green-500'>+2.1%</p>
                                </div>
                            </div>
                            <div className='flex justify-between items-center p-4 bg-gray-50 rounded-lg'>
                                <div className='flex items-center'>
                                    <div className='w-10 h-10 bg-green-500 rounded-full flex items-center justify-center text-white font-bold'>T</div>
                                    <div className='ml-3'>
                                        <h3 className='font-semibold text-gray-800'>TSLA</h3>
                                        <p className='text-sm text-gray-600'>Tesla Inc.</p>
                                    </div>
                                </div>
                                <div className='text-right'>
                                    <p className='font-semibold text-gray-800'>$242.12</p>
                                    <p className='text-sm text-red-500'>-1.3%</p>
                                </div>
                            </div>
                            <div className='flex justify-between items-center p-4 bg-gray-50 rounded-lg'>
                                <div className='flex items-center'>
                                    <div className='w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold'>M</div>
                                    <div className='ml-3'>
                                        <h3 className='font-semibold text-gray-800'>MSFT</h3>
                                        <p className='text-sm text-gray-600'>Microsoft Corp.</p>
                                    </div>
                                </div>
                                <div className='text-right'>
                                    <p className='font-semibold text-gray-800'>$312.67</p>
                                    <p className='text-sm text-green-500'>+0.8%</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Quick Actions */}
                    <div className='bg-white rounded-lg shadow-md p-6'>
                        <h2 className='text-2xl font-bold text-gray-800 mb-4'>Quick Actions</h2>
                        <div className='space-y-3'>
                            <button className='w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 transition-colors'>
                                Add New Stock
                            </button>
                            <button className='w-full bg-green-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-green-700 transition-colors'>
                                Set Alert
                            </button>
                            <button className='w-full bg-purple-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-purple-700 transition-colors'>
                                Run AI Analysis
                            </button>
                            <button className='w-full bg-orange-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-orange-700 transition-colors'>
                                View Reports
                            </button>
                        </div>
                    </div>
                </div>

                {/* Recent Activity */}
                <div className='mt-8 bg-white rounded-lg shadow-md p-6'>
                    <h2 className='text-2xl font-bold text-gray-800 mb-4'>Recent Activity</h2>
                    <div className='space-y-3'>
                        <div className='flex items-center justify-between p-3 bg-gray-50 rounded-lg'>
                            <div className='flex items-center'>
                                <div className='w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white text-sm'>✓</div>
                                <div className='ml-3'>
                                    <p className='font-medium text-gray-800'>Alert triggered: AAPL above $175</p>
                                    <p className='text-sm text-gray-600'>2 hours ago</p>
                                </div>
                            </div>
                        </div>
                        <div className='flex items-center justify-between p-3 bg-gray-50 rounded-lg'>
                            <div className='flex items-center'>
                                <div className='w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm'>📊</div>
                                <div className='ml-3'>
                                    <p className='font-medium text-gray-800'>AI prediction: TSLA bullish signal</p>
                                    <p className='text-sm text-gray-600'>4 hours ago</p>
                                </div>
                            </div>
                        </div>
                        <div className='flex items-center justify-between p-3 bg-gray-50 rounded-lg'>
                            <div className='flex items-center'>
                                <div className='w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-white text-sm'>+</div>
                                <div className='ml-3'>
                                    <p className='font-medium text-gray-800'>Added MSFT to portfolio</p>
                                    <p className='text-sm text-gray-600'>1 day ago</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard; 