import React from 'react';

const Dashboard = () => {
    return (
        <div className='min-h-screen bg-gray-50'>
            <div className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8'>
                {/* Header */}
                <div className='mb-8'>
                    <h1 className='text-4xl font-bold text-gray-900 mb-2'>Dashboard</h1>
                    <p className='text-gray-600'>Welcome back! Here's your financial overview.</p>
                </div>

                {/* Key Metrics */}
                <div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8'>
                    <div className='bg-white p-6 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200'>
                        <div className='flex items-center justify-between'>
                            <div>
                                <p className='text-sm font-medium text-gray-500 mb-1'>Portfolio Value</p>
                                <p className='text-3xl font-bold text-gray-900'>$125,430</p>
                                <p className='text-sm text-green-500 flex items-center mt-1'>
                                    <svg className='w-4 h-4 mr-1' fill='currentColor' viewBox='0 0 20 20'>
                                        <path fillRule='evenodd' d='M12 7a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0V8.414l-4.293 4.293a1 1 0 01-1.414 0L8 10.414l-4.293 4.293a1 1 0 01-1.414-1.414l5-5a1 1 0 011.414 0L12 10.586z' clipRule='evenodd' />
                                    </svg>
                                    +2.4% today
                                </p>
                            </div>
                            <div className='w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center'>
                                <svg className='w-6 h-6 text-green-600' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                    <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1' />
                                </svg>
                            </div>
                        </div>
                    </div>

                    <div className='bg-white p-6 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200'>
                        <div className='flex items-center justify-between'>
                            <div>
                                <p className='text-sm font-medium text-gray-500 mb-1'>Total Return</p>
                                <p className='text-3xl font-bold text-gray-900'>+$12,450</p>
                                <p className='text-sm text-gray-600 flex items-center mt-1'>
                                    <svg className='w-4 h-4 mr-1' fill='currentColor' viewBox='0 0 20 20'>
                                        <path fillRule='evenodd' d='M12 7a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0V8.414l-4.293 4.293a1 1 0 01-1.414 0L8 10.414l-4.293 4.293a1 1 0 01-1.414-1.414l5-5a1 1 0 011.414 0L12 10.586z' clipRule='evenodd' />
                                    </svg>
                                    +11.2% YTD
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
                                <p className='text-sm font-medium text-gray-500 mb-1'>Active Alerts</p>
                                <p className='text-3xl font-bold text-gray-900'>8</p>
                                <p className='text-sm text-orange-500 flex items-center mt-1'>
                                    <svg className='w-4 h-4 mr-1' fill='currentColor' viewBox='0 0 20 20'>
                                        <path fillRule='evenodd' d='M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z' clipRule='evenodd' />
                                    </svg>
                                    3 triggered today
                                </p>
                            </div>
                            <div className='w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center'>
                                <svg className='w-6 h-6 text-orange-600' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                    <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M15 17h5l-5 5v-5zM4.19 4.19C4.74 3.63 5.48 3.25 6.32 3.25s1.58.38 2.13.94l3.25 3.25a3.25 3.25 0 014.59 0l3.25 3.25c.56.56.87 1.3.87 2.13s-.31 1.58-.87 2.13L17.06 16.5c-.56.56-1.3.87-2.13.87s-1.58-.31-2.13-.87L9.44 13.25a3.25 3.25 0 00-4.59 0L1.69 16.5c-.56.56-.87 1.3-.87 2.13s.31 1.58.87 2.13z' />
                                </svg>
                            </div>
                        </div>
                    </div>

                    <div className='bg-white p-6 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200'>
                        <div className='flex items-center justify-between'>
                            <div>
                                <p className='text-sm font-medium text-gray-500 mb-1'>AI Predictions</p>
                                <p className='text-3xl font-bold text-gray-900'>85%</p>
                                <p className='text-sm text-gray-600 flex items-center mt-1'>
                                    <svg className='w-4 h-4 mr-1' fill='currentColor' viewBox='0 0 20 20'>
                                        <path fillRule='evenodd' d='M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z' clipRule='evenodd' />
                                    </svg>
                                    Accuracy rate
                                </p>
                            </div>
                            <div className='w-12 h-12 bg-gray-200 rounded-lg flex items-center justify-center'>
                                <svg className='w-6 h-6 text-gray-700' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                    <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z' />
                                </svg>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Main Content Grid */}
                <div className='grid grid-cols-1 lg:grid-cols-3 gap-8'>
                    {/* Portfolio Overview */}
                    <div className='lg:col-span-2 bg-white rounded-xl shadow-sm border border-gray-200 p-6'>
                        <div className='flex items-center justify-between mb-6'>
                            <h2 className='text-2xl font-bold text-gray-900'>Portfolio Overview</h2>
                            <button className='text-sm text-gray-600 hover:text-gray-800 font-medium'>
                                View Details →
                            </button>
                        </div>
                        
                        {/* Portfolio Chart Placeholder */}
                        <div className='bg-gray-50 rounded-lg p-8 text-center'>
                            <div className='w-10 h-10 bg-gradient-to-br from-gray-500 to-gray-600 rounded-lg flex items-center justify-center text-white font-bold shadow-sm'>A</div>
                            <p className='text-gray-600 mt-4'>Portfolio chart will be displayed here</p>
                        </div>
                    </div>

                    {/* Recent Activity */}
                    <div className='bg-white rounded-xl shadow-sm border border-gray-200 p-6'>
                        <h2 className='text-2xl font-bold text-gray-900 mb-6'>Recent Activity</h2>
                        <div className='space-y-4'>
                            <div className='flex items-center space-x-3'>
                                <div className='w-2 h-2 bg-green-500 rounded-full'></div>
                                <div className='flex-1'>
                                    <p className='text-sm font-medium text-gray-900'>Bought AAPL</p>
                                    <p className='text-xs text-gray-500'>2 hours ago</p>
                                </div>
                                <span className='text-sm font-medium text-gray-900'>$150.25</span>
                            </div>
                            
                            <div className='flex items-center space-x-3'>
                                <div className='w-2 h-2 bg-red-500 rounded-full'></div>
                                <div className='flex-1'>
                                    <p className='text-sm font-medium text-gray-900'>Sold TSLA</p>
                                    <p className='text-xs text-gray-500'>5 hours ago</p>
                                </div>
                                <span className='text-sm font-medium text-gray-900'>$245.80</span>
                            </div>
                            
                            <div className='flex items-center space-x-3'>
                                <div className='w-2 h-2 bg-gray-500 rounded-full'></div>
                                <div className='flex-1'>
                                    <p className='text-sm font-medium text-gray-900'>Alert triggered</p>
                                    <p className='text-xs text-gray-500'>1 day ago</p>
                                </div>
                                <span className='text-sm font-medium text-gray-900'>MSFT &gt; $350</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard; 