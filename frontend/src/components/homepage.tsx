import React from "react";

const Homepage = () => {
    return (
        <div className='min-h-screen bg-gray-50'>
            {/* Hero Section */}
            <div className='relative overflow-hidden'>
                <div className='absolute inset-0 bg-gradient-to-br from-gray-100 to-gray-200'></div>
                <div className='relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24'>
                    <div className='text-center'>
                        <h1 className='text-5xl md:text-6xl font-bold text-gray-900 mb-6'>
                            Welcome to{' '}
                            <span className='bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent'>
                                FinanceBro
                            </span>
                        </h1>
                        <p className='text-xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed'>
                            Your AI-powered financial companion. Monitor stocks, get predictions, and manage your portfolio with cutting-edge machine learning.
                        </p>
                        
                        <div className='flex gap-4 justify-center flex-wrap'>
                            <button className='bg-gradient-to-r from-gray-800 to-gray-700 hover:from-gray-900 hover:to-gray-800 text-white px-8 py-4 rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-1'>
                                Get Started
                            </button>
                            <button className='bg-white hover:bg-gray-50 text-gray-800 px-8 py-4 rounded-xl font-semibold border-2 border-gray-800 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-1'>
                                Learn More
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Features Section */}
            <div className='py-24 bg-white'>
                <div className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'>
                    <div className='text-center mb-16'>
                        <h2 className='text-4xl font-bold text-gray-900 mb-4'>
                            Powerful Features
                        </h2>
                        <p className='text-lg text-gray-600'>
                            Everything you need for intelligent investing
                        </p>
                    </div>
                    
                    <div className='grid grid-cols-1 md:grid-cols-3 gap-8'>
                        <div className='bg-gray-50 p-8 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200'>
                            <div className='w-12 h-12 bg-gray-200 rounded-lg flex items-center justify-center mb-4'>
                                <svg className='w-6 h-6 text-gray-700' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                    <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M13 7h8m0 0v8m0-8l-8 8-4-4-6 6' />
                                </svg>
                            </div>
                            <h3 className='text-xl font-semibold text-gray-900 mb-2'>Real-time Data</h3>
                            <p className='text-gray-600'>Live stock prices and market data with instant updates</p>
                        </div>
                        
                        <div className='bg-gray-50 p-8 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200'>
                            <div className='w-12 h-12 bg-gray-200 rounded-lg flex items-center justify-center mb-4'>
                                <svg className='w-6 h-6 text-gray-700' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                    <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z' />
                                </svg>
                            </div>
                            <h3 className='text-xl font-semibold text-gray-900 mb-2'>AI Predictions</h3>
                            <p className='text-gray-600'>ML-powered stock predictions and market analysis</p>
                        </div>
                        
                        <div className='bg-gray-50 p-8 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200'>
                            <div className='w-12 h-12 bg-gray-200 rounded-lg flex items-center justify-center mb-4'>
                                <svg className='w-6 h-6 text-gray-700' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                    <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M15 17h5l-5 5v-5zM4.19 4.19C4.74 3.63 5.48 3.25 6.32 3.25s1.58.38 2.13.94l3.25 3.25a3.25 3.25 0 014.59 0l3.25 3.25c.56.56.87 1.3.87 2.13s-.31 1.58-.87 2.13L17.06 16.5c-.56.56-1.3.87-2.13.87s-1.58-.31-2.13-.87L9.44 13.25a3.25 3.25 0 00-4.59 0L1.69 16.5c-.56.56-.87 1.3-.87 2.13s.31 1.58.87 2.13z' />
                                </svg>
                            </div>
                            <h3 className='text-xl font-semibold text-gray-900 mb-2'>Smart Alerts</h3>
                            <p className='text-gray-600'>Customizable price alerts and notifications</p>
                        </div>
                    </div>
                </div>
            </div>
            
            {/* About Us Section */}
            <div id="aboutus" className='py-24 bg-gray-50'>
                <div className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'>
                    <div className='text-center mb-16'>
                        <h2 className='text-4xl font-bold text-gray-900 mb-4'>
                            About Us
                        </h2>
                        <p className='text-lg text-gray-600 max-w-3xl mx-auto'>
                            We are a team of developers passionate about creating a better financial future for everyone.
                        </p>
                    </div>
                    
                    <div className='grid grid-cols-1 md:grid-cols-2 gap-12'>
                        <div className='bg-white p-8 rounded-xl shadow-sm border border-gray-200'>
                            <div className='w-12 h-12 bg-gray-200 rounded-lg flex items-center justify-center mb-4'>
                                <svg className='w-6 h-6 text-gray-700' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                    <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M13 10V3L4 14h7v7l9-11h-7z' />
                                </svg>
                            </div>
                            <h3 className='text-2xl font-semibold text-gray-900 mb-4'>Our Mission</h3>
                            <p className='text-gray-600 leading-relaxed'>
                                To democratize financial intelligence by providing AI-powered tools that help everyone make informed investment decisions. Our AI Agent is trained to analyze stock patterns and inform you of smart decisions.
                            </p>
                        </div>
                        
                        <div className='bg-white p-8 rounded-xl shadow-sm border border-gray-200'>
                            <div className='w-12 h-12 bg-gray-200 rounded-lg flex items-center justify-center mb-4'>
                                <svg className='w-6 h-6 text-gray-700' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                    <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M15 12a3 3 0 11-6 0 3 3 0 016 0z' />
                                    <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z' />
                                </svg>
                            </div>
                            <h3 className='text-2xl font-semibold text-gray-900 mb-4'>Our Vision</h3>
                            <p className='text-gray-600 leading-relaxed'>
                                To become the leading platform for AI-powered financial analysis, making sophisticated investment tools accessible to everyone, regardless of their background or experience level.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Homepage;