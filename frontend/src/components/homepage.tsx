import React from "react";

const Homepage = () => {
    return (
        <div className='min-h-screen bg-gradient-to-br from-gray-50 to-gray-100'>
            <div className='flex flex-col justify-start items-center h-screen px-4'>
                <div className='flex flex-col items-center justify-center text-center max-w-4xl mx-auto mt-10'>
                    <h1 className='text-6xl font-bold text-gray-800 mb-6'>
                        Welcome to <span className='text-green-900'>FinanceBro</span>
                    </h1>
                    <p className='text-xl text-gray-600 mb-8 max-w-2xl mx-auto'>
                        Your AI-powered financial companion. Monitor stocks, get predictions, and manage your portfolio with cutting-edge machine learning.
                    </p>
                    
                    <div className='flex gap-4 justify-center flex-wrap'>
                        <button className='bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-semibold transition-colors duration-200 shadow-lg'>
                            Get Started
                        </button>
                        <button className='bg-white hover:bg-gray-50 text-blue-600 px-8 py-3 rounded-lg font-semibold border-2 border-blue-600 transition-colors duration-200'>
                            Learn More
                        </button>
                    </div>
                    
                    <div className='mt-12 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto'>
                        <div className='bg-white p-6 rounded-lg shadow-md'>
                            <h3 className='text-xl font-semibold text-gray-800 mb-2'>📈 Real-time Data</h3>
                            <p className='text-gray-600'>Live stock prices and market data</p>
                        </div>
                        <div className='bg-white p-6 rounded-lg shadow-md'>
                            <h3 className='text-xl font-semibold text-gray-800 mb-2'>🤖 AI Predictions</h3>
                            <p className='text-gray-600'>ML-powered stock predictions</p>
                        </div>
                        <div className='bg-white p-6 rounded-lg shadow-md'>
                            <h3 className='text-xl font-semibold text-gray-800 mb-2'>🔔 Smart Alerts</h3>
                            <p className='text-gray-600'>Customizable price alerts</p>
                        </div>
                    </div>
                </div>
            </div>
            
            {/* About Us Section */}
            <div id="aboutus" className='min-h-screen bg-white py-20'>
                <div className='max-w-4xl mx-auto px-4'>
                    <div className='text-center mb-16'>
                        <h2 className='text-5xl font-bold text-gray-800 mb-6'>
                            About Us
                        </h2>
                        <p className='text-xl text-gray-600 max-w-3xl mx-auto'>
                            We are a team of developers who are passionate about creating a better financial future for everyone.
                        </p>
                    </div>
                    
                    <div className='grid grid-cols-1 md:grid-cols-2 gap-12'>
                        <div className='bg-gray-50 p-8 rounded-lg'>
                            <h3 className='text-2xl font-semibold text-gray-800 mb-4'>Our Mission</h3>
                            <p className='text-gray-600 leading-relaxed'>
                                To democratize financial intelligence by providing AI-powered tools that help everyone make informed investment decisions. Our AI Agent is trained to analyze stock patterns and inform you of smart decision.
                            </p>
                        </div>
                        <div className='bg-gray-50 p-8 rounded-lg'>
                            <h3 className='text-2xl font-semibold text-gray-800 mb-4'>Our Vision</h3>
                            <p className='text-gray-600 leading-relaxed'>
                                A world where advanced financial technology is accessible to all, not just institutional investors.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Homepage;