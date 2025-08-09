import React, { useState } from 'react';

const StockPredictorTest = () => {
    const [symbol, setSymbol] = useState('AAPL');
    const [model, setModel] = useState('random_forest');
    const [prediction, setPrediction] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string>('');

    const testStockData = async () => {
        setLoading(true);
        setError('');
        try {
            const response = await fetch(`http://localhost:8000/api/v1/stocks/${symbol}`);
            const data = await response.json();
            console.log('Stock data:', data);
            alert(`Stock data collected: ${data.data_points} data points`);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to get stock data');
        } finally {
            setLoading(false);
        }
    };

    const testPrediction = async () => {
        setLoading(true);
        setError('');
        setPrediction(null);
        try {
            const response = await fetch(`http://localhost:8000/api/v1/stocks/${symbol}/predict?model=${model}`);
            const data = await response.json();
            setPrediction(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to get prediction');
        } finally {
            setLoading(false);
        }
    };

    const trainModels = async () => {
        setLoading(true);
        setError('');
        try {
            const response = await fetch(`http://localhost:8000/api/v1/stocks/${symbol}/train`, {
                method: 'POST'
            });
            const data = await response.json();
            alert(`Training completed: ${data.message}`);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to train models');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className='min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6'>
            <div className='max-w-4xl mx-auto'>
                <h1 className='text-4xl font-bold text-gray-800 mb-8'>Stock Predictor Test</h1>
                
                <div className='bg-white rounded-lg shadow-md p-6 mb-6'>
                    <h2 className='text-2xl font-bold text-gray-800 mb-4'>Test Configuration</h2>
                    <div className='grid grid-cols-1 md:grid-cols-2 gap-4 mb-4'>
                        <div>
                            <label className='block text-sm font-medium text-gray-700 mb-2'>Stock Symbol</label>
                            <input
                                type='text'
                                value={symbol}
                                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                                className='w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent'
                                placeholder='AAPL'
                            />
                        </div>
                        <div>
                            <label className='block text-sm font-medium text-gray-700 mb-2'>Model</label>
                            <select
                                value={model}
                                onChange={(e) => setModel(e.target.value)}
                                className='w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent'
                            >
                                <option value='Random_Forest'>Random Forest</option>
                                <option value='XGBoost'>XGBoost</option>
                                <option value='LSTM'>LSTM</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div className='bg-white rounded-lg shadow-md p-6 mb-6'>
                    <h2 className='text-2xl font-bold text-gray-800 mb-4'>Test Actions</h2>
                    <div className='flex flex-wrap gap-4'>
                        <button
                            onClick={testStockData}
                            disabled={loading}
                            className='bg-gray-800 text-white px-6 py-3 rounded-lg font-medium hover:bg-gray-700 transition-colors disabled:opacity-50'
                        >
                            {loading ? 'Testing...' : 'Test Stock Data'}
                        </button>
                        
                        <button
                            onClick={testPrediction}
                            disabled={loading}
                            className='bg-gray-700 text-white px-6 py-3 rounded-lg font-medium hover:bg-gray-600 transition-colors disabled:opacity-50'
                        >
                            {loading ? 'Predicting...' : 'Test Prediction'}
                        </button>
                        
                        <button
                            onClick={trainModels}
                            disabled={loading}
                            className='bg-gray-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-gray-500 transition-colors disabled:opacity-50'
                        >
                            {loading ? 'Training...' : 'Train Models'}
                        </button>
                    </div>
                </div>

                {error && (
                    <div className='bg-red-50 border border-red-200 rounded-lg p-4 mb-6'>
                        <h3 className='text-red-800 font-semibold mb-2'>Error:</h3>
                        <p className='text-red-700'>{error}</p>
                    </div>
                )}

                {prediction && (
                    <div className='bg-green-50 border border-green-200 rounded-lg p-4 mb-6'>
                        <h3 className='text-green-800 font-semibold mb-2'>Prediction Result:</h3>
                        <div className='space-y-2 text-green-700'>
                            <p><strong>Symbol:</strong> {prediction.symbol}</p>
                            <p><strong>Model:</strong> {prediction.model}</p>
                            <p><strong>Prediction:</strong> {prediction.prediction}</p>
                            <p><strong>Confidence:</strong> {prediction.confidence}</p>
                        </div>
                    </div>
                )}

                <div className='bg-white rounded-lg shadow-md p-6'>
                    <h2 className='text-2xl font-bold text-gray-800 mb-4'>API Endpoints</h2>
                    <div className='space-y-2 text-gray-700'>
                        <p><strong>Stock Data:</strong> GET /api/v1/stocks/{symbol}</p>
                        <p><strong>Stock Features:</strong> GET /api/v1/stocks/{symbol}/features</p>
                        <p><strong>Prediction:</strong> GET /api/v1/stocks/{symbol}/predict?model={model}</p>
                        <p><strong>Train Models:</strong> POST /api/v1/stocks/{symbol}/train</p>
                    </div>
                    <div className='mt-4'>
                        <a 
                            href='http://localhost:8000/docs' 
                            target='_blank' 
                            rel='noopener noreferrer'
                            className='text-gray-600 hover:text-gray-800 underline'
                        >
                            View Full API Documentation →
                        </a>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StockPredictorTest; 