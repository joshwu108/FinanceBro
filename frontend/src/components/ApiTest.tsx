import React, { useState, useEffect } from 'react';

const ApiTest = () => {
    const [backendStatus, setBackendStatus] = useState<string>('Testing...');
    const [apiResponse, setApiResponse] = useState<any>(null);
    const [error, setError] = useState<string>('');

    const testBackendConnection = async () => {
        try {
            setBackendStatus('Testing...');
            setError('');
            
            // Test the health endpoint
            const response = await fetch('http://localhost:8000/health');
            
            if (response.ok) {
                const data = await response.json();
                setBackendStatus('✅ Connected!');
                setApiResponse(data);
            } else {
                setBackendStatus('❌ Connection failed');
                setError(`HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (err) {
            setBackendStatus('❌ Connection failed');
            setError(err instanceof Error ? err.message : 'Unknown error');
        }
    };

    useEffect(() => {
        testBackendConnection();
    }, []);

    return (
        <div className='min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6'>
            <div className='max-w-4xl mx-auto'>
                <h1 className='text-4xl font-bold text-gray-800 mb-8'>API Connection Test</h1>
                
                <div className='bg-white rounded-lg shadow-md p-6 mb-6'>
                    <h2 className='text-2xl font-bold text-gray-800 mb-4'>Backend Status</h2>
                    <div className='flex items-center space-x-4 mb-4'>
                        <span className='text-lg font-medium'>Status:</span>
                        <span className={`text-lg font-bold ${
                            backendStatus.includes('✅') ? 'text-green-600' : 'text-red-600'
                        }`}>
                            {backendStatus}
                        </span>
                    </div>
                    
                    {error && (
                        <div className='bg-red-50 border border-red-200 rounded-lg p-4 mb-4'>
                            <h3 className='text-red-800 font-semibold mb-2'>Error:</h3>
                            <p className='text-red-700'>{error}</p>
                        </div>
                    )}
                    
                    {apiResponse && (
                        <div className='bg-green-50 border border-green-200 rounded-lg p-4'>
                            <h3 className='text-green-800 font-semibold mb-2'>Response:</h3>
                            <pre className='text-green-700 text-sm overflow-auto'>
                                {JSON.stringify(apiResponse, null, 2)}
                            </pre>
                        </div>
                    )}
                    
                    <button 
                        onClick={testBackendConnection}
                        className='mt-4 bg-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors'
                    >
                        Test Again
                    </button>
                </div>

                <div className='bg-white rounded-lg shadow-md p-6'>
                    <h2 className='text-2xl font-bold text-gray-800 mb-4'>Connection Details</h2>
                    <div className='space-y-2 text-gray-700'>
                        <p><strong>Frontend URL:</strong> http://localhost:3000</p>
                        <p><strong>Backend URL:</strong> http://localhost:8000</p>
                        <p><strong>CORS:</strong> Configured for frontend origin</p>
                        <p><strong>Health Endpoint:</strong> http://localhost:8000/health</p>
                    </div>
                </div>

                <div className='bg-white rounded-lg shadow-md p-6 mt-6'>
                    <h2 className='text-2xl font-bold text-gray-800 mb-4'>Troubleshooting</h2>
                    <div className='space-y-3 text-gray-700'>
                        <div>
                            <h3 className='font-semibold text-red-600'>If connection fails:</h3>
                            <ul className='list-disc list-inside ml-4 space-y-1'>
                                <li>Make sure backend is running: <code className='bg-gray-100 px-2 py-1 rounded'>uvicorn app.main:app --reload --port 8000</code></li>
                                <li>Check if port 8000 is available</li>
                                <li>Verify CORS settings in backend</li>
                                <li>Check browser console for CORS errors</li>
                            </ul>
                        </div>
                        <div>
                            <h3 className='font-semibold text-green-600'>If connection works:</h3>
                            <ul className='list-disc list-inside ml-4 space-y-1'>
                                <li>✅ Backend is running correctly</li>
                                <li>✅ CORS is configured properly</li>
                                <li>✅ Frontend can communicate with backend</li>
                                <li>Ready to implement API calls!</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ApiTest; 