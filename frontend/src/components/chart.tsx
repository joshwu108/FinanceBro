import React, { useState, useEffect } from 'react'
import { ChartData, ChartDataPoint, RealTimeData } from '../services/api'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, TimeScale } from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import apiService from '../services/api';
import websocketService from '../services/websocket';

//Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, TimeScale);

const Chart = () => {
    const [chartData, setChartData] = useState<any>(null);
    const [chartOptions, setChartOptions] = useState<any>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [timeframe, setTimeframe] = useState<'1m' | '5m' | '15m' | '30m' | '1h' | '1d'>('1d');
    const [period, setPeriod] = useState<'1d' | '5d' | '1mo' | '3mo' | '6mo' | '1y'>('1mo');
    const [currentSymbol, setCurrentSymbol] = useState<string>('');
    const [realTimeData, setRealTimeData] = useState<RealTimeData | null>(null);

    const createChartOptions = (symbol: string) => ({
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index' as const,
            intersect: false,
        },
        plugins: {
            legend: {
                position: 'top' as const,
            },
            title: {
                display: true,
                text: `${symbol} Stock Price`,
            },
            tooltip: {
                callbacks: {
                    label: (context: any) => {
                        const point = context.parsed;
                        return `${context.dataset.label}: $${point.y.toFixed(2)}`;
                    },
                },
            },
        },
        scales: {
            x: {
                type: 'time' as const,
                time: {
                    unit: 'day' as const,
                },
                title: {
                    display: true,
                    text: 'Date',
                },
            },
            y: {
                type: 'linear' as const,
                display: true,
                position: 'left' as const,
                title: {
                    display: true,
                    text: 'Price ($)',
                },
            },
        }
    });

    const transformStockData = (data: ChartDataPoint[]) => {
        return data.map((item: ChartDataPoint) => ({
            timestamp: item.timestamp,
            open: parseFloat(item.open.toString()),
            high: parseFloat(item.high.toString()),
            low: parseFloat(item.low.toString()),
            close: parseFloat(item.close.toString()),
            volume: parseFloat(item.volume.toString()),
        }));
    };

    const createChartData = (data: ChartDataPoint[], symbol: string) => ({
        labels: data.map(point => new Date(point.timestamp)),
        datasets: [
            {
                label: `${symbol} Price`,
                data: data.map(point => ({
                    x: new Date(point.timestamp),
                    y: point.close,
                })),
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1,
            },
        ],
    });

    const handleWebSocketMessage = (event: MessageEvent) => {
        const data = JSON.parse(event.data);
        if (data.type === 'price_update' && data.symbol === currentSymbol) {
            setRealTimeData({
                symbol: data.symbol,
                price: data.price,
                change: data.change,
                change_percent: data.change_percent,
                volume: 0,
                timestamp: data.timestamp
            });
        }
    };

    const fetchStockData = async (symbol: string) => {
        if (!symbol) return;

        setIsLoading(true);
        setError(null);

        try {
            const response = await apiService.getChartData(symbol, period, timeframe);
            
            if (response.data_points && response.data_points > 0) {
                const transformedData = transformStockData(response.data);
                const chartDataConfig = createChartData(transformedData, symbol);
                const options = createChartOptions(symbol);
                
                setChartData(chartDataConfig);
                setChartOptions(options);
            } else {
                setError('No data available for this symbol');
            }
        } catch (err) {
            setError(`Failed to fetch data for ${symbol}: ${err}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSymbolSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (currentSymbol.trim()) {
            fetchStockData(currentSymbol.trim().toUpperCase());
        }
    };

    useEffect(() => {
        if (!currentSymbol) return;
        
        // Connect to WebSocket for real-time updates
        websocketService.connect('/ws/general', handleWebSocketMessage);
        
        return () => {
            websocketService.disconnect();
        };
    }, [currentSymbol]);

    return (
        <div className='w-full h-full flex flex-col'>
            {/*Header */}
            <div className='flex items-center justify-between p-4 border-b bg-white shadow-sm'>
                <h1 className='text-2xl font-bold text-gray-800'>Stock Charts</h1>
                
                {/* Symbol Input */}
                <form onSubmit={handleSymbolSubmit} className='flex items-center gap-2'>
                    <input
                        type="text"
                        placeholder="Enter stock symbol (e.g., AAPL)"
                        value={currentSymbol}
                        onChange={(e) => setCurrentSymbol(e.target.value.toUpperCase())}
                        className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <button
                        type="submit"
                        disabled={!currentSymbol || isLoading}
                        className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isLoading ? 'Loading...' : 'Load Chart'}
                    </button>
                </form>
                
                {/* Timeframe Selector */}
                <div className='flex items-center gap-2'>
                    <label className='text-sm font-medium text-gray-700'>Period:</label>
                    <select
                        value={period}
                        onChange={(e) => setPeriod(e.target.value as any)}
                        className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                        <option value="1d">1 Day</option>
                        <option value="5d">5 Days</option>
                        <option value="1mo">1 Month</option>
                        <option value="3mo">3 Months</option>
                        <option value="6mo">6 Months</option>
                        <option value="1y">1 Year</option>
                    </select>
                    
                    <label className='text-sm font-medium text-gray-700'>Interval:</label>
                    <select
                        value={timeframe}
                        onChange={(e) => setTimeframe(e.target.value as any)}
                        className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                        <option value="1m">1m</option>
                        <option value="5m">5m</option>
                        <option value="15m">15m</option>
                        <option value="30m">30m</option>
                        <option value="1h">1h</option>
                        <option value="1d">1d</option>
                    </select>
                </div>
            </div>
            
            {/* Real-time Price Display */}
            {realTimeData && (
                <div className='bg-green-50 border-b border-green-200 p-3'>
                    <div className='flex items-center justify-between'>
                        <span className='font-semibold text-green-800'>
                            {realTimeData.symbol}: ${realTimeData.price.toFixed(2)}
                        </span>
                        <span className={`font-semibold ${realTimeData.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {realTimeData.change >= 0 ? '+' : ''}{realTimeData.change.toFixed(2)} ({realTimeData.change_percent.toFixed(2)}%)
                        </span>
                    </div>
                </div>
            )}
            
            {/*Chart Container */}
            <div className='flex-1 p-4 bg-gray-50'>
                {error && (
                    <div className='text-red-500 mb-4 p-3 bg-red-50 border border-red-200 rounded-md'>
                        {error}
                    </div>
                )}
                
                {isLoading && (
                    <div className='flex items-center justify-center h-64'>
                        <div className='text-lg text-gray-600'>Loading chart data...</div>
                    </div>
                )}
                
                {chartData && chartOptions && (
                    <div className='h-96 bg-white rounded-lg shadow-sm p-4'>
                        <Line data={chartData} options={chartOptions} />
                    </div>
                )}
                
                {!chartData && !isLoading && !error && (
                    <div className='flex items-center justify-center h-64'>
                        <div className='text-lg text-gray-500'>Enter a stock symbol to view chart</div>
                    </div>
                )}
            </div>
        </div>
    )
}

export default Chart
