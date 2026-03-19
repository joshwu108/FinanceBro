import React, { useState, useEffect } from 'react'
import { ChartData, ChartDataPoint, RealTimeData } from '../services/api'
import '../syncfusion-license'; // Import license before Syncfusion components
import {
  StockChartComponent,
  StockChartSeriesCollectionDirective,
  StockChartSeriesDirective,
  Inject,
  DateTime,
  LineSeries,
  SplineSeries,
  CandleSeries,
  Tooltip
} from "@syncfusion/ej2-react-charts";
import apiService from '../services/api';
import websocketService from '../services/websocket';

const Chart = () => {
    const [chartData, setChartData] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [timeframe, setTimeframe] = useState<'1m' | '5m' | '15m' | '30m' | '1h' | '1d'>('1d');
    const [period, setPeriod] = useState<'1d' | '5d' | '1mo' | '3mo' | '6mo' | '1y'>('3mo');
    const [currentSymbol, setCurrentSymbol] = useState<string>('');
    const [realTimeData, setRealTimeData] = useState<RealTimeData | null>(null);

    const transformToSyncfusionData = (data: ChartDataPoint[]) => {
        console.log('data', data);
        return data
            .filter((item: ChartDataPoint) => {
                return item.timestamp && item.timestamp !== null && item.timestamp !== undefined;
            })
            .map((item: ChartDataPoint) => {
                const date = new Date(item.timestamp);
                if (isNaN(date.getTime())) {
                    console.warn('Invalid date:', item.timestamp);
                    return null;
                }
                
                return {
                    x: date,
                    open: Number(item.open) || 0,
                    high: Number(item.high) || 0,
                    low: Number(item.low) || 0,
                    close: Number(item.close) || 0,
                    volume: Number(item.volume) || 0
                };
            })
            .filter(item => item !== null);
    };

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
            
            if (response.data_points && response.data_points > 0 && response.data) {
                console.log(response.data);
                const syncfusionData = transformToSyncfusionData(response.data);
                
                if (syncfusionData.length === 0) {
                    setError('No valid data points found for this symbol');
                    return;
                }
                console.log('syncfusionData', syncfusionData);
                setChartData(syncfusionData);
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
        
        websocketService.connect('/ws/general', handleWebSocketMessage);
        
        return () => {
            websocketService.disconnect();
        };
    }, [currentSymbol]);

    useEffect(() => {
        if (currentSymbol && chartData.length > 0) {
            fetchStockData(currentSymbol);
        }
    }, [period, timeframe]);

    return (
        <div className='min-h-screen bg-gray-50'>
            <div className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8'>
                {/* Header */}
                <div className='mb-8'>
                    <h1 className='text-4xl font-bold text-gray-900 mb-2'>Stock Charts</h1>
                    <p className='text-gray-600'>Interactive charts and technical analysis tools</p>
                </div>

                {/* Controls */}
                <div className='bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8'>
                    <form onSubmit={handleSymbolSubmit} className='flex flex-col md:flex-row gap-4'>
                        <div className='flex-1'>
                            <input
                                type='text'
                                placeholder='Enter stock symbol (e.g., AAPL)'
                                value={currentSymbol}
                                onChange={(e) => setCurrentSymbol(e.target.value)}
                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-gray-500"
                            />
                        </div>
                        <div className='flex gap-2'>
                            <select
                                value={period}
                                onChange={(e) => setPeriod(e.target.value as any)}
                                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-gray-500"
                            >
                                <option value="1d">1 Day</option>
                                <option value="5d">5 Days</option>
                                <option value="1mo">1 Month</option>
                                <option value="3mo">3 Months</option>
                                <option value="6mo">6 Months</option>
                                <option value="1y">1 Year</option>
                            </select>
                            <select
                                value={timeframe}
                                onChange={(e) => setTimeframe(e.target.value as any)}
                                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-gray-500"
                            >
                                <option value="1m">1 Minute</option>
                                <option value="5m">5 Minutes</option>
                                <option value="15m">15 Minutes</option>
                                <option value="30m">30 Minutes</option>
                                <option value="1h">1 Hour</option>
                                <option value="1d">1 Day</option>
                            </select>
                            <button
                                type="submit"
                                className="bg-gray-800 text-white px-4 py-2 rounded-md hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                disabled={isLoading}
                            >
                                {isLoading ? 'Loading...' : 'Load Chart'}
                            </button>
                        </div>
                    </form>
                </div>

                {/* Error Display */}
                {error && (
                    <div className='bg-red-100 border border-red-300 text-red-800 px-4 py-3 rounded-lg mb-8'>
                        {error}
                    </div>
                )}

                {/* Real-time Data Display */}
                {realTimeData && (
                    <div className='bg-white rounded-xl shadow-sm border border-gray-200 p-4 mb-8'>
                        <h3 className='text-lg font-semibold text-gray-900 mb-2'>Real-time Data - {realTimeData.symbol}</h3>
                        <div className='grid grid-cols-1 md:grid-cols-4 gap-4'>
                            <div>
                                <p className='text-sm text-gray-500'>Price</p>
                                <p className='text-xl font-bold text-gray-900'>${realTimeData.price.toFixed(2)}</p>
                            </div>
                            <div>
                                <p className='text-sm text-gray-500'>Change</p>
                                <p className={`text-xl font-bold ${realTimeData.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                    {realTimeData.change >= 0 ? '+' : ''}{realTimeData.change.toFixed(2)}
                                </p>
                            </div>
                            <div>
                                <p className='text-sm text-gray-500'>Change %</p>
                                <p className={`text-xl font-bold ${realTimeData.change_percent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                    {realTimeData.change_percent >= 0 ? '+' : ''}{realTimeData.change_percent.toFixed(2)}%
                                </p>
                            </div>
                            <div>
                                <p className='text-sm text-gray-500'>Volume</p>
                                <p className='text-xl font-bold text-gray-900'>{realTimeData.volume.toLocaleString()}</p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Chart Display */}
                <div className='bg-white rounded-xl shadow-sm border border-gray-200 p-6'>
                    {isLoading ? (
                        <div className='flex items-center justify-center h-96'>
                            <div className='animate-spin rounded-full h-12 w-12 border-b-2 border-gray-600'></div>
                        </div>
                    ) : chartData.length > 0 ? (
                        <div className='h-96'>
                            <StockChartComponent
                                primaryXAxis={{
                                    valueType: 'DateTime',
                                    majorGridLines: { color: 'gray' },
                                    majorTickLines: { color: 'gray' }
                                }}
                                primaryYAxis={{
                                    majorGridLines: { color: 'gray' },
                                    majorTickLines: { color: 'gray' }
                                }}
                                chartArea={{ border: { color: 'gray' } }}
                                tooltip={{ enable: true }}
                                title={`${currentSymbol} Stock Price Chart`}
                                width='100%'
                                height='100%'
                            >
                                <Inject services={[DateTime, LineSeries, SplineSeries, CandleSeries, Tooltip]} />
                                <StockChartSeriesCollectionDirective>
                                    <StockChartSeriesDirective
                                        dataSource={chartData}
                                        type='Candle'
                                        xName='x'
                                        low='low'
                                        high='high'
                                        open='open'
                                        close='close'
                                        volume='volume'
                                        name={currentSymbol}
                                    />
                                </StockChartSeriesCollectionDirective>
                            </StockChartComponent>
                        </div>
                    ) : (
                        <div className='flex items-center justify-center h-96 text-gray-500'>
                            <p>Enter a stock symbol to view the chart</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Chart;
