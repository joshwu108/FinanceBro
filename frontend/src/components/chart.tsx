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
                        <option value="1d">Daily</option>
                        <option value="1h">Hourly</option>
                        <option value="30m">30 Min</option>
                        <option value="15m">15 Min</option>
                        <option value="5m">5 Min</option>
                        <option value="1m">1 Min</option>
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
                
                {chartData.length > 0 && (
                    <div className='h-96 bg-white rounded-lg shadow-sm p-4'>
                        <StockChartComponent
                            id="stock-chart"
                            primaryXAxis={{
                                valueType: 'DateTime',
                                majorGridLines: { color: '#f0f0f0' },
                                majorTickLines: { color: '#666' },
                                labelStyle: { color: '#666' }
                            }}
                            primaryYAxis={{
                                majorGridLines: { color: '#f0f0f0' },
                                majorTickLines: { color: '#666' },
                                labelStyle: { color: '#666' },
                                labelFormat: '${value}'
                            }}
                            chartArea={{
                                border: { width: 0 }
                            }}
                            tooltip={{
                                enable: true,
                                format: '${point.x} : ${point.y}'
                            }}
                            crosshair={{
                                enable: true,
                                lineType: 'Vertical'
                            }}
                            legendSettings={{
                                visible: false
                            }}
                            height="100%"
                            width="100%"
                        >
                            <Inject services={[DateTime, LineSeries, SplineSeries, CandleSeries, Tooltip]} />
                            <StockChartSeriesCollectionDirective>
                                <StockChartSeriesDirective
                                    dataSource={chartData}
                                    xName="x"
                                    yName="close"
                                    type="Candle"
                                    width={2}
                                    fill="#26a69a"
                                    border={{ color: '#26a69a', width: 2 }}
                                />
                            </StockChartSeriesCollectionDirective>
                        </StockChartComponent>
                    </div>
                )}
                
                {!chartData.length && !isLoading && !error && (
                    <div className='flex items-center justify-center h-64'>
                        <div className='text-lg text-gray-500'>Enter a stock symbol to view chart</div>
                    </div>
                )}
            </div>
        </div>
    )
}

export default Chart
