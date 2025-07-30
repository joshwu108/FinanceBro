import React, { useState, useEffect } from 'react';
import apiService, { Alert } from '../services/api';
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const Alerts = () => {
    const [activeTab, setActiveTab] = useState('active');
    const [alerts, setAlerts] = useState<Alert[]>([]);
    const [loading, setLoading] = useState(false);
    const [showCreateForm, setShowCreateForm] = useState(false);

    const createAlert = async(stockSymbol: string, alertType: string, targetPrice: number, targetPercent: number, message: string, userId: number) => {
        try {
            const response = await apiService.createAlert(stockSymbol, alertType, targetPrice, targetPercent, message, userId);
            console.log(response);
            toast.success('Alert created successfully');
        } catch (error) {
            toast.error(`Error creating alert: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    };

    const updateAlert = async(alertId: number, updates: {
        alert_type?: string;
        target_price?: number;
        target_percent?: number;
        message?: string;
        status?: string;
    }) => {
        try {
            const response = await apiService.updateAlert(alertId, updates);
            toast.success('Alert updated successfully');
        } catch (error) {
            toast.error(`Error updating alert: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    };

    const deleteAlert = async(alertId: number) => {
        try {
            const response = await apiService.deleteAlert(alertId);
            toast.success('Alert deleted successfully');
        } catch (error) {
            toast.error(`Error deleting alert: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    };

    const checkAlerts = async(userId: number) => {
        try {
            const response = await apiService.checkAlerts(userId);
            console.log(response);
            toast.success(`Checked ${response.checked_alerts} alerts, triggered ${response.triggered_alerts.length}`);
            await loadAlerts(); // Reload alerts after checking
        } catch (error) {
            toast.error(`Error checking alerts: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    };

    // Load alerts on component mount
    useEffect(() => {
        loadAlerts();
    }, []);

    const loadAlerts = async () => {
        try {
            setLoading(true);
            const response = await apiService.getAlerts(1); // user_id = 1
            setAlerts(response.alerts);
        } catch (error) {
            toast.error(`Error loading alerts: ${error instanceof Error ? error.message : 'Unknown error'}`);
        } finally {
            setLoading(false);
        }
    };

    // Filter alerts based on active tab
    const activeAlerts = alerts.filter(alert => alert.status === 'active');
    const triggeredAlerts = alerts.filter(alert => alert.status === 'triggered');
    const allAlerts = alerts;

    if (loading) return <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6 flex items-center justify-center">
        <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading alerts...</p>
        </div>
    </div>;

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
                        <p className='text-3xl font-bold text-blue-600'>{activeAlerts.length}</p>
                        <p className='text-sm text-blue-500'>Currently monitoring</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Triggered Today</h3>
                        <p className='text-3xl font-bold text-green-600'>{triggeredAlerts.filter(a => a.triggered_at && a.triggered_at.includes('2024-01-15')).length}</p>
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
                    <div className='flex justify-between items-center mb-4'>
                        <h2 className='text-2xl font-bold text-gray-800'>Create New Alert</h2>
                        <button
                            onClick={() => setShowCreateForm(!showCreateForm)}
                            className='bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors'
                        >
                            {showCreateForm ? 'Cancel' : 'Create Alert'}
                        </button>
                    </div>
                    
                    {showCreateForm && (
                        <form onSubmit={(e) => {
                            e.preventDefault();
                            const formData = new FormData(e.currentTarget);
                            createAlert(
                                formData.get('stockSymbol') as string,
                                formData.get('alertType') as string,
                                parseFloat(formData.get('targetPrice') as string),
                                undefined as any, // targetPercent
                                formData.get('message') as string,
                                1 // user_id
                            );
                            setShowCreateForm(false);
                        }}>
                            <div className='grid grid-cols-1 md:grid-cols-4 gap-4'>
                                <div>
                                    <label className='block text-sm font-medium text-gray-700 mb-2'>Stock Symbol</label>
                                    <input 
                                        name="stockSymbol"
                                        type='text' 
                                        placeholder='AAPL' 
                                        className='w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent'
                                        required
                                    />
                                </div>
                                <div>
                                    <label className='block text-sm font-medium text-gray-700 mb-2'>Alert Type</label>
                                    <select 
                                        name="alertType"
                                        className='w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent'
                                        required
                                    >
                                        <option value="price_above">Price Above</option>
                                        <option value="price_below">Price Below</option>
                                        <option value="percent_change">Percent Change</option>
                                    </select>
                                </div>
                                <div>
                                    <label className='block text-sm font-medium text-gray-700 mb-2'>Target Price</label>
                                    <input 
                                        name="targetPrice"
                                        type='number' 
                                        step="0.01"
                                        placeholder='175.00' 
                                        className='w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent'
                                        required
                                    />
                                </div>
                                <div>
                                    <label className='block text-sm font-medium text-gray-700 mb-2'>Message (Optional)</label>
                                    <input 
                                        name="message"
                                        type='text' 
                                        placeholder='Alert message' 
                                        className='w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent'
                                    />
                                </div>
                            </div>
                            <div className='mt-4 flex gap-4'>
                                <button 
                                    type="submit"
                                    className='bg-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors'
                                >
                                    Create Alert
                                </button>
                                <button 
                                    type="button"
                                    onClick={() => checkAlerts(1)}
                                    className='bg-green-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-green-700 transition-colors'
                                >
                                    Check Alerts
                                </button>
                            </div>
                        </form>
                    )}
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
                                Alert History ({triggeredAlerts.length})
                            </button>
                        </nav>
                    </div>

                    <div className='p-6'>
                        {activeTab === 'active' && (
                            <div className='space-y-4'>
                                {activeAlerts.map((alert) => (
                                    <div key={alert.id} className='flex items-center justify-between p-4 bg-gray-50 rounded-lg'>
                                        <div className='flex items-center'>
                                            <div className='w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold'>
                                                {alert.stock_symbol.charAt(0)}
                                            </div>
                                            <div className='ml-4'>
                                                <h3 className='font-semibold text-gray-800'>{alert.stock_symbol}</h3>
                                                <p className='text-sm text-gray-600'>
                                                    {alert.alert_type} {alert.target_price ? `$${alert.target_price}` : ''}
                                                </p>
                                            </div>
                                        </div>
                                        <div className='flex items-center space-x-4'>
                                            <span className='inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800'>
                                                {alert.status}
                                            </span>
                                            <button 
                                                onClick={() => deleteAlert(alert.id)}
                                                className='text-red-600 hover:text-red-800 text-sm font-medium'
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {activeTab === 'history' && (
                            <div className='space-y-4'>
                                {triggeredAlerts.map((alert) => (
                                    <div key={alert.id} className='flex items-center justify-between p-4 bg-gray-50 rounded-lg'>
                                        <div className='flex items-center'>
                                            <div className='w-10 h-10 bg-green-500 rounded-full flex items-center justify-center text-white font-bold'>
                                                {alert.stock_symbol.charAt(0)}
                                            </div>
                                            <div className='ml-4'>
                                                <h3 className='font-semibold text-gray-800'>{alert.stock_symbol}</h3>
                                                <p className='text-sm text-gray-600'>
                                                    {alert.alert_type} {alert.target_price ? `$${alert.target_price}` : ''}
                                                </p>
                                                <p className='text-xs text-gray-500'>Triggered: {alert.triggered_at}</p>
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