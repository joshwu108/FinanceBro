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

    if (loading) return <div className="min-h-screen bg-gray-50 p-6 flex items-center justify-center">
        <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading alerts...</p>
        </div>
    </div>;

    return (
        <div className='min-h-screen bg-gray-50'>
            <div className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8'>
                {/* Header */}
                <div className='mb-8'>
                    <h1 className='text-4xl font-bold text-gray-900 mb-2'>Alerts</h1>
                    <p className='text-gray-600'>Set up price and volume alerts with AI-powered insights</p>
                </div>

                {/* Alert Stats */}
                <div className='grid grid-cols-1 md:grid-cols-4 gap-6 mb-8'>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Active Alerts</h3>
                        <p className='text-3xl font-bold text-gray-800'>{activeAlerts.length}</p>
                        <p className='text-sm text-gray-600'>Currently monitoring</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Triggered Alerts</h3>
                        <p className='text-3xl font-bold text-orange-600'>{triggeredAlerts.length}</p>
                        <p className='text-sm text-orange-500'>Need attention</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Total Alerts</h3>
                        <p className='text-3xl font-bold text-gray-800'>{allAlerts.length}</p>
                        <p className='text-sm text-gray-600'>All time</p>
                    </div>
                    <div className='bg-white p-6 rounded-lg shadow-md'>
                        <h3 className='text-sm font-medium text-gray-500 mb-2'>Success Rate</h3>
                        <p className='text-3xl font-bold text-green-600'>85%</p>
                        <p className='text-sm text-green-500'>Accurate predictions</p>
                    </div>
                </div>

                {/* Action Buttons */}
                <div className='flex gap-4 mb-8'>
                    <button
                        onClick={() => checkAlerts(1)}
                        className='bg-gray-800 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors'
                    >
                        Check Alerts
                    </button>
                    <button
                        onClick={() => setShowCreateForm(true)}
                        className='bg-gray-800 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors'
                    >
                        Create Alert
                    </button>
                </div>

                {/* Alert Tabs */}
                <div className='bg-white rounded-lg shadow-md'>
                    <div className='border-b border-gray-200'>
                        <nav className='flex space-x-8 px-6'>
                            <button
                                onClick={() => setActiveTab('active')}
                                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                                    activeTab === 'active'
                                        ? 'border-gray-800 text-gray-800'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                }`}
                            >
                                Active Alerts ({activeAlerts.length})
                            </button>
                            <button
                                onClick={() => setActiveTab('triggered')}
                                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                                    activeTab === 'triggered'
                                        ? 'border-gray-800 text-gray-800'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                }`}
                            >
                                Triggered Alerts ({triggeredAlerts.length})
                            </button>
                            <button
                                onClick={() => setActiveTab('all')}
                                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                                    activeTab === 'all'
                                        ? 'border-gray-800 text-gray-800'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                }`}
                            >
                                All Alerts ({allAlerts.length})
                            </button>
                        </nav>
                    </div>

                    {/* Alert List */}
                    <div className='p-6'>
                        {activeTab === 'active' && (
                            <div className='space-y-4'>
                                {activeAlerts.map((alert) => (
                                    <div key={alert.id} className='flex items-center justify-between p-4 border border-gray-200 rounded-lg'>
                                        <div className='flex items-center space-x-4'>
                                            <div className='w-10 h-10 bg-gray-500 rounded-full flex items-center justify-center text-white font-bold'>
                                                {alert.stock_symbol.charAt(0)}
                                            </div>
                                            <div>
                                                <h3 className='font-semibold text-gray-900'>{alert.stock_symbol}</h3>
                                                <p className='text-sm text-gray-600'>{alert.alert_type}</p>
                                            </div>
                                        </div>
                                        <div className='text-right'>
                                            <p className='font-semibold text-gray-900'>${alert.target_price}</p>
                                            <p className='text-sm text-gray-600'>{alert.status}</p>
                                        </div>
                                        <div className='flex space-x-2'>
                                            <button
                                                onClick={() => updateAlert(alert.id, { status: 'triggered' })}
                                                className='text-gray-600 hover:text-gray-800 underline'
                                            >
                                                Edit
                                            </button>
                                            <button
                                                onClick={() => deleteAlert(alert.id)}
                                                className='text-red-600 hover:text-red-800 underline'
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {activeTab === 'triggered' && (
                            <div className='space-y-4'>
                                {triggeredAlerts.map((alert) => (
                                    <div key={alert.id} className='flex items-center justify-between p-4 border border-orange-200 bg-orange-50 rounded-lg'>
                                        <div className='flex items-center space-x-4'>
                                            <div className='w-10 h-10 bg-orange-500 rounded-full flex items-center justify-center text-white font-bold'>
                                                {alert.stock_symbol.charAt(0)}
                                            </div>
                                            <div>
                                                <h3 className='font-semibold text-gray-900'>{alert.stock_symbol}</h3>
                                                <p className='text-sm text-gray-600'>{alert.alert_type}</p>
                                            </div>
                                        </div>
                                        <div className='text-right'>
                                            <p className='font-semibold text-gray-900'>${alert.target_price}</p>
                                            <p className='text-sm text-orange-600'>{alert.status}</p>
                                        </div>
                                        <div className='flex space-x-2'>
                                            <button
                                                onClick={() => updateAlert(alert.id, { status: 'active' })}
                                                className='text-gray-600 hover:text-gray-800 underline'
                                            >
                                                Reactivate
                                            </button>
                                            <button
                                                onClick={() => deleteAlert(alert.id)}
                                                className='text-red-600 hover:text-red-800 underline'
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {activeTab === 'all' && (
                            <div className='space-y-4'>
                                {allAlerts.map((alert) => (
                                    <div key={alert.id} className='flex items-center justify-between p-4 border border-gray-200 rounded-lg'>
                                        <div className='flex items-center space-x-4'>
                                            <div className='w-10 h-10 bg-gray-500 rounded-full flex items-center justify-center text-white font-bold'>
                                                {alert.stock_symbol.charAt(0)}
                                            </div>
                                            <div>
                                                <h3 className='font-semibold text-gray-900'>{alert.stock_symbol}</h3>
                                                <p className='text-sm text-gray-600'>{alert.alert_type}</p>
                                            </div>
                                        </div>
                                        <div className='text-right'>
                                            <p className='font-semibold text-gray-900'>${alert.target_price}</p>
                                            <p className='text-sm text-gray-600'>{alert.status}</p>
                                        </div>
                                        <div className='flex space-x-2'>
                                            <button
                                                onClick={() => updateAlert(alert.id, { status: 'active' })}
                                                className='text-gray-600 hover:text-gray-800 underline'
                                            >
                                                Edit
                                            </button>
                                            <button
                                                onClick={() => deleteAlert(alert.id)}
                                                className='text-red-600 hover:text-red-800 underline'
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Alerts; 