import { registerLicense } from '@syncfusion/ej2-base';

// Get license key from environment variable or use trial
const LICENSE_KEY = process.env.REACT_APP_SYNCFUSION_LICENSE_KEY || '';

console.log('Syncfusion License Key Length:', LICENSE_KEY.length);
console.log('Syncfusion License Key Starts With:', LICENSE_KEY.substring(0, 10) + '...');

// Only register if we have a license key
if (LICENSE_KEY && LICENSE_KEY.length > 10) {
    try {
        registerLicense(LICENSE_KEY);
        console.log('✅ Syncfusion license registered successfully');
    } catch (error) {
        console.error('❌ Failed to register Syncfusion license:', error);
    }
} else {
    console.warn('⚠️ No valid Syncfusion license key found - using trial version');
    console.warn('To remove trial message, add REACT_APP_SYNCFUSION_LICENSE_KEY to your .env file');
    
    // Uncomment the line below and replace with your actual license key for testing
    // registerLicense('YOUR_ACTUAL_LICENSE_KEY_HERE');
} 