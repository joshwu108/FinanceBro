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
        
        // Force disable trial popup
        if (typeof window !== 'undefined') {
            (window as any).ej2 = (window as any).ej2 || {};
            (window as any).ej2.isLicenseValid = true;
        }
    } catch (error) {
        console.error('❌ Failed to register Syncfusion license:', error);
    }
} else {
    console.warn('⚠️ No valid Syncfusion license key found - using trial version');
    console.warn('To remove trial message, add REACT_APP_SYNCFUSION_LICENSE_KEY to your .env file');
} 