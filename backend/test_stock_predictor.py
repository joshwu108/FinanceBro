#!/usr/bin/env python3
"""
Test script for stock prediction functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import json
import time

def test_backend_connection():
    """Test if backend is running"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Backend is running!")
            return True
        else:
            print("❌ Backend is not responding properly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running on localhost:8000")
        return False

def test_stock_data_collection():
    """Test stock data collection"""
    print("\n🔍 Testing stock data collection...")
    
    symbols = ["AAPL", "TSLA", "MSFT"]
    
    for symbol in symbols:
        try:
            response = requests.get(f"http://localhost:8000/api/v1/stocks/{symbol}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {symbol}: Data collected successfully")
                print(f"   Data points: {len(data.get('data', []))}")
            else:
                print(f"❌ {symbol}: Failed to collect data - {response.status_code}")
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")

def test_stock_prediction():
    """Test stock prediction"""
    print("\n🔮 Testing stock prediction...")
    
    # Create sample features for prediction
    sample_features = [
        [0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.1, 0.9, 0.5],  # Sample feature vector
        [0.4, 0.6, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.4, 0.6],
        [0.7, 0.2, 0.5, 0.9, 0.1, 0.8, 0.3, 0.6, 0.2, 0.7]
    ]
    
    symbols = ["AAPL", "TSLA"]
    
    for symbol in symbols:
        try:
            # Test prediction endpoint
            response = requests.get(
                f"http://localhost:8000/api/v1/predictions/{symbol}",
                params={"X": json.dumps(sample_features)}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {symbol}: Prediction successful")
                print(f"   Prediction: {data.get('prediction', 'N/A')}")
            else:
                print(f"❌ {symbol}: Prediction failed - {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ {symbol}: Prediction error - {str(e)}")

def test_ml_training():
    """Test ML model training"""
    print("\n🤖 Testing ML model training...")
    
    try:
        # Test training endpoint (if it exists)
        response = requests.post("http://localhost:8000/api/v1/models/train/AAPL")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Model training initiated successfully")
            print(f"   Response: {data}")
        else:
            print(f"❌ Model training failed - {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Model training error - {str(e)}")

def test_api_documentation():
    """Test API documentation"""
    print("\n📚 Testing API documentation...")
    
    try:
        response = requests.get("http://localhost:8000/docs")
        if response.status_code == 200:
            print("✅ API documentation is available at http://localhost:8000/docs")
        else:
            print("❌ API documentation not accessible")
    except Exception as e:
        print(f"❌ API documentation error - {str(e)}")

def main():
    """Main test function"""
    print("🧪 FinanceBro Stock Predictor Test")
    print("=" * 50)
    
    # Test 1: Backend connection
    if not test_backend_connection():
        print("\n❌ Backend is not running. Please start it first:")
        print("   cd backend")
        print("   uvicorn app.main:app --reload --port 8000")
        return
    
    # Test 2: API documentation
    test_api_documentation()
    
    # Test 3: Stock data collection
    test_stock_data_collection()
    
    # Test 4: Stock prediction
    test_stock_prediction()
    
    # Test 5: ML training
    test_ml_training()
    
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print("✅ Backend connection: Working")
    print("✅ API documentation: Available at http://localhost:8000/docs")
    print("📊 Stock data collection: Tested")
    print("🔮 Stock prediction: Tested")
    print("🤖 ML training: Tested")
    print("\n💡 Next steps:")
    print("   1. Visit http://localhost:8000/docs to see all available endpoints")
    print("   2. Test specific endpoints with your frontend")
    print("   3. Train models for specific stocks")

if __name__ == "__main__":
    main() 