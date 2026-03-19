import './index.css';
import './App.css';
import { Homepage, Navbar, Dashboard, Stocks, Portfolio, Alerts, Chart } from './components';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

function App() {
    return (
        <div className="App min-h-screen bg-gray-50">
            <Router>
                <Navbar />
                <main className="pt-16">
                    <Routes>
                        <Route path='/' element={<Homepage />} />
                        <Route path='/dashboard' element={<Dashboard />} />
                        <Route path='/stocks' element={<Stocks />} />
                        <Route path='/portfolio' element={<Portfolio />} />
                        <Route path='/alerts' element={<Alerts />} />
                        <Route path='/chart' element={<Chart />} />
                    </Routes>
                </main>
            </Router>
            <ToastContainer
                position="top-right"
                autoClose={5000}
                hideProgressBar={false}
                newestOnTop={false}
                closeOnClick
                rtl={false}
                pauseOnFocusLoss
                draggable
                pauseOnHover
                theme="light"
                toastClassName="bg-white text-gray-900"
            />
        </div>
    );
}

export default App;
