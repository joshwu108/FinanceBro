import './index.css';
import './App.css';
import { Homepage, Navbar, Dashboard, Stocks, Portfolio, Alerts, Chart } from './components';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

function App() {
    return (
        <div className="App">
            <Router>
                <Navbar />
                <Routes>
                    <Route path='/' element={<Homepage />} />
                    <Route path='/dashboard' element={<Dashboard />} />
                    <Route path='/stocks' element={<Stocks />} />
                    <Route path='/portfolio' element={<Portfolio />} />
                    <Route path='/alerts' element={<Alerts />} />
                    <Route path='/chart' element={<Chart />} />
                </Routes>
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
            />
        </div>
    );
}

export default App;
