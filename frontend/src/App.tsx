import React from 'react';
import logo from './logo.svg';
import './index.css';
import './App.css';
import { Homepage, Navbar, Dashboard, Stocks, Portfolio, Alerts } from './components';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

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
        </Routes>
      </Router>
    </div>
  );
}

export default App;
