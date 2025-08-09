import React from 'react'
import { Link, useLocation } from 'react-router-dom'

const Navbar = () => {
  const location = useLocation()

  const isActive = (path: string) => location.pathname === path

  return (
    <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">F</span>
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">
                FinanceBro
              </span>
            </Link>
          </div>

          {/* Navigation Links */}
          <div className="hidden md:flex justify-end space-x-1">
            <Link
              to="/"
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                isActive('/')
                  ? 'bg-gray-100 text-gray-900'
                  : 'text-gray-700 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              Dashboard
            </Link>
            <Link
              to="/stocks"
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                isActive('/stocks')
                  ? 'bg-gray-100 text-gray-900'
                  : 'text-gray-700 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              Stocks
            </Link>
            <Link
              to="/portfolio"
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                isActive('/portfolio')
                  ? 'bg-gray-100 text-gray-900'
                  : 'text-gray-700 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              Portfolio
            </Link>
            <Link
              to="/alerts"
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                isActive('/alerts')
                  ? 'bg-gray-100 text-gray-900'
                  : 'text-gray-700 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              Alerts
            </Link>
            <Link
              to="/chart"
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                isActive('/chart')
                  ? 'bg-gray-100 text-gray-900'
                  : 'text-gray-700 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              Chart
            </Link>
          </div>

          {/* Right side - mobile menu */}
          <div className="flex items-center space-x-4">
            {/* Mobile menu button */}
            <button className="md:hidden p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors duration-200">
              <svg className="w-5 h-5 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
