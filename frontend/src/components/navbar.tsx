import React from 'react'
import { Link } from 'react-router-dom'

const navbar = () => {
  return (
    <nav className='flex justify-between items-center p-6 bg-green-900 shadow-lg'>
      <h1 className='text-2xl font-bold text-white'>FinanceBro</h1>
      <div className='flex items-center gap-6'>
        <Link to='/' className='text-white hover:text-blue-200 transition-colors duration-200 font-medium'>
          Dashboard
        </Link>
        <Link to='/stocks' className='text-white hover:text-blue-200 transition-colors duration-200 font-medium'>
          Stocks
        </Link>
        <Link to='/portfolio' className='text-white hover:text-blue-200 transition-colors duration-200 font-medium'>
          Portfolio
        </Link>
        <Link to='/alerts' className='text-white hover:text-blue-200 transition-colors duration-200 font-medium'>
          Alerts
        </Link>
        <Link to='/chart' className='text-white hover:text-blue-200 transition-colors duration-200 font-medium'>
          Chart
        </Link>
        <a href='#aboutus' className='text-white hover:text-blue-200 transition-colors duration-200 font-medium cursor-pointer'>
          About
        </a>
      </div>
    </nav>
  )
}

export default navbar
