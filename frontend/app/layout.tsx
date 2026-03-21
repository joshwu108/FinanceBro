import type { Metadata } from 'next'
import { IBM_Plex_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600', '700'],
  variable: '--font-ibm-plex-mono',
})

export const metadata: Metadata = {
  title: {
    default: 'FinanceBro',
    template: '%s | FinanceBro',
  },
  description: 'Quantitative research and trading platform — backtesting, experiment tracking, real-time inference.',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={ibmPlexMono.variable}>
      <body className="font-mono antialiased bg-[#0B0F14] text-[#E6EDF3]">
        {children}
        <Analytics />
      </body>
    </html>
  )
}
