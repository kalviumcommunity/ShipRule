import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { AuthProvider } from '@/context/AuthContext';
import { Navbar } from '@/components/layout/Navbar';
import { IntroWrapper } from '@/components/intro/IntroWrapper';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'GFS | ShipRule CDLP — Global Customs Duty & Logistics Platform',
  description: 'Enterprise AI-powered Customs Duty & Shipping Documentation RAG Platform with Grounded Answers and Source Citations.',
  icons: {
    icon: '/logo.png',
    shortcut: '/logo.png',
    apple: '/logo.png',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" data-scroll-behavior="smooth" className="dark scroll-smooth">
      <body className={`${inter.className} min-h-screen bg-[#0B192C] text-slate-100 antialiased`}>
        <AuthProvider>
          <IntroWrapper>
            <div className="flex flex-col min-h-screen">
              <Navbar />
              <main className="flex-1 w-full bg-[#0B192C]">
                {children}
              </main>
            </div>
          </IntroWrapper>
        </AuthProvider>
      </body>
    </html>
  );
}


