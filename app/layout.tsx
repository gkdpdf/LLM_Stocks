import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: '📱 AI Stock Assistant',
  description: 'Chat with your stock data like WhatsApp!',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
