'use client';

import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';

type Msg = { role: 'user' | 'assistant'; content: string; ts?: number };

export default function Page() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Auto-scroll to bottom on new messages
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, loading]);

  async function send() {
    const q = input.trim();
    if (!q) return;
    const now = Date.now();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: q, ts: now }]);
    setLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q }),
      });
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.reply ?? '⚠️ No reply', ts: Date.now() }]);
    } catch (e: any) {
      setMessages(prev => [...prev, { role: 'assistant', content: '❌ Error: ' + (e?.message || e), ts: Date.now() }]);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">📱 AI Stock Assistant</div>
        <div className="subtitle">Chat with your stock data like WhatsApp!</div>
      </header>

      <main className="stage">
        <div className="chatCard">
          <div className="messages" ref={listRef}>
            {messages.map((m, i) => (
              <div key={i} className={m.role === 'user' ? 'bubble user' : 'bubble bot'}>
                {m.role === 'assistant'
                  ? <ReactMarkdown>{m.content}</ReactMarkdown>
                  : <span>{m.content}</span>}
                {m.ts && <div className="timestamp">{new Date(m.ts).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}</div>}
              </div>
            ))}
            {loading && <div className="bubble bot">💭 Thinking…</div>}
          </div>

          <div className="composer">
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Type your message… e.g. rsi above 60"
              aria-label="Chat input"
            />
            <button onClick={send} aria-label="Send">Send</button>
          </div>
        </div>
      </main>

      <footer className="footnote">
        <span>Ask about RSI, MACD, breakouts, and trend levels 📈</span>
      </footer>
    </div>
  );
}
