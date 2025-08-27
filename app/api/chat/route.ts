import { NextRequest, NextResponse } from 'next/server';
import { getData } from '../../../lib/data';
import type { StockRow } from '../../../lib/types';
import OpenAI from 'openai';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function systemPrompt(): string {
  return [
    'You are a WhatsApp-style financial assistant that helps analyze a stock dataset.',
    'Columns include:',
    '- symbol, date, close, high, low, volume',
    '- EMA_20, EMA_50, EMA_100, EMA_200',
    '- RSI_14',
    '- MACD, MACD_Signal, MACD_Hist',
    '- BB_Upper, BB_Lower',
    '- Prev Day High',
    '- Extra derived fields: price_above_prev_day_high, price_change, price_above_200EMA, rsi_zone, macd_trend, bb_position,',
    '  price_pct_change_day, volume_pct_change_day, rsi_cross_50, rsi_cross_20, near_ath_5, total_trading_days,',
    '  w52_high, w52_low, rvol_20, range5_pct, range4_pct, range7_pct.',
    '',
    'RULE: If user asks for "stocks above previous day high", return only rows where close > Prev Day High. Do not use high > Prev Day High.',
    'Reply concisely with emojis and relevant tickers only, like a WhatsApp chat.'
  ].join('\n');
}

function pickFields(r: StockRow) {
  const keep = ['symbol','date','close','high','low','volume','EMA_50','EMA_200','RSI_14','MACD','MACD_Signal','BB_Upper','BB_Lower','Prev Day High',
    'price_above_prev_day_high','price_change','price_above_200EMA','rsi_zone','macd_trend','bb_position',
    'price_pct_change_day','volume_pct_change_day','rsi_cross_50','rsi_cross_20','near_ath_5','total_trading_days',
    'w52_high','w52_low','rvol_20','range5_pct','range4_pct','range7_pct'
  ] as const;
  const obj: Record<string, any> = {};
  for (const k of keep) obj[k] = (r as any)[k];
  return obj;
}

function sample<T>(arr: T[], n: number): T[] {
  const copy = arr.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, n);
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const q: string = String(body?.query || '');

    const rows = await getData();

    let filtered = rows;
    if (q.toLowerCase().includes('above previous day high')) {
      filtered = rows.filter(r => r.price_above_prev_day_high === true);
    }

    const subset = sample(filtered, Math.min(50, filtered.length)).map(pickFields);
    const prompt = [
      'USER QUERY:',
      q,
      '',
      '--- STOCK DATA SAMPLE START ---',
      JSON.stringify(subset, null, 2),
      '--- STOCK DATA SAMPLE END ---',
      '',
      '🎯 Respond only with relevant stocks. Format like a WhatsApp chat. Use emojis.'
    ].join('\n');

    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    const resp = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      temperature: 0.3,
      max_tokens: 1000,
      messages: [
        { role: 'system', content: systemPrompt() },
        { role: 'user', content: prompt }
      ]
    });

    const reply = resp.choices?.[0]?.message?.content ?? '⚠️ Empty response';
    return NextResponse.json({ reply });
  } catch (err: any) {
    return NextResponse.json({ error: err?.message ?? String(err) }, { status: 500 });
  }
}
