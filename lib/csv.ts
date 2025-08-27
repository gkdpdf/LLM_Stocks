import fs from 'fs';
import { parse } from 'csv-parse/sync';
import type { StockRow } from './types';

const NUMERIC_COLS = new Set<string>([
  'close','high','low','volume',
  'EMA_20','EMA_50','EMA_100','EMA_200',
  'RSI_14',
  'MACD','MACD_Signal','MACD_Hist',
  'BB_Upper','BB_Lower',
  'Prev Day High'
]);

function coerce(row: Record<string, any>): StockRow {
  const out: Record<string, any> = {};
  for (const k of Object.keys(row)) {
    const v = row[k];
    if (k === 'date') {
      const d = new Date(v);
      out[k] = isNaN(+d) ? String(v) : d.toISOString().slice(0,10);
    } else if (NUMERIC_COLS.has(k)) {
      const num = Number(v);
      out[k] = Number.isFinite(num) ? num : null;
    } else if (k.toLowerCase() === 'symbol') {
      out['symbol'] = String(v).trim();
    } else {
      out[k] = v;
    }
  }
  if (!('symbol' in out)) out['symbol'] = String(row['symbol'] ?? '');
  return out as StockRow;
}

export function loadCSV(path: string): StockRow[] {
  const txt = fs.readFileSync(path, 'utf8');
  const records = parse(txt, { columns: true, skip_empty_lines: true });
  return records.map((r: any) => coerce(r));
}
