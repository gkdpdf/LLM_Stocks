import path from 'path';
import { loadCSV } from './csv';
import { enhance } from './signals';
import type { StockRow } from './types';

let cache: Promise<StockRow[]> | null = null;

function resolveCSVPath(): string {
  const p = process.env.CSV_PATH || 'data/sample.csv';
  return path.isAbsolute(p) ? p : path.join(process.cwd(), p);
}

export function getData(): Promise<StockRow[]> {
  if (!cache) {
    cache = (async () => {
      const rows = loadCSV(resolveCSVPath());
      return enhance(rows);
    })();
  }
  return cache;
}
