import type { StockRow } from './types';

function groupBySymbol(rows: StockRow[]): Map<string, StockRow[]> {
  const map = new Map<string, StockRow[]>();
  for (const r of rows) {
    const key = r.symbol;
    if (!map.has(key)) map.set(key, []);
    map.get(key)!.push(r);
  }
  for (const arr of map.values()) {
    arr.sort((a,b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  }
  return map;
}

function rollingWindow<T>(arr: T[], i: number, n: number): T[] {
  const start = Math.max(0, i - (n - 1));
  return arr.slice(start, i + 1);
}

export function enhance(rows: StockRow[]): StockRow[] {
  const bySymbol = groupBySymbol(rows);
  const uniqueDates = new Set(rows.map(r => r.date));
  const totalTradingDays = uniqueDates.size;

  const maxDateStr = Array.from(uniqueDates).sort().at(-1)!;
  const maxDate = new Date(maxDateStr);
  const cutoff52 = new Date(maxDate.getTime() - 365 * 24 * 3600 * 1000);

  const perSymbolStats = new Map<string, {ath:number, w52high:number, w52low:number}>();

  // Precompute stats per symbol
  for (const [sym, arr] of bySymbol.entries()) {
    const highs = arr.map(r => (typeof r.high === 'number' ? r.high : -Infinity));
    const lows = arr.map(r => (typeof r.low === 'number' ? r.low : Infinity));
    const ath = Math.max(...highs);
    const last365 = arr.filter(r => new Date(r.date) >= cutoff52);
    const highs365 = last365.map(r => (typeof r.high === 'number' ? r.high : -Infinity));
    const lows365 = last365.map(r => (typeof r.low === 'number' ? r.low : Infinity));
    const w52high = highs365.length ? Math.max(...highs365) : ath;
    const w52low = lows365.length ? Math.min(...lows365) : Math.min(...lows);
    perSymbolStats.set(sym, { ath, w52high, w52low });
  }

  // Enhance each row
  for (const [sym, arr] of bySymbol.entries()) {
    const stats = perSymbolStats.get(sym)!;
    for (let i = 0; i < arr.length; i++) {
      const r = arr[i];
      const prev = i > 0 ? arr[i-1] : undefined;

      const close = typeof r.close === 'number' ? r.close : null;
      const ema50 = typeof r.EMA_50 === 'number' ? r.EMA_50 : null;
      const ema200 = typeof r.EMA_200 === 'number' ? r.EMA_200 : null;
      const prevHigh = typeof r['Prev Day High'] === 'number' ? r['Prev Day High'] : null;
      const rsi = typeof r.RSI_14 === 'number' ? r.RSI_14 : null;
      const prevRSI = typeof prev?.RSI_14 === 'number' ? prev!.RSI_14 : null;
      const macd = typeof r.MACD === 'number' ? r.MACD : null;
      const macds = typeof r.MACD_Signal === 'number' ? r.MACD_Signal : null;
      const bbU = typeof r.BB_Upper === 'number' ? r.BB_Upper : null;
      const bbL = typeof r.BB_Lower === 'number' ? r.BB_Lower : null;

      r.price_above_prev_day_high = (close != null && prevHigh != null) ? (close > prevHigh) : false;
      r.price_change = (close != null && ema50 != null) ? (close - ema50) : null;
      r.price_above_200EMA = (close != null && ema200 != null) ? (close > ema200) : false;
      r.rsi_zone = rsi == null ? null : (rsi < 30 ? 'Oversold' : rsi > 70 ? 'Overbought' : 'Neutral');
      r.macd_trend = (macd != null && macds != null) ? (macd > macds ? 'Bullish' : 'Bearish') : null;
      r.bb_position = (bbU != null && bbL != null && (bbU - bbL) !== 0) ? ((close! - bbL) / (bbU - bbL)) : null;

      if (prev && typeof prev.close === 'number' && close != null && prev.close !== 0) {
        r.price_pct_change_day = ((close - prev.close) / prev.close) * 100;
      } else {
        r.price_pct_change_day = null;
      }
      if (prev && typeof prev.volume === 'number' && typeof r.volume === 'number' && prev.volume !== 0) {
        r.volume_pct_change_day = ((r.volume - prev.volume) / prev.volume) * 100;
      } else {
        r.volume_pct_change_day = null;
      }

      // RSI crossovers at 50 and 20
      const cross = (prevVal: number|null, curVal: number|null, level: number) => {
        if (prevVal == null || curVal == null) return null;
        if (prevVal <= level && curVal > level) return 'Up';
        if (prevVal >= level && curVal < level) return 'Down';
        return 'No';
      };
      r.rsi_cross_50 = cross(prevRSI, rsi, 50);
      r.rsi_cross_20 = cross(prevRSI, rsi, 20);

      // ATH proximity (within 5%)
      const ath = stats.ath;
      r.near_ath_5 = (typeof ath === 'number' && close != null && ath > 0) ? ((ath - close) / ath <= 0.05) : null;

      // 52-week
      r.total_trading_days = totalTradingDays;
      r.w52_high = stats.w52high;
      r.w52_low = stats.w52low;

      // RVOL(20) using previous 20 days
      if (i >= 20) {
        const prev20 = arr.slice(i-20, i);
        const sum = prev20.reduce((s, x) => s + (typeof x.volume === 'number' ? x.volume : 0), 0);
        const avg20 = sum / 20;
        r.rvol_20 = avg20 ? (r.volume as number) / avg20 : null;
      } else {
        r.rvol_20 = null;
      }

      // Ranges (5, 4, 7 days) inclusive of current
      const pctRange = (win: StockRow[]) => {
        const highs = win.map(x => (typeof x.high === 'number' ? x.high : -Infinity));
        const lows  = win.map(x => (typeof x.low === 'number' ? x.low : Infinity));
        const hi = Math.max(...highs), lo = Math.min(...lows);
        return (close != null && close !== 0) ? ((hi - lo) / close) * 100 : null;
      };
      r.range5_pct = pctRange(rollingWindow(arr, i, 5));
      r.range4_pct = pctRange(rollingWindow(arr, i, 4));
      r.range7_pct = pctRange(rollingWindow(arr, i, 7));
    }
  }
  return rows;
}
