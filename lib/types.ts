export type StockRow = {
  symbol: string;
  date: string; // ISO
  close: number|null;
  high: number|null;
  low: number|null;
  volume: number|null;
  ['Prev Day High']?: number|null;
  EMA_20?: number|null;
  EMA_50?: number|null;
  EMA_100?: number|null;
  EMA_200?: number|null;
  RSI_14?: number|null;
  MACD?: number|null;
  MACD_Signal?: number|null;
  MACD_Hist?: number|null;
  BB_Upper?: number|null;
  BB_Lower?: number|null;

  // Derived fields
  price_above_prev_day_high?: boolean;
  price_change?: number|null;
  price_above_200EMA?: boolean;
  rsi_zone?: 'Oversold'|'Neutral'|'Overbought'|null;
  macd_trend?: 'Bullish'|'Bearish'|null;
  bb_position?: number|null;
  price_pct_change_day?: number|null;
  volume_pct_change_day?: number|null;
  rsi_cross_50?: 'Up'|'Down'|'No'|null;
  rsi_cross_20?: 'Up'|'Down'|'No'|null;
  near_ath_5?: boolean|null;
  total_trading_days?: number|null;
  w52_high?: number|null;
  w52_low?: number|null;
  rvol_20?: number|null;
  range5_pct?: number|null;
  range4_pct?: number|null;
  range7_pct?: number|null;
} & Record<string, any>;
