/**
 * Chart-specific types for XAI Trading Frontend
 */

// Chart.js and custom chart types
export interface ChartConfiguration {
  type: ChartType;
  data: ChartData;
  options: ChartOptions;
  plugins?: ChartPlugin[];
}

export type ChartType = 
  | 'line' 
  | 'bar' 
  | 'scatter' 
  | 'pie' 
  | 'doughnut' 
  | 'candlestick' 
  | 'heatmap' 
  | 'treemap'
  | 'waterfall'
  | 'funnel';

export interface ChartData {
  labels: string[];
  datasets: ChartDataset[];
}

export interface ChartDataset {
  label: string;
  data: (number | ChartPoint)[];
  backgroundColor?: string | string[] | CanvasGradient;
  borderColor?: string | string[];
  borderWidth?: number;
  fill?: boolean | string | number;
  tension?: number;
  pointRadius?: number | number[];
  pointHoverRadius?: number | number[];
  pointBackgroundColor?: string | string[];
  pointBorderColor?: string | string[];
  yAxisID?: string;
  type?: ChartType;
  stack?: string;
  order?: number;
  hidden?: boolean;
}

export interface ChartPoint {
  x: number | string;
  y: number;
  r?: number; // For bubble charts
}

export interface CandlestickData {
  x: string | number;
  o: number; // open
  h: number; // high
  l: number; // low
  c: number; // close
  v?: number; // volume
}

export interface ChartOptions {
  responsive?: boolean;
  maintainAspectRatio?: boolean;
  aspectRatio?: number;
  devicePixelRatio?: number;
  
  // Layout options
  layout?: {
    padding?: number | {
      left?: number;
      right?: number;
      top?: number;
      bottom?: number;
    };
  };
  
  // Scale options
  scales?: {
    [key: string]: ScaleOptions;
  };
  
  // Plugin options
  plugins?: {
    legend?: LegendOptions;
    tooltip?: TooltipOptions;
    title?: TitleOptions;
    [key: string]: any;
  };
  
  // Interaction options
  interaction?: {
    mode?: 'point' | 'nearest' | 'index' | 'dataset' | 'x' | 'y';
    intersect?: boolean;
    axis?: 'x' | 'y' | 'xy';
  };
  
  // Animation options
  animation?: {
    duration?: number;
    easing?: string;
    delay?: number;
    loop?: boolean;
  };
  
  // Event handling
  onHover?: (event: any, elements: any[]) => void;
  onClick?: (event: any, elements: any[]) => void;
}

export interface ScaleOptions {
  type?: 'linear' | 'logarithmic' | 'category' | 'time' | 'timeseries';
  position?: 'left' | 'right' | 'top' | 'bottom';
  display?: boolean;
  min?: number;
  max?: number;
  suggestedMin?: number;
  suggestedMax?: number;
  
  // Grid options
  grid?: {
    display?: boolean;
    color?: string | string[];
    lineWidth?: number | number[];
    drawBorder?: boolean;
    drawOnChartArea?: boolean;
    drawTicks?: boolean;
  };
  
  // Tick options
  ticks?: {
    display?: boolean;
    color?: string;
    font?: {
      family?: string;
      size?: number;
      style?: string;
      weight?: string;
    };
    callback?: (value: any, index: number, values: any[]) => string;
    stepSize?: number;
    maxTicksLimit?: number;
    precision?: number;
    format?: Intl.NumberFormatOptions;
  };
  
  // Title options
  title?: {
    display?: boolean;
    text?: string;
    color?: string;
    font?: {
      family?: string;
      size?: number;
      style?: string;
      weight?: string;
    };
  };
  
  // Time scale specific options
  time?: {
    unit?: 'millisecond' | 'second' | 'minute' | 'hour' | 'day' | 'week' | 'month' | 'quarter' | 'year';
    stepSize?: number;
    displayFormats?: {
      [key: string]: string;
    };
    tooltipFormat?: string;
  };
}

export interface LegendOptions {
  display?: boolean;
  position?: 'top' | 'left' | 'bottom' | 'right' | 'chartArea';
  align?: 'start' | 'center' | 'end';
  maxHeight?: number;
  maxWidth?: number;
  fullSize?: boolean;
  reverse?: boolean;
  
  labels?: {
    boxWidth?: number;
    boxHeight?: number;
    color?: string;
    font?: {
      family?: string;
      size?: number;
      style?: string;
      weight?: string;
    };
    padding?: number;
    generateLabels?: (chart: any) => any[];
    filter?: (legendItem: any, chartData: any) => boolean;
    sort?: (a: any, b: any, chartData: any) => number;
    usePointStyle?: boolean;
  };
  
  onClick?: (event: any, legendItem: any, legend: any) => void;
  onHover?: (event: any, legendItem: any, legend: any) => void;
  onLeave?: (event: any, legendItem: any, legend: any) => void;
}

export interface TooltipOptions {
  enabled?: boolean;
  external?: (context: any) => void;
  mode?: 'point' | 'nearest' | 'index' | 'dataset' | 'x' | 'y';
  intersect?: boolean;
  position?: 'average' | 'nearest';
  
  backgroundColor?: string;
  titleColor?: string;
  titleFont?: {
    family?: string;
    size?: number;
    style?: string;
    weight?: string;
  };
  titleAlign?: 'left' | 'center' | 'right';
  titleSpacing?: number;
  titleMarginBottom?: number;
  
  bodyColor?: string;
  bodyFont?: {
    family?: string;
    size?: number;
    style?: string;
    weight?: string;
  };
  bodyAlign?: 'left' | 'center' | 'right';
  bodySpacing?: number;
  
  footerColor?: string;
  footerFont?: {
    family?: string;
    size?: number;
    style?: string;
    weight?: string;
  };
  footerAlign?: 'left' | 'center' | 'right';
  footerSpacing?: number;
  footerMarginTop?: number;
  
  padding?: number;
  caretPadding?: number;
  caretSize?: number;
  cornerRadius?: number;
  multiKeyBackground?: string;
  displayColors?: boolean;
  borderColor?: string;
  borderWidth?: number;
  
  // Callback functions
  beforeTitle?: (tooltipItems: any[]) => string | string[];
  title?: (tooltipItems: any[]) => string | string[];
  afterTitle?: (tooltipItems: any[]) => string | string[];
  beforeBody?: (tooltipItems: any[]) => string | string[];
  beforeLabel?: (tooltipItem: any) => string | string[];
  label?: (tooltipItem: any) => string | string[];
  labelColor?: (tooltipItem: any) => { borderColor: string; backgroundColor: string };
  labelTextColor?: (tooltipItem: any) => string;
  afterLabel?: (tooltipItem: any) => string | string[];
  afterBody?: (tooltipItems: any[]) => string | string[];
  beforeFooter?: (tooltipItems: any[]) => string | string[];
  footer?: (tooltipItems: any[]) => string | string[];
  afterFooter?: (tooltipItems: any[]) => string | string[];
}

export interface TitleOptions {
  display?: boolean;
  text?: string | string[];
  color?: string;
  font?: {
    family?: string;
    size?: number;
    style?: string;
    weight?: string;
  };
  padding?: number;
  position?: 'top' | 'left' | 'bottom' | 'right';
  align?: 'start' | 'center' | 'end';
}

export interface ChartPlugin {
  id: string;
  beforeInit?: (chart: any, args: any, options: any) => void;
  beforeUpdate?: (chart: any, args: any, options: any) => void;
  beforeDraw?: (chart: any, args: any, options: any) => void;
  afterDraw?: (chart: any, args: any, options: any) => void;
  beforeEvent?: (chart: any, args: any, options: any) => void;
  afterEvent?: (chart: any, args: any, options: any) => void;
  resize?: (chart: any, args: any, options: any) => void;
  destroy?: (chart: any) => void;
}

// Trading-specific chart types
export interface TradingChartConfig {
  symbol: string;
  timeframe: string;
  chartType: TradingChartType;
  indicators: TechnicalIndicator[];
  overlays: ChartOverlay[];
  annotations: ChartAnnotation[];
  theme: ChartTheme;
}

export type TradingChartType = 
  | 'candlestick' 
  | 'ohlc' 
  | 'line' 
  | 'area' 
  | 'mountain' 
  | 'heikin_ashi'
  | 'renko'
  | 'point_figure';

export interface TechnicalIndicator {
  type: IndicatorType;
  params: Record<string, number>;
  color: string;
  visible: boolean;
  panel?: 'main' | 'volume' | 'oscillator';
}

export type IndicatorType = 
  | 'sma' 
  | 'ema' 
  | 'rsi' 
  | 'macd' 
  | 'bollinger_bands' 
  | 'stochastic'
  | 'atr'
  | 'volume'
  | 'vwap'
  | 'fibonacci';

export interface ChartOverlay {
  type: 'trendline' | 'horizontal_line' | 'vertical_line' | 'rectangle' | 'fibonacci';
  points: ChartPoint[];
  style: {
    color: string;
    width: number;
    dashArray?: number[];
  };
  label?: string;
}

export interface ChartAnnotation {
  type: 'arrow' | 'text' | 'callout' | 'shape';
  position: ChartPoint;
  content: string;
  style: {
    backgroundColor?: string;
    borderColor?: string;
    textColor?: string;
    fontSize?: number;
  };
}

export interface ChartTheme {
  name: string;
  backgroundColor: string;
  gridColor: string;
  textColor: string;
  candleColors: {
    up: string;
    down: string;
    upWick: string;
    downWick: string;
  };
  volumeColors: {
    up: string;
    down: string;
  };
  crosshairColor: string;
}

// Performance chart specific types
export interface PerformanceChartData {
  dates: string[];
  portfolioValue: number[];
  benchmark?: number[];
  drawdowns: number[];
  returns: number[];
  trades: TradeMarker[];
}

export interface TradeMarker {
  date: string;
  type: 'buy' | 'sell';
  price: number;
  size: number;
  symbol: string;
  reason?: string;
}

// Risk chart types
export interface RiskChartData {
  varHistory: {
    dates: string[];
    values: number[];
    confidence: number[];
  };
  correlationMatrix: {
    symbols: string[];
    correlations: number[][];
  };
  exposureBreakdown: {
    categories: string[];
    values: number[];
  };
}

// Heatmap specific types
export interface HeatmapData {
  x: string;
  y: string;
  value: number;
  color?: string;
}

export interface HeatmapConfig {
  data: HeatmapData[];
  colorScale: {
    domain: [number, number];
    range: [string, string];
  };
  cellSize: number;
  margin: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
}

// Chart utility types
export interface ChartExportOptions {
  format: 'png' | 'jpg' | 'svg' | 'pdf';
  width: number;
  height: number;
  quality?: number;
  backgroundColor?: string;
}

export interface ChartInteractionState {
  hoveredElements: any[];
  selectedElements: any[];
  zoom: {
    x: number;
    y: number;
    scale: number;
  };
  pan: {
    x: number;
    y: number;
  };
}