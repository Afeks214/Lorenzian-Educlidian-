"""
Advanced Visualization System for GrandModel
==========================================

State-of-the-art visualization system with interactive charts, signal visualization,
pattern highlighting, performance analysis, and comprehensive risk metrics.

Features:
- Interactive price charts with technical indicators
- Signal visualization and pattern highlighting
- Performance charts and equity curves
- Risk metrics and drawdown analysis
- Correlation heatmaps and analysis
- Real-time data visualization
- Multi-strategy comparison charts
- Professional presentation quality

Author: Agent 6 - Visualization and Reporting System
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
import warnings
from pathlib import Path
import json
import io
import base64
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
from dataclasses import dataclass
import concurrent.futures
import asyncio

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class ChartConfig:
    """Configuration for chart appearance and behavior"""
    width: int = 1200
    height: int = 600
    template: str = "plotly_dark"
    color_scheme: Dict[str, str] = None
    font_family: str = "Arial"
    font_size: int = 12
    show_grid: bool = True
    show_legend: bool = True
    
    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'positive': '#28A745',
                'negative': '#DC3545',
                'warning': '#FFC107',
                'info': '#17A2B8',
                'neutral': '#6C757D',
                'background': '#F8F9FA'
            }


class AdvancedVisualization:
    """
    Advanced visualization system with interactive charts and comprehensive analysis
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize advanced visualization system
        
        Args:
            config: Chart configuration settings
        """
        self.config = config or ChartConfig()
        self.charts = {}
        self.themes = {
            'light': 'plotly_white',
            'dark': 'plotly_dark',
            'professional': 'simple_white'
        }
        
        # Create results directory
        self.results_dir = Path("/home/QuantNova/GrandModel/results/visualizations")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Advanced Visualization System initialized")
    
    def create_interactive_price_chart(self, 
                                     data: pd.DataFrame,
                                     indicators: Dict[str, pd.Series] = None,
                                     signals: Dict[str, pd.Series] = None,
                                     patterns: Dict[str, List[Tuple]] = None,
                                     title: str = "Interactive Price Chart") -> go.Figure:
        """
        Create advanced interactive price chart with indicators and signals
        
        Args:
            data: OHLCV data with datetime index
            indicators: Dictionary of technical indicators
            signals: Dictionary of trading signals
            patterns: Dictionary of pattern annotations
            title: Chart title
            
        Returns:
            Interactive plotly figure
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.2, 0.15, 0.15],
                subplot_titles=('Price Action', 'Volume', 'Indicators', 'Signals'),
                specs=[[{"secondary_y": True}], [{}], [{}], [{}]]
            )
            
            # Main price chart with candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color=self.config.color_scheme['positive'],
                    decreasing_line_color=self.config.color_scheme['negative'],
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add technical indicators
            if indicators:
                for name, indicator in indicators.items():
                    if name.lower() in ['sma', 'ema', 'bollinger', 'nwrqk']:
                        fig.add_trace(
                            go.Scatter(
                                x=indicator.index,
                                y=indicator.values,
                                mode='lines',
                                name=name,
                                line=dict(width=2),
                                opacity=0.8
                            ),
                            row=1, col=1
                        )
                    elif name.lower() in ['rsi', 'macd', 'stoch']:
                        fig.add_trace(
                            go.Scatter(
                                x=indicator.index,
                                y=indicator.values,
                                mode='lines',
                                name=name,
                                line=dict(width=2)
                            ),
                            row=3, col=1
                        )
            
            # Add trading signals
            if signals:
                for signal_name, signal_data in signals.items():
                    if 'long' in signal_name.lower():
                        signal_points = signal_data[signal_data == 1]
                        if not signal_points.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=signal_points.index,
                                    y=data.loc[signal_points.index, 'Low'] * 0.995,
                                    mode='markers',
                                    name=f'{signal_name} Long',
                                    marker=dict(
                                        symbol='triangle-up',
                                        size=12,
                                        color=self.config.color_scheme['positive']
                                    )
                                ),
                                row=1, col=1
                            )
                    
                    elif 'short' in signal_name.lower():
                        signal_points = signal_data[signal_data == 1]
                        if not signal_points.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=signal_points.index,
                                    y=data.loc[signal_points.index, 'High'] * 1.005,
                                    mode='markers',
                                    name=f'{signal_name} Short',
                                    marker=dict(
                                        symbol='triangle-down',
                                        size=12,
                                        color=self.config.color_scheme['negative']
                                    )
                                ),
                                row=1, col=1
                            )
                    
                    # Add signal strength as subplot
                    fig.add_trace(
                        go.Scatter(
                            x=signal_data.index,
                            y=signal_data.values,
                            mode='lines',
                            name=f'{signal_name} Strength',
                            line=dict(width=1),
                            opacity=0.7
                        ),
                        row=4, col=1
                    )
            
            # Add volume bars
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(data['Close'], data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add pattern annotations
            if patterns:
                for pattern_name, pattern_points in patterns.items():
                    for start_time, end_time, price_level in pattern_points:
                        fig.add_shape(
                            type="rect",
                            x0=start_time, y0=price_level * 0.99,
                            x1=end_time, y1=price_level * 1.01,
                            fillcolor=self.config.color_scheme['warning'],
                            opacity=0.3,
                            line=dict(width=0),
                            row=1, col=1
                        )
                        
                        fig.add_annotation(
                            x=start_time + (end_time - start_time) / 2,
                            y=price_level,
                            text=pattern_name,
                            showarrow=False,
                            font=dict(size=10),
                            bgcolor=self.config.color_scheme['warning'],
                            opacity=0.8,
                            row=1, col=1
                        )
            
            # Update layout
            fig.update_layout(
                title=title,
                template=self.config.template,
                height=self.config.height + 200,
                width=self.config.width,
                font=dict(family=self.config.font_family, size=self.config.font_size),
                showlegend=self.config.show_legend,
                xaxis_rangeslider_visible=False
            )
            
            # Update axes
            fig.update_xaxes(showgrid=self.config.show_grid)
            fig.update_yaxes(showgrid=self.config.show_grid)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interactive price chart: {e}")
            return go.Figure()
    
    def create_performance_dashboard(self, 
                                   performance_data: Dict[str, pd.Series],
                                   risk_metrics: Dict[str, float] = None,
                                   benchmark_data: pd.Series = None,
                                   title: str = "Performance Dashboard") -> go.Figure:
        """
        Create comprehensive performance dashboard
        
        Args:
            performance_data: Dictionary of performance time series
            risk_metrics: Dictionary of risk metrics
            benchmark_data: Benchmark performance data
            title: Dashboard title
            
        Returns:
            Interactive performance dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'Cumulative Returns', 'Rolling Sharpe Ratio', 'Drawdown Analysis',
                    'Monthly Returns', 'Risk Metrics', 'Return Distribution'
                ),
                specs=[[{}, {}, {}], [{}, {}, {}]]
            )
            
            # Get main performance series
            returns = performance_data.get('returns', pd.Series())
            if returns.empty:
                logger.warning("No returns data provided")
                return go.Figure()
            
            # 1. Cumulative Returns
            cumulative_returns = (1 + returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values,
                    mode='lines',
                    name='Strategy',
                    line=dict(color=self.config.color_scheme['primary'], width=2)
                ),
                row=1, col=1
            )
            
            if benchmark_data is not None:
                benchmark_cumulative = (1 + benchmark_data).cumprod()
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_cumulative.index,
                        y=benchmark_cumulative.values,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color=self.config.color_scheme['secondary'], width=2)
                    ),
                    row=1, col=1
                )
            
            # 2. Rolling Sharpe Ratio
            rolling_sharpe = self._calculate_rolling_sharpe(returns, window=252)
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    mode='lines',
                    name='Rolling Sharpe',
                    line=dict(color=self.config.color_scheme['info'], width=2)
                ),
                row=1, col=2
            )
            
            # Add reference lines
            fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=1, col=2)
            fig.add_hline(y=0.0, line_dash="dash", line_color="gray", row=1, col=2)
            
            # 3. Drawdown Analysis
            drawdown = self._calculate_drawdown(cumulative_returns)
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    fill='tonexty',
                    name='Drawdown',
                    line=dict(color=self.config.color_scheme['negative'], width=1),
                    fillcolor=self.config.color_scheme['negative']
                ),
                row=1, col=3
            )
            
            # 4. Monthly Returns Heatmap
            monthly_returns = self._calculate_monthly_returns(returns)
            if len(monthly_returns) > 1:
                heatmap_data = self._create_monthly_heatmap_data(monthly_returns)
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_data['values'],
                        x=heatmap_data['months'],
                        y=heatmap_data['years'],
                        colorscale='RdYlGn',
                        showscale=True,
                        name='Monthly Returns'
                    ),
                    row=2, col=1
                )
            
            # 5. Risk Metrics Bar Chart
            if risk_metrics:
                fig.add_trace(
                    go.Bar(
                        x=list(risk_metrics.keys()),
                        y=list(risk_metrics.values()),
                        name='Risk Metrics',
                        marker_color=self.config.color_scheme['warning']
                    ),
                    row=2, col=2
                )
            
            # 6. Return Distribution
            fig.add_trace(
                go.Histogram(
                    x=returns.values,
                    nbinsx=50,
                    name='Return Distribution',
                    marker_color=self.config.color_scheme['primary'],
                    opacity=0.7
                ),
                row=2, col=3
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                template=self.config.template,
                height=800,
                width=1400,
                font=dict(family=self.config.font_family, size=self.config.font_size),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {e}")
            return go.Figure()
    
    def create_multi_strategy_comparison(self, 
                                       strategies: Dict[str, Dict[str, Any]],
                                       title: str = "Multi-Strategy Comparison") -> go.Figure:
        """
        Create comprehensive multi-strategy comparison dashboard
        
        Args:
            strategies: Dictionary of strategy data
            title: Dashboard title
            
        Returns:
            Multi-strategy comparison dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Cumulative Returns Comparison',
                    'Risk-Return Scatter',
                    'Performance Metrics Comparison',
                    'Correlation Matrix'
                ),
                specs=[[{}, {}], [{}, {}]]
            )
            
            colors = px.colors.qualitative.Set1
            returns_data = {}
            
            # 1. Cumulative Returns Comparison
            for i, (strategy_name, strategy_data) in enumerate(strategies.items()):
                returns = strategy_data.get('returns', pd.Series())
                if not returns.empty:
                    cumulative_returns = (1 + returns).cumprod()
                    returns_data[strategy_name] = returns
                    
                    fig.add_trace(
                        go.Scatter(
                            x=cumulative_returns.index,
                            y=cumulative_returns.values,
                            mode='lines',
                            name=strategy_name,
                            line=dict(color=colors[i % len(colors)], width=2)
                        ),
                        row=1, col=1
                    )
            
            # 2. Risk-Return Scatter Plot
            risk_return_data = []
            for strategy_name, returns in returns_data.items():
                if not returns.empty:
                    annual_return = returns.mean() * 252
                    annual_volatility = returns.std() * np.sqrt(252)
                    risk_return_data.append({
                        'strategy': strategy_name,
                        'return': annual_return,
                        'volatility': annual_volatility,
                        'sharpe': annual_return / annual_volatility if annual_volatility > 0 else 0
                    })
            
            if risk_return_data:
                for i, data in enumerate(risk_return_data):
                    fig.add_trace(
                        go.Scatter(
                            x=[data['volatility']],
                            y=[data['return']],
                            mode='markers',
                            name=data['strategy'],
                            marker=dict(
                                size=15,
                                color=colors[i % len(colors)]
                            ),
                            text=f"Sharpe: {data['sharpe']:.2f}",
                            textposition="top center"
                        ),
                        row=1, col=2
                    )
            
            # 3. Performance Metrics Comparison
            metrics_data = []
            for strategy_name, strategy_data in strategies.items():
                metrics = strategy_data.get('metrics', {})
                if metrics:
                    metrics_data.append({
                        'Strategy': strategy_name,
                        'Sharpe': metrics.get('sharpe_ratio', 0),
                        'Sortino': metrics.get('sortino_ratio', 0),
                        'Max DD': abs(metrics.get('max_drawdown', 0)),
                        'Win Rate': metrics.get('win_rate', 0)
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                for i, metric in enumerate(['Sharpe', 'Sortino', 'Max DD', 'Win Rate']):
                    fig.add_trace(
                        go.Bar(
                            x=metrics_df['Strategy'],
                            y=metrics_df[metric],
                            name=metric,
                            marker_color=colors[i % len(colors)],
                            opacity=0.7
                        ),
                        row=2, col=1
                    )
            
            # 4. Correlation Matrix
            if len(returns_data) > 1:
                correlation_matrix = pd.DataFrame(returns_data).corr()
                fig.add_trace(
                    go.Heatmap(
                        z=correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.index,
                        colorscale='RdBu',
                        showscale=True,
                        name='Correlation'
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                template=self.config.template,
                height=800,
                width=1400,
                font=dict(family=self.config.font_family, size=self.config.font_size),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating multi-strategy comparison: {e}")
            return go.Figure()
    
    def create_risk_analytics_dashboard(self, 
                                      returns: pd.Series,
                                      positions: pd.Series = None,
                                      var_confidence: float = 0.05,
                                      title: str = "Risk Analytics Dashboard") -> go.Figure:
        """
        Create comprehensive risk analytics dashboard
        
        Args:
            returns: Strategy returns
            positions: Position data
            var_confidence: VaR confidence level
            title: Dashboard title
            
        Returns:
            Risk analytics dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'Value at Risk (VaR)', 'Expected Shortfall',
                    'Rolling Volatility', 'Tail Risk Analysis',
                    'Stress Test Scenarios', 'Risk Contribution'
                ),
                specs=[[{}, {}, {}], [{}, {}, {}]]
            )
            
            # Calculate risk metrics
            var_95 = returns.quantile(var_confidence)
            var_99 = returns.quantile(0.01)
            es_95 = returns[returns <= var_95].mean()
            
            # 1. Value at Risk
            fig.add_trace(
                go.Histogram(
                    x=returns.values,
                    nbinsx=50,
                    name='Return Distribution',
                    marker_color=self.config.color_scheme['primary'],
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Add VaR lines
            fig.add_vline(
                x=var_95, line_dash="dash", line_color="red",
                annotation_text=f"VaR 95%: {var_95:.2%}",
                row=1, col=1
            )
            fig.add_vline(
                x=var_99, line_dash="dash", line_color="darkred",
                annotation_text=f"VaR 99%: {var_99:.2%}",
                row=1, col=1
            )
            
            # 2. Expected Shortfall
            tail_returns = returns[returns <= var_95]
            if not tail_returns.empty:
                fig.add_trace(
                    go.Histogram(
                        x=tail_returns.values,
                        nbinsx=20,
                        name='Tail Returns',
                        marker_color=self.config.color_scheme['negative'],
                        opacity=0.7
                    ),
                    row=1, col=2
                )
                
                fig.add_vline(
                    x=es_95, line_dash="dash", line_color="red",
                    annotation_text=f"ES 95%: {es_95:.2%}",
                    row=1, col=2
                )
            
            # 3. Rolling Volatility
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    mode='lines',
                    name='30-Day Rolling Volatility',
                    line=dict(color=self.config.color_scheme['warning'], width=2)
                ),
                row=1, col=3
            )
            
            # 4. Tail Risk Analysis
            tail_ratio = self._calculate_tail_ratio(returns)
            fig.add_trace(
                go.Scatter(
                    x=tail_ratio.index,
                    y=tail_ratio.values,
                    mode='lines',
                    name='Tail Ratio',
                    line=dict(color=self.config.color_scheme['info'], width=2)
                ),
                row=2, col=1
            )
            
            # 5. Stress Test Scenarios
            stress_scenarios = self._generate_stress_scenarios(returns)
            scenario_names = list(stress_scenarios.keys())
            scenario_losses = list(stress_scenarios.values())
            
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=scenario_losses,
                    name='Stress Test Losses',
                    marker_color=self.config.color_scheme['negative']
                ),
                row=2, col=2
            )
            
            # 6. Risk Contribution (if positions available)
            if positions is not None:
                risk_contrib = self._calculate_risk_contribution(returns, positions)
                fig.add_trace(
                    go.Pie(
                        labels=risk_contrib.index,
                        values=risk_contrib.values,
                        name='Risk Contribution'
                    ),
                    row=2, col=3
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                template=self.config.template,
                height=800,
                width=1400,
                font=dict(family=self.config.font_family, size=self.config.font_size),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk analytics dashboard: {e}")
            return go.Figure()
    
    def create_signal_analysis_dashboard(self, 
                                       signals: Dict[str, pd.Series],
                                       prices: pd.Series,
                                       title: str = "Signal Analysis Dashboard") -> go.Figure:
        """
        Create comprehensive signal analysis dashboard
        
        Args:
            signals: Dictionary of signal time series
            prices: Price data
            title: Dashboard title
            
        Returns:
            Signal analysis dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Signal Strength Over Time',
                    'Signal Correlation Matrix',
                    'Signal Frequency Analysis',
                    'Signal Performance Attribution'
                ),
                specs=[[{}, {}], [{}, {}]]
            )
            
            # 1. Signal Strength Over Time
            for i, (signal_name, signal_data) in enumerate(signals.items()):
                fig.add_trace(
                    go.Scatter(
                        x=signal_data.index,
                        y=signal_data.values,
                        mode='lines',
                        name=signal_name,
                        line=dict(width=2),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
            
            # 2. Signal Correlation Matrix
            signal_df = pd.DataFrame(signals)
            correlation_matrix = signal_df.corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='RdBu',
                    showscale=True,
                    name='Signal Correlation'
                ),
                row=1, col=2
            )
            
            # 3. Signal Frequency Analysis
            signal_frequencies = {}
            for signal_name, signal_data in signals.items():
                # Count non-zero signals
                frequency = (signal_data != 0).sum()
                signal_frequencies[signal_name] = frequency
            
            fig.add_trace(
                go.Bar(
                    x=list(signal_frequencies.keys()),
                    y=list(signal_frequencies.values()),
                    name='Signal Frequency',
                    marker_color=self.config.color_scheme['primary']
                ),
                row=2, col=1
            )
            
            # 4. Signal Performance Attribution
            signal_performance = {}
            for signal_name, signal_data in signals.items():
                # Calculate forward returns when signal is active
                forward_returns = prices.pct_change().shift(-1)
                signal_returns = forward_returns[signal_data != 0]
                if not signal_returns.empty:
                    signal_performance[signal_name] = signal_returns.mean()
            
            if signal_performance:
                fig.add_trace(
                    go.Bar(
                        x=list(signal_performance.keys()),
                        y=list(signal_performance.values()),
                        name='Signal Performance',
                        marker_color=[
                            self.config.color_scheme['positive'] if x > 0 
                            else self.config.color_scheme['negative'] 
                            for x in signal_performance.values()
                        ]
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                template=self.config.template,
                height=800,
                width=1200,
                font=dict(family=self.config.font_family, size=self.config.font_size),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating signal analysis dashboard: {e}")
            return go.Figure()
    
    def create_correlation_heatmap(self, 
                                 data: pd.DataFrame,
                                 method: str = 'pearson',
                                 title: str = "Correlation Heatmap") -> go.Figure:
        """
        Create advanced correlation heatmap with clustering
        
        Args:
            data: DataFrame with variables to correlate
            method: Correlation method ('pearson', 'spearman', 'kendall')
            title: Heatmap title
            
        Returns:
            Interactive correlation heatmap
        """
        try:
            # Calculate correlation matrix
            correlation_matrix = data.corr(method=method)
            
            # Perform hierarchical clustering
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import squareform
            
            # Convert correlation to distance
            distance_matrix = 1 - abs(correlation_matrix)
            condensed_distances = squareform(distance_matrix)
            
            # Perform clustering
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Get dendrogram order
            dendro = dendrogram(linkage_matrix, no_plot=True)
            cluster_order = dendro['leaves']
            
            # Reorder correlation matrix
            ordered_corr = correlation_matrix.iloc[cluster_order, cluster_order]
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=ordered_corr.values,
                x=ordered_corr.columns,
                y=ordered_corr.index,
                colorscale='RdBu',
                zmid=0,
                showscale=True,
                hovertemplate='<b>%{x}</b><br><b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                template=self.config.template,
                height=600,
                width=600,
                font=dict(family=self.config.font_family, size=self.config.font_size)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return go.Figure()
    
    def save_chart(self, fig: go.Figure, filename: str, format: str = 'html') -> str:
        """
        Save chart to file
        
        Args:
            fig: Plotly figure
            filename: Output filename
            format: Output format ('html', 'png', 'pdf', 'svg')
            
        Returns:
            Path to saved file
        """
        try:
            filepath = self.results_dir / f"{filename}.{format}"
            
            if format == 'html':
                fig.write_html(str(filepath))
            elif format == 'png':
                fig.write_image(str(filepath), width=1200, height=800)
            elif format == 'pdf':
                fig.write_image(str(filepath), width=1200, height=800)
            elif format == 'svg':
                fig.write_image(str(filepath), width=1200, height=800)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Chart saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving chart: {e}")
            return ""
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        return rolling_mean / rolling_std
    
    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown
    
    def _calculate_monthly_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate monthly returns"""
        return returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    def _create_monthly_heatmap_data(self, monthly_returns: pd.Series) -> Dict[str, Any]:
        """Create data for monthly returns heatmap"""
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot_table = monthly_df.pivot(index='Year', columns='Month', values='Return')
        
        return {
            'values': pivot_table.values,
            'years': pivot_table.index.tolist(),
            'months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        }
    
    def _calculate_tail_ratio(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        def tail_ratio(x):
            if len(x) < 10:
                return np.nan
            return abs(x.quantile(0.95) / x.quantile(0.05))
        
        return returns.rolling(window).apply(tail_ratio)
    
    def _generate_stress_scenarios(self, returns: pd.Series) -> Dict[str, float]:
        """Generate stress test scenarios"""
        scenarios = {}
        
        # Historical stress scenarios
        scenarios['2008 Financial Crisis'] = returns.quantile(0.01) * 21  # 21-day stress
        scenarios['Flash Crash'] = returns.quantile(0.001) * 5  # 5-day extreme stress
        scenarios['Volatility Spike'] = returns.std() * -3 * np.sqrt(5)  # 3-sigma 5-day
        scenarios['Correlation Breakdown'] = returns.quantile(0.05) * 10  # 10-day stress
        
        return scenarios
    
    def _calculate_risk_contribution(self, returns: pd.Series, positions: pd.Series) -> pd.Series:
        """Calculate risk contribution by position"""
        # This is a simplified risk contribution calculation
        # In practice, this would be more sophisticated
        position_weights = positions / positions.sum()
        return position_weights * returns.std()


# Global instance
advanced_visualization = AdvancedVisualization()