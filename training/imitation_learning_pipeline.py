"""
Imitation Learning Pipeline for Stealth Execution
=================================================

Advanced imitation learning system that analyzes natural market flow patterns
to train generative models for stealth execution. Learns to mimic human trading
patterns to make large orders invisible in market noise.

Key Components:
1. Historical market data ingestion (Time & Sales)
2. Natural trade pattern feature extraction
3. Generative model training (GAN/autoregressive)
4. Statistical validation against real market properties
5. Real-time synthetic order generation

Mathematical Foundation:
- Trade size distributions: P(size) ~ Pareto(α=1.16) + exponential tail
- Inter-arrival times: P(Δt) ~ Weibull(k=0.7, λ) + long memory
- Order flow correlation: Hawkes process with exponential decay
- Volume clustering: GARCH-type conditional heteroskedasticity

Performance Targets:
- Pattern detection accuracy: >95%
- Synthetic data indistinguishability: KS test p-value > 0.05
- Real-time generation latency: <1ms
- Statistical moment preservation: <5% error
"""

import logging


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import json
import os
from pathlib import Path

logger = structlog.get_logger()


@dataclass
class TradeRecord:
    """Individual trade record from Time & Sales data"""
    timestamp: float
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    venue: str = ''
    trade_id: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'price': self.price,
            'size': self.size,
            'side': self.side,
            'venue': self.venue,
            'trade_id': self.trade_id
        }


@dataclass
class MarketFeatures:
    """Extracted market microstructure features"""
    # Size distribution features
    mean_trade_size: float = 0.0
    std_trade_size: float = 0.0
    pareto_alpha: float = 1.16  # Empirical market constant
    size_skewness: float = 0.0
    size_kurtosis: float = 0.0
    
    # Timing distribution features
    mean_inter_arrival: float = 0.0
    std_inter_arrival: float = 0.0
    weibull_k: float = 0.7  # Shape parameter
    weibull_lambda: float = 1.0  # Scale parameter
    timing_autocorr: float = 0.0
    
    # Flow correlation features
    hawkes_intensity: float = 0.0
    decay_constant: float = 0.0
    buy_sell_imbalance: float = 0.0
    volume_clustering: float = 0.0
    
    # Intraday patterns
    morning_activity: float = 0.0
    lunch_activity: float = 0.0
    close_activity: float = 0.0
    volatility_regime: float = 0.0
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for neural network input"""
        return torch.tensor([
            self.mean_trade_size, self.std_trade_size, self.pareto_alpha,
            self.size_skewness, self.size_kurtosis, self.mean_inter_arrival,
            self.std_inter_arrival, self.weibull_k, self.weibull_lambda,
            self.timing_autocorr, self.hawkes_intensity, self.decay_constant,
            self.buy_sell_imbalance, self.volume_clustering, self.morning_activity,
            self.lunch_activity, self.close_activity, self.volatility_regime
        ], dtype=torch.float32)


class TradeDataset(Dataset):
    """PyTorch dataset for trade sequence data"""
    
    def __init__(self, trades: List[TradeRecord], sequence_length: int = 100):
        self.trades = trades
        self.sequence_length = sequence_length
        
        # Convert to tensor format
        self.data = self._prepare_sequences()
        
    def _prepare_sequences(self) -> torch.Tensor:
        """Prepare trade sequences for training"""
        sequences = []
        
        for i in range(len(self.trades) - self.sequence_length):
            sequence = []
            for j in range(i, i + self.sequence_length):
                trade = self.trades[j]
                # Normalize features
                sequence.append([
                    trade.size,
                    trade.price,
                    1.0 if trade.side == 'buy' else -1.0,
                    trade.timestamp
                ])
            sequences.append(sequence)
            
        return torch.tensor(sequences, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class GenerativeTradeModel(nn.Module):
    """
    Advanced generative model for synthetic trade generation
    
    Architecture combines:
    - Transformer encoder for sequence modeling
    - VAE for stochastic generation
    - GAN discriminator for adversarial training
    """
    
    def __init__(self, 
                 input_dim: int = 4,
                 hidden_dim: int = 256,
                 latent_dim: int = 64,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 sequence_length: int = 100):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(sequence_length, hidden_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # VAE components
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim * 2)  # mu and logvar
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence to latent space"""
        # Input embedding with positional encoding
        embedded = self.input_embedding(x) + self.positional_encoding.unsqueeze(0)
        
        # Transformer encoding
        encoded = self.transformer(embedded)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)
        
        # VAE encoding
        latent_params = self.encoder(pooled)
        mu, logvar = latent_params.chunk(2, dim=-1)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, sequence_length: Optional[int] = None) -> torch.Tensor:
        """Decode latent vector to trade sequence"""
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        batch_size = z.size(0)
        
        # Decode each time step
        sequence = []
        hidden = z
        
        for t in range(sequence_length):
            step_output = self.decoder(hidden)
            sequence.append(step_output)
            
            # Update hidden state (simple recurrence)
            hidden = hidden + 0.1 * step_output.mean(dim=-1, keepdim=True)
            
        return torch.stack(sequence, dim=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, x.size(1))
        return reconstruction, mu, logvar
    
    def generate(self, num_samples: int, sequence_length: int) -> torch.Tensor:
        """Generate synthetic trade sequences"""
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim)
            if next(self.parameters()).is_cuda:
                z = z.cuda()
                
            # Decode to sequences
            synthetic_trades = self.decode(z, sequence_length)
            
        return synthetic_trades


class TradeDiscriminator(nn.Module):
    """Discriminator for adversarial training"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 256, sequence_length: int = 100):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Calculate flattened size
        conv_out_size = 256 * (sequence_length // 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose for conv1d: (batch, features, sequence)
        x = x.transpose(1, 2)
        
        # Convolutional features
        features = self.conv_layers(x)
        
        # Flatten and classify
        features = features.view(features.size(0), -1)
        probability = self.classifier(features)
        
        return probability


class StatisticalValidator:
    """Validates synthetic data against real market statistical properties"""
    
    def __init__(self):
        self.real_stats = {}
        self.validation_results = {}
        
    def compute_real_statistics(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """Compute statistical properties of real trade data"""
        sizes = [trade.size for trade in trades]
        prices = [trade.price for trade in trades]
        timestamps = [trade.timestamp for trade in trades]
        
        # Size distribution statistics
        size_stats = {
            'mean_size': np.mean(sizes),
            'std_size': np.std(sizes),
            'size_skewness': stats.skew(sizes),
            'size_kurtosis': stats.kurtosis(sizes),
            'size_q25': np.percentile(sizes, 25),
            'size_q75': np.percentile(sizes, 75),
            'size_q95': np.percentile(sizes, 95)
        }
        
        # Inter-arrival time statistics
        inter_arrivals = np.diff(timestamps)
        timing_stats = {
            'mean_interval': np.mean(inter_arrivals),
            'std_interval': np.std(inter_arrivals),
            'interval_skewness': stats.skew(inter_arrivals),
            'autocorr_lag1': np.corrcoef(inter_arrivals[:-1], inter_arrivals[1:])[0, 1]
        }
        
        # Price dynamics
        price_returns = np.diff(np.log(prices))
        price_stats = {
            'return_volatility': np.std(price_returns),
            'return_skewness': stats.skew(price_returns),
            'return_kurtosis': stats.kurtosis(price_returns)
        }
        
        self.real_stats = {**size_stats, **timing_stats, **price_stats}
        return self.real_stats
    
    def validate_synthetic_data(self, synthetic_trades: torch.Tensor) -> Dict[str, float]:
        """Validate synthetic data against real statistics"""
        synthetic_trades = synthetic_trades.detach().cpu().numpy()
        
        validation_results = {}
        
        for batch_idx in range(synthetic_trades.shape[0]):
            batch_data = synthetic_trades[batch_idx]
            
            sizes = batch_data[:, 0]
            prices = batch_data[:, 1]
            
            # Size distribution validation
            ks_stat_size, ks_p_size = stats.ks_2samp(
                sizes, [trade.size for trade in self.real_trades[-len(sizes):]]
            )
            
            # Compute synthetic statistics
            synth_stats = {
                'mean_size': np.mean(sizes),
                'std_size': np.std(sizes),
                'size_skewness': stats.skew(sizes),
                'ks_p_value_size': ks_p_size
            }
            
            validation_results[f'batch_{batch_idx}'] = synth_stats
        
        self.validation_results = validation_results
        return validation_results
    
    def get_indistinguishability_score(self) -> float:
        """Compute overall indistinguishability score"""
        if not self.validation_results:
            return 0.0
        
        ks_p_values = []
        for batch_results in self.validation_results.values():
            if 'ks_p_value_size' in batch_results:
                ks_p_values.append(batch_results['ks_p_value_size'])
        
        if not ks_p_values:
            return 0.0
        
        # Score based on minimum acceptable p-value (0.05)
        mean_p_value = np.mean(ks_p_values)
        return min(mean_p_value / 0.05, 1.0)


class ImitationLearningPipeline:
    """
    Complete imitation learning pipeline for stealth execution
    
    Orchestrates the entire workflow from data ingestion to model deployment
    """
    
    def __init__(self, 
                 data_path: str,
                 model_save_path: str = "./models/stealth_execution",
                 batch_size: int = 32,
                 learning_rate: float = 1e-4,
                 sequence_length: int = 100):
        
        self.data_path = Path(data_path)
        self.model_save_path = Path(model_save_path)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        
        # Create model directory
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.generator = GenerativeTradeModel(sequence_length=sequence_length)
        self.discriminator = TradeDiscriminator(sequence_length=sequence_length)
        self.validator = StatisticalValidator()
        
        # Optimizers
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate * 0.1)
        
        # Data
        self.trades: List[TradeRecord] = []
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        
        # Training state
        self.training_history = {
            'gen_loss': [],
            'disc_loss': [],
            'vae_loss': [],
            'indistinguishability_score': []
        }
        
        logger.info("Imitation Learning Pipeline initialized",
                   sequence_length=sequence_length,
                   batch_size=batch_size,
                   learning_rate=learning_rate)
    
    def load_market_data(self, file_pattern: str = "*.csv") -> int:
        """
        Load historical market data from files
        
        Expected CSV format: timestamp,price,size,side,venue
        """
        data_files = list(self.data_path.glob(file_pattern))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found matching {file_pattern}")
        
        all_trades = []
        
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path)
                
                for _, row in df.iterrows():
                    trade = TradeRecord(
                        timestamp=float(row.get('timestamp', 0)),
                        price=float(row.get('price', 0)),
                        size=float(row.get('size', 0)),
                        side=str(row.get('side', 'buy')),
                        venue=str(row.get('venue', '')),
                        trade_id=str(row.get('trade_id', ''))
                    )
                    all_trades.append(trade)
                    
                logger.info(f"Loaded {len(df)} trades from {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        # Sort by timestamp
        all_trades.sort(key=lambda x: x.timestamp)
        self.trades = all_trades
        
        logger.info(f"Total trades loaded: {len(self.trades)}")
        return len(self.trades)
    
    def extract_market_features(self) -> MarketFeatures:
        """Extract statistical features from loaded trade data"""
        if not self.trades:
            raise ValueError("No trade data loaded")
        
        sizes = np.array([trade.size for trade in self.trades])
        timestamps = np.array([trade.timestamp for trade in self.trades])
        sides = [trade.side for trade in self.trades]
        
        # Size distribution analysis
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        size_skewness = stats.skew(sizes)
        size_kurtosis = stats.kurtosis(sizes)
        
        # Fit Pareto distribution to large trades
        large_trades = sizes[sizes > np.percentile(sizes, 90)]
        if len(large_trades) > 10:
            pareto_alpha = stats.pareto.fit(large_trades)[0]
        else:
            pareto_alpha = 1.16  # Default empirical value
        
        # Inter-arrival time analysis
        inter_arrivals = np.diff(timestamps)
        mean_interval = np.mean(inter_arrivals)
        std_interval = np.std(inter_arrivals)
        
        # Autocorrelation
        if len(inter_arrivals) > 1:
            timing_autocorr = np.corrcoef(inter_arrivals[:-1], inter_arrivals[1:])[0, 1]
            if np.isnan(timing_autocorr):
                timing_autocorr = 0.0
        else:
            timing_autocorr = 0.0
        
        # Weibull fitting for inter-arrival times
        try:
            weibull_params = stats.weibull_min.fit(inter_arrivals)
            weibull_k = weibull_params[0]
            weibull_lambda = weibull_params[2]
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            weibull_k = 0.7
            weibull_lambda = mean_interval
        
        # Buy/sell imbalance
        buy_count = sum(1 for side in sides if side == 'buy')
        buy_sell_imbalance = (buy_count / len(sides)) - 0.5
        
        # Volume clustering (simplified GARCH effect)
        volume_changes = np.abs(np.diff(sizes))
        volume_clustering = np.corrcoef(volume_changes[:-1], volume_changes[1:])[0, 1]
        if np.isnan(volume_clustering):
            volume_clustering = 0.0
        
        # Intraday patterns (simplified)
        morning_activity = 0.8  # Placeholder
        lunch_activity = 0.5    # Placeholder
        close_activity = 1.2    # Placeholder
        volatility_regime = std_size / mean_size
        
        features = MarketFeatures(
            mean_trade_size=mean_size,
            std_trade_size=std_size,
            pareto_alpha=pareto_alpha,
            size_skewness=size_skewness,
            size_kurtosis=size_kurtosis,
            mean_inter_arrival=mean_interval,
            std_inter_arrival=std_interval,
            weibull_k=weibull_k,
            weibull_lambda=weibull_lambda,
            timing_autocorr=timing_autocorr,
            buy_sell_imbalance=buy_sell_imbalance,
            volume_clustering=volume_clustering,
            morning_activity=morning_activity,
            lunch_activity=lunch_activity,
            close_activity=close_activity,
            volatility_regime=volatility_regime
        )
        
        logger.info("Market features extracted",
                   mean_size=mean_size,
                   pareto_alpha=pareto_alpha,
                   timing_autocorr=timing_autocorr)
        
        return features
    
    def prepare_training_data(self, validation_split: float = 0.2) -> None:
        """Prepare training and validation datasets"""
        if not self.trades:
            raise ValueError("No trade data loaded")
        
        # Create dataset
        dataset = TradeDataset(self.trades, self.sequence_length)
        
        # Train/validation split
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info("Training data prepared",
                   train_samples=train_size,
                   val_samples=val_size,
                   batch_size=self.batch_size)
    
    def train_generator(self, real_batch: torch.Tensor) -> Dict[str, float]:
        """Train generator with VAE + GAN losses"""
        self.gen_optimizer.zero_grad()
        
        # Forward pass
        reconstruction, mu, logvar = self.generator(real_batch)
        
        # VAE loss
        recon_loss = F.mse_loss(reconstruction, real_batch)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / real_batch.size(0)  # Normalize by batch size
        vae_loss = recon_loss + 0.1 * kl_loss
        
        # GAN loss (generator wants discriminator to think synthetic is real)
        synthetic_batch = self.generator.generate(real_batch.size(0), real_batch.size(1))
        disc_fake = self.discriminator(synthetic_batch)
        gan_loss = F.binary_cross_entropy(disc_fake, torch.ones_like(disc_fake))
        
        # Total generator loss
        total_loss = vae_loss + 0.1 * gan_loss
        
        total_loss.backward()
        self.gen_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'vae_loss': vae_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'gan_loss': gan_loss.item()
        }
    
    def train_discriminator(self, real_batch: torch.Tensor) -> Dict[str, float]:
        """Train discriminator to distinguish real from synthetic"""
        self.disc_optimizer.zero_grad()
        
        batch_size = real_batch.size(0)
        
        # Real data
        disc_real = self.discriminator(real_batch)
        real_loss = F.binary_cross_entropy(disc_real, torch.ones_like(disc_real))
        
        # Synthetic data
        with torch.no_grad():
            synthetic_batch = self.generator.generate(batch_size, real_batch.size(1))
        disc_fake = self.discriminator(synthetic_batch)
        fake_loss = F.binary_cross_entropy(disc_fake, torch.zeros_like(disc_fake))
        
        # Total discriminator loss
        disc_loss = (real_loss + fake_loss) / 2
        
        disc_loss.backward()
        self.disc_optimizer.step()
        
        return {
            'disc_loss': disc_loss.item(),
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'real_accuracy': (disc_real > 0.5).float().mean().item(),
            'fake_accuracy': (disc_fake < 0.5).float().mean().item()
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'gen_total_loss': 0.0,
            'gen_vae_loss': 0.0,
            'disc_loss': 0.0,
            'batches': 0
        }
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.float()
            
            # Train discriminator
            disc_metrics = self.train_discriminator(batch)
            
            # Train generator (less frequently to balance training)
            if batch_idx % 2 == 0:
                gen_metrics = self.train_generator(batch)
                epoch_metrics['gen_total_loss'] += gen_metrics['total_loss']
                epoch_metrics['gen_vae_loss'] += gen_metrics['vae_loss']
            
            epoch_metrics['disc_loss'] += disc_metrics['disc_loss']
            epoch_metrics['batches'] += 1
        
        # Average metrics
        for key in epoch_metrics:
            if key != 'batches':
                epoch_metrics[key] /= epoch_metrics['batches']
        
        return epoch_metrics
    
    def validate_model(self) -> Dict[str, float]:
        """Validate model on validation set"""
        self.generator.eval()
        
        val_metrics = {
            'val_loss': 0.0,
            'val_batches': 0
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.float()
                reconstruction, mu, logvar = self.generator(batch)
                
                recon_loss = F.mse_loss(reconstruction, batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / batch.size(0)
                val_loss = recon_loss + 0.1 * kl_loss
                
                val_metrics['val_loss'] += val_loss.item()
                val_metrics['val_batches'] += 1
        
        val_metrics['val_loss'] /= val_metrics['val_batches']
        
        # Statistical validation
        synthetic_batch = self.generator.generate(self.batch_size, self.sequence_length)
        self.validator.real_trades = self.trades
        validation_results = self.validator.validate_synthetic_data(synthetic_batch)
        indistinguishability_score = self.validator.get_indistinguishability_score()
        
        val_metrics['indistinguishability_score'] = indistinguishability_score
        
        return val_metrics
    
    def train(self, num_epochs: int = 100, save_frequency: int = 10) -> None:
        """Complete training loop"""
        if self.train_loader is None:
            raise ValueError("Training data not prepared. Call prepare_training_data() first.")
        
        logger.info("Starting imitation learning training", num_epochs=num_epochs)
        
        best_score = 0.0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_model()
            
            # Update history
            self.training_history['gen_loss'].append(train_metrics['gen_total_loss'])
            self.training_history['disc_loss'].append(train_metrics['disc_loss'])
            self.training_history['vae_loss'].append(train_metrics['gen_vae_loss'])
            self.training_history['indistinguishability_score'].append(
                val_metrics['indistinguishability_score']
            )
            
            epoch_time = time.time() - start_time
            
            logger.info("Epoch completed",
                       epoch=epoch + 1,
                       gen_loss=train_metrics['gen_total_loss'],
                       disc_loss=train_metrics['disc_loss'],
                       val_loss=val_metrics['val_loss'],
                       indistinguishability_score=val_metrics['indistinguishability_score'],
                       epoch_time=epoch_time)
            
            # Save best model
            if val_metrics['indistinguishability_score'] > best_score:
                best_score = val_metrics['indistinguishability_score']
                self.save_model(f"best_model_epoch_{epoch + 1}")
            
            # Periodic save
            if (epoch + 1) % save_frequency == 0:
                self.save_model(f"checkpoint_epoch_{epoch + 1}")
        
        logger.info("Training completed", best_indistinguishability_score=best_score)
    
    def save_model(self, model_name: str) -> None:
        """Save trained model and metadata"""
        save_path = self.model_save_path / model_name
        save_path.mkdir(exist_ok=True)
        
        # Save model state
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'training_history': self.training_history,
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }, save_path / "model.pt")
        
        # Save market features
        if self.trades:
            features = self.extract_market_features()
            with open(save_path / "market_features.json", 'w') as f:
                json.dump(features.__dict__, f, indent=2)
        
        # Save validation results
        with open(save_path / "validation_results.json", 'w') as f:
            json.dump(self.validator.validation_results, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load pre-trained model"""
        model_path = Path(model_path)
        
        # Load model state
        checkpoint = torch.load(model_path / "model.pt", map_location='cpu')
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {model_path}")
    
    def generate_synthetic_trades(self, 
                                 num_sequences: int, 
                                 sequence_length: int) -> torch.Tensor:
        """Generate synthetic trade sequences for stealth execution"""
        self.generator.eval()
        
        with torch.no_grad():
            synthetic_trades = self.generator.generate(num_sequences, sequence_length)
        
        logger.info("Synthetic trades generated",
                   num_sequences=num_sequences,
                   sequence_length=sequence_length)
        
        return synthetic_trades
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.training_history['gen_loss']:
            return {}
        
        metrics = {
            'final_gen_loss': self.training_history['gen_loss'][-1],
            'final_disc_loss': self.training_history['disc_loss'][-1],
            'final_indistinguishability_score': self.training_history['indistinguishability_score'][-1],
            'best_indistinguishability_score': max(self.training_history['indistinguishability_score']),
            'training_epochs': len(self.training_history['gen_loss']),
            'convergence_check': {
                'gen_loss_stable': np.std(self.training_history['gen_loss'][-10:]) < 0.01,
                'score_improving': self.training_history['indistinguishability_score'][-1] > 0.5
            }
        }
        
        return metrics


# Export classes and functions
__all__ = [
    'ImitationLearningPipeline',
    'GenerativeTradeModel',
    'TradeDiscriminator',
    'StatisticalValidator',
    'TradeRecord',
    'MarketFeatures',
    'TradeDataset'
]