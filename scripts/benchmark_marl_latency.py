#!/usr/bin/env python3
"""
MARL Latency Benchmark Script
=============================

This script benchmarks the MARL inference pool performance to validate
the P99 latency optimization from 11.227ms to under 2ms.

Usage:
    python scripts/benchmark_marl_latency.py [options]

Options:
    --iterations N      Number of benchmark iterations (default: 1000)
    --warmup N         Number of warmup iterations (default: 100)
    --output FILE      Output file for results (default: stdout)
    --gpu              Force GPU usage if available
    --detailed         Show detailed performance breakdown
"""

import asyncio
import argparse
import json
import time
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from tactical.async_inference_pool import (
    AsyncInferencePool,
    benchmark_latency,
    get_latency_stats,
    reset_latency_monitoring
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_comprehensive_benchmark(args) -> Dict[str, Any]:
    """Run comprehensive MARL latency benchmark."""
    
    logger.info("🚀 Starting MARL Ultra-Low Latency Benchmark")
    logger.info(f"Configuration: {args.iterations} iterations, {args.warmup} warmup")
    
    # Initialize inference pool
    pool = AsyncInferencePool(
        max_workers_per_type=2,
        max_queue_size=1000,
        batch_timeout_ms=5.0,
        max_batch_size=16
    )
    
    try:
        await pool.initialize()
        await pool.start()
        
        # System information
        system_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
            "pool_workers": sum(len(workers) for workers in pool.workers.values())
        }
        
        logger.info(f"System Info: GPU={'Available' if system_info['cuda_available'] else 'Not Available'}")
        if system_info["cuda_available"]:
            logger.info(f"GPU Device: {system_info['cuda_device_name']}")
        
        # Generate test data
        test_matrix = np.random.randn(60, 7).astype(np.float32)
        test_synergy = {
            "synergy_type": "benchmark",
            "direction": 1,
            "confidence": 0.8,
            "timestamp": time.time()
        }
        
        logger.info("📊 Starting warmup phase...")
        
        # Warmup phase
        for i in range(args.warmup):
            await pool.submit_inference_jobs_vectorized(
                test_matrix, test_synergy, f"warmup_{i}"
            )
            if (i + 1) % 20 == 0:
                logger.info(f"Warmup progress: {i + 1}/{args.warmup}")
        
        logger.info("🎯 Starting main benchmark...")
        
        # Main benchmark
        benchmark_results = await benchmark_latency(pool, iterations=args.iterations)
        
        # Get detailed statistics
        latency_stats = benchmark_results["latency_results"]
        pool_stats = pool.get_stats()
        latency_report = pool.get_latency_report()
        
        # Performance validation
        p99_latency = latency_stats.get("p99_latency_ms", float('inf'))
        target_achieved = p99_latency < 2.0
        
        # Compile comprehensive results
        results = {
            "benchmark_metadata": {
                "timestamp": time.time(),
                "iterations": args.iterations,
                "warmup_iterations": args.warmup,
                "target_p99_latency_ms": 2.0,
                "baseline_p99_latency_ms": 11.227
            },
            "system_information": system_info,
            "performance_results": {
                "p99_latency_ms": p99_latency,
                "p95_latency_ms": latency_stats.get("p95_latency_ms"),
                "p90_latency_ms": latency_stats.get("p90_latency_ms"),
                "mean_latency_ms": latency_stats.get("mean_latency_ms"),
                "median_latency_ms": latency_stats.get("median_latency_ms"),
                "min_latency_ms": latency_stats.get("min_latency_ms"),
                "max_latency_ms": latency_stats.get("max_latency_ms"),
                "sub_2ms_percentage": latency_stats.get("sub_2ms_percentage"),
                "throughput_rps": latency_stats.get("throughput_rps")
            },
            "optimization_validation": {
                "target_achieved": target_achieved,
                "improvement_factor": 11.227 / p99_latency if p99_latency > 0 else float('inf'),
                "latency_reduction_ms": 11.227 - p99_latency,
                "latency_reduction_percentage": ((11.227 - p99_latency) / 11.227) * 100 if p99_latency > 0 else 100
            },
            "detailed_metrics": benchmark_results,
            "pool_statistics": pool_stats,
            "optimization_report": latency_report
        }
        
        # Performance summary
        logger.info("📈 Benchmark Results Summary:")
        logger.info(f"  🎯 Target: P99 < 2.0ms")
        logger.info(f"  📊 Achieved: P99 = {p99_latency:.3f}ms")
        logger.info(f"  ✅ Target Met: {'YES' if target_achieved else 'NO'}")
        logger.info(f"  🚀 Improvement: {results['optimization_validation']['improvement_factor']:.1f}x faster")
        logger.info(f"  📉 Reduction: {results['optimization_validation']['latency_reduction_percentage']:.1f}%")
        logger.info(f"  ⚡ Throughput: {latency_stats.get('throughput_rps', 0):.1f} RPS")
        
        if args.detailed:
            logger.info("📋 Detailed Performance Breakdown:")
            logger.info(f"  P95: {latency_stats.get('p95_latency_ms'):.3f}ms")
            logger.info(f"  P90: {latency_stats.get('p90_latency_ms'):.3f}ms")
            logger.info(f"  Mean: {latency_stats.get('mean_latency_ms'):.3f}ms")
            logger.info(f"  Median: {latency_stats.get('median_latency_ms'):.3f}ms")
            logger.info(f"  Min: {latency_stats.get('min_latency_ms'):.3f}ms")
            logger.info(f"  Max: {latency_stats.get('max_latency_ms'):.3f}ms")
            logger.info(f"  Sub-2ms %: {latency_stats.get('sub_2ms_percentage'):.1f}%")
        
        return results
        
    finally:
        await pool.stop()

def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description="MARL Ultra-Low Latency Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=1000,
        help="Number of benchmark iterations (default: 1000)"
    )
    
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=100,
        help="Number of warmup iterations (default: 100)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (default: stdout)"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU usage if available"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed performance breakdown"
    )
    
    args = parser.parse_args()
    
    # Set GPU preference
    if args.gpu and torch.cuda.is_available():
        torch.cuda.set_device(0)
        logger.info("🔥 GPU mode enabled")
    
    # Run benchmark
    try:
        results = asyncio.run(run_comprehensive_benchmark(args))
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"📄 Results saved to: {output_path}")
        else:
            print("\n" + "="*60)
            print("MARL LATENCY BENCHMARK RESULTS")
            print("="*60)
            print(json.dumps(results, indent=2, default=str))
        
        # Exit code based on success
        target_achieved = results["optimization_validation"]["target_achieved"]
        sys.exit(0 if target_achieved else 1)
        
    except KeyboardInterrupt:
        logger.info("⏹️  Benchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()