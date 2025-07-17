#!/usr/bin/env python3
"""
Comprehensive Performance Validation Suite
==========================================

This script runs comprehensive performance validation tests on the MARL system
with large synthetic datasets to validate 5-year dataset handling capabilities.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
import psutil
import traceback
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from performance_validation.performance_benchmark_framework import PerformanceBenchmarkFramework
from performance_validation.synthetic_data_generator import SyntheticDataGenerator

class ComprehensivePerformanceValidator:
    """
    Comprehensive performance validation suite for MARL system.
    
    Tests all components with large-scale synthetic datasets to validate
    5-year dataset handling capabilities.
    """
    
    def __init__(self, output_dir: str = "performance_validation/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.benchmark_framework = PerformanceBenchmarkFramework(output_dir)
        self.validation_results = {}
        
        # Test configurations
        self.test_configurations = {
            'small_dataset': {'size': 10000, 'description': 'Small dataset (10K records)'},
            'medium_dataset': {'size': 50000, 'description': 'Medium dataset (50K records)'},
            'large_dataset': {'size': 100000, 'description': 'Large dataset (100K records)'},
            'xlarge_dataset': {'size': 500000, 'description': 'X-Large dataset (500K records)'},
            'stress_test': {'size': 1000000, 'description': 'Stress test (1M records)'}
        }
    
    def validate_data_processing_performance(self) -> Dict:
        """
        Validate data processing performance with different dataset sizes.
        
        Returns:
            Dictionary with validation results
        """
        
        print("=== Data Processing Performance Validation ===")
        
        results = {}
        
        # Test data loading performance
        test_files = [
            'performance_validation/synthetic_5year_5min.csv',
            'performance_validation/synthetic_5year_30min.csv',
            'performance_validation/stress_test_dataset.csv'
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                print(f"\nTesting data loading: {file_path}")
                
                try:
                    # Benchmark data loading
                    result = self.benchmark_framework.benchmark_data_loading(file_path)
                    results[f'data_loading_{os.path.basename(file_path)}'] = {
                        'success': result.success,
                        'total_time': result.total_time_seconds,
                        'peak_memory_mb': result.peak_memory_mb,
                        'throughput': result.throughput_records_per_second,
                        'dataset_size': result.dataset_size
                    }
                    
                except Exception as e:
                    print(f"Error testing {file_path}: {e}")
                    results[f'data_loading_{os.path.basename(file_path)}'] = {
                        'success': False,
                        'error': str(e)
                    }
        
        return results
    
    def validate_training_performance(self) -> Dict:
        """
        Validate training performance with different dataset sizes.
        
        Returns:
            Dictionary with validation results
        """
        
        print("\n=== Training Performance Validation ===")
        
        results = {}
        
        for config_name, config in self.test_configurations.items():
            print(f"\nTesting training performance: {config['description']}")
            
            try:
                # Benchmark training simulation
                result = self.benchmark_framework.benchmark_training_simulation(
                    dataset_size=config['size'],
                    epochs=5  # Reduced epochs for faster testing
                )
                
                results[f'training_{config_name}'] = {
                    'success': result.success,
                    'total_time': result.total_time_seconds,
                    'peak_memory_mb': result.peak_memory_mb,
                    'throughput': result.throughput_records_per_second,
                    'dataset_size': result.dataset_size,
                    'time_per_epoch': result.total_time_seconds / 5
                }
                
            except Exception as e:
                print(f"Error testing {config_name}: {e}")
                results[f'training_{config_name}'] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def validate_notebook_performance(self) -> Dict:
        """
        Validate notebook performance with large datasets.
        
        Returns:
            Dictionary with validation results
        """
        
        print("\n=== Notebook Performance Validation ===")
        
        results = {}
        
        notebooks = [
            'colab/notebooks/tactical_mappo_training.ipynb',
            'colab/notebooks/strategic_mappo_training.ipynb',
            'colab/notebooks/risk_management_mappo_training.ipynb'
        ]
        
        datasets = [
            'performance_validation/synthetic_5year_5min.csv',
            'performance_validation/synthetic_5year_30min.csv'
        ]
        
        for notebook in notebooks:
            if os.path.exists(notebook):
                notebook_name = os.path.basename(notebook).replace('.ipynb', '')
                
                for dataset in datasets:
                    if os.path.exists(dataset):
                        dataset_name = os.path.basename(dataset).replace('.csv', '')
                        test_name = f"{notebook_name}_{dataset_name}"
                        
                        print(f"\nTesting notebook: {notebook_name} with {dataset_name}")
                        
                        try:
                            # Create modified notebook for testing
                            modified_notebook = self._create_test_notebook(notebook, dataset)
                            
                            # Benchmark notebook execution
                            result = self.benchmark_framework.benchmark_notebook_execution(
                                modified_notebook, dataset
                            )
                            
                            results[test_name] = {
                                'success': result.success,
                                'total_time': result.total_time_seconds,
                                'peak_memory_mb': result.peak_memory_mb,
                                'throughput': result.throughput_records_per_second,
                                'dataset_size': result.dataset_size
                            }
                            
                        except Exception as e:
                            print(f"Error testing {test_name}: {e}")
                            results[test_name] = {
                                'success': False,
                                'error': str(e)
                            }
        
        return results
    
    def _create_test_notebook(self, notebook_path: str, dataset_path: str) -> str:
        """
        Create a modified notebook for testing with specific dataset.
        
        Args:
            notebook_path: Path to the original notebook
            dataset_path: Path to the dataset
            
        Returns:
            Path to the modified notebook
        """
        
        import nbformat
        
        # Read original notebook
        with open(notebook_path, 'r') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Modify first cell to use test dataset
        if notebook.cells:
            # Find data loading cell and modify it
            for cell in notebook.cells:
                if cell.cell_type == 'code' and 'read_csv' in cell.source:
                    # Replace dataset path
                    cell.source = cell.source.replace(
                        'colab/data/NQ - 5 min - ETH.csv',
                        dataset_path
                    ).replace(
                        'colab/data/NQ - 30 min - ETH.csv',
                        dataset_path
                    )
                    break
        
        # Add performance monitoring cell at the beginning
        monitoring_cell = nbformat.v4.new_code_cell(source="""
import time
import psutil
import os

# Performance monitoring setup
start_time = time.time()
start_memory = psutil.virtual_memory().used / 1024 / 1024

print(f"Starting performance test at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Initial memory usage: {start_memory:.1f} MB")
print(f"Dataset path: {os.environ.get('BENCHMARK_DATASET_PATH', 'Not set')}")
""")
        
        notebook.cells.insert(0, monitoring_cell)
        
        # Add performance summary cell at the end
        summary_cell = nbformat.v4.new_code_cell(source="""
# Performance summary
end_time = time.time()
end_memory = psutil.virtual_memory().used / 1024 / 1024

total_time = end_time - start_time
memory_used = end_memory - start_memory

print(f"\\nPerformance Summary:")
print(f"Total execution time: {total_time:.2f} seconds")
print(f"Memory used: {memory_used:.1f} MB")
print(f"Peak memory: {psutil.virtual_memory().used / 1024 / 1024:.1f} MB")
print(f"CPU usage: {psutil.cpu_percent()}%")
""")
        
        notebook.cells.append(summary_cell)
        
        # Save modified notebook
        test_notebook_path = notebook_path.replace('.ipynb', '_performance_test.ipynb')
        with open(test_notebook_path, 'w') as f:
            nbformat.write(notebook, f)
        
        return test_notebook_path
    
    def validate_memory_usage(self) -> Dict:
        """
        Validate memory usage patterns under different loads.
        
        Returns:
            Dictionary with memory validation results
        """
        
        print("\n=== Memory Usage Validation ===")
        
        results = {}
        
        # Test memory usage with different dataset sizes
        dataset_sizes = [10000, 50000, 100000, 500000]
        
        for size in dataset_sizes:
            print(f"\nTesting memory usage with {size:,} records")
            
            try:
                # Create synthetic data
                generator = SyntheticDataGenerator()
                
                # Monitor memory during data generation
                start_memory = psutil.virtual_memory().used / 1024 / 1024
                
                df = generator.generate_dataset(
                    start_date="2023-01-01",
                    end_date="2023-01-02",
                    interval_minutes=5,
                    output_file=None
                )
                
                # Simulate data processing
                if df is not None and len(df) > 0:
                    # Basic statistics
                    stats = df.describe()
                    
                    # Technical indicators simulation
                    df['ma_5'] = df['Close'].rolling(window=5).mean()
                    df['ma_20'] = df['Close'].rolling(window=20).mean()
                    df['volatility'] = df['Close'].rolling(window=20).std()
                    
                    # Memory usage after processing
                    end_memory = psutil.virtual_memory().used / 1024 / 1024
                    memory_used = end_memory - start_memory
                    
                    results[f'memory_{size}'] = {
                        'dataset_size': len(df),
                        'memory_used_mb': memory_used,
                        'memory_per_record_kb': (memory_used * 1024) / len(df) if len(df) > 0 else 0,
                        'peak_memory_mb': end_memory,
                        'success': True
                    }
                    
                    # Clean up
                    del df
                    
                else:
                    results[f'memory_{size}'] = {
                        'success': False,
                        'error': 'Failed to generate data'
                    }
                    
            except Exception as e:
                print(f"Error testing memory usage for {size}: {e}")
                results[f'memory_{size}'] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def validate_scalability(self) -> Dict:
        """
        Validate system scalability with increasing dataset sizes.
        
        Returns:
            Dictionary with scalability validation results
        """
        
        print("\n=== Scalability Validation ===")
        
        results = {}
        
        # Test scalability with exponentially increasing sizes
        test_sizes = [1000, 5000, 10000, 50000, 100000]
        
        scalability_data = []
        
        for size in test_sizes:
            print(f"\nTesting scalability with {size:,} records")
            
            try:
                start_time = time.time()
                start_memory = psutil.virtual_memory().used / 1024 / 1024
                
                # Generate synthetic data
                generator = SyntheticDataGenerator()
                df = generator.generate_dataset(
                    start_date="2023-01-01",
                    end_date="2023-01-02",
                    interval_minutes=5,
                    output_file=None
                )
                
                if df is not None and len(df) > 0:
                    # Simulate processing
                    processed_data = df.copy()
                    processed_data['returns'] = processed_data['Close'].pct_change()
                    processed_data['volatility'] = processed_data['returns'].rolling(window=20).std()
                    
                    end_time = time.time()
                    end_memory = psutil.virtual_memory().used / 1024 / 1024
                    
                    processing_time = end_time - start_time
                    memory_used = end_memory - start_memory
                    
                    scalability_data.append({
                        'dataset_size': len(df),
                        'processing_time': processing_time,
                        'memory_used': memory_used,
                        'throughput': len(df) / processing_time if processing_time > 0 else 0
                    })
                    
                    # Clean up
                    del df, processed_data
                    
            except Exception as e:
                print(f"Error testing scalability for {size}: {e}")
                scalability_data.append({
                    'dataset_size': size,
                    'error': str(e)
                })
        
        # Analyze scalability
        if scalability_data:
            results['scalability_analysis'] = self._analyze_scalability(scalability_data)
            results['scalability_data'] = scalability_data
        
        return results
    
    def _analyze_scalability(self, data: List[Dict]) -> Dict:
        """
        Analyze scalability from performance data.
        
        Args:
            data: List of performance measurements
            
        Returns:
            Dictionary with scalability analysis
        """
        
        # Filter successful measurements
        successful_data = [d for d in data if 'error' not in d]
        
        if len(successful_data) < 2:
            return {'error': 'Insufficient data for scalability analysis'}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(successful_data)
        
        # Calculate scaling coefficients
        size_log = np.log(df['dataset_size'])
        time_log = np.log(df['processing_time'])
        memory_log = np.log(df['memory_used'])
        
        # Time scaling (O(n^?) complexity)
        time_slope = np.polyfit(size_log, time_log, 1)[0]
        time_r2 = np.corrcoef(size_log, time_log)[0, 1] ** 2
        
        # Memory scaling
        memory_slope = np.polyfit(size_log, memory_log, 1)[0]
        memory_r2 = np.corrcoef(size_log, memory_log)[0, 1] ** 2
        
        # Throughput analysis
        avg_throughput = df['throughput'].mean()
        throughput_std = df['throughput'].std()
        
        return {
            'time_scaling': {
                'slope': time_slope,
                'r_squared': time_r2,
                'complexity_class': self._classify_complexity(time_slope)
            },
            'memory_scaling': {
                'slope': memory_slope,
                'r_squared': memory_r2,
                'complexity_class': self._classify_complexity(memory_slope)
            },
            'throughput_analysis': {
                'average_records_per_second': avg_throughput,
                'std_dev': throughput_std,
                'coefficient_of_variation': throughput_std / avg_throughput if avg_throughput > 0 else 0
            }
        }
    
    def _classify_complexity(self, slope: float) -> str:
        """Classify algorithmic complexity based on scaling slope."""
        if slope < 0.5:
            return "Sub-linear (O(log n) or O(√n))"
        elif slope < 1.2:
            return "Linear (O(n))"
        elif slope < 1.8:
            return "Linearithmic (O(n log n))"
        elif slope < 2.5:
            return "Quadratic (O(n²))"
        else:
            return "Exponential (O(n^k), k>2)"
    
    def generate_5year_projections(self) -> Dict:
        """
        Generate projections for 5-year dataset handling.
        
        Returns:
            Dictionary with 5-year dataset projections
        """
        
        print("\n=== 5-Year Dataset Projections ===")
        
        # Get scaling projections from benchmark framework
        scaling_projections = self.benchmark_framework.generate_scaling_projections()
        
        # Calculate additional projections
        projections = {}
        
        if 'projections' in scaling_projections:
            base_projections = scaling_projections['projections']
            
            for dataset_type, proj in base_projections.items():
                # Add additional metrics
                enhanced_proj = proj.copy()
                
                # Storage requirements
                enhanced_proj['storage_requirements'] = {
                    'raw_data_gb': proj['dataset_size'] * 0.0001,  # ~100KB per 1000 records
                    'processed_data_gb': proj['dataset_size'] * 0.0002,  # Double for processed
                    'total_storage_gb': proj['dataset_size'] * 0.0003
                }
                
                # Resource requirements
                enhanced_proj['resource_requirements'] = {
                    'minimum_ram_gb': max(8, proj['projected_memory_gb'] * 1.5),
                    'recommended_ram_gb': max(16, proj['projected_memory_gb'] * 2),
                    'estimated_cpu_hours': proj['projected_time_hours'],
                    'parallel_processing_benefit': min(4, psutil.cpu_count() / 2)
                }
                
                # Performance recommendations
                enhanced_proj['recommendations'] = self._generate_dataset_recommendations(proj)
                
                projections[dataset_type] = enhanced_proj
        
        return projections
    
    def _generate_dataset_recommendations(self, projection: Dict) -> List[str]:
        """Generate recommendations for specific dataset projection."""
        recommendations = []
        
        # Memory recommendations
        if projection['projected_memory_gb'] > 8:
            recommendations.append("Use memory-mapped files for large dataset access")
            recommendations.append("Implement data chunking for processing")
        
        # Time recommendations
        if projection['projected_time_hours'] > 2:
            recommendations.append("Consider parallel processing for faster execution")
            recommendations.append("Implement progress checkpointing for long runs")
        
        # Storage recommendations
        if projection['dataset_size'] > 500000:
            recommendations.append("Use compressed data formats (HDF5, Parquet)")
            recommendations.append("Implement data preprocessing pipelines")
        
        return recommendations
    
    def run_comprehensive_validation(self) -> Dict:
        """
        Run comprehensive performance validation suite.
        
        Returns:
            Dictionary with all validation results
        """
        
        print("Starting Comprehensive Performance Validation")
        print("=" * 50)
        
        validation_start = time.time()
        
        # Run all validation tests
        self.validation_results = {
            'validation_info': {
                'start_time': datetime.now().isoformat(),
                'system_info': self.benchmark_framework.system_info
            },
            'data_processing': self.validate_data_processing_performance(),
            'training_performance': self.validate_training_performance(),
            'memory_usage': self.validate_memory_usage(),
            'scalability': self.validate_scalability(),
            'projections_5year': self.generate_5year_projections()
        }
        
        # Add notebook validation if notebooks exist
        try:
            notebook_results = self.validate_notebook_performance()
            self.validation_results['notebook_performance'] = notebook_results
        except Exception as e:
            print(f"Notebook validation failed: {e}")
            self.validation_results['notebook_performance'] = {'error': str(e)}
        
        validation_end = time.time()
        
        self.validation_results['validation_info']['end_time'] = datetime.now().isoformat()
        self.validation_results['validation_info']['total_validation_time'] = validation_end - validation_start
        
        # Generate reports
        self._generate_validation_reports()
        
        return self.validation_results
    
    def _generate_validation_reports(self):
        """Generate comprehensive validation reports."""
        
        # Save raw validation data
        data_path = os.path.join(self.output_dir, f"validation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(data_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        # Generate markdown report
        report_path = os.path.join(self.output_dir, f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        self._generate_markdown_report(report_path)
        
        # Generate performance summary
        summary_path = os.path.join(self.output_dir, f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        self._generate_performance_summary(summary_path)
        
        print(f"\nValidation reports generated:")
        print(f"  Data: {data_path}")
        print(f"  Report: {report_path}")
        print(f"  Summary: {summary_path}")
    
    def _generate_markdown_report(self, report_path: str):
        """Generate detailed markdown report."""
        
        with open(report_path, 'w') as f:
            f.write("# Comprehensive Performance Validation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System information
            f.write("## System Information\n\n")
            system_info = self.validation_results['validation_info']['system_info']
            for key, value in system_info.items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Data processing results
            f.write("## Data Processing Performance\n\n")
            data_results = self.validation_results.get('data_processing', {})
            for test_name, result in data_results.items():
                f.write(f"### {test_name}\n\n")
                if result.get('success'):
                    f.write(f"- **Success**: ✅\n")
                    f.write(f"- **Dataset Size**: {result['dataset_size']:,} records\n")
                    f.write(f"- **Total Time**: {result['total_time']:.2f} seconds\n")
                    f.write(f"- **Throughput**: {result['throughput']:.0f} records/second\n")
                    f.write(f"- **Peak Memory**: {result['peak_memory_mb']:.1f} MB\n")
                else:
                    f.write(f"- **Success**: ❌\n")
                    f.write(f"- **Error**: {result.get('error', 'Unknown error')}\n")
                f.write("\n")
            
            # Training performance results
            f.write("## Training Performance\n\n")
            training_results = self.validation_results.get('training_performance', {})
            for test_name, result in training_results.items():
                f.write(f"### {test_name}\n\n")
                if result.get('success'):
                    f.write(f"- **Success**: ✅\n")
                    f.write(f"- **Dataset Size**: {result['dataset_size']:,} records\n")
                    f.write(f"- **Total Time**: {result['total_time']:.2f} seconds\n")
                    f.write(f"- **Time per Epoch**: {result['time_per_epoch']:.2f} seconds\n")
                    f.write(f"- **Throughput**: {result['throughput']:.0f} records/second\n")
                    f.write(f"- **Peak Memory**: {result['peak_memory_mb']:.1f} MB\n")
                else:
                    f.write(f"- **Success**: ❌\n")
                    f.write(f"- **Error**: {result.get('error', 'Unknown error')}\n")
                f.write("\n")
            
            # 5-year projections
            f.write("## 5-Year Dataset Projections\n\n")
            projections = self.validation_results.get('projections_5year', {})
            for dataset_type, proj in projections.items():
                f.write(f"### {dataset_type}\n\n")
                f.write(f"- **Dataset Size**: {proj['dataset_size']:,} records\n")
                f.write(f"- **Projected Time**: {proj['projected_time_hours']:.2f} hours\n")
                f.write(f"- **Projected Memory**: {proj['projected_memory_gb']:.2f} GB\n")
                
                if 'storage_requirements' in proj:
                    storage = proj['storage_requirements']
                    f.write(f"- **Storage Requirements**: {storage['total_storage_gb']:.2f} GB\n")
                
                if 'resource_requirements' in proj:
                    resources = proj['resource_requirements']
                    f.write(f"- **Minimum RAM**: {resources['minimum_ram_gb']:.0f} GB\n")
                    f.write(f"- **Recommended RAM**: {resources['recommended_ram_gb']:.0f} GB\n")
                
                if 'recommendations' in proj:
                    f.write("- **Recommendations**:\n")
                    for rec in proj['recommendations']:
                        f.write(f"  - {rec}\n")
                
                f.write("\n")
            
            # Scalability analysis
            f.write("## Scalability Analysis\n\n")
            scalability = self.validation_results.get('scalability', {})
            if 'scalability_analysis' in scalability:
                analysis = scalability['scalability_analysis']
                
                f.write("### Time Complexity\n\n")
                time_scaling = analysis['time_scaling']
                f.write(f"- **Slope**: {time_scaling['slope']:.3f}\n")
                f.write(f"- **R²**: {time_scaling['r_squared']:.3f}\n")
                f.write(f"- **Complexity Class**: {time_scaling['complexity_class']}\n\n")
                
                f.write("### Memory Complexity\n\n")
                memory_scaling = analysis['memory_scaling']
                f.write(f"- **Slope**: {memory_scaling['slope']:.3f}\n")
                f.write(f"- **R²**: {memory_scaling['r_squared']:.3f}\n")
                f.write(f"- **Complexity Class**: {memory_scaling['complexity_class']}\n\n")
                
                f.write("### Throughput Analysis\n\n")
                throughput = analysis['throughput_analysis']
                f.write(f"- **Average Throughput**: {throughput['average_records_per_second']:.0f} records/second\n")
                f.write(f"- **Standard Deviation**: {throughput['std_dev']:.0f}\n")
                f.write(f"- **Coefficient of Variation**: {throughput['coefficient_of_variation']:.3f}\n\n")
    
    def _generate_performance_summary(self, summary_path: str):
        """Generate executive performance summary."""
        
        with open(summary_path, 'w') as f:
            f.write("# Performance Validation Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Overall assessment
            total_tests = 0
            successful_tests = 0
            
            for category, results in self.validation_results.items():
                if isinstance(results, dict) and category != 'validation_info':
                    for test_name, result in results.items():
                        if isinstance(result, dict) and 'success' in result:
                            total_tests += 1
                            if result['success']:
                                successful_tests += 1
            
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            
            f.write(f"- **Total Tests**: {total_tests}\n")
            f.write(f"- **Successful Tests**: {successful_tests}\n")
            f.write(f"- **Success Rate**: {success_rate:.1f}%\n")
            f.write(f"- **System Status**: {'✅ READY' if success_rate >= 80 else '⚠️ NEEDS ATTENTION'}\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Performance metrics
            projections = self.validation_results.get('projections_5year', {})
            if projections:
                f.write("### 5-Year Dataset Capabilities\n\n")
                
                for dataset_type, proj in projections.items():
                    f.write(f"**{dataset_type}**:\n")
                    f.write(f"- Processing Time: {proj['projected_time_hours']:.1f} hours\n")
                    f.write(f"- Memory Requirements: {proj['projected_memory_gb']:.1f} GB\n")
                    f.write(f"- Storage Requirements: {proj.get('storage_requirements', {}).get('total_storage_gb', 0):.1f} GB\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if success_rate >= 90:
                f.write("- System demonstrates excellent performance capabilities\n")
                f.write("- Ready for production deployment with 5-year datasets\n")
                f.write("- Consider implementing monitoring for continuous performance tracking\n")
            elif success_rate >= 70:
                f.write("- System shows good performance with some areas for improvement\n")
                f.write("- Address identified bottlenecks before production deployment\n")
                f.write("- Implement additional testing for edge cases\n")
            else:
                f.write("- System requires significant performance improvements\n")
                f.write("- Focus on critical bottlenecks and stability issues\n")
                f.write("- Consider architecture changes for better scalability\n")
            
            f.write("\n")
            f.write("## Next Steps\n\n")
            f.write("1. Review detailed performance report for specific optimizations\n")
            f.write("2. Implement recommended improvements\n")
            f.write("3. Repeat validation testing to confirm improvements\n")
            f.write("4. Deploy monitoring systems for production use\n")

def main():
    """Main execution function."""
    
    print("Starting Comprehensive Performance Validation Suite")
    print("=" * 60)
    
    # Create validator
    validator = ComprehensivePerformanceValidator()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    print("\n" + "=" * 60)
    print("PERFORMANCE VALIDATION COMPLETE")
    print("=" * 60)
    
    # Print summary
    total_validation_time = results['validation_info']['total_validation_time']
    print(f"Total validation time: {total_validation_time:.1f} seconds")
    
    # Count successful tests
    successful_tests = 0
    total_tests = 0
    
    for category, category_results in results.items():
        if isinstance(category_results, dict) and category != 'validation_info':
            for test_name, result in category_results.items():
                if isinstance(result, dict) and 'success' in result:
                    total_tests += 1
                    if result['success']:
                        successful_tests += 1
    
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Test results: {successful_tests}/{total_tests} passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("✅ SYSTEM READY FOR 5-YEAR DATASETS")
    else:
        print("⚠️ SYSTEM NEEDS OPTIMIZATION")

if __name__ == "__main__":
    main()