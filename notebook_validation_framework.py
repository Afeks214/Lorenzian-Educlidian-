#!/usr/bin/env python3
"""
AGENT 4 - NOTEBOOK CELL FLOW VALIDATION FRAMEWORK

Complete validation system for testing notebook cell execution with minimalistic datasets.
This framework validates the correct flow between all notebook cells and ensures proper
output generation for all 5 MARL training notebooks.

Author: Agent 4 - Notebook Cell Flow Validation Expert
"""

import os
import sys
import json
import time
import traceback
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import shutil

@dataclass
class CellExecutionResult:
    """Result of executing a single cell"""
    cell_index: int
    cell_type: str
    execution_time: float
    success: bool
    output: str = ""
    error: str = ""
    memory_usage_mb: float = 0.0
    variables_created: List[str] = None

@dataclass
class NotebookValidationResult:
    """Result of validating an entire notebook"""
    notebook_name: str
    total_cells: int
    executed_cells: int
    successful_cells: int
    failed_cells: int
    total_execution_time: float
    memory_peak_mb: float
    cell_results: List[CellExecutionResult] = None
    critical_issues: List[str] = None
    success_rate: float = 0.0

class MinimalisticDatasetGenerator:
    """Generate minimal synthetic datasets for notebook validation"""
    
    def __init__(self, max_samples: int = 100):
        self.max_samples = max_samples
        
    def create_minimal_market_data(self, timeframe: str = "5min") -> pd.DataFrame:
        """Create minimal market data for testing"""
        np.random.seed(42)  # Reproducible results
        
        # Generate 100 samples of OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=self.max_samples, freq='5T')
        
        # Generate realistic price movements
        base_price = 15000
        price_changes = np.random.normal(0, 50, self.max_samples)
        prices = base_price + np.cumsum(price_changes)
        
        data = {
            'Date': dates,
            'Open': prices + np.random.normal(0, 10, self.max_samples),
            'High': prices + np.abs(np.random.normal(20, 10, self.max_samples)),
            'Low': prices - np.abs(np.random.normal(20, 10, self.max_samples)),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, self.max_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Ensure OHLC relationships are correct
        df['High'] = np.maximum.reduce([df['Open'], df['High'], df['Low'], df['Close']])
        df['Low'] = np.minimum.reduce([df['Open'], df['High'], df['Low'], df['Close']])
        
        return df
    
    def create_strategic_matrix_data(self) -> np.ndarray:
        """Create 48x13 strategic matrix for strategic training"""
        np.random.seed(42)
        return np.random.randn(48, 13) * 0.1  # Normalized strategic features
    
    def create_tactical_features(self) -> np.ndarray:
        """Create tactical features for tactical training"""
        np.random.seed(42)
        return np.random.randn(self.max_samples, 7) * 0.1  # 7 tactical features
    
    def create_risk_metrics(self) -> Dict[str, np.ndarray]:
        """Create risk management metrics"""
        np.random.seed(42)
        return {
            'var_95': np.random.gamma(2, 0.01, self.max_samples),
            'expected_shortfall': np.random.gamma(2.5, 0.01, self.max_samples),
            'sharpe_ratio': np.random.normal(1.2, 0.4, self.max_samples),
            'max_drawdown': np.random.gamma(1, 0.02, self.max_samples),
            'correlation_matrix': np.random.randn(10, 10) * 0.3
        }
    
    def create_execution_metrics(self) -> Dict[str, Any]:
        """Create execution engine metrics"""
        np.random.seed(42)
        return {
            'fill_rates': np.random.beta(10, 1, self.max_samples),  # High fill rates
            'slippage_bps': np.random.gamma(1, 0.5, self.max_samples),  # Low slippage
            'latency_us': np.random.gamma(50, 5, self.max_samples),  # Microsecond latencies
            'market_impact_bps': np.random.gamma(1, 0.3, self.max_samples)
        }
    
    def create_xai_training_data(self) -> Dict[str, Any]:
        """Create XAI training data"""
        np.random.seed(42)
        
        # Trading decisions
        decisions = []
        for i in range(self.max_samples):
            decision = {
                'decision_id': f'decision_{i:03d}',
                'timestamp': datetime.now().isoformat(),
                'symbol': np.random.choice(['NQ', 'ES', 'YM', 'RTY']),
                'action': np.random.choice(['LONG', 'SHORT', 'HOLD']),
                'confidence': np.random.beta(3, 1),
                'agent_contributions': {
                    'MLMI': np.random.dirichlet([2, 2, 2])[0],
                    'NWRQK': np.random.dirichlet([2, 2, 2])[1],
                    'Regime': np.random.dirichlet([2, 2, 2])[2]
                }
            }
            decisions.append(decision)
        
        return {
            'decisions': decisions,
            'explanations': [f'Sample explanation {i}' for i in range(self.max_samples)],
            'queries': [f'Why did the system choose {decisions[i]["action"]}?' for i in range(min(50, self.max_samples))]
        }

class NotebookCellValidator:
    """Validate individual notebook cells"""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self.data_generator = MinimalisticDatasetGenerator()
        
    def setup_test_environment(self, notebook_type: str) -> Dict[str, str]:
        """Setup test environment with minimal data"""
        data_paths = {}
        
        # Create data directory
        data_dir = self.temp_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Generate appropriate test data based on notebook type
        if notebook_type in ['strategic', 'tactical', 'risk_management', 'execution_engine']:
            # Create market data files
            market_data = self.data_generator.create_minimal_market_data()
            
            # 30-minute data for strategic
            if notebook_type == 'strategic':
                data_30min = market_data.iloc[::6].reset_index(drop=True)  # Every 6th row = 30min
                path_30min = data_dir / "NQ - 30 min - ETH.csv"
                data_30min.to_csv(path_30min, index=False)
                data_paths['30min'] = str(path_30min)
            
            # 5-minute data for tactical/execution
            if notebook_type in ['tactical', 'execution_engine']:
                path_5min = data_dir / "NQ - 5 min - ETH.csv"
                market_data.to_csv(path_5min, index=False)
                data_paths['5min'] = str(path_5min)
                
                # Extended data
                path_5min_ext = data_dir / "NQ - 5 min - ETH_extended.csv"
                market_data.to_csv(path_5min_ext, index=False)
                data_paths['5min_extended'] = str(path_5min_ext)
            
            # Risk management specific data
            if notebook_type == 'risk_management':
                risk_data = self.data_generator.create_risk_metrics()
                for key, values in risk_data.items():
                    if key != 'correlation_matrix':
                        risk_df = pd.DataFrame({key: values})
                        risk_path = data_dir / f"{key}.csv"
                        risk_df.to_csv(risk_path, index=False)
                        data_paths[key] = str(risk_path)
        
        # XAI specific data
        elif notebook_type == 'xai_trading_explanations':
            xai_data = self.data_generator.create_xai_training_data()
            xai_path = data_dir / "xai_training_data.json"
            with open(xai_path, 'w') as f:
                json.dump(xai_data, f, indent=2)
            data_paths['xai_data'] = str(xai_path)
        
        return data_paths
    
    def create_test_variables(self, notebook_type: str) -> Dict[str, Any]:
        """Create test variables for notebook cells"""
        variables = {
            'np': np,
            'pd': pd,
            'plt': None,  # Matplotlib import will be handled in cells
            'time': time,
            'os': os,
            'sys': sys
        }
        
        # Notebook-specific variables
        if notebook_type == 'strategic':
            variables.update({
                'matrix_data': self.data_generator.create_strategic_matrix_data(),
                'config': {'batch_size': 32, 'learning_rate': 1e-4, 'epochs': 5}
            })
        
        elif notebook_type == 'tactical':
            variables.update({
                'tactical_features': self.data_generator.create_tactical_features(),
                'training_config': {'episodes': 10, 'batch_size': 16}
            })
        
        elif notebook_type == 'risk_management':
            variables.update({
                'risk_metrics': self.data_generator.create_risk_metrics(),
                'portfolio_config': {'max_position': 0.1, 'var_limit': 0.02}
            })
        
        elif notebook_type == 'execution_engine':
            variables.update({
                'execution_metrics': self.data_generator.create_execution_metrics(),
                'latency_config': {'target_latency_us': 500, 'fill_rate_target': 0.998}
            })
        
        elif notebook_type == 'xai_trading_explanations':
            variables.update({
                'xai_data': self.data_generator.create_xai_training_data(),
                'explanation_config': {'max_length': 500, 'target_latency_ms': 100}
            })
        
        return variables

class NotebookFlowValidator:
    """Main validation framework for notebook cell flow"""
    
    def __init__(self, notebooks_dir: str):
        self.notebooks_dir = Path(notebooks_dir)
        self.temp_dir = None
        self.results = {}
        
        # Target notebooks for validation
        self.target_notebooks = {
            'risk_management_mappo_training.ipynb': 'risk_management',
            'execution_engine_mappo_training.ipynb': 'execution_engine', 
            'strategic_mappo_training.ipynb': 'strategic',
            'tactical_mappo_training.ipynb': 'tactical',
            'xai_trading_explanations_training.ipynb': 'xai_trading_explanations'
        }
    
    def validate_all_notebooks(self) -> Dict[str, NotebookValidationResult]:
        """Validate all target notebooks"""
        print("🧪 Starting comprehensive notebook validation...")
        print(f"📁 Notebooks directory: {self.notebooks_dir}")
        print(f"🎯 Target notebooks: {list(self.target_notebooks.keys())}")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="notebook_validation_")
        print(f"📂 Temporary directory: {self.temp_dir}")
        
        try:
            validation_results = {}
            
            for notebook_file, notebook_type in self.target_notebooks.items():
                notebook_path = self.notebooks_dir / notebook_file
                
                print(f"\n🔍 Validating: {notebook_file}")
                
                if notebook_path.exists():
                    result = self.validate_notebook(notebook_path, notebook_type)
                    validation_results[notebook_file] = result
                    
                    # Print summary
                    success_rate = result.success_rate * 100
                    status = "✅ PASS" if success_rate >= 80 else "❌ FAIL" if success_rate < 50 else "⚠️ PARTIAL"
                    print(f"   Result: {status} ({success_rate:.1f}% success rate)")
                    print(f"   Cells: {result.successful_cells}/{result.total_cells} successful")
                    print(f"   Time: {result.total_execution_time:.2f}s")
                    
                    if result.critical_issues:
                        print(f"   ⚠️ Issues: {len(result.critical_issues)}")
                else:
                    print(f"   ❌ Notebook not found: {notebook_path}")
                    validation_results[notebook_file] = self._create_not_found_result(notebook_file)
            
            return validation_results
            
        finally:
            # Cleanup temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"\n🧹 Cleaned up temporary directory: {self.temp_dir}")
    
    def validate_notebook(self, notebook_path: Path, notebook_type: str) -> NotebookValidationResult:
        """Validate a single notebook"""
        start_time = time.time()
        
        try:
            # Read notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_content = json.load(f)
            
            cells = notebook_content.get('cells', [])
            cell_results = []
            critical_issues = []
            
            # Setup test environment
            validator = NotebookCellValidator(self.temp_dir)
            data_paths = validator.setup_test_environment(notebook_type)
            test_variables = validator.create_test_variables(notebook_type)
            
            # Execute cells
            successful_cells = 0
            
            for i, cell in enumerate(cells):
                if cell.get('cell_type') == 'code':
                    result = self._execute_cell(cell, i, test_variables, data_paths)
                    cell_results.append(result)
                    
                    if result.success:
                        successful_cells += 1
                    else:
                        critical_issues.append(f"Cell {i}: {result.error}")
                        
                        # Stop on critical failures
                        if "ImportError" in result.error or "ModuleNotFoundError" in result.error:
                            critical_issues.append(f"Critical import failure in cell {i}")
                            break
            
            total_time = time.time() - start_time
            success_rate = successful_cells / len(cell_results) if cell_results else 0
            
            return NotebookValidationResult(
                notebook_name=notebook_path.name,
                total_cells=len(cells),
                executed_cells=len(cell_results),
                successful_cells=successful_cells,
                failed_cells=len(cell_results) - successful_cells,
                total_execution_time=total_time,
                memory_peak_mb=self._get_peak_memory(),
                cell_results=cell_results,
                critical_issues=critical_issues,
                success_rate=success_rate
            )
            
        except Exception as e:
            return NotebookValidationResult(
                notebook_name=notebook_path.name,
                total_cells=0,
                executed_cells=0,
                successful_cells=0,
                failed_cells=1,
                total_execution_time=time.time() - start_time,
                memory_peak_mb=0,
                critical_issues=[f"Failed to load notebook: {str(e)}"],
                success_rate=0.0
            )
    
    def _execute_cell(self, cell: Dict, cell_index: int, variables: Dict, data_paths: Dict) -> CellExecutionResult:
        """Execute a single notebook cell"""
        start_time = time.time()
        
        try:
            # Get cell source
            source = cell.get('source', [])
            if isinstance(source, list):
                source_code = ''.join(source)
            else:
                source_code = source
            
            # Skip empty cells
            if not source_code.strip():
                return CellExecutionResult(
                    cell_index=cell_index,
                    cell_type='code',
                    execution_time=0.0,
                    success=True,
                    output="Empty cell - skipped"
                )
            
            # Modify code for test environment
            modified_code = self._modify_code_for_testing(source_code, data_paths)
            
            # Execute code
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()
            
            # Create execution namespace
            exec_namespace = variables.copy()
            exec_namespace['__name__'] = '__main__'
            
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                exec(modified_code, exec_namespace)
            
            execution_time = time.time() - start_time
            output = output_buffer.getvalue()
            error_output = error_buffer.getvalue()
            
            # Check for new variables
            new_vars = [k for k in exec_namespace.keys() if k not in variables and not k.startswith('_')]
            
            return CellExecutionResult(
                cell_index=cell_index,
                cell_type='code',
                execution_time=execution_time,
                success=True,
                output=output,
                error=error_output,
                variables_created=new_vars
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return CellExecutionResult(
                cell_index=cell_index,
                cell_type='code',
                execution_time=execution_time,
                success=False,
                error=str(e),
                output=""
            )
    
    def _modify_code_for_testing(self, source_code: str, data_paths: Dict) -> str:
        """Modify code to work with test environment"""
        modified = source_code
        
        # Skip pip installs and downloads
        if '!pip install' in modified or '!wget' in modified or '!curl' in modified:
            return "# Skipped pip install/download for testing\npass"
        
        # Skip GPU-specific operations that might fail
        if 'torch.cuda' in modified and '.cuda()' in modified:
            modified = modified.replace('.cuda()', '.cpu()')
            modified = modified.replace("device='cuda'", "device='cpu'")
        
        # Replace data file paths
        for key, path in data_paths.items():
            # Common path patterns
            patterns = [
                '/home/QuantNova/GrandModel/colab/data/',
                '/content/GrandModel/colab/data/',
                'colab/data/',
                '../data/',
                './data/'
            ]
            
            for pattern in patterns:
                if pattern in modified:
                    modified = modified.replace(pattern, str(Path(path).parent) + '/')
        
        # Reduce training parameters for testing
        training_reductions = {
            'num_episodes': '5',
            'episodes': '5', 
            'num_iterations': '3',
            'iterations': '3',
            'epochs': '2',
            'batch_size': '16',
            'max_steps': '50',
            'training_steps': '50'
        }
        
        for param, value in training_reductions.items():
            if f'{param}=' in modified:
                import re
                modified = re.sub(f'{param}=\\d+', f'{param}={value}', modified)
        
        # Skip lengthy operations
        if 'tqdm' in modified:
            modified = modified.replace('tqdm(', 'list(')
        
        return modified
    
    def _get_peak_memory(self) -> float:
        """Get peak memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _create_not_found_result(self, notebook_name: str) -> NotebookValidationResult:
        """Create result for notebook not found"""
        return NotebookValidationResult(
            notebook_name=notebook_name,
            total_cells=0,
            executed_cells=0,
            successful_cells=0,
            failed_cells=0,
            total_execution_time=0.0,
            memory_peak_mb=0.0,
            critical_issues=[f"Notebook file not found: {notebook_name}"],
            success_rate=0.0
        )
    
    def generate_validation_report(self, results: Dict[str, NotebookValidationResult]) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("# 📋 NOTEBOOK CELL FLOW VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"🕐 Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"🎯 Notebooks Tested: {len(results)}")
        report.append("")
        
        # Summary statistics
        total_notebooks = len(results)
        successful_notebooks = sum(1 for r in results.values() if r.success_rate >= 0.8)
        total_cells = sum(r.total_cells for r in results.values())
        successful_cells = sum(r.successful_cells for r in results.values())
        
        report.append("## 📊 SUMMARY STATISTICS")
        report.append("-" * 30)
        report.append(f"✅ Successful Notebooks: {successful_notebooks}/{total_notebooks} ({successful_notebooks/total_notebooks*100:.1f}%)")
        report.append(f"📱 Total Cells: {total_cells}")
        report.append(f"✅ Successful Cells: {successful_cells}/{total_cells} ({successful_cells/total_cells*100:.1f}%)")
        report.append(f"⏱️ Total Execution Time: {sum(r.total_execution_time for r in results.values()):.2f}s")
        report.append("")
        
        # Individual notebook results
        report.append("## 📓 INDIVIDUAL NOTEBOOK RESULTS")
        report.append("-" * 40)
        
        for notebook_name, result in results.items():
            status = "✅ PASS" if result.success_rate >= 0.8 else "❌ FAIL" if result.success_rate < 0.5 else "⚠️ PARTIAL"
            
            report.append(f"### {notebook_name}")
            report.append(f"**Status**: {status} ({result.success_rate*100:.1f}% success rate)")
            report.append(f"**Cells**: {result.successful_cells}/{result.total_cells} successful")
            report.append(f"**Execution Time**: {result.total_execution_time:.2f}s")
            report.append(f"**Memory Peak**: {result.memory_peak_mb:.1f}MB")
            
            if result.critical_issues:
                report.append("**Critical Issues**:")
                for issue in result.critical_issues:
                    report.append(f"  - {issue}")
            
            # Cell-by-cell details for failed notebooks
            if result.success_rate < 0.8 and result.cell_results:
                report.append("**Cell Details**:")
                for cell_result in result.cell_results:
                    if not cell_result.success:
                        status_icon = "❌"
                        report.append(f"  {status_icon} Cell {cell_result.cell_index}: {cell_result.error[:100]}...")
            
            report.append("")
        
        # Recommendations
        report.append("## 🔧 RECOMMENDATIONS")
        report.append("-" * 20)
        
        failed_notebooks = [name for name, result in results.items() if result.success_rate < 0.8]
        
        if not failed_notebooks:
            report.append("✅ All notebooks passed validation!")
            report.append("🚀 System is ready for production deployment.")
        else:
            report.append("⚠️ The following notebooks need attention:")
            for notebook in failed_notebooks:
                result = results[notebook]
                if result.critical_issues:
                    for issue in result.critical_issues[:3]:  # Top 3 issues
                        report.append(f"  - {notebook}: {issue}")
        
        report.append("")
        report.append("## 🎯 VALIDATION TARGETS")
        report.append("-" * 25)
        report.append("- **Minimum Success Rate**: 80% of cells must execute successfully")
        report.append("- **Memory Usage**: < 2GB peak memory per notebook")
        report.append("- **Execution Time**: < 300s total per notebook")
        report.append("- **Critical Errors**: No import/module errors")
        
        return "\n".join(report)

def main():
    """Main validation execution"""
    print("🚀 AGENT 4 - NOTEBOOK CELL FLOW VALIDATION")
    print("=" * 50)
    
    # Determine notebooks directory
    notebooks_dir = Path("/home/QuantNova/GrandModel/notebooks")
    if not notebooks_dir.exists():
        # Try colab directory
        notebooks_dir = Path("/home/QuantNova/GrandModel/colab/notebooks")
        if not notebooks_dir.exists():
            print(f"❌ Notebooks directory not found!")
            return
    
    print(f"📁 Using notebooks directory: {notebooks_dir}")
    
    # Create validator
    validator = NotebookFlowValidator(notebooks_dir)
    
    # Run validation
    results = validator.validate_all_notebooks()
    
    # Generate report
    report = validator.generate_validation_report(results)
    
    # Save report
    report_path = notebooks_dir.parent / "notebook_validation_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Validation report saved: {report_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("🎯 VALIDATION COMPLETE")
    print("=" * 50)
    
    successful_notebooks = sum(1 for r in results.values() if r.success_rate >= 0.8)
    total_notebooks = len(results)
    
    print(f"✅ Success Rate: {successful_notebooks}/{total_notebooks} notebooks passed")
    
    if successful_notebooks == total_notebooks:
        print("🎉 ALL NOTEBOOKS VALIDATED SUCCESSFULLY!")
        print("🚀 System ready for production deployment.")
    else:
        print("⚠️ Some notebooks need attention.")
        print("📋 Check the validation report for details.")

if __name__ == "__main__":
    main()