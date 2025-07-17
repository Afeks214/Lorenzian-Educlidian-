"""
Report Archival and Historical Access System
Manages long-term storage, retrieval, and analysis of test reports
"""

import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import asyncio
import aiofiles
import shutil
import zipfile
import tarfile
import gzip
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
import schedule
import threading
import time
from .advanced_test_reporting import TestResult, TestSuite, TestStatus
from .coverage_analyzer import CoverageReport


class ArchiveFormat(Enum):
    """Supported archive formats"""
    ZIP = "zip"
    TAR = "tar"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"


class RetentionPolicy(Enum):
    """Data retention policies"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


@dataclass
class ArchiveConfig:
    """Configuration for report archival"""
    enabled: bool = True
    archive_format: str = ArchiveFormat.ZIP.value
    retention_policy: str = RetentionPolicy.MONTHLY.value
    retention_days: int = 90
    max_archive_size_mb: int = 1000
    compression_level: int = 6
    storage_path: str = "archives"
    backup_locations: List[str] = None
    auto_cleanup: bool = True
    cleanup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    metadata_retention_days: int = 365
    
    def __post_init__(self):
        if self.backup_locations is None:
            self.backup_locations = []


@dataclass
class ArchiveEntry:
    """Archive entry metadata"""
    archive_id: str
    suite_name: str
    generated_at: datetime
    archive_path: str
    archive_size: int
    compression_ratio: float
    file_count: int
    retention_date: datetime
    metadata: Dict[str, Any]
    tags: List[str]
    checksum: str


class ReportArchiver:
    """Manages archival and retrieval of test reports"""
    
    def __init__(self, config: ArchiveConfig):
        self.config = config
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.db_path = self.storage_path / "archive_catalog.db"
        self._init_database()
        
        # Setup scheduled cleanup
        if config.auto_cleanup:
            self._setup_cleanup_schedule()
    
    def _init_database(self):
        """Initialize archive catalog database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS archive_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                archive_id TEXT UNIQUE,
                suite_name TEXT,
                generated_at TIMESTAMP,
                archive_path TEXT,
                archive_size INTEGER,
                compression_ratio REAL,
                file_count INTEGER,
                retention_date TIMESTAMP,
                metadata TEXT,
                tags TEXT,
                checksum TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS archive_contents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                archive_id TEXT,
                file_path TEXT,
                file_size INTEGER,
                file_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (archive_id) REFERENCES archive_catalog (archive_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                archive_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                metric_type TEXT,
                period_start TIMESTAMP,
                period_end TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (archive_id) REFERENCES archive_catalog (archive_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS archive_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                archive_id TEXT,
                accessed_by TEXT,
                access_type TEXT,
                access_details TEXT,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (archive_id) REFERENCES archive_catalog (archive_id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_archive_date ON archive_catalog(generated_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_archive_suite ON archive_catalog(suite_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_archive_retention ON archive_catalog(retention_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name ON historical_metrics(metric_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_period ON historical_metrics(period_start, period_end)')
        
        conn.commit()
        conn.close()
    
    def _setup_cleanup_schedule(self):
        """Setup scheduled cleanup task"""
        def run_cleanup():
            asyncio.run(self.cleanup_expired_archives())
        
        # Schedule cleanup
        schedule.every().day.at("02:00").do(run_cleanup)
        
        # Start scheduler in background thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    async def archive_reports(self, 
                             suite: TestSuite,
                             reports: Dict[str, str],
                             coverage_report: Optional[CoverageReport] = None,
                             additional_files: List[str] = None) -> ArchiveEntry:
        """Archive test reports and related files"""
        
        if not self.config.enabled:
            self.logger.info("Archival is disabled")
            return None
        
        # Generate archive ID
        archive_id = self._generate_archive_id(suite)
        
        # Prepare files for archival
        files_to_archive = self._prepare_files_for_archive(
            suite, reports, coverage_report, additional_files
        )
        
        # Create archive
        archive_path = await self._create_archive(archive_id, files_to_archive)
        
        # Calculate metadata
        metadata = self._calculate_archive_metadata(suite, coverage_report)
        
        # Create archive entry
        archive_entry = ArchiveEntry(
            archive_id=archive_id,
            suite_name=suite.suite_name,
            generated_at=suite.end_time,
            archive_path=str(archive_path),
            archive_size=archive_path.stat().st_size,
            compression_ratio=self._calculate_compression_ratio(files_to_archive, archive_path),
            file_count=len(files_to_archive),
            retention_date=self._calculate_retention_date(suite.end_time),
            metadata=metadata,
            tags=self._generate_tags(suite),
            checksum=self._calculate_checksum(archive_path)
        )
        
        # Store in catalog
        await self._store_archive_entry(archive_entry, files_to_archive)
        
        # Store historical metrics
        await self._store_historical_metrics(archive_id, suite, coverage_report)
        
        # Create backup copies if configured
        if self.config.backup_locations:
            await self._create_backups(archive_path)
        
        self.logger.info(f"Archive created: {archive_id} ({archive_entry.archive_size} bytes)")
        
        return archive_entry
    
    def _generate_archive_id(self, suite: TestSuite) -> str:
        """Generate unique archive ID"""
        import hashlib
        data = f"{suite.suite_name}_{suite.end_time.isoformat()}_{suite.total_tests}"
        return f"ARCH_{hashlib.md5(data.encode()).hexdigest()[:12]}"
    
    def _prepare_files_for_archive(self,
                                  suite: TestSuite,
                                  reports: Dict[str, str],
                                  coverage_report: Optional[CoverageReport],
                                  additional_files: List[str]) -> List[Tuple[str, str]]:
        """Prepare files for archival (source_path, archive_path)"""
        
        files_to_archive = []
        
        # Add report files
        for format_name, report_path in reports.items():
            if Path(report_path).exists():
                archive_path = f"reports/{format_name}_{Path(report_path).name}"
                files_to_archive.append((report_path, archive_path))
        
        # Add suite metadata
        suite_metadata = {
            'suite_name': suite.suite_name,
            'start_time': suite.start_time.isoformat(),
            'end_time': suite.end_time.isoformat(),
            'total_tests': suite.total_tests,
            'passed': suite.passed,
            'failed': suite.failed,
            'skipped': suite.skipped,
            'errors': suite.errors,
            'total_duration': suite.total_duration,
            'coverage_percentage': suite.coverage_percentage,
            'success_rate': suite.success_rate,
            'results': [asdict(result) for result in suite.results]
        }
        
        # Create temporary metadata file
        temp_metadata_path = self.storage_path / f"temp_metadata_{suite.suite_name}.json"
        with open(temp_metadata_path, 'w') as f:
            json.dump(suite_metadata, f, indent=2, default=str)
        
        files_to_archive.append((str(temp_metadata_path), "metadata/suite_metadata.json"))
        
        # Add coverage report if available
        if coverage_report:
            coverage_metadata = {
                'report_id': coverage_report.report_id,
                'generated_at': coverage_report.generated_at.isoformat(),
                'overall_coverage': coverage_report.overall_coverage,
                'line_coverage': coverage_report.line_coverage,
                'branch_coverage': coverage_report.branch_coverage,
                'function_coverage': coverage_report.function_coverage,
                'total_files': coverage_report.total_files,
                'covered_files': coverage_report.covered_files,
                'modules': [asdict(module) for module in coverage_report.modules]
            }
            
            temp_coverage_path = self.storage_path / f"temp_coverage_{suite.suite_name}.json"
            with open(temp_coverage_path, 'w') as f:
                json.dump(coverage_metadata, f, indent=2, default=str)
            
            files_to_archive.append((str(temp_coverage_path), "metadata/coverage_metadata.json"))
        
        # Add additional files
        if additional_files:
            for file_path in additional_files:
                if Path(file_path).exists():
                    archive_path = f"additional/{Path(file_path).name}"
                    files_to_archive.append((file_path, archive_path))
        
        return files_to_archive
    
    async def _create_archive(self, archive_id: str, files_to_archive: List[Tuple[str, str]]) -> Path:
        """Create compressed archive from files"""
        
        archive_filename = f"{archive_id}.{self.config.archive_format}"
        archive_path = self.storage_path / archive_filename
        
        if self.config.archive_format == ArchiveFormat.ZIP.value:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, 
                               compresslevel=self.config.compression_level) as zf:
                for source_path, archive_path_in_zip in files_to_archive:
                    zf.write(source_path, archive_path_in_zip)
        
        elif self.config.archive_format in [ArchiveFormat.TAR.value, ArchiveFormat.TAR_GZ.value, ArchiveFormat.TAR_BZ2.value]:
            mode = 'w'
            if self.config.archive_format == ArchiveFormat.TAR_GZ.value:
                mode = 'w:gz'
            elif self.config.archive_format == ArchiveFormat.TAR_BZ2.value:
                mode = 'w:bz2'
            
            with tarfile.open(archive_path, mode) as tf:
                for source_path, archive_path_in_tar in files_to_archive:
                    tf.add(source_path, arcname=archive_path_in_tar)
        
        # Clean up temporary files
        for source_path, _ in files_to_archive:
            if "temp_" in source_path:
                Path(source_path).unlink(missing_ok=True)
        
        return archive_path
    
    def _calculate_compression_ratio(self, files_to_archive: List[Tuple[str, str]], archive_path: Path) -> float:
        """Calculate compression ratio"""
        
        total_original_size = sum(
            Path(source_path).stat().st_size 
            for source_path, _ in files_to_archive
            if Path(source_path).exists()
        )
        
        archive_size = archive_path.stat().st_size
        
        if total_original_size > 0:
            return archive_size / total_original_size
        else:
            return 1.0
    
    def _calculate_retention_date(self, generated_at: datetime) -> datetime:
        """Calculate retention date based on policy"""
        
        if self.config.retention_policy == RetentionPolicy.DAILY.value:
            return generated_at + timedelta(days=self.config.retention_days)
        elif self.config.retention_policy == RetentionPolicy.WEEKLY.value:
            return generated_at + timedelta(weeks=self.config.retention_days // 7)
        elif self.config.retention_policy == RetentionPolicy.MONTHLY.value:
            return generated_at + timedelta(days=self.config.retention_days * 30)
        elif self.config.retention_policy == RetentionPolicy.QUARTERLY.value:
            return generated_at + timedelta(days=self.config.retention_days * 90)
        elif self.config.retention_policy == RetentionPolicy.YEARLY.value:
            return generated_at + timedelta(days=self.config.retention_days * 365)
        else:  # CUSTOM
            return generated_at + timedelta(days=self.config.retention_days)
    
    def _calculate_archive_metadata(self, suite: TestSuite, coverage_report: Optional[CoverageReport]) -> Dict[str, Any]:
        """Calculate metadata for archive"""
        
        metadata = {
            'suite_summary': {
                'total_tests': suite.total_tests,
                'success_rate': suite.success_rate,
                'duration': suite.total_duration,
                'coverage': suite.coverage_percentage
            },
            'performance_metrics': {
                'avg_test_duration': statistics.mean(r.duration for r in suite.results) if suite.results else 0,
                'max_test_duration': max(r.duration for r in suite.results) if suite.results else 0,
                'slow_tests_count': len([r for r in suite.results if r.duration > 5.0]),
                'failed_tests_count': suite.failed
            },
            'test_distribution': {
                'by_module': self._calculate_module_distribution(suite.results),
                'by_status': {
                    'passed': suite.passed,
                    'failed': suite.failed,
                    'skipped': suite.skipped,
                    'errors': suite.errors
                }
            }
        }
        
        if coverage_report:
            metadata['coverage_summary'] = {
                'overall_coverage': coverage_report.overall_coverage,
                'line_coverage': coverage_report.line_coverage,
                'branch_coverage': coverage_report.branch_coverage,
                'function_coverage': coverage_report.function_coverage,
                'modules_count': len(coverage_report.modules)
            }
        
        return metadata
    
    def _calculate_module_distribution(self, results: List[TestResult]) -> Dict[str, int]:
        """Calculate test distribution by module"""
        
        module_counts = {}
        for result in results:
            module = result.test_module
            module_counts[module] = module_counts.get(module, 0) + 1
        
        return module_counts
    
    def _generate_tags(self, suite: TestSuite) -> List[str]:
        """Generate tags for archive"""
        
        tags = []
        
        # Add status-based tags
        if suite.success_rate >= 95:
            tags.append("high_success")
        elif suite.success_rate >= 80:
            tags.append("medium_success")
        else:
            tags.append("low_success")
        
        # Add performance tags
        if suite.total_duration > 300:  # 5 minutes
            tags.append("slow_execution")
        elif suite.total_duration < 30:  # 30 seconds
            tags.append("fast_execution")
        
        # Add coverage tags
        if suite.coverage_percentage >= 90:
            tags.append("high_coverage")
        elif suite.coverage_percentage >= 70:
            tags.append("medium_coverage")
        else:
            tags.append("low_coverage")
        
        # Add size tags
        if suite.total_tests > 1000:
            tags.append("large_suite")
        elif suite.total_tests > 100:
            tags.append("medium_suite")
        else:
            tags.append("small_suite")
        
        return tags
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        
        import hashlib
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def _store_archive_entry(self, entry: ArchiveEntry, files_to_archive: List[Tuple[str, str]]):
        """Store archive entry in catalog"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store main entry
        cursor.execute('''
            INSERT INTO archive_catalog 
            (archive_id, suite_name, generated_at, archive_path, archive_size, 
             compression_ratio, file_count, retention_date, metadata, tags, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.archive_id, entry.suite_name, entry.generated_at, entry.archive_path,
            entry.archive_size, entry.compression_ratio, entry.file_count,
            entry.retention_date, json.dumps(entry.metadata), json.dumps(entry.tags),
            entry.checksum
        ))
        
        # Store file contents
        for source_path, archive_path in files_to_archive:
            file_size = Path(source_path).stat().st_size if Path(source_path).exists() else 0
            file_type = Path(source_path).suffix.lower()
            
            cursor.execute('''
                INSERT INTO archive_contents 
                (archive_id, file_path, file_size, file_type)
                VALUES (?, ?, ?, ?)
            ''', (entry.archive_id, archive_path, file_size, file_type))
        
        conn.commit()
        conn.close()
    
    async def _store_historical_metrics(self, archive_id: str, suite: TestSuite, coverage_report: Optional[CoverageReport]):
        """Store historical metrics for trend analysis"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metrics = [
            ('test_count', suite.total_tests, 'count'),
            ('success_rate', suite.success_rate, 'percentage'),
            ('duration', suite.total_duration, 'seconds'),
            ('coverage', suite.coverage_percentage, 'percentage'),
            ('failed_tests', suite.failed, 'count'),
            ('skipped_tests', suite.skipped, 'count'),
            ('avg_test_duration', statistics.mean(r.duration for r in suite.results) if suite.results else 0, 'seconds')
        ]
        
        if coverage_report:
            metrics.extend([
                ('line_coverage', coverage_report.line_coverage, 'percentage'),
                ('branch_coverage', coverage_report.branch_coverage, 'percentage'),
                ('function_coverage', coverage_report.function_coverage, 'percentage'),
                ('covered_files', coverage_report.covered_files, 'count')
            ])
        
        for metric_name, metric_value, metric_type in metrics:
            cursor.execute('''
                INSERT INTO historical_metrics 
                (archive_id, metric_name, metric_value, metric_type, period_start, period_end)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (archive_id, metric_name, metric_value, metric_type, suite.start_time, suite.end_time))
        
        conn.commit()
        conn.close()
    
    async def _create_backups(self, archive_path: Path):
        """Create backup copies of archive"""
        
        for backup_location in self.config.backup_locations:
            backup_path = Path(backup_location)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            backup_file = backup_path / archive_path.name
            
            try:
                shutil.copy2(archive_path, backup_file)
                self.logger.info(f"Backup created: {backup_file}")
            except Exception as e:
                self.logger.error(f"Failed to create backup at {backup_location}: {e}")
    
    async def retrieve_archive(self, archive_id: str, extract_to: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve and optionally extract archive"""
        
        # Get archive info
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM archive_catalog WHERE archive_id = ?
        ''', (archive_id,))
        
        archive_row = cursor.fetchone()
        if not archive_row:
            conn.close()
            return None
        
        # Convert to dictionary
        columns = [desc[0] for desc in cursor.description]
        archive_info = dict(zip(columns, archive_row))
        
        # Get file contents
        cursor.execute('''
            SELECT file_path, file_size, file_type FROM archive_contents 
            WHERE archive_id = ?
        ''', (archive_id,))
        
        file_contents = cursor.fetchall()
        conn.close()
        
        # Log access
        await self._log_access(archive_id, "retrieve", {"extract_to": extract_to})
        
        archive_path = Path(archive_info['archive_path'])
        
        if not archive_path.exists():
            self.logger.error(f"Archive file not found: {archive_path}")
            return None
        
        # Verify checksum
        current_checksum = self._calculate_checksum(archive_path)
        if current_checksum != archive_info['checksum']:
            self.logger.error(f"Checksum mismatch for archive {archive_id}")
            return None
        
        result = {
            'archive_info': archive_info,
            'file_contents': [
                {'file_path': row[0], 'file_size': row[1], 'file_type': row[2]}
                for row in file_contents
            ],
            'archive_path': str(archive_path)
        }
        
        # Extract if requested
        if extract_to:
            extract_path = Path(extract_to)
            extract_path.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(extract_path)
            elif archive_path.suffix in ['.tar', '.gz', '.bz2']:
                with tarfile.open(archive_path, 'r') as tf:
                    tf.extractall(extract_path)
            
            result['extracted_to'] = str(extract_path)
        
        return result
    
    async def _log_access(self, archive_id: str, access_type: str, access_details: Dict[str, Any]):
        """Log archive access"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO archive_access_log 
            (archive_id, accessed_by, access_type, access_details)
            VALUES (?, ?, ?, ?)
        ''', (archive_id, "system", access_type, json.dumps(access_details)))
        
        conn.commit()
        conn.close()
    
    async def search_archives(self, 
                            suite_name: Optional[str] = None,
                            date_range: Optional[Tuple[datetime, datetime]] = None,
                            tags: Optional[List[str]] = None,
                            success_rate_range: Optional[Tuple[float, float]] = None,
                            limit: int = 50) -> List[Dict[str, Any]]:
        """Search archives based on criteria"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM archive_catalog WHERE 1=1"
        params = []
        
        if suite_name:
            query += " AND suite_name LIKE ?"
            params.append(f"%{suite_name}%")
        
        if date_range:
            query += " AND generated_at BETWEEN ? AND ?"
            params.extend(date_range)
        
        if tags:
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f"%{tag}%")
        
        if success_rate_range:
            query += " AND JSON_EXTRACT(metadata, '$.suite_summary.success_rate') BETWEEN ? AND ?"
            params.extend(success_rate_range)
        
        query += " ORDER BY generated_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Convert to dictionaries
        columns = [desc[0] for desc in cursor.description]
        archives = [dict(zip(columns, row)) for row in results]
        
        conn.close()
        
        return archives
    
    async def get_historical_trends(self, 
                                  metric_names: List[str],
                                  days: int = 30,
                                  suite_name: Optional[str] = None) -> Dict[str, Any]:
        """Get historical trends for specified metrics"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                hm.metric_name, 
                hm.metric_value, 
                hm.period_start,
                ac.suite_name
            FROM historical_metrics hm
            JOIN archive_catalog ac ON hm.archive_id = ac.archive_id
            WHERE hm.metric_name IN ({})
            AND hm.period_start > datetime('now', '-{} days')
        '''.format(','.join('?' * len(metric_names)), days)
        
        params = metric_names
        
        if suite_name:
            query += " AND ac.suite_name = ?"
            params.append(suite_name)
        
        query += " ORDER BY hm.period_start"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return {"message": "No historical data found"}
        
        # Process trends
        trends = {}
        
        for metric_name in metric_names:
            metric_data = df[df['metric_name'] == metric_name]
            
            if not metric_data.empty:
                trends[metric_name] = {
                    'values': metric_data['metric_value'].tolist(),
                    'timestamps': metric_data['period_start'].tolist(),
                    'trend_direction': self._calculate_trend_direction(metric_data['metric_value'].tolist()),
                    'average': metric_data['metric_value'].mean(),
                    'latest': metric_data['metric_value'].iloc[-1],
                    'change_percent': self._calculate_change_percent(metric_data['metric_value'].tolist())
                }
        
        return trends
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction"""
        
        if len(values) < 2:
            return "stable"
        
        recent_avg = statistics.mean(values[-5:])
        older_avg = statistics.mean(values[:-5]) if len(values) > 5 else values[0]
        
        diff = recent_avg - older_avg
        
        if diff > 0.05:
            return "increasing"
        elif diff < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_change_percent(self, values: List[float]) -> float:
        """Calculate percentage change from first to last value"""
        
        if len(values) < 2 or values[0] == 0:
            return 0.0
        
        return ((values[-1] - values[0]) / values[0]) * 100
    
    async def generate_historical_report(self, 
                                       suite_name: Optional[str] = None,
                                       days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive historical report"""
        
        # Get archive summary
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                COUNT(*) as total_archives,
                AVG(archive_size) as avg_archive_size,
                SUM(archive_size) as total_storage_used,
                MIN(generated_at) as earliest_date,
                MAX(generated_at) as latest_date
            FROM archive_catalog
            WHERE generated_at > datetime('now', '-{} days')
        '''.format(days)
        
        params = []
        
        if suite_name:
            query += " AND suite_name = ?"
            params.append(suite_name)
        
        cursor = conn.cursor()
        cursor.execute(query, params)
        summary = cursor.fetchone()
        
        # Get metric trends
        key_metrics = ['success_rate', 'duration', 'coverage', 'test_count']
        trends = await self.get_historical_trends(key_metrics, days, suite_name)
        
        # Get tag distribution
        cursor.execute('''
            SELECT tags FROM archive_catalog 
            WHERE generated_at > datetime('now', '-{} days')
        '''.format(days))
        
        tag_results = cursor.fetchall()
        tag_counts = {}
        
        for (tags_json,) in tag_results:
            if tags_json:
                tags = json.loads(tags_json)
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        conn.close()
        
        # Create visualizations
        charts = await self._generate_historical_charts(trends)
        
        return {
            'summary': {
                'total_archives': summary[0] or 0,
                'avg_archive_size_mb': (summary[1] or 0) / (1024 * 1024),
                'total_storage_mb': (summary[2] or 0) / (1024 * 1024),
                'date_range': {
                    'earliest': summary[3],
                    'latest': summary[4]
                }
            },
            'trends': trends,
            'tag_distribution': tag_counts,
            'charts': charts
        }
    
    async def _generate_historical_charts(self, trends: Dict[str, Any]) -> Dict[str, str]:
        """Generate charts for historical data"""
        
        charts = {}
        chart_dir = self.storage_path / "charts"
        chart_dir.mkdir(exist_ok=True)
        
        for metric_name, trend_data in trends.items():
            if 'values' in trend_data and 'timestamps' in trend_data:
                # Create time series chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=trend_data['timestamps'],
                    y=trend_data['values'],
                    mode='lines+markers',
                    name=metric_name,
                    line=dict(width=2)
                ))
                
                fig.update_layout(
                    title=f'{metric_name.replace("_", " ").title()} Trend',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    height=400
                )
                
                chart_path = chart_dir / f"{metric_name}_trend.html"
                fig.write_html(chart_path)
                charts[metric_name] = str(chart_path)
        
        return charts
    
    async def cleanup_expired_archives(self) -> Dict[str, Any]:
        """Clean up expired archives"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find expired archives
        cursor.execute('''
            SELECT archive_id, archive_path, archive_size 
            FROM archive_catalog 
            WHERE retention_date < datetime('now')
        ''')
        
        expired_archives = cursor.fetchall()
        
        cleanup_results = {
            'archives_removed': 0,
            'space_freed_mb': 0,
            'errors': []
        }
        
        for archive_id, archive_path, archive_size in expired_archives:
            try:
                # Remove archive file
                if Path(archive_path).exists():
                    Path(archive_path).unlink()
                    cleanup_results['space_freed_mb'] += archive_size / (1024 * 1024)
                
                # Remove from database
                cursor.execute('DELETE FROM archive_catalog WHERE archive_id = ?', (archive_id,))
                cursor.execute('DELETE FROM archive_contents WHERE archive_id = ?', (archive_id,))
                cursor.execute('DELETE FROM historical_metrics WHERE archive_id = ?', (archive_id,))
                cursor.execute('DELETE FROM archive_access_log WHERE archive_id = ?', (archive_id,))
                
                cleanup_results['archives_removed'] += 1
                
            except Exception as e:
                cleanup_results['errors'].append(f"Failed to remove {archive_id}: {str(e)}")
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Cleanup completed: {cleanup_results['archives_removed']} archives removed, "
                        f"{cleanup_results['space_freed_mb']:.2f} MB freed")
        
        return cleanup_results
    
    async def get_archive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive archive statistics"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Basic statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_archives,
                SUM(archive_size) as total_size,
                AVG(archive_size) as avg_size,
                AVG(compression_ratio) as avg_compression,
                AVG(file_count) as avg_file_count,
                MIN(generated_at) as earliest,
                MAX(generated_at) as latest
            FROM archive_catalog
        ''')
        
        basic_stats = cursor.fetchone()
        
        # Size distribution
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN archive_size < 1048576 THEN 'Small (<1MB)'
                    WHEN archive_size < 10485760 THEN 'Medium (1-10MB)'
                    WHEN archive_size < 104857600 THEN 'Large (10-100MB)'
                    ELSE 'Very Large (>100MB)'
                END as size_category,
                COUNT(*) as count
            FROM archive_catalog
            GROUP BY size_category
        ''')
        
        size_distribution = cursor.fetchall()
        
        # Monthly archive counts
        cursor.execute('''
            SELECT 
                strftime('%Y-%m', generated_at) as month,
                COUNT(*) as count
            FROM archive_catalog
            GROUP BY month
            ORDER BY month DESC
            LIMIT 12
        ''')
        
        monthly_counts = cursor.fetchall()
        
        # Top suites by archive count
        cursor.execute('''
            SELECT 
                suite_name,
                COUNT(*) as archive_count,
                AVG(JSON_EXTRACT(metadata, '$.suite_summary.success_rate')) as avg_success_rate
            FROM archive_catalog
            GROUP BY suite_name
            ORDER BY archive_count DESC
            LIMIT 10
        ''')
        
        top_suites = cursor.fetchall()
        
        conn.close()
        
        return {
            'basic_statistics': {
                'total_archives': basic_stats[0] or 0,
                'total_size_mb': (basic_stats[1] or 0) / (1024 * 1024),
                'avg_size_mb': (basic_stats[2] or 0) / (1024 * 1024),
                'avg_compression_ratio': basic_stats[3] or 0,
                'avg_file_count': basic_stats[4] or 0,
                'earliest_archive': basic_stats[5],
                'latest_archive': basic_stats[6]
            },
            'size_distribution': [
                {'category': row[0], 'count': row[1]} 
                for row in size_distribution
            ],
            'monthly_counts': [
                {'month': row[0], 'count': row[1]} 
                for row in monthly_counts
            ],
            'top_suites': [
                {'suite_name': row[0], 'archive_count': row[1], 'avg_success_rate': row[2]} 
                for row in top_suites
            ]
        }