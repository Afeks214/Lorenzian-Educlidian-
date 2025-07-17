"""
Regulatory Reporter for Compliance Reporting

This module provides automated regulatory reporting capabilities for various
financial regulatory frameworks and requirements.
"""

import logging


import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import structlog
from pathlib import Path
import uuid
import zipfile
import io

from .compliance_monitor import ComplianceMonitor, RegulatoryFramework, ComplianceViolation
from .audit_system import AuditSystem, AuditEvent, AuditEventType
from ..core.event_bus import EventBus

logger = structlog.get_logger()


class ReportType(Enum):
    """Types of regulatory reports"""
    TRADE_REPORTING = "trade_reporting"
    POSITION_REPORTING = "position_reporting"
    RISK_REPORTING = "risk_reporting"
    COMPLIANCE_SUMMARY = "compliance_summary"
    AUDIT_TRAIL = "audit_trail"
    TRANSACTION_REPORTING = "transaction_reporting"
    BEST_EXECUTION = "best_execution"
    MARKET_DATA_USAGE = "market_data_usage"
    OPERATIONAL_RISK = "operational_risk"
    LIQUIDITY_REPORTING = "liquidity_reporting"


class ReportFormat(Enum):
    """Report output formats"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    PDF = "pdf"
    EXCEL = "excel"
    FIXED_WIDTH = "fixed_width"


class ReportFrequency(Enum):
    """Report generation frequencies"""
    REAL_TIME = "real_time"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    ON_DEMAND = "on_demand"


@dataclass
class ReportTemplate:
    """Template for regulatory reports"""
    template_id: str
    template_name: str
    report_type: ReportType
    framework: RegulatoryFramework
    format: ReportFormat
    frequency: ReportFrequency
    fields: List[str]
    required_fields: List[str]
    field_mappings: Dict[str, str]
    validation_rules: Dict[str, Any]
    submission_details: Dict[str, Any]
    active: bool = True
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class RegulatoryReport:
    """Generated regulatory report"""
    report_id: str
    template_id: str
    report_type: ReportType
    framework: RegulatoryFramework
    format: ReportFormat
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    data: Any  # Can be dict, DataFrame, or file content
    metadata: Dict[str, Any]
    validation_status: str
    validation_errors: List[str]
    submission_status: str
    submission_timestamp: Optional[datetime] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None


class TradeReportingTemplate:
    """Template for trade reporting (FINRA, SEC, etc.)"""
    
    @staticmethod
    def create_template() -> ReportTemplate:
        """Create trade reporting template"""
        return ReportTemplate(
            template_id="trade_reporting_finra",
            template_name="FINRA Trade Reporting",
            report_type=ReportType.TRADE_REPORTING,
            framework=RegulatoryFramework.FINRA,
            format=ReportFormat.CSV,
            frequency=ReportFrequency.DAILY,
            fields=[
                "trade_id", "symbol", "trade_date", "trade_time", "quantity",
                "price", "side", "counterparty", "venue", "settlement_date",
                "commission", "fees", "trade_type", "reporting_party"
            ],
            required_fields=[
                "trade_id", "symbol", "trade_date", "trade_time", "quantity",
                "price", "side", "venue"
            ],
            field_mappings={
                "trade_id": "TRADE_ID",
                "symbol": "SYMBOL",
                "trade_date": "TRADE_DATE",
                "trade_time": "TRADE_TIME",
                "quantity": "QUANTITY",
                "price": "PRICE",
                "side": "SIDE",
                "counterparty": "COUNTERPARTY",
                "venue": "VENUE"
            },
            validation_rules={
                "trade_id": {"type": "string", "max_length": 50},
                "symbol": {"type": "string", "max_length": 12},
                "quantity": {"type": "number", "min": 0},
                "price": {"type": "number", "min": 0},
                "side": {"type": "enum", "values": ["BUY", "SELL"]}
            },
            submission_details={
                "endpoint": "https://reporting.finra.org/trades",
                "auth_method": "certificate",
                "submission_window": "T+1",
                "retry_attempts": 3
            }
        )


class PositionReportingTemplate:
    """Template for position reporting (SEC, CFTC, etc.)"""
    
    @staticmethod
    def create_template() -> ReportTemplate:
        """Create position reporting template"""
        return ReportTemplate(
            template_id="position_reporting_sec",
            template_name="SEC Position Reporting",
            report_type=ReportType.POSITION_REPORTING,
            framework=RegulatoryFramework.SEC,
            format=ReportFormat.XML,
            frequency=ReportFrequency.DAILY,
            fields=[
                "position_id", "symbol", "position_date", "quantity",
                "market_value", "cost_basis", "unrealized_pnl",
                "asset_class", "sector", "country", "currency"
            ],
            required_fields=[
                "position_id", "symbol", "position_date", "quantity",
                "market_value", "asset_class"
            ],
            field_mappings={
                "position_id": "POSITION_ID",
                "symbol": "INSTRUMENT_ID",
                "position_date": "POSITION_DATE",
                "quantity": "QUANTITY",
                "market_value": "MARKET_VALUE",
                "asset_class": "ASSET_CLASS"
            },
            validation_rules={
                "position_id": {"type": "string", "max_length": 50},
                "symbol": {"type": "string", "max_length": 12},
                "quantity": {"type": "number"},
                "market_value": {"type": "number"},
                "asset_class": {"type": "enum", "values": ["EQUITY", "BOND", "DERIVATIVE", "COMMODITY"]}
            },
            submission_details={
                "endpoint": "https://reporting.sec.gov/positions",
                "auth_method": "api_key",
                "submission_window": "T+1",
                "retry_attempts": 3
            }
        )


class RiskReportingTemplate:
    """Template for risk reporting (Basel III, etc.)"""
    
    @staticmethod
    def create_template() -> ReportTemplate:
        """Create risk reporting template"""
        return ReportTemplate(
            template_id="risk_reporting_basel",
            template_name="Basel III Risk Reporting",
            report_type=ReportType.RISK_REPORTING,
            framework=RegulatoryFramework.BASEL_III,
            format=ReportFormat.EXCEL,
            frequency=ReportFrequency.MONTHLY,
            fields=[
                "report_date", "portfolio_value", "var_95", "var_99",
                "expected_shortfall", "leverage_ratio", "liquidity_ratio",
                "concentration_risk", "market_risk", "credit_risk",
                "operational_risk", "capital_adequacy_ratio"
            ],
            required_fields=[
                "report_date", "portfolio_value", "var_95", "leverage_ratio",
                "capital_adequacy_ratio"
            ],
            field_mappings={
                "report_date": "REPORT_DATE",
                "portfolio_value": "PORTFOLIO_VALUE",
                "var_95": "VAR_95",
                "leverage_ratio": "LEVERAGE_RATIO",
                "capital_adequacy_ratio": "CAR"
            },
            validation_rules={
                "report_date": {"type": "date"},
                "portfolio_value": {"type": "number", "min": 0},
                "var_95": {"type": "number", "min": 0, "max": 1},
                "leverage_ratio": {"type": "number", "min": 0}
            },
            submission_details={
                "endpoint": "https://reporting.bis.org/risk",
                "auth_method": "certificate",
                "submission_window": "M+10",
                "retry_attempts": 3
            }
        )


class ReportValidator:
    """Validator for regulatory reports"""
    
    def __init__(self, template: ReportTemplate):
        self.template = template
    
    def validate_data(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate report data against template"""
        errors = []
        
        try:
            # Convert data to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                errors.append("Invalid data format")
                return False, errors
            
            # Check required fields
            missing_fields = []
            for field in self.template.required_fields:
                if field not in df.columns:
                    missing_fields.append(field)
            
            if missing_fields:
                errors.append(f"Missing required fields: {', '.join(missing_fields)}")
            
            # Validate field values
            for field, rules in self.template.validation_rules.items():
                if field in df.columns:
                    field_errors = self._validate_field(df[field], field, rules)
                    errors.extend(field_errors)
            
            # Check field mappings
            for internal_field, external_field in self.template.field_mappings.items():
                if internal_field in df.columns:
                    # Field mapping is valid
                    pass
            
            is_valid = len(errors) == 0
            return is_valid, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def _validate_field(self, series: pd.Series, field_name: str, rules: Dict[str, Any]) -> List[str]:
        """Validate a specific field"""
        errors = []
        
        try:
            # Check data type
            if rules.get("type") == "string":
                if not series.dtype == "object":
                    errors.append(f"Field {field_name} must be string")
                
                # Check max length
                if "max_length" in rules:
                    max_len = rules["max_length"]
                    if series.str.len().max() > max_len:
                        errors.append(f"Field {field_name} exceeds maximum length of {max_len}")
            
            elif rules.get("type") == "number":
                if not pd.api.types.is_numeric_dtype(series):
                    errors.append(f"Field {field_name} must be numeric")
                
                # Check min value
                if "min" in rules:
                    min_val = rules["min"]
                    if series.min() < min_val:
                        errors.append(f"Field {field_name} has values below minimum of {min_val}")
                
                # Check max value
                if "max" in rules:
                    max_val = rules["max"]
                    if series.max() > max_val:
                        errors.append(f"Field {field_name} has values above maximum of {max_val}")
            
            elif rules.get("type") == "date":
                try:
                    pd.to_datetime(series)
                except (ValueError, KeyError, AttributeError) as e:
                    errors.append(f"Field {field_name} contains invalid dates")
            
            elif rules.get("type") == "enum":
                allowed_values = rules.get("values", [])
                invalid_values = series[~series.isin(allowed_values)]
                if not invalid_values.empty:
                    errors.append(f"Field {field_name} contains invalid values: {list(invalid_values.unique())}")
            
            # Check for null values in required fields
            if field_name in self.template.required_fields:
                if series.isnull().any():
                    errors.append(f"Field {field_name} contains null values")
            
        except Exception as e:
            errors.append(f"Error validating field {field_name}: {str(e)}")
        
        return errors


class ReportGenerator:
    """Generator for regulatory reports"""
    
    def __init__(self, template: ReportTemplate, output_dir: str = "reports"):
        self.template = template
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.validator = ReportValidator(template)
    
    def generate_report(
        self,
        data: Any,
        period_start: datetime,
        period_end: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RegulatoryReport:
        """Generate regulatory report"""
        try:
            report_id = str(uuid.uuid4())
            
            # Validate data
            is_valid, validation_errors = self.validator.validate_data(data)
            
            # Transform data according to template
            transformed_data = self._transform_data(data)
            
            # Generate report in specified format
            file_path, file_size, checksum = self._generate_output(
                transformed_data, report_id, period_start, period_end
            )
            
            # Create report object
            report = RegulatoryReport(
                report_id=report_id,
                template_id=self.template.template_id,
                report_type=self.template.report_type,
                framework=self.template.framework,
                format=self.template.format,
                period_start=period_start,
                period_end=period_end,
                generated_at=datetime.now(),
                data=transformed_data,
                metadata=metadata or {},
                validation_status="VALID" if is_valid else "INVALID",
                validation_errors=validation_errors,
                submission_status="READY" if is_valid else "VALIDATION_FAILED",
                file_path=file_path,
                file_size=file_size,
                checksum=checksum
            )
            
            logger.info(
                "Regulatory report generated",
                report_id=report_id,
                template_id=self.template.template_id,
                validation_status=report.validation_status
            )
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate regulatory report", error=str(e))
            raise
    
    def _transform_data(self, data: Any) -> Any:
        """Transform data according to template mappings"""
        try:
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                return data
            
            # Apply field mappings
            for internal_field, external_field in self.template.field_mappings.items():
                if internal_field in df.columns:
                    df.rename(columns={internal_field: external_field}, inplace=True)
            
            # Filter to template fields
            available_fields = [
                self.template.field_mappings.get(field, field) 
                for field in self.template.fields
                if field in df.columns or self.template.field_mappings.get(field, field) in df.columns
            ]
            
            df = df[available_fields]
            
            return df
            
        except Exception as e:
            logger.error("Failed to transform data", error=str(e))
            return data
    
    def _generate_output(
        self,
        data: Any,
        report_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> Tuple[str, int, str]:
        """Generate report output file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.template.template_id}_{report_id}_{timestamp}"
            
            if self.template.format == ReportFormat.CSV:
                file_path = self.output_dir / f"{filename}.csv"
                data.to_csv(file_path, index=False)
            
            elif self.template.format == ReportFormat.JSON:
                file_path = self.output_dir / f"{filename}.json"
                if isinstance(data, pd.DataFrame):
                    data.to_json(file_path, orient="records", indent=2)
                else:
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
            
            elif self.template.format == ReportFormat.XML:
                file_path = self.output_dir / f"{filename}.xml"
                self._generate_xml_report(data, file_path, period_start, period_end)
            
            elif self.template.format == ReportFormat.EXCEL:
                file_path = self.output_dir / f"{filename}.xlsx"
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    data.to_excel(writer, sheet_name='Report', index=False)
            
            elif self.template.format == ReportFormat.FIXED_WIDTH:
                file_path = self.output_dir / f"{filename}.txt"
                self._generate_fixed_width_report(data, file_path)
            
            else:
                raise ValueError(f"Unsupported report format: {self.template.format}")
            
            # Calculate file size and checksum
            file_size = file_path.stat().st_size
            checksum = self._calculate_file_checksum(file_path)
            
            return str(file_path), file_size, checksum
            
        except Exception as e:
            logger.error("Failed to generate output file", error=str(e))
            raise
    
    def _generate_xml_report(self, data: pd.DataFrame, file_path: Path, period_start: datetime, period_end: datetime):
        """Generate XML report"""
        root = ET.Element("Report")
        root.set("xmlns", "http://www.regulatory-reporting.com/schema")
        root.set("version", "1.0")
        
        # Header
        header = ET.SubElement(root, "Header")
        ET.SubElement(header, "ReportID").text = str(uuid.uuid4())
        ET.SubElement(header, "Framework").text = self.template.framework.value
        ET.SubElement(header, "ReportType").text = self.template.report_type.value
        ET.SubElement(header, "PeriodStart").text = period_start.isoformat()
        ET.SubElement(header, "PeriodEnd").text = period_end.isoformat()
        ET.SubElement(header, "GeneratedAt").text = datetime.now().isoformat()
        
        # Data
        data_element = ET.SubElement(root, "Data")
        for _, row in data.iterrows():
            record = ET.SubElement(data_element, "Record")
            for column, value in row.items():
                field = ET.SubElement(record, column)
                field.text = str(value)
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
    
    def _generate_fixed_width_report(self, data: pd.DataFrame, file_path: Path):
        """Generate fixed-width report"""
        # Define field widths (this would be template-specific)
        field_widths = {
            'TRADE_ID': 20,
            'SYMBOL': 12,
            'TRADE_DATE': 10,
            'TRADE_TIME': 8,
            'QUANTITY': 15,
            'PRICE': 15,
            'SIDE': 4,
            'VENUE': 10
        }
        
        with open(file_path, 'w') as f:
            for _, row in data.iterrows():
                line = ""
                for column in data.columns:
                    value = str(row[column])
                    width = field_widths.get(column, 20)
                    line += value.ljust(width)[:width]
                f.write(line + "\n")
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        import hashlib
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class RegulatoryReporter:
    """Main regulatory reporting system"""
    
    def __init__(
        self,
        event_bus: EventBus,
        compliance_monitor: ComplianceMonitor,
        audit_system: AuditSystem,
        output_dir: str = "regulatory_reports"
    ):
        self.event_bus = event_bus
        self.compliance_monitor = compliance_monitor
        self.audit_system = audit_system
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Report templates
        self.templates: Dict[str, ReportTemplate] = {}
        self.generators: Dict[str, ReportGenerator] = {}
        
        # Generated reports
        self.reports: List[RegulatoryReport] = []
        
        # Scheduling
        self.scheduled_reports: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default templates
        self._initialize_default_templates()
        
        logger.info("Regulatory Reporter initialized")
    
    def _initialize_default_templates(self):
        """Initialize default report templates"""
        # Trade reporting
        trade_template = TradeReportingTemplate.create_template()
        self.add_template(trade_template)
        
        # Position reporting
        position_template = PositionReportingTemplate.create_template()
        self.add_template(position_template)
        
        # Risk reporting
        risk_template = RiskReportingTemplate.create_template()
        self.add_template(risk_template)
    
    def add_template(self, template: ReportTemplate) -> bool:
        """Add a report template"""
        try:
            self.templates[template.template_id] = template
            self.generators[template.template_id] = ReportGenerator(template, str(self.output_dir))
            
            logger.info("Report template added", template_id=template.template_id)
            return True
            
        except Exception as e:
            logger.error("Failed to add report template", template_id=template.template_id, error=str(e))
            return False
    
    def generate_report(
        self,
        template_id: str,
        data_source: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[RegulatoryReport]:
        """Generate a regulatory report"""
        try:
            if template_id not in self.templates:
                logger.error("Template not found", template_id=template_id)
                return None
            
            template = self.templates[template_id]
            generator = self.generators[template_id]
            
            # Set default time periods
            if period_end is None:
                period_end = datetime.now()
            if period_start is None:
                if template.frequency == ReportFrequency.DAILY:
                    period_start = period_end - timedelta(days=1)
                elif template.frequency == ReportFrequency.WEEKLY:
                    period_start = period_end - timedelta(weeks=1)
                elif template.frequency == ReportFrequency.MONTHLY:
                    period_start = period_end - timedelta(days=30)
                else:
                    period_start = period_end - timedelta(days=1)
            
            # Get data based on source
            data = self._get_report_data(data_source, template, period_start, period_end)
            
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                logger.warning("No data available for report", template_id=template_id)
                return None
            
            # Generate report
            report = generator.generate_report(data, period_start, period_end, metadata)
            
            # Store report
            self.reports.append(report)
            
            # Log report generation
            self._log_report_generation(report)
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate report", template_id=template_id, error=str(e))
            return None
    
    def _get_report_data(
        self,
        data_source: str,
        template: ReportTemplate,
        period_start: datetime,
        period_end: datetime
    ) -> Optional[Any]:
        """Get data for report generation"""
        try:
            if data_source == "trades":
                return self._get_trade_data(period_start, period_end)
            elif data_source == "positions":
                return self._get_position_data(period_start, period_end)
            elif data_source == "risk":
                return self._get_risk_data(period_start, period_end)
            elif data_source == "compliance":
                return self._get_compliance_data(period_start, period_end)
            elif data_source == "audit":
                return self._get_audit_data(period_start, period_end)
            else:
                logger.error("Unknown data source", data_source=data_source)
                return None
                
        except Exception as e:
            logger.error("Failed to get report data", data_source=data_source, error=str(e))
            return None
    
    def _get_trade_data(self, period_start: datetime, period_end: datetime) -> pd.DataFrame:
        """Get trade data for reporting"""
        # Mock trade data - in production, this would query actual trade database
        trades = []
        
        # Generate sample trades
        for i in range(100):
            trade = {
                "trade_id": f"TRD{i:06d}",
                "symbol": "ETH-USD",
                "trade_date": period_start.strftime("%Y-%m-%d"),
                "trade_time": "09:30:00",
                "quantity": 1000 + i,
                "price": 2000.00 + i,
                "side": "BUY" if i % 2 == 0 else "SELL",
                "counterparty": "EXCHANGE",
                "venue": "COINBASE",
                "settlement_date": (period_start + timedelta(days=1)).strftime("%Y-%m-%d"),
                "commission": 10.00,
                "fees": 5.00,
                "trade_type": "MARKET",
                "reporting_party": "FIRM"
            }
            trades.append(trade)
        
        return pd.DataFrame(trades)
    
    def _get_position_data(self, period_start: datetime, period_end: datetime) -> pd.DataFrame:
        """Get position data for reporting"""
        # Mock position data
        positions = []
        
        symbols = ["ETH-USD", "BTC-USD", "AAPL", "GOOGL", "MSFT"]
        
        for i, symbol in enumerate(symbols):
            position = {
                "position_id": f"POS{i:06d}",
                "symbol": symbol,
                "position_date": period_end.strftime("%Y-%m-%d"),
                "quantity": 1000 * (i + 1),
                "market_value": 2000000 * (i + 1),
                "cost_basis": 1900000 * (i + 1),
                "unrealized_pnl": 100000 * (i + 1),
                "asset_class": "EQUITY" if symbol in ["AAPL", "GOOGL", "MSFT"] else "COMMODITY",
                "sector": "TECHNOLOGY",
                "country": "US",
                "currency": "USD"
            }
            positions.append(position)
        
        return pd.DataFrame(positions)
    
    def _get_risk_data(self, period_start: datetime, period_end: datetime) -> pd.DataFrame:
        """Get risk data for reporting"""
        # Mock risk data
        risk_data = [{
            "report_date": period_end.strftime("%Y-%m-%d"),
            "portfolio_value": 10000000.00,
            "var_95": 0.045,
            "var_99": 0.065,
            "expected_shortfall": 0.055,
            "leverage_ratio": 3.5,
            "liquidity_ratio": 0.85,
            "concentration_risk": 0.15,
            "market_risk": 0.035,
            "credit_risk": 0.015,
            "operational_risk": 0.005,
            "capital_adequacy_ratio": 0.12
        }]
        
        return pd.DataFrame(risk_data)
    
    def _get_compliance_data(self, period_start: datetime, period_end: datetime) -> pd.DataFrame:
        """Get compliance data for reporting"""
        violations = self.compliance_monitor.get_violations()
        
        # Filter by period
        period_violations = [
            v for v in violations 
            if period_start <= v.timestamp <= period_end
        ]
        
        # Convert to DataFrame
        compliance_data = []
        for violation in period_violations:
            compliance_data.append({
                "violation_id": violation.violation_id,
                "rule_id": violation.rule_id,
                "framework": violation.framework.value,
                "severity": violation.severity.value,
                "status": violation.status.value,
                "description": violation.description,
                "timestamp": violation.timestamp.isoformat(),
                "resolved": violation.resolved,
                "resolution_timestamp": violation.resolution_timestamp.isoformat() if violation.resolution_timestamp else None
            })
        
        return pd.DataFrame(compliance_data)
    
    def _get_audit_data(self, period_start: datetime, period_end: datetime) -> pd.DataFrame:
        """Get audit data for reporting"""
        events = self.audit_system.storage.get_events(
            start_time=period_start,
            end_time=period_end
        )
        
        # Convert to DataFrame
        audit_data = []
        for event in events:
            audit_data.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "outcome": event.outcome.value,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.context.user_id,
                "system_component": event.context.system_component,
                "action": event.context.action,
                "resource": event.context.resource,
                "description": event.description
            })
        
        return pd.DataFrame(audit_data)
    
    def _log_report_generation(self, report: RegulatoryReport):
        """Log report generation in audit system"""
        try:
            from .audit_system import AuditContext, AuditEventType, AuditSeverity, AuditOutcome
            
            context = AuditContext(
                user_id="system",
                session_id="regulatory_reporter",
                ip_address="127.0.0.1",
                user_agent="regulatory_reporter",
                system_component="regulatory_reporter",
                action="generate_report",
                resource=report.template_id
            )
            
            self.audit_system.log_event(
                event_type=AuditEventType.MANUAL_INTERVENTION,
                severity=AuditSeverity.INFO,
                outcome=AuditOutcome.SUCCESS,
                context=context,
                description=f"Regulatory report generated: {report.report_type.value}",
                details={
                    "report_id": report.report_id,
                    "template_id": report.template_id,
                    "framework": report.framework.value,
                    "validation_status": report.validation_status,
                    "file_path": report.file_path,
                    "file_size": report.file_size
                }
            )
            
        except Exception as e:
            logger.error("Failed to log report generation", error=str(e))
    
    def get_report_status(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific report"""
        try:
            report = next((r for r in self.reports if r.report_id == report_id), None)
            
            if not report:
                return None
            
            return {
                "report_id": report.report_id,
                "template_id": report.template_id,
                "report_type": report.report_type.value,
                "framework": report.framework.value,
                "format": report.format.value,
                "generated_at": report.generated_at.isoformat(),
                "validation_status": report.validation_status,
                "validation_errors": report.validation_errors,
                "submission_status": report.submission_status,
                "submission_timestamp": report.submission_timestamp.isoformat() if report.submission_timestamp else None,
                "file_path": report.file_path,
                "file_size": report.file_size,
                "checksum": report.checksum
            }
            
        except Exception as e:
            logger.error("Failed to get report status", report_id=report_id, error=str(e))
            return None
    
    def get_all_reports(
        self,
        framework: Optional[RegulatoryFramework] = None,
        report_type: Optional[ReportType] = None,
        validation_status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[RegulatoryReport]:
        """Get all reports with optional filtering"""
        try:
            reports = self.reports.copy()
            
            # Apply filters
            if framework:
                reports = [r for r in reports if r.framework == framework]
            
            if report_type:
                reports = [r for r in reports if r.report_type == report_type]
            
            if validation_status:
                reports = [r for r in reports if r.validation_status == validation_status]
            
            # Sort by generated_at (newest first)
            reports.sort(key=lambda x: x.generated_at, reverse=True)
            
            # Apply limit
            if limit:
                reports = reports[:limit]
            
            return reports
            
        except Exception as e:
            logger.error("Failed to get reports", error=str(e))
            return []
    
    def get_reporting_summary(self) -> Dict[str, Any]:
        """Get summary of regulatory reporting"""
        try:
            total_reports = len(self.reports)
            
            # Count by framework
            reports_by_framework = {}
            for report in self.reports:
                framework = report.framework.value
                reports_by_framework[framework] = reports_by_framework.get(framework, 0) + 1
            
            # Count by validation status
            reports_by_validation = {}
            for report in self.reports:
                status = report.validation_status
                reports_by_validation[status] = reports_by_validation.get(status, 0) + 1
            
            # Count by submission status
            reports_by_submission = {}
            for report in self.reports:
                status = report.submission_status
                reports_by_submission[status] = reports_by_submission.get(status, 0) + 1
            
            # Recent reports
            recent_reports = sorted(self.reports, key=lambda x: x.generated_at, reverse=True)[:10]
            
            return {
                "total_reports": total_reports,
                "active_templates": len(self.templates),
                "reports_by_framework": reports_by_framework,
                "reports_by_validation": reports_by_validation,
                "reports_by_submission": reports_by_submission,
                "recent_reports": [
                    {
                        "report_id": r.report_id,
                        "template_id": r.template_id,
                        "framework": r.framework.value,
                        "generated_at": r.generated_at.isoformat(),
                        "validation_status": r.validation_status
                    }
                    for r in recent_reports
                ]
            }
            
        except Exception as e:
            logger.error("Failed to get reporting summary", error=str(e))
            return {}
    
    def cleanup_old_reports(self, days_old: int = 90):
        """Clean up old report files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            reports_to_remove = []
            for report in self.reports:
                if report.generated_at < cutoff_date:
                    # Remove file if it exists
                    if report.file_path and Path(report.file_path).exists():
                        Path(report.file_path).unlink()
                    
                    reports_to_remove.append(report)
            
            # Remove from reports list
            for report in reports_to_remove:
                self.reports.remove(report)
            
            logger.info("Cleaned up old reports", count=len(reports_to_remove))
            
        except Exception as e:
            logger.error("Failed to cleanup old reports", error=str(e))