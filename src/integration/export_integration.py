"""
Export and Integration System for GrandModel
==========================================

Comprehensive export and integration capabilities including PDF generation,
HTML reports, CSV data export, API integration, and notebook integration.

Features:
- PDF report generation with professional formatting
- HTML interactive reports with embedded charts
- CSV data export for external analysis
- API integration for real-time data access
- Notebook integration for Jupyter environments
- Email report distribution
- Cloud storage integration
- Data pipeline integration
- Real-time streaming capabilities

Author: Agent 6 - Visualization and Reporting System
"""

import pandas as pd
import numpy as np
import json
import csv
import io
import base64
import zipfile
import tempfile
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
import aiohttp
import aiofiles
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs
import dropbox
import requests
import plotly.graph_objects as go
import plotly.io as pio
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from jinja2 import Template
import xlsxwriter
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for export and integration"""
    output_directory: str = "/home/QuantNova/GrandModel/results/exports"
    
    # PDF settings
    pdf_page_size: str = "A4"
    pdf_orientation: str = "portrait"
    pdf_margins: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)  # top, right, bottom, left
    
    # HTML settings
    html_template: str = "professional"
    embed_charts: bool = True
    responsive_design: bool = True
    
    # Excel settings
    excel_chart_support: bool = True
    excel_formatting: bool = True
    
    # Email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    
    # Cloud storage settings
    aws_access_key: str = ""
    aws_secret_key: str = ""
    aws_bucket: str = ""
    azure_connection_string: str = ""
    gcp_credentials_path: str = ""
    gcp_bucket: str = ""
    
    # API settings
    api_base_url: str = "http://localhost:8000"
    api_key: str = ""
    webhook_urls: List[str] = None
    
    def __post_init__(self):
        if self.webhook_urls is None:
            self.webhook_urls = []


class ExportIntegration:
    """
    Comprehensive export and integration system
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize export and integration system
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
        
        # Create output directory
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cloud storage clients
        self.cloud_clients = {}
        self._init_cloud_clients()
        
        # Initialize executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Export history
        self.export_history = []
        
        logger.info("Export and Integration System initialized")
    
    def _init_cloud_clients(self):
        """Initialize cloud storage clients"""
        try:
            # AWS S3
            if self.config.aws_access_key and self.config.aws_secret_key:
                self.cloud_clients['s3'] = boto3.client(
                    's3',
                    aws_access_key_id=self.config.aws_access_key,
                    aws_secret_access_key=self.config.aws_secret_key
                )
            
            # Azure Blob Storage
            if self.config.azure_connection_string:
                self.cloud_clients['azure'] = BlobServiceClient.from_connection_string(
                    self.config.azure_connection_string
                )
            
            # Google Cloud Storage
            if self.config.gcp_credentials_path:
                self.cloud_clients['gcs'] = gcs.Client.from_service_account_json(
                    self.config.gcp_credentials_path
                )
            
            logger.info("Cloud storage clients initialized")
            
        except Exception as e:
            logger.error(f"Error initializing cloud clients: {e}")
    
    def export_to_pdf(self, 
                     report_data: Dict[str, Any],
                     filename: str,
                     charts: Dict[str, go.Figure] = None,
                     include_charts: bool = True) -> str:
        """
        Export report to PDF format
        
        Args:
            report_data: Report data to export
            filename: Output filename
            charts: Optional charts to include
            include_charts: Whether to include charts
            
        Returns:
            Path to generated PDF file
        """
        try:
            # Create PDF path
            pdf_path = self.output_dir / f"{filename}.pdf"
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=A4 if self.config.pdf_page_size == "A4" else letter,
                topMargin=self.config.pdf_margins[0] * inch,
                rightMargin=self.config.pdf_margins[1] * inch,
                bottomMargin=self.config.pdf_margins[2] * inch,
                leftMargin=self.config.pdf_margins[3] * inch
            )
            
            # Build PDF content
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.navy,
                alignment=1,  # Center alignment
                spaceAfter=30
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.darkblue,
                spaceBefore=20,
                spaceAfter=10
            )
            
            # Title page
            story.append(Paragraph("GrandModel Trading System", title_style))
            story.append(Paragraph("Performance Analysis Report", styles['Heading2']))
            story.append(Spacer(1, 20))
            
            # Report metadata
            metadata = report_data.get('report_metadata', {})
            if metadata:
                story.append(Paragraph("Report Information", heading_style))
                
                metadata_data = []
                for key, value in metadata.items():
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:50] + "..."
                    metadata_data.append([key.replace('_', ' ').title(), str(value)])
                
                metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
                metadata_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(metadata_table)
                story.append(Spacer(1, 20))
            
            # Executive Summary
            if 'executive_summary' in report_data or 'key_metrics' in report_data:
                story.append(Paragraph("Executive Summary", heading_style))
                
                # Key metrics
                if 'key_metrics' in report_data:
                    metrics_data = []
                    for key, value in report_data['key_metrics'].items():
                        if isinstance(value, (int, float)):
                            if 'return' in key.lower() or 'ratio' in key.lower():
                                formatted_value = f"{value:.2%}"
                            else:
                                formatted_value = f"{value:.4f}"
                        else:
                            formatted_value = str(value)
                        metrics_data.append([key.replace('_', ' ').title(), formatted_value])
                    
                    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 3.5*inch])
                    metrics_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(metrics_table)
                    story.append(Spacer(1, 20))
            
            # Charts
            if include_charts and charts:
                story.append(Paragraph("Performance Charts", heading_style))
                
                for chart_name, chart_fig in charts.items():
                    # Convert chart to image
                    img_bytes = pio.to_image(chart_fig, format='png', width=600, height=400)
                    img_path = self.output_dir / f"temp_chart_{chart_name}.png"
                    
                    with open(img_path, 'wb') as f:
                        f.write(img_bytes)
                    
                    # Add chart to PDF
                    chart_img = Image(str(img_path), width=5*inch, height=3.33*inch)
                    story.append(chart_img)
                    story.append(Paragraph(f"Figure: {chart_name.replace('_', ' ').title()}", styles['Normal']))
                    story.append(Spacer(1, 20))
                    
                    # Clean up temporary file
                    img_path.unlink()
            
            # Performance Analysis
            if 'performance_analysis' in report_data:
                story.append(Paragraph("Performance Analysis", heading_style))
                
                perf_analysis = report_data['performance_analysis']
                for section_name, section_data in perf_analysis.items():
                    if isinstance(section_data, dict):
                        story.append(Paragraph(section_name.replace('_', ' ').title(), styles['Heading3']))
                        
                        section_table_data = []
                        for key, value in section_data.items():
                            if isinstance(value, (int, float)):
                                formatted_value = f"{value:.4f}"
                            else:
                                formatted_value = str(value)
                            section_table_data.append([key.replace('_', ' ').title(), formatted_value])
                        
                        if section_table_data:
                            section_table = Table(section_table_data, colWidths=[2.5*inch, 3.5*inch])
                            section_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, 0), 9),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black)
                            ]))
                            story.append(section_table)
                            story.append(Spacer(1, 15))
            
            # Recommendations
            if 'recommendations' in report_data:
                story.append(Paragraph("Recommendations", heading_style))
                
                for i, rec in enumerate(report_data['recommendations'], 1):
                    story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                    story.append(Spacer(1, 10))
            
            # Footer
            story.append(Spacer(1, 50))
            story.append(Paragraph("Generated by GrandModel Export System", styles['Normal']))
            story.append(Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            # Record export
            self._record_export('pdf', str(pdf_path), report_data.get('report_metadata', {}))
            
            logger.info(f"PDF report exported to {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Error exporting PDF: {e}")
            return ""
    
    def export_to_html(self,
                      report_data: Dict[str, Any],
                      filename: str,
                      charts: Dict[str, go.Figure] = None,
                      template_name: str = "professional") -> str:
        """
        Export report to HTML format
        
        Args:
            report_data: Report data to export
            filename: Output filename
            charts: Optional charts to include
            template_name: Template to use
            
        Returns:
            Path to generated HTML file
        """
        try:
            html_path = self.output_dir / f"{filename}.html"
            
            # Get template
            template = self._get_html_template(template_name)
            
            # Convert charts to HTML
            chart_html = {}
            if charts:
                for chart_name, chart_fig in charts.items():
                    chart_html[chart_name] = pio.to_html(chart_fig, include_plotlyjs='cdn')
            
            # Render template
            html_content = template.render(
                report=report_data,
                charts=chart_html,
                timestamp=datetime.now(),
                config=self.config
            )
            
            # Write to file
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Record export
            self._record_export('html', str(html_path), report_data.get('report_metadata', {}))
            
            logger.info(f"HTML report exported to {html_path}")
            return str(html_path)
            
        except Exception as e:
            logger.error(f"Error exporting HTML: {e}")
            return ""
    
    def export_to_excel(self,
                       report_data: Dict[str, Any],
                       filename: str,
                       charts: Dict[str, go.Figure] = None,
                       include_charts: bool = True) -> str:
        """
        Export report to Excel format
        
        Args:
            report_data: Report data to export
            filename: Output filename
            charts: Optional charts to include
            include_charts: Whether to include charts
            
        Returns:
            Path to generated Excel file
        """
        try:
            excel_path = self.output_dir / f"{filename}.xlsx"
            
            # Create Excel workbook
            workbook = xlsxwriter.Workbook(str(excel_path))
            
            # Define formats
            title_format = workbook.add_format({
                'bold': True,
                'font_size': 16,
                'align': 'center',
                'valign': 'vcenter',
                'bg_color': '#4F81BD',
                'font_color': 'white'
            })
            
            header_format = workbook.add_format({
                'bold': True,
                'font_size': 12,
                'align': 'center',
                'bg_color': '#D9E1F2',
                'border': 1
            })
            
            data_format = workbook.add_format({
                'font_size': 10,
                'border': 1
            })
            
            # Summary sheet
            summary_sheet = workbook.add_worksheet('Summary')
            summary_sheet.set_column('A:A', 20)
            summary_sheet.set_column('B:B', 15)
            
            # Title
            summary_sheet.merge_range('A1:B1', 'Performance Report Summary', title_format)
            
            # Key metrics
            if 'key_metrics' in report_data:
                row = 3
                summary_sheet.write(row, 0, 'Metric', header_format)
                summary_sheet.write(row, 1, 'Value', header_format)
                row += 1
                
                for key, value in report_data['key_metrics'].items():
                    summary_sheet.write(row, 0, key.replace('_', ' ').title(), data_format)
                    if isinstance(value, (int, float)):
                        if 'return' in key.lower() or 'ratio' in key.lower():
                            summary_sheet.write(row, 1, value, workbook.add_format({'num_format': '0.00%', 'border': 1}))
                        else:
                            summary_sheet.write(row, 1, value, workbook.add_format({'num_format': '0.0000', 'border': 1}))
                    else:
                        summary_sheet.write(row, 1, str(value), data_format)
                    row += 1
            
            # Performance data sheet
            if 'performance_analysis' in report_data:
                perf_sheet = workbook.add_worksheet('Performance Analysis')
                perf_sheet.set_column('A:A', 25)
                perf_sheet.set_column('B:B', 15)
                
                perf_sheet.merge_range('A1:B1', 'Performance Analysis', title_format)
                
                row = 3
                for section_name, section_data in report_data['performance_analysis'].items():
                    if isinstance(section_data, dict):
                        # Section header
                        perf_sheet.merge_range(f'A{row}:B{row}', section_name.replace('_', ' ').title(), header_format)
                        row += 1
                        
                        # Section data
                        for key, value in section_data.items():
                            perf_sheet.write(row, 0, key.replace('_', ' ').title(), data_format)
                            if isinstance(value, (int, float)):
                                perf_sheet.write(row, 1, value, workbook.add_format({'num_format': '0.0000', 'border': 1}))
                            else:
                                perf_sheet.write(row, 1, str(value), data_format)
                            row += 1
                        
                        row += 1  # Add space between sections
            
            # Charts sheet
            if include_charts and charts and self.config.excel_chart_support:
                charts_sheet = workbook.add_worksheet('Charts')
                charts_sheet.merge_range('A1:E1', 'Performance Charts', title_format)
                
                row = 3
                for chart_name, chart_fig in charts.items():
                    # Convert chart to image
                    img_bytes = pio.to_image(chart_fig, format='png', width=600, height=400)
                    img_path = self.output_dir / f"temp_chart_{chart_name}.png"
                    
                    with open(img_path, 'wb') as f:
                        f.write(img_bytes)
                    
                    # Insert image
                    charts_sheet.write(row, 0, chart_name.replace('_', ' ').title(), header_format)
                    charts_sheet.insert_image(row + 1, 0, str(img_path), {'x_scale': 0.8, 'y_scale': 0.8})
                    
                    row += 25  # Move to next chart position
                    
                    # Clean up temporary file
                    img_path.unlink()
            
            # Close workbook
            workbook.close()
            
            # Record export
            self._record_export('excel', str(excel_path), report_data.get('report_metadata', {}))
            
            logger.info(f"Excel report exported to {excel_path}")
            return str(excel_path)
            
        except Exception as e:
            logger.error(f"Error exporting Excel: {e}")
            return ""
    
    def export_to_csv(self,
                     data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                     filename: str,
                     include_metadata: bool = True) -> str:
        """
        Export data to CSV format
        
        Args:
            data: Data to export (DataFrame or dict of DataFrames)
            filename: Output filename
            include_metadata: Whether to include metadata
            
        Returns:
            Path to generated CSV file or ZIP file
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Single DataFrame
                csv_path = self.output_dir / f"{filename}.csv"
                data.to_csv(csv_path, index=True)
                
                # Record export
                self._record_export('csv', str(csv_path), {'rows': len(data), 'columns': len(data.columns)})
                
                logger.info(f"CSV data exported to {csv_path}")
                return str(csv_path)
            
            elif isinstance(data, dict):
                # Multiple DataFrames - create ZIP file
                zip_path = self.output_dir / f"{filename}.zip"
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for sheet_name, df in data.items():
                        # Create CSV content
                        csv_content = df.to_csv(index=True)
                        
                        # Add to ZIP
                        zipf.writestr(f"{sheet_name}.csv", csv_content)
                    
                    # Add metadata if requested
                    if include_metadata:
                        metadata = {
                            'export_timestamp': datetime.now().isoformat(),
                            'sheets': {name: {'rows': len(df), 'columns': len(df.columns)} 
                                     for name, df in data.items()}
                        }
                        zipf.writestr('metadata.json', json.dumps(metadata, indent=2))
                
                # Record export
                self._record_export('zip', str(zip_path), {'sheets': len(data)})
                
                logger.info(f"CSV data exported to {zip_path}")
                return str(zip_path)
            
            else:
                raise ValueError("Data must be DataFrame or dict of DataFrames")
                
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            return ""
    
    def export_to_json(self,
                      data: Dict[str, Any],
                      filename: str,
                      pretty_print: bool = True) -> str:
        """
        Export data to JSON format
        
        Args:
            data: Data to export
            filename: Output filename
            pretty_print: Whether to format JSON nicely
            
        Returns:
            Path to generated JSON file
        """
        try:
            json_path = self.output_dir / f"{filename}.json"
            
            # Convert to JSON-serializable format
            json_data = self._convert_to_json_serializable(data)
            
            # Write to file
            with open(json_path, 'w', encoding='utf-8') as f:
                if pretty_print:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(json_data, f, ensure_ascii=False)
            
            # Record export
            self._record_export('json', str(json_path), {'size_bytes': json_path.stat().st_size})
            
            logger.info(f"JSON data exported to {json_path}")
            return str(json_path)
            
        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")
            return ""
    
    def send_email_report(self,
                         report_path: str,
                         recipients: List[str],
                         subject: str,
                         body: str = "",
                         attachments: List[str] = None) -> bool:
        """
        Send report via email
        
        Args:
            report_path: Path to report file
            recipients: List of email recipients
            subject: Email subject
            body: Email body
            attachments: Additional attachments
            
        Returns:
            Success status
        """
        try:
            if not self.config.smtp_username or not self.config.smtp_password:
                logger.error("SMTP credentials not configured")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.smtp_username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add main report
            with open(report_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {Path(report_path).name}'
                )
                msg.attach(part)
            
            # Add additional attachments
            if attachments:
                for attachment_path in attachments:
                    with open(attachment_path, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {Path(attachment_path).name}'
                        )
                        msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            
            text = msg.as_string()
            server.sendmail(self.config.smtp_username, recipients, text)
            server.quit()
            
            logger.info(f"Email sent successfully to {recipients}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def upload_to_cloud(self,
                       file_path: str,
                       cloud_provider: str,
                       remote_path: str = None) -> bool:
        """
        Upload file to cloud storage
        
        Args:
            file_path: Local file path
            cloud_provider: Cloud provider ('s3', 'azure', 'gcs')
            remote_path: Remote path (optional)
            
        Returns:
            Success status
        """
        try:
            if cloud_provider not in self.cloud_clients:
                logger.error(f"Cloud provider {cloud_provider} not configured")
                return False
            
            file_name = Path(file_path).name
            if remote_path is None:
                remote_path = f"exports/{datetime.now().strftime('%Y/%m/%d')}/{file_name}"
            
            if cloud_provider == 's3':
                self.cloud_clients['s3'].upload_file(
                    file_path,
                    self.config.aws_bucket,
                    remote_path
                )
            
            elif cloud_provider == 'azure':
                blob_client = self.cloud_clients['azure'].get_blob_client(
                    container="exports",
                    blob=remote_path
                )
                with open(file_path, 'rb') as data:
                    blob_client.upload_blob(data, overwrite=True)
            
            elif cloud_provider == 'gcs':
                bucket = self.cloud_clients['gcs'].bucket(self.config.gcp_bucket)
                blob = bucket.blob(remote_path)
                blob.upload_from_filename(file_path)
            
            logger.info(f"File uploaded to {cloud_provider}: {remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading to cloud: {e}")
            return False
    
    def create_notebook_integration(self,
                                  report_data: Dict[str, Any],
                                  notebook_path: str) -> str:
        """
        Create Jupyter notebook with integrated report
        
        Args:
            report_data: Report data
            notebook_path: Output notebook path
            
        Returns:
            Path to generated notebook
        """
        try:
            # Create notebook structure
            notebook = {
                "cells": [],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "name": "python",
                        "version": "3.8.0"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
            
            # Title cell
            title_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {report_data.get('report_metadata', {}).get('strategy_name', 'Strategy')} Performance Report\n\n",
                    f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                ]
            }
            notebook["cells"].append(title_cell)
            
            # Import cell
            import_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import plotly.graph_objects as go\n",
                    "import plotly.express as px\n",
                    "from plotly.subplots import make_subplots\n",
                    "import json\n",
                    "from datetime import datetime\n",
                    "\n",
                    "# Load report data\n",
                    f"report_data = {json.dumps(report_data, indent=2, default=str)}\n"
                ]
            }
            notebook["cells"].append(import_cell)
            
            # Key metrics cell
            if 'key_metrics' in report_data:
                metrics_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Key Performance Metrics\n\n"]
                }
                
                for key, value in report_data['key_metrics'].items():
                    if isinstance(value, (int, float)):
                        if 'return' in key.lower() or 'ratio' in key.lower():
                            formatted_value = f"{value:.2%}"
                        else:
                            formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    
                    metrics_cell["source"].append(f"- **{key.replace('_', ' ').title()}**: {formatted_value}\n")
                
                notebook["cells"].append(metrics_cell)
            
            # Chart generation cells
            chart_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generate performance charts\n",
                    "fig = make_subplots(\n",
                    "    rows=2, cols=2,\n",
                    "    subplot_titles=('Performance Overview', 'Risk Metrics', 'Analysis', 'Summary')\n",
                    ")\n",
                    "\n",
                    "# Add sample data visualization\n",
                    "fig.add_trace(\n",
                    "    go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], name='Sample Data'),\n",
                    "    row=1, col=1\n",
                    ")\n",
                    "\n",
                    "fig.update_layout(height=600, showlegend=True)\n",
                    "fig.show()\n"
                ]
            }
            notebook["cells"].append(chart_cell)
            
            # Analysis cell
            analysis_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Analysis\n\n"]
            }
            
            if 'key_insights' in report_data:
                analysis_cell["source"].append("### Key Insights\n\n")
                for insight in report_data['key_insights']:
                    analysis_cell["source"].append(f"- {insight}\n")
                analysis_cell["source"].append("\n")
            
            if 'recommendations' in report_data:
                analysis_cell["source"].append("### Recommendations\n\n")
                for i, rec in enumerate(report_data['recommendations'], 1):
                    analysis_cell["source"].append(f"{i}. {rec}\n")
            
            notebook["cells"].append(analysis_cell)
            
            # Save notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Notebook created at {notebook_path}")
            return notebook_path
            
        except Exception as e:
            logger.error(f"Error creating notebook: {e}")
            return ""
    
    def _get_html_template(self, template_name: str) -> Template:
        """Get HTML template"""
        
        if template_name == "professional":
            template_str = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{{ report.report_metadata.strategy_name }} - Performance Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                        line-height: 1.6;
                    }
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    }
                    .header {
                        text-align: center;
                        border-bottom: 3px solid #2E86AB;
                        padding-bottom: 20px;
                        margin-bottom: 30px;
                    }
                    .header h1 {
                        color: #2E86AB;
                        margin: 0;
                        font-size: 2.5em;
                    }
                    .section {
                        margin: 30px 0;
                        padding: 20px;
                        border-left: 4px solid #2E86AB;
                        background-color: #f9f9f9;
                    }
                    .section h2 {
                        color: #2E86AB;
                        margin-top: 0;
                    }
                    .metrics-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 20px;
                        margin: 20px 0;
                    }
                    .metric-card {
                        background: white;
                        padding: 15px;
                        border-radius: 8px;
                        text-align: center;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }
                    .metric-value {
                        font-size: 1.8em;
                        font-weight: bold;
                        color: #2E86AB;
                    }
                    .metric-label {
                        color: #666;
                        font-size: 0.9em;
                    }
                    .chart-container {
                        margin: 20px 0;
                        padding: 20px;
                        background: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }
                    .insights {
                        background-color: #e8f4f8;
                        padding: 20px;
                        border-radius: 8px;
                        margin: 20px 0;
                    }
                    .insights h3 {
                        color: #2E86AB;
                        margin-top: 0;
                    }
                    .insights ul {
                        list-style-type: none;
                        padding: 0;
                    }
                    .insights li {
                        padding: 8px 0;
                        border-bottom: 1px solid #ddd;
                    }
                    .insights li:last-child {
                        border-bottom: none;
                    }
                    .recommendations {
                        background-color: #fff3cd;
                        padding: 20px;
                        border-radius: 8px;
                        margin: 20px 0;
                    }
                    .recommendations h3 {
                        color: #856404;
                        margin-top: 0;
                    }
                    .footer {
                        text-align: center;
                        margin-top: 50px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                        color: #666;
                        font-size: 0.9em;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>{{ report.report_metadata.strategy_name }}</h1>
                        <p>Performance Analysis Report</p>
                        <p>Generated: {{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                    </div>
                    
                    {% if report.key_metrics %}
                    <div class="section">
                        <h2>Key Performance Metrics</h2>
                        <div class="metrics-grid">
                            {% for key, value in report.key_metrics.items() %}
                            <div class="metric-card">
                                <div class="metric-value">
                                    {% if 'return' in key.lower() or 'ratio' in key.lower() %}
                                        {{ "%.2f%%"|format(value * 100) }}
                                    {% else %}
                                        {{ "%.4f"|format(value) }}
                                    {% endif %}
                                </div>
                                <div class="metric-label">{{ key.replace('_', ' ').title() }}</div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    {% endif %}
                    
                    {% if charts %}
                    <div class="section">
                        <h2>Performance Charts</h2>
                        {% for chart_name, chart_html in charts.items() %}
                        <div class="chart-container">
                            <h3>{{ chart_name.replace('_', ' ').title() }}</h3>
                            {{ chart_html|safe }}
                        </div>
                        {% endfor %}
                    </div>
                    {% endif %}
                    
                    {% if report.key_insights %}
                    <div class="insights">
                        <h3>Key Insights</h3>
                        <ul>
                            {% for insight in report.key_insights %}
                            <li>{{ insight }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                    {% endif %}
                    
                    {% if report.recommendations %}
                    <div class="recommendations">
                        <h3>Recommendations</h3>
                        <ol>
                            {% for recommendation in report.recommendations %}
                            <li>{{ recommendation }}</li>
                            {% endfor %}
                        </ol>
                    </div>
                    {% endif %}
                    
                    <div class="footer">
                        <p>Generated by GrandModel Export and Integration System</p>
                        <p>Report ID: {{ report.report_metadata.get('report_id', 'N/A') }}</p>
                    </div>
                </div>
            </body>
            </html>
            """
        else:
            # Default template
            template_str = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{{ report.report_metadata.strategy_name }} - Report</title>
            </head>
            <body>
                <h1>{{ report.report_metadata.strategy_name }}</h1>
                <p>Generated: {{ timestamp }}</p>
                
                {% if report.key_metrics %}
                <h2>Key Metrics</h2>
                <ul>
                    {% for key, value in report.key_metrics.items() %}
                    <li>{{ key }}: {{ value }}</li>
                    {% endfor %}
                </ul>
                {% endif %}
                
                {% if charts %}
                <h2>Charts</h2>
                {% for chart_name, chart_html in charts.items() %}
                <h3>{{ chart_name }}</h3>
                {{ chart_html|safe }}
                {% endfor %}
                {% endif %}
            </body>
            </html>
            """
        
        return Template(template_str)
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return self._convert_to_json_serializable(obj.__dict__)
        else:
            return obj
    
    def _record_export(self, format_type: str, file_path: str, metadata: Dict[str, Any]):
        """Record export in history"""
        export_record = {
            'timestamp': datetime.now().isoformat(),
            'format': format_type,
            'file_path': file_path,
            'file_size': Path(file_path).stat().st_size if Path(file_path).exists() else 0,
            'metadata': metadata
        }
        self.export_history.append(export_record)
        
        # Keep only last 100 records
        if len(self.export_history) > 100:
            self.export_history = self.export_history[-100:]
    
    def get_export_history(self) -> List[Dict[str, Any]]:
        """Get export history"""
        return self.export_history.copy()
    
    def cleanup_old_exports(self, days_to_keep: int = 30):
        """Clean up old export files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for file_path in self.output_dir.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    logger.info(f"Cleaned up old export: {file_path}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old exports: {e}")


# Global instance
export_integration = ExportIntegration()