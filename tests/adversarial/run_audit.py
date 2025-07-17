"""
Zero Defect Adversarial Audit - Execution Script
================================================

This script executes the comprehensive adversarial audit of the Tactical MARL System.
It coordinates all attack phases and generates a final security assessment report.

Usage:
    python tests/adversarial/run_audit.py [--target-host HOST] [--target-port PORT] [--output REPORT_FILE]

CRITICAL SECURITY AUDIT PHASES:
1. Race Condition & Concurrency Exploits
2. Input Validation & Boundary Testing  
3. Dependency Chain Vulnerabilities
4. Memory Exhaustion Attacks
5. Event Bus Flood Attacks

Author: Zero Defect Security Audit
Version: 1.0.0
Classification: CRITICAL SECURITY AUDIT
"""

import argparse
import asyncio
import json
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.security.attack_detection import TacticalMARLAttackDetector, VulnerabilitySeverity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('adversarial_audit.log')
    ]
)

logger = logging.getLogger(__name__)


class ZeroDefectAdversarialAudit:
    """
    Comprehensive adversarial audit orchestrator for the Tactical MARL System.
    
    This class coordinates all attack phases and generates a unified security
    assessment report with vulnerability findings and remediation guidance.
    """

    def __init__(self, target_host: str = "localhost", target_port: int = 8001):
        """
        Initialize the Zero Defect Adversarial Audit system.
        
        Args:
            target_host: Target system hostname
            target_port: Target system port
        """
        self.target_host = target_host
        self.target_port = target_port
        self.attack_detector = TacticalMARLAttackDetector(target_host, target_port)
        
        # Audit results storage
        self.audit_results = {}
        self.all_vulnerabilities = []
        self.audit_metadata = {
            "audit_type": "Zero Defect Adversarial Audit - Phase 1",
            "target_system": "Tactical MARL System",
            "target_endpoint": f"http://{target_host}:{target_port}",
            "auditor": "ZeroDefectAdversarialAudit v1.0.0",
            "start_time": None,
            "end_time": None,
            "total_duration": None
        }
        
        logger.info(f"🚨 Zero Defect Adversarial Audit initialized for {target_host}:{target_port}")

    async def execute_full_audit(self) -> Dict[str, Any]:
        """
        Execute the complete adversarial audit across all attack vectors.
        
        Returns:
            Comprehensive audit report with all findings
        """
        logger.info("🚨🚨🚨 STARTING ZERO DEFECT ADVERSARIAL AUDIT 🚨🚨🚨")
        logger.info("=" * 80)
        logger.info("MISSION: Assume the Tactical MARL System is brittle and will break under stress.")
        logger.info("GOAL: Find every possible vulnerability with ZERO false positives.")
        logger.info("STANDARD: Perfection - failure to find a flaw is a failure of the audit.")
        logger.info("=" * 80)
        
        self.audit_metadata["start_time"] = time.time()
        
        try:
            # Phase 1.1: Race Condition & Concurrency Exploits
            await self._execute_phase_1_1()
            
            # Phase 1.2: Input Validation & Boundary Testing
            await self._execute_phase_1_2()
            
            # Phase 1.3: Dependency Chain Vulnerabilities
            await self._execute_phase_1_3()
            
            # Phase 1.4: Memory Exhaustion Attacks
            await self._execute_phase_1_4()
            
            # Phase 1.5: Event Bus Flood Attacks
            await self._execute_phase_1_5()
            
        except Exception as e:
            logger.critical(f"CRITICAL AUDIT FAILURE: {e}")
            raise
        
        finally:
            self.audit_metadata["end_time"] = time.time()
            self.audit_metadata["total_duration"] = self.audit_metadata["end_time"] - self.audit_metadata["start_time"]
        
        # Generate final comprehensive report
        final_report = await self._generate_final_report()
        
        logger.info("🚨🚨🚨 ZERO DEFECT ADVERSARIAL AUDIT COMPLETED 🚨🚨🚨")
        logger.info(f"⏱️ Total audit duration: {self.audit_metadata['total_duration']:.2f} seconds")
        logger.info(f"🔍 Total vulnerabilities found: {len(self.all_vulnerabilities)}")
        
        # Log vulnerability summary
        critical_vulns = len([v for v in self.all_vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL])
        high_vulns = len([v for v in self.all_vulnerabilities if v.severity == VulnerabilitySeverity.HIGH])
        medium_vulns = len([v for v in self.all_vulnerabilities if v.severity == VulnerabilitySeverity.MEDIUM])
        
        logger.critical(f"🚨 CRITICAL vulnerabilities: {critical_vulns}")
        logger.warning(f"⚠️ HIGH vulnerabilities: {high_vulns}")
        logger.info(f"📋 MEDIUM vulnerabilities: {medium_vulns}")
        
        if critical_vulns > 0:
            logger.critical("🚨 IMMEDIATE ACTION REQUIRED: CRITICAL vulnerabilities found!")
            logger.critical("🚨 System is NOT PRODUCTION READY until CRITICAL issues are resolved!")
        
        return final_report

    async def _execute_phase_1_1(self):
        """Execute Phase 1.1: Race Condition & Concurrency Exploits"""
        logger.info("🏃 PHASE 1.1: Race Condition & Concurrency Exploits")
        logger.info("-" * 60)
        
        try:
            # Execute all race condition attack patterns
            race_results = await self.attack_detector._execute_race_condition_attacks()
            
            self.audit_results["phase_1_1_race_conditions"] = {
                "phase_name": "Race Condition & Concurrency Exploits",
                "attack_results": race_results,
                "vulnerabilities_found": [],
                "phase_success": len(race_results) > 0
            }
            
            # Collect vulnerabilities from race condition attacks
            for attack_result in race_results:
                self.audit_results["phase_1_1_race_conditions"]["vulnerabilities_found"].extend(
                    attack_result.vulnerabilities_found
                )
                self.all_vulnerabilities.extend(attack_result.vulnerabilities_found)
                
                logger.info(f"✅ {attack_result.attack_name}: {len(attack_result.vulnerabilities_found)} vulnerabilities")
        
        except Exception as e:
            logger.error(f"❌ Phase 1.1 failed: {e}")
            self.audit_results["phase_1_1_race_conditions"] = {
                "phase_name": "Race Condition & Concurrency Exploits",
                "error": str(e),
                "phase_success": False
            }

    async def _execute_phase_1_2(self):
        """Execute Phase 1.2: Input Validation & Boundary Testing"""
        logger.info("🎯 PHASE 1.2: Input Validation & Boundary Testing")
        logger.info("-" * 60)
        
        try:
            # Execute all input validation attack patterns
            input_results = await self.attack_detector._execute_input_validation_attacks()
            
            self.audit_results["phase_1_2_input_validation"] = {
                "phase_name": "Input Validation & Boundary Testing",
                "attack_results": input_results,
                "vulnerabilities_found": [],
                "phase_success": len(input_results) > 0
            }
            
            # Collect vulnerabilities from input validation attacks
            for attack_result in input_results:
                self.audit_results["phase_1_2_input_validation"]["vulnerabilities_found"].extend(
                    attack_result.vulnerabilities_found
                )
                self.all_vulnerabilities.extend(attack_result.vulnerabilities_found)
                
                logger.info(f"✅ {attack_result.attack_name}: {len(attack_result.vulnerabilities_found)} vulnerabilities")
        
        except Exception as e:
            logger.error(f"❌ Phase 1.2 failed: {e}")
            self.audit_results["phase_1_2_input_validation"] = {
                "phase_name": "Input Validation & Boundary Testing",
                "error": str(e),
                "phase_success": False
            }

    async def _execute_phase_1_3(self):
        """Execute Phase 1.3: Dependency Chain Vulnerabilities"""
        logger.info("🔗 PHASE 1.3: Dependency Chain Vulnerabilities")
        logger.info("-" * 60)
        
        try:
            # Execute all dependency chain attack patterns
            dependency_results = await self.attack_detector._execute_dependency_attacks()
            
            self.audit_results["phase_1_3_dependency_chain"] = {
                "phase_name": "Dependency Chain Vulnerabilities",
                "attack_results": dependency_results,
                "vulnerabilities_found": [],
                "phase_success": len(dependency_results) > 0
            }
            
            # Collect vulnerabilities from dependency attacks
            for attack_result in dependency_results:
                self.audit_results["phase_1_3_dependency_chain"]["vulnerabilities_found"].extend(
                    attack_result.vulnerabilities_found
                )
                self.all_vulnerabilities.extend(attack_result.vulnerabilities_found)
                
                logger.info(f"✅ {attack_result.attack_name}: {len(attack_result.vulnerabilities_found)} vulnerabilities")
        
        except Exception as e:
            logger.error(f"❌ Phase 1.3 failed: {e}")
            self.audit_results["phase_1_3_dependency_chain"] = {
                "phase_name": "Dependency Chain Vulnerabilities",
                "error": str(e),
                "phase_success": False
            }

    async def _execute_phase_1_4(self):
        """Execute Phase 1.4: Memory Exhaustion Attacks"""
        logger.info("💥 PHASE 1.4: Memory Exhaustion Attacks")
        logger.info("-" * 60)
        
        try:
            # Execute all memory exhaustion attack patterns
            memory_results = await self.attack_detector._execute_memory_attacks()
            
            self.audit_results["phase_1_4_memory_exhaustion"] = {
                "phase_name": "Memory Exhaustion Attacks",
                "attack_results": memory_results,
                "vulnerabilities_found": [],
                "phase_success": len(memory_results) > 0
            }
            
            # Collect vulnerabilities from memory attacks
            for attack_result in memory_results:
                self.audit_results["phase_1_4_memory_exhaustion"]["vulnerabilities_found"].extend(
                    attack_result.vulnerabilities_found
                )
                self.all_vulnerabilities.extend(attack_result.vulnerabilities_found)
                
                logger.info(f"✅ {attack_result.attack_name}: {len(attack_result.vulnerabilities_found)} vulnerabilities")
        
        except Exception as e:
            logger.error(f"❌ Phase 1.4 failed: {e}")
            self.audit_results["phase_1_4_memory_exhaustion"] = {
                "phase_name": "Memory Exhaustion Attacks",
                "error": str(e),
                "phase_success": False
            }

    async def _execute_phase_1_5(self):
        """Execute Phase 1.5: Event Bus Flood Attacks"""
        logger.info("🌊 PHASE 1.5: Event Bus Flood Attacks")
        logger.info("-" * 60)
        
        try:
            # Execute all event bus flood attack patterns
            event_results = await self.attack_detector._execute_event_bus_attacks()
            
            self.audit_results["phase_1_5_event_bus_floods"] = {
                "phase_name": "Event Bus Flood Attacks",
                "attack_results": event_results,
                "vulnerabilities_found": [],
                "phase_success": len(event_results) > 0
            }
            
            # Collect vulnerabilities from event bus attacks
            for attack_result in event_results:
                self.audit_results["phase_1_5_event_bus_floods"]["vulnerabilities_found"].extend(
                    attack_result.vulnerabilities_found
                )
                self.all_vulnerabilities.extend(attack_result.vulnerabilities_found)
                
                logger.info(f"✅ {attack_result.attack_name}: {len(attack_result.vulnerabilities_found)} vulnerabilities")
        
        except Exception as e:
            logger.error(f"❌ Phase 1.5 failed: {e}")
            self.audit_results["phase_1_5_event_bus_floods"] = {
                "phase_name": "Event Bus Flood Attacks",
                "error": str(e),
                "phase_success": False
            }

    async def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final audit report"""
        logger.info("📊 Generating Final Audit Report")
        
        # Calculate vulnerability statistics
        total_vulnerabilities = len(self.all_vulnerabilities)
        critical_vulns = [v for v in self.all_vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL]
        high_vulns = [v for v in self.all_vulnerabilities if v.severity == VulnerabilitySeverity.HIGH]
        medium_vulns = [v for v in self.all_vulnerabilities if v.severity == VulnerabilitySeverity.MEDIUM]
        
        # Calculate phase success rates
        successful_phases = len([phase for phase in self.audit_results.values() if phase.get("phase_success", False)])
        total_phases = len(self.audit_results)
        
        # Generate executive summary
        executive_summary = {
            "audit_outcome": self._determine_audit_outcome(len(critical_vulns), len(high_vulns)),
            "total_vulnerabilities_found": total_vulnerabilities,
            "critical_vulnerabilities": len(critical_vulns),
            "high_vulnerabilities": len(high_vulns),
            "medium_vulnerabilities": len(medium_vulns),
            "phases_executed": total_phases,
            "phases_successful": successful_phases,
            "phase_success_rate": successful_phases / total_phases if total_phases > 0 else 0.0,
            "production_readiness": self._assess_production_readiness(len(critical_vulns), len(high_vulns)),
            "immediate_action_required": len(critical_vulns) > 0 or len(high_vulns) > 3
        }
        
        # Generate vulnerability analysis
        vulnerability_analysis = self._analyze_vulnerabilities()
        
        # Generate security recommendations
        security_recommendations = self._generate_security_recommendations()
        
        # Generate remediation roadmap
        remediation_roadmap = self._generate_remediation_roadmap()
        
        # Compile final report
        final_report = {
            "audit_metadata": self.audit_metadata,
            "executive_summary": executive_summary,
            "vulnerability_analysis": vulnerability_analysis,
            "phase_results": self.audit_results,
            "security_recommendations": security_recommendations,
            "remediation_roadmap": remediation_roadmap,
            "detailed_vulnerabilities": [
                {
                    "vulnerability_id": v.vulnerability_id,
                    "severity": v.severity.value,
                    "attack_vector": v.attack_vector.value,
                    "description": v.description,
                    "reproduction_steps": v.reproduction_steps,
                    "impact_assessment": v.impact_assessment,
                    "remediation": v.remediation,
                    "cve_references": v.cve_references,
                    "affected_components": v.affected_components,
                    "discovery_timestamp": v.discovery_timestamp
                }
                for v in self.all_vulnerabilities
            ],
            "compliance_assessment": self._generate_compliance_assessment(len(critical_vulns), len(high_vulns))
        }
        
        return final_report

    def _determine_audit_outcome(self, critical_count: int, high_count: int) -> str:
        """Determine overall audit outcome"""
        if critical_count > 0:
            return "AUDIT FAILED - CRITICAL VULNERABILITIES FOUND"
        elif high_count > 3:
            return "AUDIT WARNING - MULTIPLE HIGH SEVERITY VULNERABILITIES"
        elif high_count > 0:
            return "AUDIT PASSED WITH CONCERNS - HIGH SEVERITY VULNERABILITIES"
        else:
            return "AUDIT PASSED - NO CRITICAL OR HIGH SEVERITY VULNERABILITIES"

    def _assess_production_readiness(self, critical_count: int, high_count: int) -> Dict[str, Any]:
        """Assess production readiness"""
        if critical_count > 0:
            status = "NOT READY"
            reason = f"{critical_count} CRITICAL vulnerabilities must be fixed"
            confidence = 0.0
        elif high_count > 2:
            status = "NOT RECOMMENDED"
            reason = f"{high_count} HIGH severity vulnerabilities should be addressed"
            confidence = 0.3
        elif high_count > 0:
            status = "CONDITIONAL"
            reason = f"{high_count} HIGH severity vulnerabilities present but manageable"
            confidence = 0.7
        else:
            status = "READY"
            reason = "No critical or high severity vulnerabilities found"
            confidence = 0.9
        
        return {
            "status": status,
            "reason": reason,
            "confidence_score": confidence,
            "recommendation": self._get_production_recommendation(status)
        }

    def _get_production_recommendation(self, status: str) -> str:
        """Get production deployment recommendation"""
        recommendations = {
            "NOT READY": "DO NOT DEPLOY - Fix all CRITICAL vulnerabilities first",
            "NOT RECOMMENDED": "DELAY DEPLOYMENT - Address HIGH severity issues",
            "CONDITIONAL": "PROCEED WITH CAUTION - Monitor and patch HIGH severity issues",
            "READY": "APPROVED FOR DEPLOYMENT - Continue monitoring"
        }
        return recommendations.get(status, "MANUAL REVIEW REQUIRED")

    def _analyze_vulnerabilities(self) -> Dict[str, Any]:
        """Analyze vulnerability patterns and trends"""
        if not self.all_vulnerabilities:
            return {"total": 0, "analysis": "No vulnerabilities found"}
        
        # Group by attack vector
        by_attack_vector = {}
        for vuln in self.all_vulnerabilities:
            vector = vuln.attack_vector.value
            if vector not in by_attack_vector:
                by_attack_vector[vector] = []
            by_attack_vector[vector].append(vuln)
        
        # Group by severity
        by_severity = {}
        for vuln in self.all_vulnerabilities:
            severity = vuln.severity.value
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(vuln)
        
        # Group by affected components
        affected_components = set()
        for vuln in self.all_vulnerabilities:
            affected_components.update(vuln.affected_components)
        
        return {
            "total_vulnerabilities": len(self.all_vulnerabilities),
            "by_attack_vector": {vector: len(vulns) for vector, vulns in by_attack_vector.items()},
            "by_severity": {severity: len(vulns) for severity, vulns in by_severity.items()},
            "affected_components": list(affected_components),
            "most_vulnerable_component": self._find_most_vulnerable_component(),
            "vulnerability_density": len(self.all_vulnerabilities) / len(affected_components) if affected_components else 0
        }

    def _find_most_vulnerable_component(self) -> str:
        """Find the component with the most vulnerabilities"""
        component_counts = {}
        for vuln in self.all_vulnerabilities:
            for component in vuln.affected_components:
                component_counts[component] = component_counts.get(component, 0) + 1
        
        if not component_counts:
            return "No components identified"
        
        return max(component_counts, key=component_counts.get)

    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []
        
        critical_vulns = [v for v in self.all_vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL]
        high_vulns = [v for v in self.all_vulnerabilities if v.severity == VulnerabilitySeverity.HIGH]
        
        if critical_vulns:
            recommendations.append("🚨 IMMEDIATE: Fix all CRITICAL vulnerabilities before any deployment")
            recommendations.append("🚨 IMMEDIATE: Conduct security code review of all affected components")
            recommendations.append("🚨 IMMEDIATE: Implement emergency monitoring for exploited vulnerabilities")
        
        if high_vulns:
            recommendations.append("⚠️ URGENT: Address all HIGH severity vulnerabilities within 48 hours")
            recommendations.append("⚠️ URGENT: Implement additional security controls for affected components")
        
        # General recommendations
        recommendations.extend([
            "Implement comprehensive input validation at all API endpoints",
            "Add rate limiting and request throttling mechanisms", 
            "Implement proper error handling and logging without information disclosure",
            "Add security monitoring and alerting for all attack vectors",
            "Conduct regular security audits and penetration testing",
            "Implement security testing in CI/CD pipeline",
            "Provide security training for development team",
            "Establish incident response procedures for security events"
        ])
        
        return recommendations

    def _generate_remediation_roadmap(self) -> Dict[str, Any]:
        """Generate prioritized remediation roadmap"""
        critical_vulns = [v for v in self.all_vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL]
        high_vulns = [v for v in self.all_vulnerabilities if v.severity == VulnerabilitySeverity.HIGH]
        medium_vulns = [v for v in self.all_vulnerabilities if v.severity == VulnerabilitySeverity.MEDIUM]
        
        roadmap = {
            "immediate_phase": {
                "timeframe": "0-24 hours",
                "priority": "CRITICAL",
                "vulnerabilities": [v.vulnerability_id for v in critical_vulns],
                "actions": [
                    "Fix all CRITICAL vulnerabilities",
                    "Deploy emergency patches",
                    "Implement monitoring for exploitation attempts"
                ]
            },
            "urgent_phase": {
                "timeframe": "1-7 days", 
                "priority": "HIGH",
                "vulnerabilities": [v.vulnerability_id for v in high_vulns],
                "actions": [
                    "Address all HIGH severity vulnerabilities",
                    "Implement additional security controls",
                    "Conduct thorough testing of fixes"
                ]
            },
            "planned_phase": {
                "timeframe": "1-4 weeks",
                "priority": "MEDIUM",
                "vulnerabilities": [v.vulnerability_id for v in medium_vulns],
                "actions": [
                    "Address MEDIUM severity vulnerabilities",
                    "Implement security hardening measures",
                    "Conduct comprehensive security review"
                ]
            },
            "ongoing_phase": {
                "timeframe": "Continuous",
                "priority": "MAINTENANCE",
                "vulnerabilities": [],
                "actions": [
                    "Implement regular security audits",
                    "Maintain security monitoring",
                    "Update security controls as needed"
                ]
            }
        }
        
        return roadmap

    def _generate_compliance_assessment(self, critical_count: int, high_count: int) -> Dict[str, Any]:
        """Generate compliance assessment"""
        return {
            "security_standards_compliance": {
                "owasp_top_10": "NON-COMPLIANT" if critical_count > 0 else "PARTIALLY COMPLIANT",
                "iso_27001": "NON-COMPLIANT" if critical_count > 0 or high_count > 2 else "CONDITIONALLY COMPLIANT",
                "nist_cybersecurity_framework": "IMPROVING" if high_count == 0 else "PARTIALLY IMPLEMENTED"
            },
            "risk_assessment": {
                "overall_risk_level": "CRITICAL" if critical_count > 0 else "HIGH" if high_count > 2 else "MEDIUM",
                "exploitation_likelihood": "HIGH" if critical_count > 0 else "MEDIUM" if high_count > 0 else "LOW",
                "business_impact": "HIGH" if critical_count > 0 else "MEDIUM"
            },
            "regulatory_compliance": {
                "financial_services": "NON-COMPLIANT" if critical_count > 0 else "UNDER REVIEW",
                "data_protection": "NON-COMPLIANT" if critical_count > 0 else "CONDITIONALLY COMPLIANT"
            }
        }

    def save_report(self, report: Dict[str, Any], output_file: str):
        """Save audit report to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"📄 Audit report saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Zero Defect Adversarial Audit for Tactical MARL System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_audit.py                                    # Default localhost:8001
  python run_audit.py --target-host 192.168.1.100       # Custom host
  python run_audit.py --target-port 9000                # Custom port
  python run_audit.py --output security_audit.json      # Custom output file
        """
    )
    
    parser.add_argument(
        '--target-host',
        default='localhost',
        help='Target system hostname (default: localhost)'
    )
    
    parser.add_argument(
        '--target-port',
        type=int,
        default=8001,
        help='Target system port (default: 8001)'
    )
    
    parser.add_argument(
        '--output',
        default='zero_defect_audit_report.json',
        help='Output report file (default: zero_defect_audit_report.json)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and execute audit
    audit = ZeroDefectAdversarialAudit(args.target_host, args.target_port)
    
    try:
        # Execute comprehensive audit
        final_report = await audit.execute_full_audit()
        
        # Save report to file
        audit.save_report(final_report, args.output)
        
        # Print executive summary
        print("\n" + "=" * 80)
        print("ZERO DEFECT ADVERSARIAL AUDIT - EXECUTIVE SUMMARY")
        print("=" * 80)
        print(f"Audit Outcome: {final_report['executive_summary']['audit_outcome']}")
        print(f"Total Vulnerabilities: {final_report['executive_summary']['total_vulnerabilities_found']}")
        print(f"Critical: {final_report['executive_summary']['critical_vulnerabilities']}")
        print(f"High: {final_report['executive_summary']['high_vulnerabilities']}")
        print(f"Medium: {final_report['executive_summary']['medium_vulnerabilities']}")
        print(f"Production Readiness: {final_report['executive_summary']['production_readiness']['status']}")
        print(f"Recommendation: {final_report['executive_summary']['production_readiness']['recommendation']}")
        print("=" * 80)
        
        # Exit with appropriate code
        if final_report['executive_summary']['critical_vulnerabilities'] > 0:
            sys.exit(1)  # Critical vulnerabilities found
        elif final_report['executive_summary']['high_vulnerabilities'] > 3:
            sys.exit(2)  # Too many high severity vulnerabilities
        else:
            sys.exit(0)  # Audit passed
            
    except Exception as e:
        logger.critical(f"AUDIT EXECUTION FAILED: {e}")
        print(f"\n🚨 AUDIT EXECUTION FAILED: {e}")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())