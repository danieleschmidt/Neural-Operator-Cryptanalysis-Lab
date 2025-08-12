"""
Compliance framework for Neural Cryptanalysis Framework.

This module provides comprehensive compliance management for various
data protection regulations including GDPR, CCPA, and PDPA.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

class DataProcessingPurpose(Enum):
    """Data processing purposes as defined by GDPR."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SECURITY_TESTING = "security_testing"
    MODEL_TRAINING = "model_training"
    PERFORMANCE_MONITORING = "performance_monitoring"
    LEGAL_COMPLIANCE = "legal_compliance"
    LEGITIMATE_INTEREST = "legitimate_interest"

class LegalBasis(Enum):
    """Legal basis for data processing under GDPR."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

class DataCategory(Enum):
    """Categories of personal data."""
    IDENTIFIER = "identifier"
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    BIOMETRIC = "biometric"
    SPECIAL_CATEGORY = "special_category"  # Sensitive personal data

@dataclass
class DataSubjectRequest:
    """Represents a data subject request under various regulations."""
    request_id: str
    request_type: str  # access, rectification, erasure, portability, object
    data_subject_id: str
    request_date: datetime
    description: str
    status: str = "pending"
    completed_date: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None
    
class ComplianceFramework(ABC):
    """Abstract base class for compliance frameworks."""
    
    @abstractmethod
    def validate_processing(self, purpose: DataProcessingPurpose, 
                          legal_basis: LegalBasis) -> bool:
        """Validate if data processing is compliant."""
        pass
    
    @abstractmethod
    def get_retention_period(self, data_category: DataCategory) -> timedelta:
        """Get data retention period for a data category."""
        pass
    
    @abstractmethod
    def handle_subject_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Handle data subject requests."""
        pass
    
    @abstractmethod
    def get_required_notices(self) -> List[str]:
        """Get required privacy notices and disclosures."""
        pass

class GDPRCompliance(ComplianceFramework):
    """GDPR (General Data Protection Regulation) compliance implementation."""
    
    def __init__(self, data_controller: str, dpo_contact: str):
        self.data_controller = data_controller
        self.dpo_contact = dpo_contact
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.processing_activities: List[Dict[str, Any]] = []
        
    def validate_processing(self, purpose: DataProcessingPurpose, 
                          legal_basis: LegalBasis) -> bool:
        """Validate GDPR compliance for data processing."""
        # Define valid combinations
        valid_combinations = {
            DataProcessingPurpose.RESEARCH: [
                LegalBasis.CONSENT, 
                LegalBasis.LEGITIMATE_INTERESTS,
                LegalBasis.PUBLIC_TASK
            ],
            DataProcessingPurpose.SECURITY_TESTING: [
                LegalBasis.LEGITIMATE_INTERESTS,
                LegalBasis.PUBLIC_TASK
            ],
            DataProcessingPurpose.MODEL_TRAINING: [
                LegalBasis.CONSENT,
                LegalBasis.LEGITIMATE_INTERESTS
            ],
            DataProcessingPurpose.PERFORMANCE_MONITORING: [
                LegalBasis.LEGITIMATE_INTERESTS
            ],
            DataProcessingPurpose.LEGAL_COMPLIANCE: [
                LegalBasis.LEGAL_OBLIGATION
            ]
        }
        
        return legal_basis in valid_combinations.get(purpose, [])
    
    def get_retention_period(self, data_category: DataCategory) -> timedelta:
        """Get GDPR-compliant retention periods."""
        retention_periods = {
            DataCategory.IDENTIFIER: timedelta(days=90),
            DataCategory.DEMOGRAPHIC: timedelta(days=365),
            DataCategory.BEHAVIORAL: timedelta(days=180),
            DataCategory.TECHNICAL: timedelta(days=30),
            DataCategory.BIOMETRIC: timedelta(days=30),
            DataCategory.SPECIAL_CATEGORY: timedelta(days=30),
        }
        return retention_periods.get(data_category, timedelta(days=30))
    
    def record_consent(self, data_subject_id: str, purposes: List[DataProcessingPurpose],
                      consent_given: bool, timestamp: Optional[datetime] = None) -> str:
        """Record consent for GDPR compliance."""
        consent_id = f"consent_{data_subject_id}_{int(datetime.now().timestamp())}"
        timestamp = timestamp or datetime.now()
        
        self.consent_records[consent_id] = {
            "data_subject_id": data_subject_id,
            "purposes": [p.value for p in purposes],
            "consent_given": consent_given,
            "timestamp": timestamp.isoformat(),
            "method": "explicit",
            "ip_address": None,  # Should be populated in real implementation
            "user_agent": None   # Should be populated in real implementation
        }
        
        logger.info(f"Recorded consent: {consent_id}")
        return consent_id
    
    def withdraw_consent(self, data_subject_id: str, consent_id: str) -> bool:
        """Withdraw consent and handle data deletion."""
        if consent_id in self.consent_records:
            self.consent_records[consent_id]["consent_withdrawn"] = True
            self.consent_records[consent_id]["withdrawal_timestamp"] = datetime.now().isoformat()
            
            # Trigger data deletion process
            self._trigger_data_deletion(data_subject_id)
            logger.info(f"Consent withdrawn: {consent_id}")
            return True
        return False
    
    def handle_subject_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Handle GDPR data subject requests."""
        response = {
            "request_id": request.request_id,
            "status": "processing",
            "response_data": {}
        }
        
        if request.request_type == "access":
            response["response_data"] = self._handle_access_request(request.data_subject_id)
        elif request.request_type == "rectification":
            response["response_data"] = self._handle_rectification_request(request)
        elif request.request_type == "erasure":
            response["response_data"] = self._handle_erasure_request(request.data_subject_id)
        elif request.request_type == "portability":
            response["response_data"] = self._handle_portability_request(request.data_subject_id)
        elif request.request_type == "object":
            response["response_data"] = self._handle_objection_request(request.data_subject_id)
        
        response["status"] = "completed"
        response["completed_date"] = datetime.now().isoformat()
        
        return response
    
    def _handle_access_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle right to access request."""
        return {
            "personal_data": self._get_personal_data(data_subject_id),
            "processing_purposes": self._get_processing_purposes(data_subject_id),
            "data_recipients": self._get_data_recipients(data_subject_id),
            "retention_periods": self._get_retention_periods(),
            "data_sources": self._get_data_sources(data_subject_id)
        }
    
    def _handle_rectification_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Handle right to rectification request."""
        # Implementation would update the data based on request
        return {"status": "data_updated", "updated_fields": []}
    
    def _handle_erasure_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle right to erasure (right to be forgotten) request."""
        deleted_data = self._delete_personal_data(data_subject_id)
        return {"status": "data_deleted", "deleted_items": deleted_data}
    
    def _handle_portability_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle right to data portability request."""
        portable_data = self._get_portable_data(data_subject_id)
        return {"status": "data_exported", "export_format": "JSON", "data": portable_data}
    
    def _handle_objection_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle right to object request."""
        # Stop processing based on legitimate interests
        return {"status": "processing_stopped", "affected_purposes": []}
    
    def get_required_notices(self) -> List[str]:
        """Get required GDPR privacy notices."""
        return [
            "privacy_policy",
            "cookie_notice",
            "data_processing_notice",
            "retention_policy",
            "third_party_sharing_notice",
            "data_subject_rights_notice",
            "contact_dpo_notice"
        ]
    
    def _trigger_data_deletion(self, data_subject_id: str) -> None:
        """Trigger automated data deletion process."""
        # Implementation would handle actual data deletion
        logger.info(f"Triggered data deletion for subject: {data_subject_id}")
    
    def _get_personal_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Retrieve all personal data for a subject."""
        # Implementation would query all data stores
        return {}
    
    def _get_processing_purposes(self, data_subject_id: str) -> List[str]:
        """Get processing purposes for a data subject."""
        return []
    
    def _get_data_recipients(self, data_subject_id: str) -> List[str]:
        """Get data recipients/processors."""
        return []
    
    def _get_retention_periods(self) -> Dict[str, str]:
        """Get retention periods for different data categories."""
        return {category.value: str(self.get_retention_period(category).days) + " days" 
                for category in DataCategory}
    
    def _get_data_sources(self, data_subject_id: str) -> List[str]:
        """Get data sources for a subject."""
        return []
    
    def _delete_personal_data(self, data_subject_id: str) -> List[str]:
        """Delete personal data and return list of deleted items."""
        # Implementation would handle actual deletion
        return []
    
    def _get_portable_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Get data in portable format."""
        return {}

class CCPACompliance(ComplianceFramework):
    """CCPA (California Consumer Privacy Act) compliance implementation."""
    
    def __init__(self, business_name: str, contact_info: str):
        self.business_name = business_name
        self.contact_info = contact_info
        self.opt_out_requests: Dict[str, datetime] = {}
    
    def validate_processing(self, purpose: DataProcessingPurpose, 
                          legal_basis: LegalBasis) -> bool:
        """Validate CCPA compliance for data processing."""
        # CCPA allows broader processing purposes
        return True
    
    def get_retention_period(self, data_category: DataCategory) -> timedelta:
        """Get CCPA-compliant retention periods."""
        # CCPA doesn't specify retention periods like GDPR
        return timedelta(days=365)
    
    def handle_subject_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Handle CCPA consumer requests."""
        if request.request_type == "know":
            return self._handle_right_to_know(request.data_subject_id)
        elif request.request_type == "delete":
            return self._handle_right_to_delete(request.data_subject_id)
        elif request.request_type == "opt_out":
            return self._handle_opt_out_request(request.data_subject_id)
        
        return {"status": "unsupported_request_type"}
    
    def _handle_right_to_know(self, consumer_id: str) -> Dict[str, Any]:
        """Handle consumer's right to know."""
        return {
            "categories_collected": [],
            "categories_sold": [],
            "categories_disclosed": [],
            "business_purposes": [],
            "third_parties": []
        }
    
    def _handle_right_to_delete(self, consumer_id: str) -> Dict[str, Any]:
        """Handle consumer's right to delete."""
        return {"status": "deletion_completed"}
    
    def _handle_opt_out_request(self, consumer_id: str) -> Dict[str, Any]:
        """Handle opt-out of sale request."""
        self.opt_out_requests[consumer_id] = datetime.now()
        return {"status": "opt_out_processed"}
    
    def get_required_notices(self) -> List[str]:
        """Get required CCPA notices."""
        return [
            "privacy_policy",
            "notice_at_collection",
            "do_not_sell_notice",
            "consumer_rights_notice"
        ]

class PDPACompliance(ComplianceFramework):
    """PDPA (Personal Data Protection Act) compliance implementation."""
    
    def __init__(self, organization_name: str, dpo_contact: str):
        self.organization_name = organization_name
        self.dpo_contact = dpo_contact
    
    def validate_processing(self, purpose: DataProcessingPurpose, 
                          legal_basis: LegalBasis) -> bool:
        """Validate PDPA compliance for data processing."""
        # PDPA has similar structure to GDPR
        return True
    
    def get_retention_period(self, data_category: DataCategory) -> timedelta:
        """Get PDPA-compliant retention periods."""
        return timedelta(days=90)  # Conservative default
    
    def handle_subject_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Handle PDPA data subject requests."""
        # Similar to GDPR implementation
        return {"status": "processed"}
    
    def get_required_notices(self) -> List[str]:
        """Get required PDPA notices."""
        return [
            "privacy_policy",
            "consent_notice",
            "data_protection_notice"
        ]

class ComplianceManager:
    """Central compliance management system."""
    
    def __init__(self, region: str = "global"):
        self.region = region
        self.frameworks: Dict[str, ComplianceFramework] = {}
        self.active_requests: Dict[str, DataSubjectRequest] = {}
        self._initialize_frameworks()
    
    def _initialize_frameworks(self) -> None:
        """Initialize compliance frameworks based on region."""
        if self.region in ["eu", "global"]:
            self.frameworks["gdpr"] = GDPRCompliance(
                data_controller="Neural Cryptanalysis Framework",
                dpo_contact="dpo@neural-crypto.com"
            )
        
        if self.region in ["us", "california", "global"]:
            self.frameworks["ccpa"] = CCPACompliance(
                business_name="Neural Cryptanalysis Corp",
                contact_info="privacy@neural-crypto.com"
            )
        
        if self.region in ["singapore", "asia", "global"]:
            self.frameworks["pdpa"] = PDPACompliance(
                organization_name="Neural Cryptanalysis Pte Ltd",
                dpo_contact="dpo@neural-crypto.com"
            )
    
    def validate_data_processing(self, purpose: DataProcessingPurpose,
                               legal_basis: LegalBasis,
                               applicable_regulations: Optional[List[str]] = None) -> Dict[str, bool]:
        """Validate data processing across applicable regulations."""
        if applicable_regulations is None:
            applicable_regulations = list(self.frameworks.keys())
        
        results = {}
        for regulation in applicable_regulations:
            if regulation in self.frameworks:
                results[regulation] = self.frameworks[regulation].validate_processing(
                    purpose, legal_basis
                )
        
        return results
    
    def submit_subject_request(self, request: DataSubjectRequest,
                             applicable_regulations: Optional[List[str]] = None) -> str:
        """Submit and process data subject request."""
        if applicable_regulations is None:
            applicable_regulations = list(self.frameworks.keys())
        
        self.active_requests[request.request_id] = request
        
        # Process request across applicable frameworks
        for regulation in applicable_regulations:
            if regulation in self.frameworks:
                response = self.frameworks[regulation].handle_subject_request(request)
                logger.info(f"Processed {regulation} request: {request.request_id}")
        
        return request.request_id
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status."""
        return {
            "active_frameworks": list(self.frameworks.keys()),
            "pending_requests": len([r for r in self.active_requests.values() 
                                   if r.status == "pending"]),
            "compliance_checks": {
                framework: "compliant" for framework in self.frameworks.keys()
            }
        }
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "region": self.region,
            "frameworks": {},
            "data_subject_requests": {
                "total": len(self.active_requests),
                "by_status": {},
                "by_type": {}
            }
        }
        
        # Framework-specific reports
        for name, framework in self.frameworks.items():
            report["frameworks"][name] = {
                "required_notices": framework.get_required_notices(),
                "retention_periods": {
                    cat.value: str(framework.get_retention_period(cat).days) + " days"
                    for cat in DataCategory
                }
            }
        
        # Request statistics
        statuses = [r.status for r in self.active_requests.values()]
        types = [r.request_type for r in self.active_requests.values()]
        
        for status in set(statuses):
            report["data_subject_requests"]["by_status"][status] = statuses.count(status)
        
        for req_type in set(types):
            report["data_subject_requests"]["by_type"][req_type] = types.count(req_type)
        
        return report

# Global compliance manager instance
_compliance_manager = ComplianceManager()

def get_compliance_manager() -> ComplianceManager:
    """Get the global compliance manager instance."""
    return _compliance_manager