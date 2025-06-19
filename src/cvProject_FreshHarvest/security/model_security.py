"""
Model Security for FreshHarvest
==============================

This module provides comprehensive security measures for the FreshHarvest
fruit freshness classification system, protecting the 96.50% accuracy model.

Author: FreshHarvest Team
Version: 1.0.0
"""

import hashlib
import hmac
import secrets
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import sys
import json
import time
from datetime import datetime, timedelta
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from cvProject_FreshHarvest.utils.common import read_yaml

logger = logging.getLogger(__name__)

class ModelSecurityManager:
    """
    Comprehensive security manager for FreshHarvest model protection.

    Provides encryption, authentication, access control, and audit logging
    for the 96.50% accuracy model and associated data.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize model security manager.

        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.security_config = self.config.get('security', {})

        # Security settings
        self.encryption_enabled = self.security_config.get('encryption_enabled', True)
        self.access_control_enabled = self.security_config.get('access_control_enabled', True)
        self.audit_logging_enabled = self.security_config.get('audit_logging_enabled', True)
        self.rate_limiting_enabled = self.security_config.get('rate_limiting_enabled', True)

        # Rate limiting settings
        self.max_requests_per_minute = self.security_config.get('max_requests_per_minute', 100)
        self.max_requests_per_hour = self.security_config.get('max_requests_per_hour', 1000)

        # Access tracking
        self.access_log = []
        self.failed_attempts = {}
        self.blocked_ips = set()

        # Initialize encryption
        self.encryption_key = None
        if self.encryption_enabled:
            self._initialize_encryption()

        # Audit log file
        self.audit_log_file = self.security_config.get('audit_log_file', 'logs/security_audit.log')

        logger.info("Model security manager initialized for 96.50% accuracy model")

    def _initialize_encryption(self):
        """Initialize encryption system."""
        try:
            # Generate or load encryption key
            key_file = self.security_config.get('encryption_key_file', 'security/model.key')
            key_path = Path(key_file)

            if key_path.exists():
                # Load existing key
                with open(key_path, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = Fernet.generate_key()

                # Save key securely
                key_path.parent.mkdir(parents=True, exist_ok=True)
                with open(key_path, 'wb') as f:
                    f.write(self.encryption_key)

                # Set restrictive permissions
                key_path.chmod(0o600)

            self.cipher_suite = Fernet(self.encryption_key)
            logger.info("✅ Encryption system initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize encryption: {e}")
            self.encryption_enabled = False

    def encrypt_data(self, data: Union[str, bytes, dict]) -> str:
        """
        Encrypt sensitive data.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data as base64 string
        """
        try:
            if not self.encryption_enabled:
                logger.warning("Encryption disabled - returning plain data")
                return str(data)

            # Convert data to bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data

            # Encrypt data
            encrypted_data = self.cipher_suite.encrypt(data_bytes)

            # Return as base64 string
            return base64.b64encode(encrypted_data).decode('utf-8')

        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise SecurityError(f"Failed to encrypt data: {e}")

    def decrypt_data(self, encrypted_data: str) -> Union[str, dict]:
        """
        Decrypt encrypted data.

        Args:
            encrypted_data: Base64 encoded encrypted data

        Returns:
            Decrypted data
        """
        try:
            if not self.encryption_enabled:
                logger.warning("Encryption disabled - returning data as-is")
                return encrypted_data

            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))

            # Decrypt data
            decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
            decrypted_str = decrypted_bytes.decode('utf-8')

            # Try to parse as JSON
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise SecurityError(f"Failed to decrypt data: {e}")

    def generate_api_key(self, user_id: str, permissions: List[str] = None) -> Dict[str, Any]:
        """
        Generate secure API key for user access.

        Args:
            user_id: User identifier
            permissions: List of permissions

        Returns:
            API key information
        """
        try:
            # Generate secure random key
            api_key = secrets.token_urlsafe(32)

            # Create key metadata
            key_data = {
                'user_id': user_id,
                'api_key': api_key,
                'permissions': permissions or ['predict'],
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=365)).isoformat(),
                'is_active': True,
                'usage_count': 0,
                'last_used': None
            }

            # Generate key hash for storage
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Log key generation
            self._audit_log('api_key_generated', {
                'user_id': user_id,
                'key_hash': key_hash,
                'permissions': permissions
            })

            return {
                'api_key': api_key,
                'key_hash': key_hash,
                'metadata': key_data
            }

        except Exception as e:
            logger.error(f"API key generation failed: {e}")
            raise SecurityError(f"Failed to generate API key: {e}")

    def validate_api_key(self, api_key: str, required_permission: str = 'predict') -> Dict[str, Any]:
        """
        Validate API key and check permissions.

        Args:
            api_key: API key to validate
            required_permission: Required permission

        Returns:
            Validation result
        """
        try:
            # Generate key hash
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # In production, this would check against a secure database
            # For now, we'll simulate validation
            validation_result = {
                'valid': True,
                'user_id': 'demo_user',
                'permissions': ['predict', 'explain', 'monitor'],
                'rate_limit_remaining': self.max_requests_per_minute,
                'expires_at': (datetime.now() + timedelta(days=365)).isoformat()
            }

            # Check if key has required permission
            if required_permission not in validation_result['permissions']:
                validation_result['valid'] = False
                validation_result['error'] = f"Insufficient permissions: {required_permission} required"

            # Log access attempt
            self._audit_log('api_key_validation', {
                'key_hash': key_hash,
                'valid': validation_result['valid'],
                'required_permission': required_permission
            })

            return validation_result

        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return {
                'valid': False,
                'error': f"Validation failed: {e}"
            }

    def check_rate_limit(self, client_id: str, endpoint: str = 'predict') -> Dict[str, Any]:
        """
        Check rate limiting for client requests.

        Args:
            client_id: Client identifier
            endpoint: API endpoint

        Returns:
            Rate limit status
        """
        try:
            if not self.rate_limiting_enabled:
                return {
                    'allowed': True,
                    'remaining': self.max_requests_per_minute,
                    'reset_time': None
                }

            current_time = time.time()
            current_minute = int(current_time // 60)
            current_hour = int(current_time // 3600)

            # Initialize client tracking if not exists
            if client_id not in self.access_log:
                self.access_log[client_id] = {
                    'minute_requests': {},
                    'hour_requests': {},
                    'total_requests': 0
                }

            client_log = self.access_log[client_id]

            # Clean old entries
            client_log['minute_requests'] = {
                minute: count for minute, count in client_log['minute_requests'].items()
                if minute >= current_minute - 1
            }
            client_log['hour_requests'] = {
                hour: count for hour, count in client_log['hour_requests'].items()
                if hour >= current_hour - 1
            }

            # Get current counts
            minute_count = client_log['minute_requests'].get(current_minute, 0)
            hour_count = client_log['hour_requests'].get(current_hour, 0)

            # Check limits
            if minute_count >= self.max_requests_per_minute:
                return {
                    'allowed': False,
                    'remaining': 0,
                    'reset_time': (current_minute + 1) * 60,
                    'error': 'Rate limit exceeded: too many requests per minute'
                }

            if hour_count >= self.max_requests_per_hour:
                return {
                    'allowed': False,
                    'remaining': 0,
                    'reset_time': (current_hour + 1) * 3600,
                    'error': 'Rate limit exceeded: too many requests per hour'
                }

            # Update counts
            client_log['minute_requests'][current_minute] = minute_count + 1
            client_log['hour_requests'][current_hour] = hour_count + 1
            client_log['total_requests'] += 1

            return {
                'allowed': True,
                'remaining': self.max_requests_per_minute - minute_count - 1,
                'reset_time': (current_minute + 1) * 60,
                'total_requests': client_log['total_requests']
            }

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return {
                'allowed': False,
                'error': f"Rate limit check failed: {e}"
            }

    def detect_anomalous_requests(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect potentially malicious or anomalous requests.

        Args:
            request_data: Request data to analyze

        Returns:
            Anomaly detection results
        """
        try:
            anomalies = []
            risk_score = 0.0

            # Check request size
            if 'image_size' in request_data:
                image_size = request_data['image_size']
                if image_size > 10 * 1024 * 1024:  # 10MB
                    anomalies.append("Unusually large image size")
                    risk_score += 0.3

            # Check request frequency
            client_id = request_data.get('client_id', 'unknown')
            if client_id in self.access_log:
                recent_requests = sum(self.access_log[client_id]['minute_requests'].values())
                if recent_requests > self.max_requests_per_minute * 0.8:
                    anomalies.append("High request frequency")
                    risk_score += 0.2

            # Check for suspicious patterns
            user_agent = request_data.get('user_agent', '')
            if 'bot' in user_agent.lower() or 'crawler' in user_agent.lower():
                anomalies.append("Bot-like user agent")
                risk_score += 0.1

            # Check IP reputation (simplified)
            client_ip = request_data.get('client_ip', '')
            if client_ip in self.blocked_ips:
                anomalies.append("Request from blocked IP")
                risk_score += 1.0

            # Determine threat level
            if risk_score >= 0.8:
                threat_level = "HIGH"
            elif risk_score >= 0.5:
                threat_level = "MEDIUM"
            elif risk_score >= 0.2:
                threat_level = "LOW"
            else:
                threat_level = "NONE"

            # Log anomaly detection
            if anomalies:
                self._audit_log('anomaly_detected', {
                    'client_id': client_id,
                    'client_ip': client_ip,
                    'anomalies': anomalies,
                    'risk_score': risk_score,
                    'threat_level': threat_level
                })

            return {
                'anomalies_detected': len(anomalies) > 0,
                'anomalies': anomalies,
                'risk_score': risk_score,
                'threat_level': threat_level,
                'action_required': risk_score >= 0.8
            }

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {
                'anomalies_detected': False,
                'error': f"Anomaly detection failed: {e}"
            }

    def secure_model_access(self, model_path: str, user_id: str, action: str) -> bool:
        """
        Control access to model files and operations.

        Args:
            model_path: Path to model file
            user_id: User requesting access
            action: Action being performed (read, write, execute)

        Returns:
            Access granted status
        """
        try:
            if not self.access_control_enabled:
                return True

            # Define access permissions
            permissions = {
                'admin': ['read', 'write', 'execute', 'delete'],
                'user': ['read', 'execute'],
                'guest': ['execute']
            }

            # Get user role (simplified - in production, this would be from a database)
            user_role = 'user'  # Default role
            if user_id == 'admin':
                user_role = 'admin'
            elif user_id.startswith('guest_'):
                user_role = 'guest'

            # Check permissions
            allowed_actions = permissions.get(user_role, [])
            access_granted = action in allowed_actions

            # Log access attempt
            self._audit_log('model_access_attempt', {
                'user_id': user_id,
                'user_role': user_role,
                'model_path': model_path,
                'action': action,
                'access_granted': access_granted
            })

            if not access_granted:
                logger.warning(f"Access denied: {user_id} attempted {action} on {model_path}")

            return access_granted

        except Exception as e:
            logger.error(f"Access control check failed: {e}")
            return False

    def _audit_log(self, event_type: str, event_data: Dict[str, Any]):
        """Log security events for audit purposes."""
        try:
            if not self.audit_logging_enabled:
                return

            # Create audit entry
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'event_data': event_data,
                'model_version': '96.50% accuracy model'
            }

            # Log to file
            audit_log_path = Path(self.audit_log_file)
            audit_log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(audit_log_path, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')

            # Also log to application logger
            logger.info(f"Security audit: {event_type} - {event_data}")

        except Exception as e:
            logger.error(f"Audit logging failed: {e}")

    def generate_security_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive security report.

        Returns:
            Security status report
        """
        try:
            # Calculate statistics
            total_requests = sum(
                client_data['total_requests']
                for client_data in self.access_log.values()
            )

            active_clients = len(self.access_log)
            blocked_ips_count = len(self.blocked_ips)

            # Security status
            security_status = {
                'encryption_enabled': self.encryption_enabled,
                'access_control_enabled': self.access_control_enabled,
                'audit_logging_enabled': self.audit_logging_enabled,
                'rate_limiting_enabled': self.rate_limiting_enabled
            }

            # Usage statistics
            usage_stats = {
                'total_requests': total_requests,
                'active_clients': active_clients,
                'blocked_ips': blocked_ips_count,
                'max_requests_per_minute': self.max_requests_per_minute,
                'max_requests_per_hour': self.max_requests_per_hour
            }

            # Security recommendations
            recommendations = []
            if not self.encryption_enabled:
                recommendations.append("Enable encryption for sensitive data protection")
            if not self.access_control_enabled:
                recommendations.append("Enable access control for model protection")
            if not self.audit_logging_enabled:
                recommendations.append("Enable audit logging for security monitoring")
            if not self.rate_limiting_enabled:
                recommendations.append("Enable rate limiting to prevent abuse")

            if not recommendations:
                recommendations.append("All security measures are properly configured")

            return {
                'model_version': '96.50% accuracy model',
                'security_status': security_status,
                'usage_statistics': usage_stats,
                'recommendations': recommendations,
                'report_generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Security report generation failed: {e}")
            return {'error': str(e)}

class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass

def secure_model_prediction(model_func, security_manager: ModelSecurityManager):
    """
    Decorator to add security to model prediction functions.

    Args:
        model_func: Model prediction function to secure
        security_manager: Security manager instance

    Returns:
        Secured prediction function
    """
    def wrapper(*args, **kwargs):
        try:
            # Extract security context
            api_key = kwargs.pop('api_key', None)
            client_id = kwargs.pop('client_id', 'unknown')
            client_ip = kwargs.pop('client_ip', 'unknown')

            # Validate API key
            if api_key:
                validation = security_manager.validate_api_key(api_key, 'predict')
                if not validation['valid']:
                    return {'error': 'Invalid API key', 'details': validation.get('error')}
                client_id = validation['user_id']

            # Check rate limits
            rate_limit = security_manager.check_rate_limit(client_id, 'predict')
            if not rate_limit['allowed']:
                return {'error': 'Rate limit exceeded', 'details': rate_limit.get('error')}

            # Detect anomalies
            request_data = {
                'client_id': client_id,
                'client_ip': client_ip,
                'image_size': kwargs.get('image_size', 0),
                'user_agent': kwargs.get('user_agent', '')
            }
            anomaly_result = security_manager.detect_anomalous_requests(request_data)

            if anomaly_result['action_required']:
                return {
                    'error': 'Request blocked due to security concerns',
                    'threat_level': anomaly_result['threat_level']
                }

            # Execute model prediction
            result = model_func(*args, **kwargs)

            # Add security metadata to result
            result['security'] = {
                'rate_limit_remaining': rate_limit['remaining'],
                'threat_level': anomaly_result['threat_level'],
                'model_version': '96.50% accuracy model'
            }

            return result

        except Exception as e:
            logger.error(f"Secured prediction failed: {e}")
            return {'error': f'Prediction failed: {e}'}

    return wrapper