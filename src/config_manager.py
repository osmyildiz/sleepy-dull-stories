"""
Sleepy Dull Stories - CENTRAL CONFIGURATION MANAGER
Production-ready centralized configuration and credentials management
Secure API key management with environment-specific settings
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv
import secrets
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigurationError(Exception):
    """Configuration-specific exceptions"""
    pass


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    path: str
    timeout_seconds: int = 30
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    wal_mode: bool = True
    synchronous_mode: str = "NORMAL"


@dataclass
class APIConfig:
    """API configuration for external services"""
    openai_api_key: Optional[str] = None
    google_cloud_credentials_path: Optional[str] = None
    youtube_client_secrets_path: Optional[str] = None
    youtube_token_path: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Rate limiting
    openai_requests_per_minute: int = 50
    google_tts_requests_per_minute: int = 100
    youtube_uploads_per_day: int = 50


@dataclass
class PipelineConfig:
    """Pipeline execution configuration"""
    max_retry_attempts: int = 3
    stage_timeout_minutes: int = 60
    inter_stage_delay_seconds: int = 5
    auto_detect_interval_minutes: int = 10
    batch_size: int = 5

    # Budget controls
    max_concurrent_productions: int = 3
    max_daily_productions: int = 20
    max_session_duration_hours: int = 8
    max_cost_per_story_usd: float = 5.0
    max_cost_per_session_usd: float = 25.0


@dataclass
class QualityConfig:
    """Quality control settings"""
    min_story_completion_rate: float = 0.8
    min_character_count: int = 3
    min_scene_count: int = 20
    max_scene_count: int = 50
    min_audio_duration_minutes: float = 5.0
    max_audio_duration_minutes: float = 120.0

    # TTS quality
    tts_voice: str = "en-US-Chirp3-HD-Enceladus"
    audio_bitrate: str = "192k"
    audio_sample_rate: int = 44100


@dataclass
class SecurityConfig:
    """Security and privacy settings"""
    default_youtube_privacy: str = "private"
    enable_content_filtering: bool = True
    log_sensitive_data: bool = False
    token_encryption_enabled: bool = True
    session_timeout_minutes: int = 480  # 8 hours


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    dashboard_enabled: bool = True
    dashboard_port: int = 5000
    dashboard_host: str = "0.0.0.0"
    health_check_interval_seconds: int = 30

    # Alerting
    slack_webhook_url: Optional[str] = None
    email_alerts_enabled: bool = False
    email_smtp_server: Optional[str] = None
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    alert_on_failure: bool = True
    alert_on_success: bool = False


@dataclass
class ServerConfig:
    """Server and deployment configuration"""
    environment: str = Environment.DEVELOPMENT.value
    debug_mode: bool = False
    log_level: str = "INFO"
    max_workers: int = 4

    # Paths
    base_dir: str = ""
    data_dir: str = ""
    output_dir: str = ""
    logs_dir: str = ""
    credentials_dir: str = ""
    fonts_dir: str = ""


class ConfigurationManager:
    """Central configuration management system"""

    def __init__(self, environment: Environment = Environment.DEVELOPMENT, config_file: str = None):
        self.environment = environment

        # Setup paths
        self.setup_paths()

        # Load environment variables first
        self.load_environment_variables()

        # Load configuration
        self.config_file = config_file or self.get_default_config_path()
        self.config_data = self.load_configuration()

        # Initialize encryption for sensitive data
        self.encryption_key = self.get_or_create_encryption_key()

        # Validate configuration
        self.validate_configuration()

        print(f"✅ Configuration Manager initialized ({environment.value})")

    def setup_paths(self):
        """Setup project paths"""
        current_file = Path(__file__).resolve()

        # Detect if we're in src/ directory or project root
        if current_file.parent.name == 'src':
            self.project_root = current_file.parent.parent
        elif 'src' in str(current_file.parent):
            # Find the src directory and go up one level
            src_path = current_file
            while src_path.name != 'src' and src_path.parent != src_path:
                src_path = src_path.parent
            self.project_root = src_path.parent
        else:
            self.project_root = current_file.parent.parent

        self.paths = {
            'project_root': self.project_root,
            'src_dir': self.project_root / 'src',
            'config_dir': self.project_root / 'config',
            'data_dir': self.project_root / 'data',
            'output_dir': self.project_root / 'output',
            'logs_dir': self.project_root / 'logs',
            'credentials_dir': self.project_root / 'credentials',
            'fonts_dir': self.project_root / 'fonts'
        }

        # Ensure directories exist
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

        print(f"📁 Project root: {self.project_root}")

    def load_environment_variables(self):
        """Load environment variables from .env files"""
        env_files = [
            self.project_root / '.env',
            self.project_root / f'.env.{self.environment.value}',
            self.paths['config_dir'] / '.env',
            self.paths['config_dir'] / f'.env.{self.environment.value}'
        ]

        loaded_files = []
        for env_file in env_files:
            if env_file.exists():
                load_dotenv(env_file)
                loaded_files.append(str(env_file))

        if loaded_files:
            print(f"🔧 Loaded environment files: {loaded_files}")
        else:
            print("⚠️  No .env files found - using system environment variables")

    def get_default_config_path(self) -> Path:
        """Get default configuration file path"""
        return self.paths['config_dir'] / f'config.{self.environment.value}.json'

    def load_configuration(self) -> Dict:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"📋 Configuration loaded from: {self.config_file}")
                return config
            except Exception as e:
                print(f"⚠️  Error loading config file: {e}")
                print("📋 Creating default configuration...")
        else:
            print(f"📋 Config file not found: {self.config_file}")
            print("📋 Creating default configuration...")

        # Create default configuration
        default_config = self.create_default_configuration()
        self.save_configuration(default_config)
        return default_config

    def create_default_configuration(self) -> Dict:
        """Create default configuration"""

        default_config = {
            "database": asdict(DatabaseConfig(
                path=str(self.paths['data_dir'] / 'production.db')
            )),

            "api": asdict(APIConfig(
                google_cloud_credentials_path=str(self.paths['credentials_dir'] / 'google_cloud.json'),
                youtube_client_secrets_path=str(self.paths['credentials_dir'] / 'youtube_credentials.json'),
                youtube_token_path=str(self.paths['credentials_dir'] / 'youtube_token.json')
            )),

            "pipeline": asdict(PipelineConfig()),

            "quality": asdict(QualityConfig()),

            "security": asdict(SecurityConfig()),

            "monitoring": asdict(MonitoringConfig()),

            "server": asdict(ServerConfig(
                environment=self.environment.value,
                debug_mode=(self.environment == Environment.DEVELOPMENT),
                base_dir=str(self.project_root),
                data_dir=str(self.paths['data_dir']),
                output_dir=str(self.paths['output_dir']),
                logs_dir=str(self.paths['logs_dir']),
                credentials_dir=str(self.paths['credentials_dir']),
                fonts_dir=str(self.paths['fonts_dir'])
            ))
        }

        return default_config

    def save_configuration(self, config: Dict = None):
        """Save configuration to file"""
        if config is None:
            config = self.config_data

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"💾 Configuration saved to: {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")

    def get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for sensitive data"""
        key_file = self.paths['credentials_dir'] / '.encryption_key'

        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                print(f"⚠️  Error reading encryption key: {e}")

        # Create new encryption key
        password = os.environ.get('CONFIG_ENCRYPTION_PASSWORD', 'sleepy-dull-stories-2025').encode()
        salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(kdf.derive(password))

        try:
            # Save key securely
            with open(key_file, 'wb') as f:
                f.write(salt + key)

            # Set secure file permissions (Unix systems)
            if hasattr(os, 'chmod'):
                os.chmod(key_file, 0o600)

            print(f"🔐 New encryption key created: {key_file}")
            return key

        except Exception as e:
            print(f"❌ Error saving encryption key: {e}")
            return key

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            print(f"⚠️  Encryption warning: {e}")
            return data  # Return original if encryption fails

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            fernet = Fernet(self.encryption_key)
            decrypted_data = fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            print(f"⚠️  Decryption warning: {e}")
            return encrypted_data  # Return original if decryption fails

    def get_api_key(self, service: str, encrypted: bool = True) -> Optional[str]:
        """Get API key for service"""
        env_var_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google_cloud': 'GOOGLE_CLOUD_CREDENTIALS_PATH',
            'youtube': 'YOUTUBE_CREDENTIALS_PATH'
        }

        # Try environment variable first
        env_var = env_var_map.get(service)
        if env_var:
            api_key = os.environ.get(env_var)
            if api_key:
                return api_key

        # Try configuration file
        api_config = self.config_data.get('api', {})
        config_key = f"{service}_api_key"

        if config_key in api_config and api_config[config_key]:
            encrypted_key = api_config[config_key]
            if encrypted and self.config_data.get('security', {}).get('token_encryption_enabled', True):
                return self.decrypt_sensitive_data(encrypted_key)
            else:
                return encrypted_key

        return None

    def set_api_key(self, service: str, api_key: str, encrypt: bool = True):
        """Set API key for service"""
        if 'api' not in self.config_data:
            self.config_data['api'] = {}

        config_key = f"{service}_api_key"

        if encrypt and self.config_data.get('security', {}).get('token_encryption_enabled', True):
            encrypted_key = self.encrypt_sensitive_data(api_key)
            self.config_data['api'][config_key] = encrypted_key
        else:
            self.config_data['api'][config_key] = api_key

        self.save_configuration()
        print(f"🔐 API key set for {service} ({'encrypted' if encrypt else 'plain'})")

    def validate_configuration(self):
        """Validate configuration and show warnings for missing items"""
        missing_items = []
        warnings = []

        # Required API keys for production
        if self.environment == Environment.PRODUCTION:
            required_api_keys = ['openai', 'google_cloud', 'youtube']
            for service in required_api_keys:
                if not self.get_api_key(service):
                    missing_items.append(f"{service.upper()}_API_KEY")

        # Required paths
        required_paths = ['google_cloud_credentials_path', 'youtube_client_secrets_path']
        api_config = self.config_data.get('api', {})

        for path_key in required_paths:
            if path_key in api_config:
                path = Path(api_config[path_key])
                if not path.exists():
                    warnings.append(f"Credentials file not found: {path}")

        # Database validation
        db_path = Path(self.config_data.get('database', {}).get('path', ''))
        if not db_path.parent.exists():
            warnings.append(f"Database directory does not exist: {db_path.parent}")

        # Show validation results
        if missing_items:
            print(f"⚠️  Missing required configuration:")
            for item in missing_items:
                print(f"   ❌ {item}")

        if warnings:
            print(f"⚠️  Configuration warnings:")
            for warning in warnings:
                print(f"   ⚠️  {warning}")

        if not missing_items and not warnings:
            print(f"✅ Configuration validation passed")

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        db_config = self.config_data.get('database', {})
        return DatabaseConfig(**db_config)

    def get_api_config(self) -> APIConfig:
        """Get API configuration with decrypted keys"""
        api_config = self.config_data.get('api', {}).copy()

        # Decrypt API keys
        for key in ['openai_api_key', 'anthropic_api_key']:
            if key in api_config and api_config[key]:
                if self.config_data.get('security', {}).get('token_encryption_enabled', True):
                    api_config[key] = self.decrypt_sensitive_data(api_config[key])

        return APIConfig(**api_config)

    def get_pipeline_config(self) -> PipelineConfig:
        """Get pipeline configuration"""
        pipeline_config = self.config_data.get('pipeline', {})
        return PipelineConfig(**pipeline_config)

    def get_quality_config(self) -> QualityConfig:
        """Get quality configuration"""
        quality_config = self.config_data.get('quality', {})
        return QualityConfig(**quality_config)

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        security_config = self.config_data.get('security', {})
        return SecurityConfig(**security_config)

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        monitoring_config = self.config_data.get('monitoring', {})

        # Decrypt sensitive monitoring data
        if 'slack_webhook_url' in monitoring_config and monitoring_config['slack_webhook_url']:
            if self.config_data.get('security', {}).get('token_encryption_enabled', True):
                monitoring_config['slack_webhook_url'] = self.decrypt_sensitive_data(
                    monitoring_config['slack_webhook_url'])

        return MonitoringConfig(**monitoring_config)

    def get_server_config(self) -> ServerConfig:
        """Get server configuration"""
        server_config = self.config_data.get('server', {})
        return ServerConfig(**server_config)

    def update_config_section(self, section: str, config: Dict):
        """Update a configuration section"""
        if section not in self.config_data:
            self.config_data[section] = {}

        self.config_data[section].update(config)
        self.save_configuration()
        print(f"🔧 Updated {section} configuration")

    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configuration objects"""
        return {
            'database': self.get_database_config(),
            'api': self.get_api_config(),
            'pipeline': self.get_pipeline_config(),
            'quality': self.get_quality_config(),
            'security': self.get_security_config(),
            'monitoring': self.get_monitoring_config(),
            'server': self.get_server_config()
        }

    def print_configuration_summary(self):
        """Print comprehensive configuration summary"""
        print(f"\n🔧 CONFIGURATION SUMMARY ({self.environment.value.upper()})")
        print("=" * 60)

        configs = self.get_all_configs()

        # API Configuration
        api_config = configs['api']
        print(f"🔑 API CONFIGURATION:")
        print(f"   OpenAI: {'✅ Set' if api_config.openai_api_key else '❌ Not set'}")
        print(f"   Anthropic: {'✅ Set' if api_config.anthropic_api_key else '❌ Not set'}")
        print(f"   Google Cloud: {'✅ Set' if api_config.google_cloud_credentials_path else '❌ Not set'}")
        print(f"   YouTube: {'✅ Set' if api_config.youtube_client_secrets_path else '❌ Not set'}")

        # Database Configuration
        db_config = configs['database']
        print(f"\n🗄️  DATABASE CONFIGURATION:")
        print(f"   Path: {db_config.path}")
        print(f"   Timeout: {db_config.timeout_seconds}s")
        print(f"   WAL Mode: {'✅' if db_config.wal_mode else '❌'}")

        # Pipeline Configuration
        pipeline_config = configs['pipeline']
        print(f"\n🚀 PIPELINE CONFIGURATION:")
        print(f"   Max retries: {pipeline_config.max_retry_attempts}")
        print(f"   Stage timeout: {pipeline_config.stage_timeout_minutes} min")
        print(f"   Max concurrent: {pipeline_config.max_concurrent_productions}")
        print(f"   Daily limit: {pipeline_config.max_daily_productions}")

        # Quality Configuration
        quality_config = configs['quality']
        print(f"\n🎯 QUALITY CONFIGURATION:")
        print(f"   TTS Voice: {quality_config.tts_voice}")
        print(f"   Audio bitrate: {quality_config.audio_bitrate}")
        print(f"   Min scenes: {quality_config.min_scene_count}")
        print(f"   Max cost/story: ${quality_config.max_cost_per_story_usd}")

        # Security Configuration
        security_config = configs['security']
        print(f"\n🔒 SECURITY CONFIGURATION:")
        print(f"   YouTube privacy: {security_config.default_youtube_privacy}")
        print(f"   Content filtering: {'✅' if security_config.enable_content_filtering else '❌'}")
        print(f"   Token encryption: {'✅' if security_config.token_encryption_enabled else '❌'}")

        # Monitoring Configuration
        monitoring_config = configs['monitoring']
        print(f"\n📊 MONITORING CONFIGURATION:")
        print(f"   Dashboard: {'✅' if monitoring_config.dashboard_enabled else '❌'}")
        print(f"   Dashboard port: {monitoring_config.dashboard_port}")
        print(f"   Slack alerts: {'✅' if monitoring_config.slack_webhook_url else '❌'}")
        print(f"   Email alerts: {'✅' if monitoring_config.email_alerts_enabled else '❌'}")


def create_sample_env_file(environment: Environment = Environment.DEVELOPMENT):
    """Create sample .env file"""

    env_content = f"""# Sleepy Dull Stories - Environment Configuration ({environment.value})
# Copy this file to .env and fill in your actual values

# =============================================================================
# API KEYS & CREDENTIALS
# =============================================================================

# OpenAI API Key (for story generation)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (if using Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Cloud Credentials (for TTS)
GOOGLE_CLOUD_CREDENTIALS_PATH=./credentials/google_cloud.json

# YouTube API Credentials
YOUTUBE_CREDENTIALS_PATH=./credentials/youtube_credentials.json

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DATABASE_PATH=./data/production.db
DATABASE_BACKUP_ENABLED=true

# =============================================================================
# SECURITY SETTINGS
# =============================================================================

# Encryption password for sensitive data
CONFIG_ENCRYPTION_PASSWORD=your_secure_encryption_password_here

# Default YouTube video privacy
DEFAULT_YOUTUBE_PRIVACY=private

# =============================================================================
# MONITORING & ALERTS
# =============================================================================

# Slack webhook for alerts (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url

# Email settings for alerts (optional)
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# =============================================================================
# PIPELINE SETTINGS
# =============================================================================

# Budget controls
MAX_COST_PER_STORY_USD=5.0
MAX_COST_PER_SESSION_USD=25.0
MAX_DAILY_PRODUCTIONS=20

# Performance settings
MAX_CONCURRENT_PRODUCTIONS=3
STAGE_TIMEOUT_MINUTES=60

# =============================================================================
# ENVIRONMENT SPECIFIC SETTINGS
# =============================================================================

ENVIRONMENT={environment.value}
DEBUG_MODE={"true" if environment == Environment.DEVELOPMENT else "false"}
LOG_LEVEL={"DEBUG" if environment == Environment.DEVELOPMENT else "INFO"}
"""

    return env_content


def main():
    """Main function for configuration management"""
    import argparse

    parser = argparse.ArgumentParser(description="Configuration Manager - Sleepy Dull Stories")

    parser.add_argument('--env',
                        choices=['development', 'staging', 'production'],
                        default='development',
                        help='Environment (default: development)')

    parser.add_argument('--create-env',
                        action='store_true',
                        help='Create sample .env file')

    parser.add_argument('--validate',
                        action='store_true',
                        help='Validate configuration')

    parser.add_argument('--set-api-key',
                        nargs=2,
                        metavar=('SERVICE', 'KEY'),
                        help='Set API key for service')

    args = parser.parse_args()

    print("🔧 SLEEPY DULL STORIES - CONFIGURATION MANAGER")
    print("=" * 60)

    # Create sample .env file
    if args.create_env:
        env_map = {
            'development': Environment.DEVELOPMENT,
            'staging': Environment.STAGING,
            'production': Environment.PRODUCTION
        }

        environment = env_map[args.env]
        env_content = create_sample_env_file(environment)

        env_file = Path(f'.env.{environment.value}')
        with open(env_file, 'w') as f:
            f.write(env_content)

        print(f"📝 Sample environment file created: {env_file}")
        print(f"🔧 Edit this file with your actual API keys and settings")
        return

    # Initialize configuration manager
    environment = Environment(args.env)
    config_manager = ConfigurationManager(environment)

    # Set API key
    if args.set_api_key:
        service, api_key = args.set_api_key
        config_manager.set_api_key(service, api_key)
        print(f"✅ API key set for {service}")

    # Validate configuration
    if args.validate:
        config_manager.validate_configuration()

    # Print configuration summary
    config_manager.print_configuration_summary()


if __name__ == "__main__":
    main()