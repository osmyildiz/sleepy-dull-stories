"""
Sleepy Dull Stories - CENTRAL CONFIGURATION UTILITY
Simplified configuration access for all generators
Import this in your generators for centralized config management
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src directory to path for imports
current_dir = Path(__file__).parent
if 'src' in str(current_dir):
    src_dir = current_dir
    while src_dir.name != 'src' and src_dir.parent != src_dir:
        src_dir = src_dir.parent
    if src_dir.name == 'src':
        sys.path.insert(0, str(src_dir))

try:
    from config_manager import ConfigurationManager, Environment

    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️  Config Manager not found - using fallback configuration")
    CONFIG_MANAGER_AVAILABLE = False


class CentralConfig:
    """Simplified configuration access for all generators"""

    _instance = None
    _config_manager = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._setup_config()

    def _setup_config(self):
        """Setup configuration manager"""
        if CONFIG_MANAGER_AVAILABLE:
            # Determine environment
            env_name = os.environ.get('ENVIRONMENT', 'development').lower()
            try:
                environment = Environment(env_name)
            except ValueError:
                environment = Environment.DEVELOPMENT

            try:
                self._config_manager = ConfigurationManager(environment)
                print(f"✅ Central Config initialized ({environment.value})")
            except Exception as e:
                print(f"⚠️  Config Manager initialization failed: {e}")
                self._config_manager = None
        else:
            self._config_manager = None

        # Setup fallback configuration
        self._setup_fallback_config()

    def _setup_fallback_config(self):
        """Setup fallback configuration when Config Manager is not available"""
        current_file = Path(__file__).resolve()

        # Detect project root
        if 'src' in str(current_file.parent):
            src_path = current_file
            while src_path.name != 'src' and src_path.parent != src_path:
                src_path = src_path.parent
            self.project_root = src_path.parent
        else:
            self.project_root = current_file.parent.parent

        # Fallback paths
        self.fallback_paths = {
            'BASE_DIR': str(self.project_root),
            'SRC_DIR': str(self.project_root / 'src'),
            'DATA_DIR': str(self.project_root / 'data'),
            'OUTPUT_DIR': str(self.project_root / 'output'),
            'LOGS_DIR': str(self.project_root / 'logs'),
            'CONFIG_DIR': str(self.project_root / 'config'),
            'CREDENTIALS_DIR': str(self.project_root / 'credentials'),
            'FONTS_DIR': str(self.project_root / 'fonts')
        }

        # Ensure directories exist
        for path_str in self.fallback_paths.values():
            Path(path_str).mkdir(parents=True, exist_ok=True)

    @property
    def paths(self) -> Dict[str, str]:
        """Get project paths"""
        if self._config_manager:
            server_config = self._config_manager.get_server_config()
            return {
                'BASE_DIR': server_config.base_dir,
                'SRC_DIR': str(Path(server_config.base_dir) / 'src'),
                'DATA_DIR': server_config.data_dir,
                'OUTPUT_DIR': server_config.output_dir,
                'LOGS_DIR': server_config.logs_dir,
                'CONFIG_DIR': str(Path(server_config.base_dir) / 'config'),
                'CREDENTIALS_DIR': server_config.credentials_dir,
                'FONTS_DIR': server_config.fonts_dir
            }
        else:
            return self.fallback_paths

    @property
    def database_path(self) -> str:
        """Get database path"""
        if self._config_manager:
            db_config = self._config_manager.get_database_config()
            return db_config.path
        else:
            return str(Path(self.fallback_paths['DATA_DIR']) / 'production.db')

    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for service"""
        if self._config_manager:
            return self._config_manager.get_api_key(service)
        else:
            # Fallback to environment variables
            env_var_map = {
                'openai': 'OPENAI_API_KEY',
                'anthropic': 'ANTHROPIC_API_KEY',
                'google_cloud': 'GOOGLE_CLOUD_CREDENTIALS_PATH',
                'youtube': 'YOUTUBE_CREDENTIALS_PATH'
            }

            env_var = env_var_map.get(service)
            return os.environ.get(env_var) if env_var else None

    @property
    def google_cloud_credentials_path(self) -> Optional[str]:
        """Get Google Cloud credentials path"""
        if self._config_manager:
            api_config = self._config_manager.get_api_config()
            return api_config.google_cloud_credentials_path
        else:
            return os.environ.get('GOOGLE_CLOUD_CREDENTIALS_PATH') or str(
                Path(self.fallback_paths['CREDENTIALS_DIR']) / 'google_cloud.json')

    @property
    def youtube_credentials_path(self) -> Optional[str]:
        """Get YouTube credentials path"""
        if self._config_manager:
            api_config = self._config_manager.get_api_config()
            return api_config.youtube_client_secrets_path
        else:
            return os.environ.get('YOUTUBE_CREDENTIALS_PATH') or str(
                Path(self.fallback_paths['CREDENTIALS_DIR']) / 'youtube_credentials.json')

    @property
    def youtube_token_path(self) -> Optional[str]:
        """Get YouTube token path"""
        if self._config_manager:
            api_config = self._config_manager.get_api_config()
            return api_config.youtube_token_path
        else:
            return str(Path(self.fallback_paths['CREDENTIALS_DIR']) / 'youtube_token.json')

    @property
    def tts_voice(self) -> str:
        """Get TTS voice setting"""
        if self._config_manager:
            quality_config = self._config_manager.get_quality_config()
            return quality_config.tts_voice
        else:
            return os.environ.get('TTS_VOICE', 'en-US-Chirp3-HD-Enceladus')

    @property
    def audio_bitrate(self) -> str:
        """Get audio bitrate setting"""
        if self._config_manager:
            quality_config = self._config_manager.get_quality_config()
            return quality_config.audio_bitrate
        else:
            return os.environ.get('AUDIO_BITRATE', '192k')

    @property
    def max_cost_per_story(self) -> float:
        """Get maximum cost per story"""
        if self._config_manager:
            pipeline_config = self._config_manager.get_pipeline_config()
            return pipeline_config.max_cost_per_story_usd
        else:
            return float(os.environ.get('MAX_COST_PER_STORY_USD', '5.0'))

    @property
    def max_cost_per_session(self) -> float:
        """Get maximum cost per session"""
        if self._config_manager:
            pipeline_config = self._config_manager.get_pipeline_config()
            return pipeline_config.max_cost_per_session_usd
        else:
            return float(os.environ.get('MAX_COST_PER_SESSION_USD', '25.0'))

    @property
    def max_retry_attempts(self) -> int:
        """Get maximum retry attempts"""
        if self._config_manager:
            pipeline_config = self._config_manager.get_pipeline_config()
            return pipeline_config.max_retry_attempts
        else:
            return int(os.environ.get('MAX_RETRY_ATTEMPTS', '3'))

    @property
    def stage_timeout_minutes(self) -> int:
        """Get stage timeout in minutes"""
        if self._config_manager:
            pipeline_config = self._config_manager.get_pipeline_config()
            return pipeline_config.stage_timeout_minutes
        else:
            return int(os.environ.get('STAGE_TIMEOUT_MINUTES', '60'))

    @property
    def default_youtube_privacy(self) -> str:
        """Get default YouTube privacy setting"""
        if self._config_manager:
            security_config = self._config_manager.get_security_config()
            return security_config.default_youtube_privacy
        else:
            return os.environ.get('DEFAULT_YOUTUBE_PRIVACY', 'private')

    @property
    def content_filtering_enabled(self) -> bool:
        """Get content filtering setting"""
        if self._config_manager:
            security_config = self._config_manager.get_security_config()
            return security_config.enable_content_filtering
        else:
            return os.environ.get('CONTENT_FILTERING_ENABLED', 'true').lower() == 'true'

    @property
    def dashboard_port(self) -> int:
        """Get dashboard port"""
        if self._config_manager:
            monitoring_config = self._config_manager.get_monitoring_config()
            return monitoring_config.dashboard_port
        else:
            return int(os.environ.get('DASHBOARD_PORT', '5000'))

    @property
    def dashboard_host(self) -> str:
        """Get dashboard host"""
        if self._config_manager:
            monitoring_config = self._config_manager.get_monitoring_config()
            return monitoring_config.dashboard_host
        else:
            return os.environ.get('DASHBOARD_HOST', '0.0.0.0')

    def get_budget_controls(self) -> Dict[str, Any]:
        """Get budget control settings"""
        if self._config_manager:
            pipeline_config = self._config_manager.get_pipeline_config()
            return {
                'max_cost_per_story_usd': pipeline_config.max_cost_per_story_usd,
                'max_cost_per_session_usd': pipeline_config.max_cost_per_session_usd,
                'max_concurrent_productions': pipeline_config.max_concurrent_productions,
                'max_daily_productions': pipeline_config.max_daily_productions,
                'max_session_duration_hours': pipeline_config.max_session_duration_hours
            }
        else:
            return {
                'max_cost_per_story_usd': float(os.environ.get('MAX_COST_PER_STORY_USD', '5.0')),
                'max_cost_per_session_usd': float(os.environ.get('MAX_COST_PER_SESSION_USD', '25.0')),
                'max_concurrent_productions': int(os.environ.get('MAX_CONCURRENT_PRODUCTIONS', '3')),
                'max_daily_productions': int(os.environ.get('MAX_DAILY_PRODUCTIONS', '20')),
                'max_session_duration_hours': int(os.environ.get('MAX_SESSION_DURATION_HOURS', '8'))
            }

    def get_quality_settings(self) -> Dict[str, Any]:
        """Get quality control settings"""
        if self._config_manager:
            quality_config = self._config_manager.get_quality_config()
            return {
                'min_story_completion_rate': quality_config.min_story_completion_rate,
                'min_character_count': quality_config.min_character_count,
                'min_scene_count': quality_config.min_scene_count,
                'max_scene_count': quality_config.max_scene_count,
                'tts_voice': quality_config.tts_voice,
                'audio_bitrate': quality_config.audio_bitrate,
                'audio_sample_rate': quality_config.audio_sample_rate
            }
        else:
            return {
                'min_story_completion_rate': float(os.environ.get('MIN_STORY_COMPLETION_RATE', '0.8')),
                'min_character_count': int(os.environ.get('MIN_CHARACTER_COUNT', '3')),
                'min_scene_count': int(os.environ.get('MIN_SCENE_COUNT', '20')),
                'max_scene_count': int(os.environ.get('MAX_SCENE_COUNT', '50')),
                'tts_voice': os.environ.get('TTS_VOICE', 'en-US-Chirp3-HD-Enceladus'),
                'audio_bitrate': os.environ.get('AUDIO_BITRATE', '192k'),
                'audio_sample_rate': int(os.environ.get('AUDIO_SAMPLE_RATE', '44100'))
            }

    def is_development_mode(self) -> bool:
        """Check if running in development mode"""
        if self._config_manager:
            server_config = self._config_manager.get_server_config()
            return server_config.debug_mode
        else:
            return os.environ.get('DEBUG_MODE', 'false').lower() == 'true'

    def get_environment(self) -> str:
        """Get current environment"""
        if self._config_manager:
            server_config = self._config_manager.get_server_config()
            return server_config.environment
        else:
            return os.environ.get('ENVIRONMENT', 'development')

    def print_config_summary(self):
        """Print configuration summary"""
        print(f"\n🔧 CENTRAL CONFIG SUMMARY")
        print("=" * 40)
        print(f"Environment: {self.get_environment()}")
        print(f"Config Manager: {'✅ Available' if self._config_manager else '❌ Fallback mode'}")
        print(f"Database: {self.database_path}")
        print(f"TTS Voice: {self.tts_voice}")
        print(f"Max Cost/Story: ${self.max_cost_per_story}")
        print(f"YouTube Privacy: {self.default_youtube_privacy}")
        print(f"Dashboard: {self.dashboard_host}:{self.dashboard_port}")


# Global config instance
config = CentralConfig()


# Convenience functions for common operations
def get_paths() -> Dict[str, str]:
    """Get all project paths"""
    return config.paths


def get_database_path() -> str:
    """Get database path"""
    return config.database_path


def get_api_key(service: str) -> Optional[str]:
    """Get API key for service"""
    return config.get_api_key(service)


def get_google_cloud_credentials() -> Optional[str]:
    """Get Google Cloud credentials path"""
    return config.google_cloud_credentials_path


def get_youtube_credentials() -> Optional[str]:
    """Get YouTube credentials path"""
    return config.youtube_credentials_path


def get_budget_controls() -> Dict[str, Any]:
    """Get budget control settings"""
    return config.get_budget_controls()


def get_quality_settings() -> Dict[str, Any]:
    """Get quality control settings"""
    return config.get_quality_settings()


def is_development() -> bool:
    """Check if in development mode"""
    return config.is_development_mode()


# Example usage in generators
def example_generator_usage():
    """Example of how to use central config in generators"""

    print("📝 EXAMPLE: Using Central Config in Generators")
    print("=" * 50)

    # Import central config (add this to the top of your generator files)
    from central_config import config, get_paths, get_api_key, get_budget_controls

    # Get paths
    paths = get_paths()
    print(f"Output directory: {paths['OUTPUT_DIR']}")
    print(f"Database: {config.database_path}")

    # Get API keys
    openai_key = get_api_key('openai')
    print(f"OpenAI API: {'✅ Set' if openai_key else '❌ Not set'}")

    # Get budget controls
    budget = get_budget_controls()
    print(f"Max cost per story: ${budget['max_cost_per_story_usd']}")

    # Get quality settings
    quality = config.get_quality_settings()
    print(f"TTS Voice: {quality['tts_voice']}")

    # Check environment
    if config.is_development_mode():
        print("🔧 Development mode - verbose logging enabled")
    else:
        print("🚀 Production mode - optimized for performance")


if __name__ == "__main__":
    # Show example usage
    example_generator_usage()

    # Print config summary
    config.print_config_summary()