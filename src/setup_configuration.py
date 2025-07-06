"""
Sleepy Dull Stories - CONFIGURATION SETUP SCRIPT
Interactive setup for production-ready configuration
Guides users through complete system configuration
"""

import os
import json
import getpass
from pathlib import Path
from typing import Dict, Any, Optional
import requests

try:
    from config_manager import ConfigurationManager, Environment, create_sample_env_file
    from central_config import CentralConfig

    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️  Configuration modules not found")
    CONFIG_AVAILABLE = False


class ConfigurationSetup:
    """Interactive configuration setup wizard"""

    def __init__(self):
        self.setup_data = {}
        self.environment = Environment.DEVELOPMENT

        print("🔧 SLEEPY DULL STORIES - CONFIGURATION SETUP WIZARD")
        print("=" * 60)
        print("This wizard will guide you through setting up your production environment")
        print()

    def welcome_screen(self):
        """Show welcome screen and get basic info"""
        print("👋 Welcome to Sleepy Dull Stories Configuration Setup!")
        print()
        print("This setup will configure:")
        print("  🔑 API Keys (OpenAI, Google Cloud, YouTube)")
        print("  🗄️  Database settings")
        print("  🎛️  Pipeline configuration")
        print("  🔒 Security settings")
        print("  📊 Monitoring setup")
        print()

        # Get environment
        while True:
            env_input = input("Select environment (development/staging/production) [development]: ").strip().lower()
            if not env_input:
                env_input = "development"

            try:
                self.environment = Environment(env_input)
                break
            except ValueError:
                print("❌ Invalid environment. Please choose: development, staging, or production")

        print(f"✅ Environment selected: {self.environment.value}")
        print()

    def setup_api_keys(self):
        """Setup API keys interactively"""
        print("🔑 API KEYS CONFIGURATION")
        print("-" * 30)
        print("We'll set up your API keys for external services.")
        print("You can skip any service you don't plan to use immediately.")
        print()

        # OpenAI API Key
        print("1. OpenAI API Key (for story generation)")
        print("   Get your key from: https://platform.openai.com/api-keys")
        openai_key = getpass.getpass("   Enter OpenAI API key (or press Enter to skip): ").strip()
        if openai_key:
            if self.validate_openai_key(openai_key):
                self.setup_data['openai_api_key'] = openai_key
                print("   ✅ OpenAI API key validated and saved")
            else:
                print("   ⚠️  Invalid OpenAI API key, but saving anyway")
                self.setup_data['openai_api_key'] = openai_key
        else:
            print("   ⏭️  Skipped OpenAI API key")
        print()

        # Anthropic API Key
        print("2. Anthropic API Key (optional, for Claude)")
        print("   Get your key from: https://console.anthropic.com/")
        anthropic_key = getpass.getpass("   Enter Anthropic API key (or press Enter to skip): ").strip()
        if anthropic_key:
            self.setup_data['anthropic_api_key'] = anthropic_key
            print("   ✅ Anthropic API key saved")
        else:
            print("   ⏭️  Skipped Anthropic API key")
        print()

        # Google Cloud Setup
        print("3. Google Cloud (for Text-to-Speech)")
        print("   You need to:")
        print("   - Create a project at: https://console.cloud.google.com/")
        print("   - Enable Text-to-Speech API")
        print("   - Create a service account and download JSON credentials")
        print()

        setup_google = input("   Have you set up Google Cloud credentials? (y/n) [n]: ").strip().lower()
        if setup_google == 'y':
            while True:
                cred_path = input("   Enter path to Google Cloud credentials JSON file: ").strip()
                if cred_path and Path(cred_path).exists():
                    self.setup_data['google_cloud_credentials_path'] = cred_path
                    print("   ✅ Google Cloud credentials path saved")
                    break
                elif not cred_path:
                    print("   ⏭️  Skipped Google Cloud setup")
                    break
                else:
                    print("   ❌ File not found, please check the path")
        else:
            print("   ⏭️  Skipped Google Cloud setup")
        print()

        # YouTube Setup
        print("4. YouTube API (for video uploads)")
        print("   You need to:")
        print("   - Create a project at: https://console.developers.google.com/")
        print("   - Enable YouTube Data API v3")
        print("   - Create OAuth 2.0 credentials")
        print("   - Download credentials JSON file")
        print()

        setup_youtube = input("   Have you set up YouTube API credentials? (y/n) [n]: ").strip().lower()
        if setup_youtube == 'y':
            while True:
                cred_path = input("   Enter path to YouTube credentials JSON file: ").strip()
                if cred_path and Path(cred_path).exists():
                    self.setup_data['youtube_credentials_path'] = cred_path
                    print("   ✅ YouTube credentials path saved")
                    break
                elif not cred_path:
                    print("   ⏭️  Skipped YouTube setup")
                    break
                else:
                    print("   ❌ File not found, please check the path")
        else:
            print("   ⏭️  Skipped YouTube setup")
        print()

    def validate_openai_key(self, api_key: str) -> bool:
        """Validate OpenAI API key"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Simple API call to validate key
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=10
            )

            return response.status_code == 200
        except Exception:
            return False

    def setup_pipeline_config(self):
        """Setup pipeline configuration"""
        print("🚀 PIPELINE CONFIGURATION")
        print("-" * 30)
        print("Configure pipeline behavior and limits.")
        print()

        # Budget controls
        print("Budget Controls:")

        max_cost_story = input(f"   Max cost per story (USD) [5.0]: ").strip()
        self.setup_data['max_cost_per_story_usd'] = float(max_cost_story) if max_cost_story else 5.0

        max_cost_session = input(f"   Max cost per session (USD) [25.0]: ").strip()
        self.setup_data['max_cost_per_session_usd'] = float(max_cost_session) if max_cost_session else 25.0

        max_daily = input(f"   Max daily productions [20]: ").strip()
        self.setup_data['max_daily_productions'] = int(max_daily) if max_daily else 20

        max_concurrent = input(f"   Max concurrent productions [3]: ").strip()
        self.setup_data['max_concurrent_productions'] = int(max_concurrent) if max_concurrent else 3

        print()

        # Performance settings
        print("Performance Settings:")

        retry_attempts = input(f"   Max retry attempts per stage [3]: ").strip()
        self.setup_data['max_retry_attempts'] = int(retry_attempts) if retry_attempts else 3

        stage_timeout = input(f"   Stage timeout (minutes) [60]: ").strip()
        self.setup_data['stage_timeout_minutes'] = int(stage_timeout) if stage_timeout else 60

        print()

    def setup_quality_config(self):
        """Setup quality configuration"""
        print("🎯 QUALITY CONFIGURATION")
        print("-" * 30)
        print("Configure quality standards and TTS settings.")
        print()

        # TTS Settings
        print("Text-to-Speech Settings:")

        print("   Available voices:")
        print("   1. en-US-Chirp3-HD-Enceladus (Premium, HD quality)")
        print("   2. en-US-Studio-O (Standard quality)")
        print("   3. en-US-Wavenet-D (Standard quality)")

        voice_choice = input("   Select voice (1-3) [1]: ").strip()
        voice_map = {
            '1': 'en-US-Chirp3-HD-Enceladus',
            '2': 'en-US-Studio-O',
            '3': 'en-US-Wavenet-D'
        }
        self.setup_data['tts_voice'] = voice_map.get(voice_choice, 'en-US-Chirp3-HD-Enceladus')

        audio_bitrate = input("   Audio bitrate (96k/128k/192k) [192k]: ").strip()
        self.setup_data['audio_bitrate'] = audio_bitrate if audio_bitrate in ['96k', '128k', '192k'] else '192k'

        print()

        # Quality thresholds
        print("Quality Thresholds:")

        min_scenes = input("   Minimum scenes per story [20]: ").strip()
        self.setup_data['min_scene_count'] = int(min_scenes) if min_scenes else 20

        max_scenes = input("   Maximum scenes per story [50]: ").strip()
        self.setup_data['max_scene_count'] = int(max_scenes) if max_scenes else 50

        min_characters = input("   Minimum characters per story [3]: ").strip()
        self.setup_data['min_character_count'] = int(min_characters) if min_characters else 3

        print()

    def setup_security_config(self):
        """Setup security configuration"""
        print("🔒 SECURITY CONFIGURATION")
        print("-" * 30)
        print("Configure security and privacy settings.")
        print()

        # YouTube privacy
        print("YouTube Privacy Settings:")
        print("   1. private (videos start as private, manual review)")
        print("   2. unlisted (videos are unlisted by default)")
        print("   3. public (videos are immediately public - not recommended)")

        privacy_choice = input("   Default YouTube privacy (1-3) [1]: ").strip()
        privacy_map = {'1': 'private', '2': 'unlisted', '3': 'public'}
        self.setup_data['default_youtube_privacy'] = privacy_map.get(privacy_choice, 'private')

        # Content filtering
        content_filter = input("   Enable content filtering? (y/n) [y]: ").strip().lower()
        self.setup_data['enable_content_filtering'] = content_filter != 'n'

        # Token encryption
        token_encryption = input("   Enable token encryption? (y/n) [y]: ").strip().lower()
        self.setup_data['token_encryption_enabled'] = token_encryption != 'n'

        if self.setup_data['token_encryption_enabled']:
            encryption_password = getpass.getpass("   Enter encryption password (or press Enter for default): ").strip()
            if encryption_password:
                self.setup_data['encryption_password'] = encryption_password

        print()

    def setup_monitoring_config(self):
        """Setup monitoring configuration"""
        print("📊 MONITORING CONFIGURATION")
        print("-" * 30)
        print("Configure monitoring and alerting.")
        print()

        # Dashboard settings
        dashboard_enabled = input("   Enable web dashboard? (y/n) [y]: ").strip().lower()
        self.setup_data['dashboard_enabled'] = dashboard_enabled != 'n'

        if self.setup_data['dashboard_enabled']:
            dashboard_port = input("   Dashboard port [5000]: ").strip()
            self.setup_data['dashboard_port'] = int(dashboard_port) if dashboard_port else 5000

        # Slack alerts
        print("\n   Slack Notifications (optional):")
        slack_setup = input("   Setup Slack alerts? (y/n) [n]: ").strip().lower()
        if slack_setup == 'y':
            slack_webhook = input("   Enter Slack webhook URL: ").strip()
            if slack_webhook:
                self.setup_data['slack_webhook_url'] = slack_webhook

        # Email alerts
        print("\n   Email Notifications (optional):")
        email_setup = input("   Setup email alerts? (y/n) [n]: ").strip().lower()
        if email_setup == 'y':
            email_server = input("   SMTP server [smtp.gmail.com]: ").strip()
            self.setup_data['email_smtp_server'] = email_server if email_server else 'smtp.gmail.com'

            email_port = input("   SMTP port [587]: ").strip()
            self.setup_data['email_smtp_port'] = int(email_port) if email_port else 587

            email_username = input("   Email username: ").strip()
            if email_username:
                self.setup_data['email_username'] = email_username

                email_password = getpass.getpass("   Email password (app password recommended): ").strip()
                if email_password:
                    self.setup_data['email_password'] = email_password

        print()

    def create_configuration_files(self):
        """Create configuration files"""
        print("💾 CREATING CONFIGURATION FILES")
        print("-" * 40)

        if not CONFIG_AVAILABLE:
            print("❌ Configuration modules not available")
            return False

        try:
            # Initialize config manager
            config_manager = ConfigurationManager(self.environment)

            # Update API configuration
            if 'openai_api_key' in self.setup_data:
                config_manager.set_api_key('openai', self.setup_data['openai_api_key'])

            if 'anthropic_api_key' in self.setup_data:
                config_manager.set_api_key('anthropic', self.setup_data['anthropic_api_key'])

            # Update pipeline configuration
            pipeline_updates = {}
            for key in ['max_cost_per_story_usd', 'max_cost_per_session_usd', 'max_daily_productions',
                        'max_concurrent_productions', 'max_retry_attempts', 'stage_timeout_minutes']:
                if key in self.setup_data:
                    pipeline_updates[key] = self.setup_data[key]

            if pipeline_updates:
                config_manager.update_config_section('pipeline', pipeline_updates)

            # Update quality configuration
            quality_updates = {}
            for key in ['tts_voice', 'audio_bitrate', 'min_scene_count', 'max_scene_count', 'min_character_count']:
                if key in self.setup_data:
                    quality_updates[key] = self.setup_data[key]

            if quality_updates:
                config_manager.update_config_section('quality', quality_updates)

            # Update security configuration
            security_updates = {}
            for key in ['default_youtube_privacy', 'enable_content_filtering', 'token_encryption_enabled']:
                if key in self.setup_data:
                    security_updates[key] = self.setup_data[key]

            if security_updates:
                config_manager.update_config_section('security', security_updates)

            # Update monitoring configuration
            monitoring_updates = {}
            for key in ['dashboard_enabled', 'dashboard_port', 'slack_webhook_url',
                        'email_smtp_server', 'email_smtp_port', 'email_username', 'email_password']:
                if key in self.setup_data:
                    monitoring_updates[key] = self.setup_data[key]

            if monitoring_updates:
                config_manager.update_config_section('monitoring', monitoring_updates)

            # Update API paths
            api_updates = {}
            if 'google_cloud_credentials_path' in self.setup_data:
                api_updates['google_cloud_credentials_path'] = self.setup_data['google_cloud_credentials_path']

            if 'youtube_credentials_path' in self.setup_data:
                api_updates['youtube_client_secrets_path'] = self.setup_data['youtube_credentials_path']

            if api_updates:
                config_manager.update_config_section('api', api_updates)

            print("✅ Configuration files created successfully")

            # Create .env file
            env_content = self.create_env_file_content()
            env_file = Path(f'.env.{self.environment.value}')

            with open(env_file, 'w') as f:
                f.write(env_content)

            print(f"✅ Environment file created: {env_file}")

            return True

        except Exception as e:
            print(f"❌ Error creating configuration: {e}")
            return False

    def create_env_file_content(self) -> str:
        """Create .env file content from setup data"""
        lines = [
            f"# Sleepy Dull Stories - Environment Configuration ({self.environment.value})",
            f"# Generated by Configuration Setup Wizard",
            f"# {Path().absolute()}",
            "",
            "# =============================================================================",
            "# ENVIRONMENT SETTINGS",
            "# =============================================================================",
            "",
            f"ENVIRONMENT={self.environment.value}",
            f"DEBUG_MODE={'true' if self.environment == Environment.DEVELOPMENT else 'false'}",
            f"LOG_LEVEL={'DEBUG' if self.environment == Environment.DEVELOPMENT else 'INFO'}",
            "",
            "# =============================================================================",
            "# API KEYS & CREDENTIALS",
            "# =============================================================================",
            ""
        ]

        # Add API keys
        if 'openai_api_key' in self.setup_data:
            lines.append(f"OPENAI_API_KEY={self.setup_data['openai_api_key']}")
        else:
            lines.append("# OPENAI_API_KEY=your_openai_api_key_here")

        if 'anthropic_api_key' in self.setup_data:
            lines.append(f"ANTHROPIC_API_KEY={self.setup_data['anthropic_api_key']}")
        else:
            lines.append("# ANTHROPIC_API_KEY=your_anthropic_api_key_here")

        if 'google_cloud_credentials_path' in self.setup_data:
            lines.append(f"GOOGLE_CLOUD_CREDENTIALS_PATH={self.setup_data['google_cloud_credentials_path']}")
        else:
            lines.append("# GOOGLE_CLOUD_CREDENTIALS_PATH=./credentials/google_cloud.json")

        if 'youtube_credentials_path' in self.setup_data:
            lines.append(f"YOUTUBE_CREDENTIALS_PATH={self.setup_data['youtube_credentials_path']}")
        else:
            lines.append("# YOUTUBE_CREDENTIALS_PATH=./credentials/youtube_credentials.json")

        lines.extend([
            "",
            "# =============================================================================",
            "# SECURITY SETTINGS",
            "# =============================================================================",
            ""
        ])

        if 'encryption_password' in self.setup_data:
            lines.append(f"CONFIG_ENCRYPTION_PASSWORD={self.setup_data['encryption_password']}")

        if 'default_youtube_privacy' in self.setup_data:
            lines.append(f"DEFAULT_YOUTUBE_PRIVACY={self.setup_data['default_youtube_privacy']}")

        lines.extend([
            "",
            "# =============================================================================",
            "# MONITORING & ALERTS",
            "# =============================================================================",
            ""
        ])

        if 'slack_webhook_url' in self.setup_data:
            lines.append(f"SLACK_WEBHOOK_URL={self.setup_data['slack_webhook_url']}")

        if 'email_username' in self.setup_data:
            lines.append(f"EMAIL_USERNAME={self.setup_data['email_username']}")

        if 'email_password' in self.setup_data:
            lines.append(f"EMAIL_PASSWORD={self.setup_data['email_password']}")

        return "\n".join(lines)

    def show_completion_summary(self):
        """Show setup completion summary"""
        print("\n" + "🎉" * 60)
        print("CONFIGURATION SETUP COMPLETED SUCCESSFULLY!")
        print("🎉" * 60)
        print()

        print("📋 What was configured:")
        if 'openai_api_key' in self.setup_data:
            print("   ✅ OpenAI API key")
        if 'anthropic_api_key' in self.setup_data:
            print("   ✅ Anthropic API key")
        if 'google_cloud_credentials_path' in self.setup_data:
            print("   ✅ Google Cloud credentials")
        if 'youtube_credentials_path' in self.setup_data:
            print("   ✅ YouTube API credentials")
        if 'slack_webhook_url' in self.setup_data:
            print("   ✅ Slack notifications")
        if 'email_username' in self.setup_data:
            print("   ✅ Email notifications")

        print(f"\n🎯 Environment: {self.environment.value}")
        print(f"🔧 Configuration files created")
        print(f"🌍 Environment variables set")
        print()

        print("🚀 NEXT STEPS:")
        print("1. Test your configuration:")
        print("   python config_manager.py --validate")
        print()
        print("2. Initialize the database:")
        print("   python database_setup.py")
        print()
        print("3. Start the monitoring dashboard:")
        if 'dashboard_port' in self.setup_data:
            print(f"   python production_dashboard_server.py")
            print(f"   Then open: http://localhost:{self.setup_data['dashboard_port']}")
        else:
            print("   python production_dashboard_server.py")
        print()
        print("4. Run the Master Orchestrator:")
        print("   python master_orchestrator.py --mode auto")
        print()
        print("🎬 Your autonomous YouTube content factory is ready!")

    def run_setup(self):
        """Run the complete setup wizard"""
        try:
            self.welcome_screen()
            self.setup_api_keys()
            self.setup_pipeline_config()
            self.setup_quality_config()
            self.setup_security_config()
            self.setup_monitoring_config()

            success = self.create_configuration_files()

            if success:
                self.show_completion_summary()
                return True
            else:
                print("❌ Configuration setup failed")
                return False

        except KeyboardInterrupt:
            print("\n⏹️  Setup cancelled by user")
            return False
        except Exception as e:
            print(f"\n❌ Setup error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main setup function"""
    setup = ConfigurationSetup()
    success = setup.run_setup()

    if success:
        print("\n✅ Configuration setup completed successfully!")
    else:
        print("\n❌ Configuration setup failed or was cancelled")

    return success


if __name__ == "__main__":
    main()