# 🔧 Configuration Management System

Complete centralized configuration management for Sleepy Dull Stories production environment.

## 📋 Overview

The Configuration Management System provides:
- **Centralized API key management** with encryption
- **Environment-specific settings** (development/staging/production)
- **Secure credential storage** with automatic encryption
- **Interactive setup wizard** for easy deployment
- **Fallback configuration** when config manager is unavailable

## 🚀 Quick Start

### 1. Interactive Setup (Recommended)

Run the interactive setup wizard to configure everything:

```bash
# Complete guided setup
python setup_configuration.py

# Setup for specific environment
python setup_configuration.py --env production
```

The wizard will guide you through:
- ✅ API keys (OpenAI, Google Cloud, YouTube)
- ✅ Pipeline configuration (budgets, timeouts)
- ✅ Quality settings (TTS voice, thresholds)
- ✅ Security settings (privacy, encryption)
- ✅ Monitoring setup (dashboard, alerts)

### 2. Manual Configuration

Create configuration files manually:

```bash
# Create sample environment file
python config_manager.py --create-env --env development

# Edit the created .env file with your settings
# Then validate configuration
python config_manager.py --validate
```

## 📁 File Structure

```
project_root/
├── config/
│   ├── config.development.json    # Development settings
│   ├── config.staging.json        # Staging settings
│   └── config.production.json     # Production settings
├── credentials/
│   ├── .encryption_key            # Encryption key (auto-generated)
│   ├── google_cloud.json          # Google Cloud service account
│   ├── youtube_credentials.json   # YouTube OAuth2 credentials
│   └── youtube_token.json         # YouTube access token (auto-generated)
├── .env                           # Main environment variables
├── .env.development              # Development-specific variables
├── .env.staging                  # Staging-specific variables
└── .env.production              # Production-specific variables
```

## 🔑 API Keys Setup

### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add to configuration: `OPENAI_API_KEY=sk-...`

### Google Cloud TTS
1. Create project at [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Text-to-Speech API
3. Create service account and download JSON credentials
4. Set path: `GOOGLE_CLOUD_CREDENTIALS_PATH=./credentials/google_cloud.json`

### YouTube API
1. Create project at [Google Developers Console](https://console.developers.google.com/)
2. Enable YouTube Data API v3
3. Create OAuth 2.0 credentials
4. Download credentials JSON file
5. Set path: `YOUTUBE_CREDENTIALS_PATH=./credentials/youtube_credentials.json`

## 🔧 Configuration Sections

### Database Configuration
```json
{
  "database": {
    "path": "./data/production.db",
    "timeout_seconds": 30,
    "backup_enabled": true,
    "wal_mode": true
  }
}
```

### API Configuration
```json
{
  "api": {
    "openai_api_key": "encrypted_key_here",
    "google_cloud_credentials_path": "./credentials/google_cloud.json",
    "youtube_client_secrets_path": "./credentials/youtube_credentials.json",
    "openai_requests_per_minute": 50,
    "youtube_uploads_per_day": 50
  }
}
```

### Pipeline Configuration
```json
{
  "pipeline": {
    "max_retry_attempts": 3,
    "stage_timeout_minutes": 60,
    "max_concurrent_productions": 3,
    "max_daily_productions": 20,
    "max_cost_per_story_usd": 5.0,
    "max_cost_per_session_usd": 25.0
  }
}
```

### Quality Configuration
```json
{
  "quality": {
    "tts_voice": "en-US-Chirp3-HD-Enceladus",
    "audio_bitrate": "192k",
    "min_scene_count": 20,
    "max_scene_count": 50,
    "min_character_count": 3
  }
}
```

### Security Configuration
```json
{
  "security": {
    "default_youtube_privacy": "private",
    "enable_content_filtering": true,
    "token_encryption_enabled": true,
    "session_timeout_minutes": 480
  }
}
```

### Monitoring Configuration
```json
{
  "monitoring": {
    "dashboard_enabled": true,
    "dashboard_port": 5000,
    "slack_webhook_url": "encrypted_webhook_here",
    "email_alerts_enabled": false,
    "alert_on_failure": true
  }
}
```

## 🔒 Security Features

### Automatic Encryption
- API keys are automatically encrypted using Fernet encryption
- Encryption key is derived from a password using PBKDF2
- Sensitive data is encrypted before storing in config files

### Environment Variables
- Sensitive data can be stored in environment variables
- Automatic fallback to environment variables if config unavailable
- Support for environment-specific .env files

### Secure Defaults
- YouTube videos default to "private" for manual review
- Content filtering enabled by default
- Token encryption enabled by default

## 📋 Usage in Generators

Import central configuration in your generator scripts:

```python
from central_config import config, get_paths, get_api_key, get_budget_controls

# Get project paths
paths = get_paths()
output_dir = paths['OUTPUT_DIR']
database_path = config.database_path

# Get API keys
openai_key = get_api_key('openai')
google_creds = config.google_cloud_credentials_path

# Get budget controls
budget = get_budget_controls()
max_cost = budget['max_cost_per_story_usd']

# Get quality settings
quality = config.get_quality_settings()
tts_voice = quality['tts_voice']

# Check environment
if config.is_development_mode():
    print("Development mode - verbose logging")
```

## 🛠️ Management Commands

### Validate Configuration
```bash
python config_manager.py --validate
```

### Set API Keys
```bash
python config_manager.py --set-api-key openai sk-your-key-here
python config_manager.py --set-api-key anthropic your-claude-key
```

### Environment Management
```bash
# Development environment
python config_manager.py --env development

# Production environment  
python config_manager.py --env production

# Create sample .env file
python config_manager.py --create-env --env production
```

### Show Configuration Summary
```bash
python config_manager.py --env production
```

## 🌍 Environment Configurations

### Development
- Debug mode enabled
- Verbose logging
- Lower budget limits
- Local file paths
- Relaxed security settings

### Staging
- Production-like settings
- Moderate logging
- Testing budget limits
- Content filtering enabled
- Private YouTube uploads

### Production
- Optimized for performance
- Error-only logging
- Full budget limits
- Maximum security
- Encrypted credentials

## 🔧 Troubleshooting

### Common Issues

**Config Manager not found:**
```bash
# Ensure config files are in the right location
ls -la config_manager.py central_config.py

# Check Python path
python -c "import sys; print(sys.path)"
```

**API Key validation fails:**
```bash
# Test API key manually
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

**Database path issues:**
```bash
# Check database path
python -c "from central_config import config; print(config.database_path)"

# Create database directory
mkdir -p data
```

**Permission errors:**
```bash
# Fix credential file permissions
chmod 600 credentials/*
chmod 700 credentials/
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Set debug mode in environment
export DEBUG_MODE=true
export LOG_LEVEL=DEBUG

# Or in .env file
echo "DEBUG_MODE=true" >> .env
echo "LOG_LEVEL=DEBUG" >> .env
```

## 📊 Integration with Other Components

### Master Orchestrator
The Master Orchestrator automatically uses centralized configuration:
```python
# Orchestrator uses config for all pipeline settings
python master_orchestrator.py --mode auto
```

### Database Setup
Database initialization uses configuration settings:
```python
# Database setup reads config for paths and settings
python database_setup.py
```

### Production Dashboard
Dashboard uses monitoring configuration:
```python
# Dashboard reads port and security settings from config
python production_dashboard_server.py
```

## 🎯 Best Practices

### Security
1. **Never commit API keys to version control**
2. **Use environment variables in production**
3. **Enable token encryption in production**
4. **Set secure file permissions (600) on credential files**
5. **Use strong encryption passwords**

### Environment Management
1. **Use separate configurations for each environment**
2. **Test configuration changes in staging first**
3. **Validate configuration after changes**
4. **Keep backups of working configurations**

### API Key Management
1. **Rotate API keys regularly**
2. **Use least-privilege access policies**
3. **Monitor API usage and costs**
4. **Set appropriate rate limits**

### Monitoring
1. **Enable monitoring dashboard in production**
2. **Set up failure alerts**
3. **Monitor budget usage regularly**
4. **Review logs periodically**

## 🆘 Support

If you encounter issues:

1. **Run configuration validation:**
   ```bash
   python config_manager.py --validate
   ```

2. **Check configuration summary:**
   ```bash
   python config_manager.py --env production
   ```

3. **Review logs:**
   ```bash
   tail -f logs/orchestrator/master_orchestrator_*.log
   ```

4. **Test individual components:**
   ```bash
   python central_config.py  # Test config loading
   ```

For additional help, check the main project documentation or create an issue with your configuration details (without sensitive information).