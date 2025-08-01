#!/usr/bin/env python3
"""
YouTube Authentication Checker & Setup
Kontrol eder ve eksik olanları düzeltir
Project structure: src/generators/youtube_auth_checker.py
"""

import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()


class YouTubeAuthChecker:
    def __init__(self):
        # Current file: src/generators/youtube_auth_checker.py
        self.current_file = Path(__file__).resolve()
        self.generators_dir = self.current_file.parent  # src/generators/
        self.src_dir = self.generators_dir.parent  # src/
        self.project_root = self.src_dir.parent  # project root

        self.credentials_dir = self.project_root / 'credentials'
        self.setup_paths()

    def setup_paths(self):
        """Setup paths based on project structure"""
        # Current file: src/generators/youtube_auth_checker.py
        # Project root: sleepy-dull-stories/
        # Credentials: sleepy-dull-stories/credentials/ (ROOT LEVEL)

        self.credentials_dir = self.project_root / 'credentials'  # ROOT level credentials

        # Create credentials directory if needed
        self.credentials_dir.mkdir(exist_ok=True)

        # Check existing files in credentials
        existing_files = list(self.credentials_dir.glob('*.json'))

        print(f"📁 Project structure:")
        print(f"   📄 Current script: {self.current_file}")
        print(f"   📁 Project root: {self.project_root}")
        print(f"   📁 Credentials dir: {self.credentials_dir}")
        print(f"   📄 Existing files: {[f.name for f in existing_files]}")

        # File paths - check existing names
        possible_oauth_names = [
            'youtube_client_secret.json',
            'youtube_credentials.json',
            'client_secret.json'
        ]

        self.oauth_file = None
        for name in possible_oauth_names:
            candidate = self.credentials_dir / name
            if candidate.exists():
                self.oauth_file = candidate
                print(f"✅ Found OAuth file: {name}")
                break

        if not self.oauth_file:
            # Default to standard name
            self.oauth_file = self.credentials_dir / 'youtube_credentials.json'
            print(f"⚠️  OAuth file not found, will look for: youtube_credentials.json")

        self.token_file = self.credentials_dir / 'youtube_token.json'

        print(f"🔐 OAuth file path: {self.oauth_file}")
        print(f"🔑 Token file path: {self.token_file}")

    def check_environment_variables(self):
        """Check environment variables"""
        print("\n🔍 Checking Environment Variables")
        print("=" * 40)

        # Google Application Credentials (Service Account)
        google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if google_creds:
            print(f"📄 GOOGLE_APPLICATION_CREDENTIALS: {google_creds}")

            # Check if file exists
            if Path(google_creds).exists():
                print("✅ Service account credentials file found")

                # Load and check
                try:
                    with open(google_creds, 'r') as f:
                        creds = json.load(f)

                    if 'type' in creds and creds['type'] == 'service_account':
                        print(f"✅ Valid service account credentials")
                        print(f"📧 Service account email: {creds.get('client_email', 'N/A')}")
                        print(f"🆔 Project ID: {creds.get('project_id', 'N/A')}")
                    else:
                        print("⚠️  Not a service account credential")

                except Exception as e:
                    print(f"❌ Error reading service account credentials: {e}")
            else:
                print(f"❌ Service account credentials file not found: {google_creds}")
        else:
            print("⚠️  GOOGLE_APPLICATION_CREDENTIALS not set")

        # Update path for server
        print(f"\n🔧 Server Path Suggestions:")
        print(f"📝 Current local project: {self.project_root}")

        # Server path mapping
        server_project_root = "/home/youtube-automation/channels/sleepy-dull-stories"

        print(f"📝 Server project root: {server_project_root}")

        if google_creds:
            # Convert local path to server path
            if str(self.project_root) in google_creds:
                server_creds_path = google_creds.replace(str(self.project_root), server_project_root)
            else:
                server_creds_path = f"{server_project_root}/credentials/service-account.json"

            print(f"📝 Server .env should have:")
            print(f"GOOGLE_APPLICATION_CREDENTIALS={server_creds_path}")

        return google_creds

    def check_youtube_packages(self):
        """Check required Python packages"""
        print("\n🐍 Checking Python Packages")
        print("=" * 40)

        required_packages = {
            'google-api-python-client': 'googleapiclient',
            'google-auth-oauthlib': 'google_auth_oauthlib',
            'google-auth-httplib2': 'google.auth.transport.requests'
        }

        missing = []
        for package_name, import_name in required_packages.items():
            try:
                if import_name == 'googleapiclient':
                    import googleapiclient.discovery
                elif import_name == 'google_auth_oauthlib':
                    import google_auth_oauthlib.flow
                elif import_name == 'google.auth.transport.requests':
                    import google.auth.transport.requests

                print(f"✅ {package_name}")
            except ImportError:
                print(f"❌ {package_name}")
                missing.append(package_name)

        if missing:
            print(f"\n📦 Install missing packages:")
            print(f"pip install {' '.join(missing)}")
            return False
        else:
            print(f"✅ All packages installed")
            return True

    def check_oauth_credentials(self):
        """Check OAuth 2.0 credentials for YouTube upload"""
        print("\n🔐 Checking YouTube OAuth Credentials")
        print("=" * 40)

        # Check for existing file
        if self.oauth_file and self.oauth_file.exists():
            print(f"✅ Found OAuth file: {self.oauth_file.name}")
        else:
            print(f"❌ OAuth credentials not found")
            print(f"📁 Looking for files in: {self.credentials_dir}")

            # List all JSON files in credentials
            json_files = list(self.credentials_dir.glob('*.json'))
            if json_files:
                print(f"📄 Found JSON files: {[f.name for f in json_files]}")

                # Check if youtube_client_secret.json is the OAuth file
                client_secret_file = self.credentials_dir / 'youtube_client_secret.json'
                if client_secret_file.exists():
                    print(f"🔍 Checking youtube_client_secret.json...")
                    self.oauth_file = client_secret_file
                else:
                    print("\n🔧 You need OAuth 2.0 credentials for YouTube upload:")
                    print("1. Go to: https://console.cloud.google.com/")
                    print("2. Select your project")
                    print("3. APIs & Services > Credentials")
                    print("4. Create Credentials > OAuth 2.0 Client ID")
                    print("5. Application type: Desktop Application")
                    print("6. Download JSON file")
                    print(f"7. Save as: {self.credentials_dir}/youtube_credentials.json")
                    return False
            else:
                print("📄 No JSON files found in credentials directory")
                return False

        # Validate OAuth file
        try:
            with open(self.oauth_file, 'r') as f:
                oauth_data = json.load(f)

            # Check format - could be 'installed' or 'web'
            if 'installed' in oauth_data:
                client_info = oauth_data['installed']
                print("✅ Desktop application OAuth credentials")
            elif 'web' in oauth_data:
                client_info = oauth_data['web']
                print("⚠️  Web application OAuth credentials (Desktop preferred)")
            else:
                print("❌ Unknown OAuth format")
                print(f"🔍 Available keys: {list(oauth_data.keys())}")
                return False

            required_fields = ['client_id', 'client_secret', 'auth_uri', 'token_uri']
            missing_fields = [field for field in required_fields if field not in client_info]

            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
                return False
            else:
                print("✅ OAuth credentials format is correct")
                print(f"🆔 Client ID: {client_info['client_id'][:20]}...")
                return True

        except json.JSONDecodeError:
            print("❌ Invalid JSON format")
            return False
        except Exception as e:
            print(f"❌ Error reading OAuth credentials: {e}")
            return False

    def check_token_file(self):
        """Check authorization token"""
        print("\n🔑 Checking Authorization Token")
        print("=" * 40)

        if not self.token_file.exists():
            print(f"❌ Authorization token not found: {self.token_file.name}")
            print("🔧 Need to run authorization flow")
            return False

        try:
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)

            required_fields = ['token', 'refresh_token', 'client_id', 'client_secret']
            missing_fields = [field for field in required_fields if field not in token_data]

            if missing_fields:
                print(f"❌ Token missing fields: {missing_fields}")
                return False
            else:
                print("✅ Authorization token exists and valid format")

                # Check if token is expired
                if 'expiry' in token_data:
                    from datetime import datetime
                    try:
                        expiry = datetime.fromisoformat(token_data['expiry'].replace('Z', '+00:00'))
                        now = datetime.now(expiry.tzinfo)

                        if expiry < now:
                            print("⚠️  Token expired but has refresh token")
                        else:
                            print("✅ Token is still valid")
                    except:
                        print("⚠️  Could not parse expiry date")

                return True

        except Exception as e:
            print(f"❌ Error reading token: {e}")
            return False

    def test_youtube_api(self):
        """Test YouTube API access"""
        print("\n📺 Testing YouTube API Access")
        print("=" * 40)

        try:
            from googleapiclient.discovery import build
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request

            # Load credentials
            if not self.token_file.exists():
                print("❌ No authorization token found")
                return False

            creds = Credentials.from_authorized_user_file(str(self.token_file))

            # Refresh if expired
            if not creds.valid:
                if creds.expired and creds.refresh_token:
                    print("🔄 Refreshing expired token...")
                    creds.refresh(Request())

                    # Save refreshed token
                    with open(self.token_file, 'w') as token:
                        token.write(creds.to_json())
                    print("✅ Token refreshed and saved")
                else:
                    print("❌ Token invalid and cannot refresh")
                    return False

            # Test API
            print("🧪 Testing YouTube API...")
            service = build('youtube', 'v3', credentials=creds)

            # Get channel info
            request = service.channels().list(part='snippet', mine=True)
            response = request.execute()

            if response['items']:
                channel = response['items'][0]['snippet']
                print(f"✅ YouTube API access successful!")
                print(f"📺 Channel: {channel['title']}")
                print(f"🆔 Channel ID: {response['items'][0]['id']}")
                return True
            else:
                print("⚠️  No YouTube channel found for this account")
                return False

        except Exception as e:
            print(f"❌ YouTube API test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_oauth_flow(self):
        """Run OAuth 2.0 authorization flow"""
        print("\n🔐 Running OAuth 2.0 Authorization Flow")
        print("=" * 40)

        if not self.oauth_file or not self.oauth_file.exists():
            print("❌ OAuth credentials file required first")
            return False

        try:
            from google_auth_oauthlib.flow import InstalledAppFlow

            SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

            print("🌐 Starting OAuth flow...")
            flow = InstalledAppFlow.from_client_secrets_file(str(self.oauth_file), SCOPES)

            print("⚠️  This will open a browser window")
            print("📝 Please authorize the application")

            input("Press Enter to continue...")

            # Run OAuth flow
            creds = flow.run_local_server(port=0)

            # Save token
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())

            print(f"✅ Authorization successful!")
            print(f"💾 Token saved to: {self.token_file}")
            return True

        except Exception as e:
            print(f"❌ OAuth flow failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_server_env_file(self):
        """Create .env file for server"""
        print("\n📝 Creating Server Environment File")
        print("=" * 40)

        env_file = self.project_root / '.env.server'

        # Get current Google credentials path
        google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')

        # Server project mapping
        server_root = '/home/youtube-automation/channels/sleepy-dull-stories'

        # Convert paths
        if google_creds and str(self.project_root) in google_creds:
            server_creds = google_creds.replace(str(self.project_root), server_root)
        else:
            server_creds = f'{server_root}/credentials/service-account.json'

        env_content = f'''# Server Environment Configuration
# YouTube Automation Server
# Generated from: {self.current_file}

# Google Service Account (for other APIs if needed)
GOOGLE_APPLICATION_CREDENTIALS={server_creds}

# YouTube API uses OAuth2 files in credentials/ directory:
# - credentials/youtube_credentials.json (or youtube_client_secret.json)
# - credentials/youtube_token.json

# Project paths
PROJECT_ROOT={server_root}
CREDENTIALS_DIR={server_root}/credentials
OUTPUT_DIR={server_root}/output
DATA_DIR={server_root}/data

# Other API keys (add as needed)
# PIAPI_KEY=your_piapi_key_here
# OPENAI_API_KEY=your_openai_key_here
'''

        with open(env_file, 'w') as f:
            f.write(env_content)

        print(f"✅ Server .env file created: {env_file}")
        print(f"🔧 Copy this to your server as .env")

    def run_full_check(self):
        """Run complete authentication check"""
        print("🔐 YouTube Authentication Complete Check")
        print("=" * 50)

        # Check environment
        self.check_environment_variables()

        # Check packages
        packages_ok = self.check_youtube_packages()
        if not packages_ok:
            print("\n❌ Install missing packages first")
            return False

        # Check OAuth credentials
        oauth_ok = self.check_oauth_credentials()

        # Check token
        token_ok = self.check_token_file()

        # Test API if both exist
        if oauth_ok and token_ok:
            api_ok = self.test_youtube_api()
        else:
            api_ok = False

        # Summary
        print("\n📊 AUTHENTICATION STATUS SUMMARY")
        print("=" * 50)
        print(f"🐍 Python packages: {'✅' if packages_ok else '❌'}")
        print(f"🔐 OAuth credentials: {'✅' if oauth_ok else '❌'}")
        print(f"🔑 Authorization token: {'✅' if token_ok else '❌'}")
        print(f"📺 YouTube API access: {'✅' if api_ok else '❌'}")

        if all([packages_ok, oauth_ok, token_ok, api_ok]):
            print("\n🎉 ALL CHECKS PASSED!")
            print("✅ Ready for YouTube uploads")

            # Create server env file
            self.create_server_env_file()
            return True
        else:
            print("\n🔧 FIXES NEEDED:")
            if not oauth_ok:
                print("1. OAuth 2.0 credentials needed")
                if self.oauth_file and self.oauth_file.exists():
                    print(f"   Found: {self.oauth_file.name} - checking format...")
                else:
                    print("   Create Desktop Application OAuth in Google Cloud Console")

            if oauth_ok and not token_ok:
                print("2. Authorization token needed")
                choice = input("\n▶️  Run OAuth flow now? (y/N): ").strip().lower()
                if choice == 'y':
                    if self.run_oauth_flow():
                        return self.test_youtube_api()

            return False


def main():
    print("🚀 YouTube Authentication Checker")
    print("📁 Running from: src/generators/")
    print("=" * 50)

    checker = YouTubeAuthChecker()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == '--oauth':
            # Just run OAuth flow
            checker.run_oauth_flow()
        elif command == '--test':
            # Just test API
            checker.test_youtube_api()
        elif command == '--env':
            # Just create server env
            checker.create_server_env_file()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python3 src/generators/youtube_auth_checker.py [--oauth|--test|--env]")
    else:
        # Full check
        success = checker.run_full_check()

        if success:
            print("\n🎯 NEXT STEPS:")
            print("1. Copy .env.server to your server as .env")
            print("2. Copy credentials/ directory to server")
            print("3. Test upload: python3 src/generators/7_youtube_uploader_autonomous.py")
        else:
            print("\n🔧 Fix the issues above and run again")


if __name__ == "__main__":
    main()


    def check_environment_variables(self):
        """Check environment variables"""
        print("\n🔍 Checking Environment Variables")
        print("=" * 40)

        # Google Application Credentials (Service Account)
        google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if google_creds:
            print(f"📄 GOOGLE_APPLICATION_CREDENTIALS: {google_creds}")

            # Check if file exists
            if Path(google_creds).exists():
                print("✅ Service account credentials file found")

                # Load and check
                try:
                    with open(google_creds, 'r') as f:
                        creds = json.load(f)

                    if 'type' in creds and creds['type'] == 'service_account':
                        print(f"✅ Valid service account credentials")
                        print(f"📧 Service account email: {creds.get('client_email', 'N/A')}")
                        print(f"🆔 Project ID: {creds.get('project_id', 'N/A')}")
                    else:
                        print("⚠️  Not a service account credential")

                except Exception as e:
                    print(f"❌ Error reading service account credentials: {e}")
            else:
                print(f"❌ Service account credentials file not found: {google_creds}")
        else:
            print("⚠️  GOOGLE_APPLICATION_CREDENTIALS not set")

        # Update path for server
        print(f"\n🔧 Server Path Suggestion:")
        if google_creds:
            # Convert local path to server path
            server_path = google_creds.replace('/Users/nilgun/PycharmProjects/', '/home/youtube-automation/channels/')
            print(f"📝 Server .env should have:")
            print(f"GOOGLE_APPLICATION_CREDENTIALS={server_path}")

        return google_creds


    def check_youtube_packages(self):
        """Check required Python packages"""
        print("\n🐍 Checking Python Packages")
        print("=" * 40)

        required_packages = {
            'google-api-python-client': 'googleapiclient',
            'google-auth-oauthlib': 'google_auth_oauthlib',
            'google-auth-httplib2': 'google.auth.transport.requests'
        }

        missing = []
        for package_name, import_name in required_packages.items():
            try:
                if import_name == 'googleapiclient':
                    import googleapiclient.discovery
                elif import_name == 'google_auth_oauthlib':
                    import google_auth_oauthlib.flow
                elif import_name == 'google.auth.transport.requests':
                    import google.auth.transport.requests

                print(f"✅ {package_name}")
            except ImportError:
                print(f"❌ {package_name}")
                missing.append(package_name)

        if missing:
            print(f"\n📦 Install missing packages:")
            print(f"pip install {' '.join(missing)}")
            return False
        else:
            print(f"✅ All packages installed")
            return True


    def check_oauth_credentials(self):
        """Check OAuth 2.0 credentials for YouTube upload"""
        print("\n🔐 Checking YouTube OAuth Credentials")
        print("=" * 40)

        if not self.oauth_file.exists():
            print(f"❌ OAuth credentials not found: {self.oauth_file}")
            print("\n🔧 You need to create OAuth 2.0 credentials:")
            print("1. Go to: https://console.cloud.google.com/")
            print("2. Select your project")
            print("3. APIs & Services > Credentials")
            print("4. Create Credentials > OAuth 2.0 Client ID")
            print("5. Application type: Desktop Application")
            print("6. Download JSON file")
            print(f"7. Save as: {self.oauth_file}")
            return False

        # Validate OAuth file
        try:
            with open(self.oauth_file, 'r') as f:
                oauth_data = json.load(f)

            if 'installed' in oauth_data:
                client_info = oauth_data['installed']
                required_fields = ['client_id', 'client_secret', 'auth_uri', 'token_uri']

                missing_fields = [field for field in required_fields if field not in client_info]

                if missing_fields:
                    print(f"❌ Missing fields: {missing_fields}")
                    return False
                else:
                    print("✅ OAuth credentials format is correct")
                    print(f"🆔 Client ID: {client_info['client_id'][:20]}...")
                    return True
            else:
                print("❌ Invalid OAuth format - should be Desktop Application")
                return False

        except json.JSONDecodeError:
            print("❌ Invalid JSON format")
            return False
        except Exception as e:
            print(f"❌ Error reading OAuth credentials: {e}")
            return False


    def check_token_file(self):
        """Check authorization token"""
        print("\n🔑 Checking Authorization Token")
        print("=" * 40)

        if not self.token_file.exists():
            print(f"❌ Authorization token not found: {self.token_file}")
            print("🔧 Need to run authorization flow")
            return False

        try:
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)

            required_fields = ['token', 'refresh_token', 'client_id', 'client_secret']
            missing_fields = [field for field in required_fields if field not in token_data]

            if missing_fields:
                print(f"❌ Token missing fields: {missing_fields}")
                return False
            else:
                print("✅ Authorization token exists and valid format")

                # Check if token is expired
                if 'expiry' in token_data:
                    from datetime import datetime
                    expiry = datetime.fromisoformat(token_data['expiry'].replace('Z', '+00:00'))
                    now = datetime.now(expiry.tzinfo)

                    if expiry < now:
                        print("⚠️  Token expired but has refresh token")
                    else:
                        print("✅ Token is still valid")

                return True

        except Exception as e:
            print(f"❌ Error reading token: {e}")
            return False


    def test_youtube_api(self):
        """Test YouTube API access"""
        print("\n📺 Testing YouTube API Access")
        print("=" * 40)

        try:
            from googleapiclient.discovery import build
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request

            # Load credentials
            if not self.token_file.exists():
                print("❌ No authorization token found")
                return False

            creds = Credentials.from_authorized_user_file(str(self.token_file))

            # Refresh if expired
            if not creds.valid:
                if creds.expired and creds.refresh_token:
                    print("🔄 Refreshing expired token...")
                    creds.refresh(Request())

                    # Save refreshed token
                    with open(self.token_file, 'w') as token:
                        token.write(creds.to_json())
                    print("✅ Token refreshed and saved")
                else:
                    print("❌ Token invalid and cannot refresh")
                    return False

            # Test API
            print("🧪 Testing YouTube API...")
            service = build('youtube', 'v3', credentials=creds)

            # Get channel info
            request = service.channels().list(part='snippet', mine=True)
            response = request.execute()

            if response['items']:
                channel = response['items'][0]['snippet']
                print(f"✅ YouTube API access successful!")
                print(f"📺 Channel: {channel['title']}")
                print(f"🆔 Channel ID: {response['items'][0]['id']}")
                return True
            else:
                print("⚠️  No YouTube channel found for this account")
                return False

        except Exception as e:
            print(f"❌ YouTube API test failed: {e}")
            return False


    def run_oauth_flow(self):
        """Run OAuth 2.0 authorization flow"""
        print("\n🔐 Running OAuth 2.0 Authorization Flow")
        print("=" * 40)

        if not self.oauth_file.exists():
            print("❌ OAuth credentials file required first")
            return False

        try:
            from google_auth_oauthlib.flow import InstalledAppFlow

            SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

            print("🌐 Starting OAuth flow...")
            flow = InstalledAppFlow.from_client_secrets_file(str(self.oauth_file), SCOPES)

            print("⚠️  This will open a browser window")
            print("📝 Please authorize the application")

            input("Press Enter to continue...")

            # Run OAuth flow
            creds = flow.run_local_server(port=0)

            # Save token
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())

            print(f"✅ Authorization successful!")
            print(f"💾 Token saved to: {self.token_file}")
            return True

        except Exception as e:
            print(f"❌ OAuth flow failed: {e}")
            return False


    def create_server_env_file(self):
        """Create .env file for server"""
        print("\n📝 Creating Server Environment File")
        print("=" * 40)

        env_file = self.project_root / '.env.server'

        # Get current Google credentials path
        google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')

        # Convert to server path
        if 'nilgun' in google_creds:
            server_creds = google_creds.replace(
                '/Users/nilgun/PycharmProjects/',
                '/home/youtube-automation/channels/'
            )
        else:
            server_creds = '/home/youtube-automation/channels/sleepy-dull-stories/credentials/service-account.json'

        env_content = f'''# Server Environment Configuration
# YouTube Automation Server

# Google Service Account (for other APIs)
GOOGLE_APPLICATION_CREDENTIALS={server_creds}

# YouTube API (uses OAuth2 files in credentials/)
# OAuth files: credentials/youtube_credentials.json & youtube_token.json

# Other API keys (if needed)
# PIAPI_KEY=your_piapi_key_here
# OPENAI_API_KEY=your_openai_key_here
'''

        with open(env_file, 'w') as f:
            f.write(env_content)

        print(f"✅ Server .env file created: {env_file}")
        print(f"📝 Content preview:")
        print("─" * 40)
        print(env_content)
        print("─" * 40)
        print(f"🔧 Copy this to your server as .env")


    def run_full_check(self):
        """Run complete authentication check"""
        print("🔐 YouTube Authentication Complete Check")
        print("=" * 50)

        # Check environment
        self.check_environment_variables()

        # Check packages
        packages_ok = self.check_youtube_packages()
        if not packages_ok:
            print("\n❌ Install missing packages first")
            return False

        # Check OAuth credentials
        oauth_ok = self.check_oauth_credentials()

        # Check token
        token_ok = self.check_token_file()

        # Test API if both exist
        if oauth_ok and token_ok:
            api_ok = self.test_youtube_api()
        else:
            api_ok = False

        # Summary
        print("\n📊 AUTHENTICATION STATUS SUMMARY")
        print("=" * 50)
        print(f"🐍 Python packages: {'✅' if packages_ok else '❌'}")
        print(f"🔐 OAuth credentials: {'✅' if oauth_ok else '❌'}")
        print(f"🔑 Authorization token: {'✅' if token_ok else '❌'}")
        print(f"📺 YouTube API access: {'✅' if api_ok else '❌'}")

        if all([packages_ok, oauth_ok, token_ok, api_ok]):
            print("\n🎉 ALL CHECKS PASSED!")
            print("✅ Ready for YouTube uploads")

            # Create server env file
            self.create_server_env_file()
            return True
        else:
            print("\n🔧 FIXES NEEDED:")
            if not oauth_ok:
                print("1. Create OAuth 2.0 credentials in Google Cloud Console")
                print(f"   Save as: {self.oauth_file}")
            if not token_ok and oauth_ok:
                print("2. Run authorization flow")
                choice = input("\n▶️  Run OAuth flow now? (y/N): ").strip().lower()
                if choice == 'y':
                    if self.run_oauth_flow():
                        return self.test_youtube_api()

            return False


def main():
    print("🚀 YouTube Authentication Checker")
    print("=" * 50)

    checker = YouTubeAuthChecker()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == '--oauth':
            # Just run OAuth flow
            checker.run_oauth_flow()
        elif command == '--test':
            # Just test API
            checker.test_youtube_api()
        elif command == '--env':
            # Just create server env
            checker.create_server_env_file()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python youtube_auth_checker.py [--oauth|--test|--env]")
    else:
        # Full check
        success = checker.run_full_check()

        if success:
            print("\n🎯 NEXT STEPS:")
            print("1. Copy .env.server to your server as .env")
            print("2. Copy credentials/ directory to server")
            print("3. Test upload: python3 7_youtube_uploader_autonomous.py")
        else:
            print("\n🔧 Fix the issues above and run again")


if __name__ == "__main__":
    main()