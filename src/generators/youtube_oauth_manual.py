#!/usr/bin/env python3
"""
YouTube OAuth Manual Flow for Server (No Browser)
Headless OAuth flow for Ubuntu server without GUI
"""

import json
import os
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


def manual_oauth_flow():
    """Manual OAuth flow without browser"""

    # Paths
    project_root = Path(__file__).parent.parent.parent
    credentials_dir = project_root / 'credentials'
    oauth_file = credentials_dir / 'youtube_credentials1.json'
    token_file = credentials_dir / 'youtube_token.json'

    print("🔐 YouTube OAuth Manual Flow (No Browser)")
    print("=" * 50)
    print(f"📁 Project: {project_root}")
    print(f"🔑 OAuth file: {oauth_file}")
    print(f"💾 Token file: {token_file}")

    if not oauth_file.exists():
        print(f"❌ OAuth file not found: {oauth_file}")
        return False

    try:
        # Setup OAuth flow
        SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
        flow = InstalledAppFlow.from_client_secrets_file(str(oauth_file), SCOPES)

        # MANUAL FLOW - No automatic browser
        print("\n🌐 Manual OAuth Process:")
        print("1. Copy the URL below")
        print("2. Open it in YOUR browser (laptop/phone)")
        print("3. Authorize the application")
        print("4. Copy the authorization code back here")

        # Get authorization URL
        flow.redirect_uri = 'http://localhost'   # Out-of-band flow
        auth_url, _ = flow.authorization_url(prompt='consent')

        print(f"\n🔗 AUTHORIZATION URL:")
        print("=" * 50)
        print(auth_url)
        print("=" * 50)

        # Get authorization code from user
        print("\n📝 After authorizing, you'll get an authorization code")
        auth_code = input("🔑 Enter the authorization code: ").strip()

        if not auth_code:
            print("❌ No authorization code provided")
            return False

        # Exchange code for token
        print("🔄 Exchanging code for token...")
        flow.fetch_token(code=auth_code)

        creds = flow.credentials

        # Save token
        with open(token_file, 'w') as f:
            f.write(creds.to_json())

        print(f"✅ Token saved successfully: {token_file}")

        # Test the token
        print("\n🧪 Testing YouTube API access...")
        from googleapiclient.discovery import build

        service = build('youtube', 'v3', credentials=creds)
        request = service.channels().list(part='snippet', mine=True)
        response = request.execute()

        if response['items']:
            channel = response['items'][0]['snippet']
            print(f"✅ YouTube API test successful!")
            print(f"📺 Channel: {channel['title']}")
            print(f"🆔 Channel ID: {response['items'][0]['id']}")
        else:
            print("⚠️  No YouTube channel found")

        print("\n🎉 OAuth setup complete!")
        return True

    except Exception as e:
        print(f"❌ OAuth flow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_existing_token():
    """Check if token already exists and is valid"""
    project_root = Path(__file__).parent.parent.parent
    token_file = project_root / 'credentials' / 'youtube_token.json'

    if not token_file.exists():
        return False

    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        # Load existing token
        creds = Credentials.from_authorized_user_file(str(token_file))

        # Refresh if needed
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                print("🔄 Refreshing token...")
                creds.refresh(Request())

                # Save refreshed token
                with open(token_file, 'w') as f:
                    f.write(creds.to_json())
                print("✅ Token refreshed")
            else:
                print("❌ Token invalid, need new authorization")
                return False

        # Test API
        print("🧪 Testing existing token...")
        service = build('youtube', 'v3', credentials=creds)
        request = service.channels().list(part='snippet', mine=True)
        response = request.execute()

        if response['items']:
            channel = response['items'][0]['snippet']
            print(f"✅ Existing token works!")
            print(f"📺 Channel: {channel['title']}")
            return True
        else:
            print("⚠️  Token works but no channel found")
            return True

    except Exception as e:
        print(f"❌ Token test failed: {e}")
        return False


def main():
    print("🚀 YouTube OAuth Setup for Server")
    print("=" * 40)

    # Check if token already exists
    if check_existing_token():
        print("\n✅ You already have a working YouTube token!")
        print("🎯 Ready to upload videos")
        return

    print("\n🔑 Need to create YouTube authorization token")

    choice = input("▶️  Run manual OAuth flow? (y/N): ").strip().lower()
    if choice == 'y':
        success = manual_oauth_flow()

        if success:
            print("\n🎉 SUCCESS!")
            print("✅ YouTube OAuth setup complete")
            print("🚀 Ready to upload videos")
            print("\n🎯 Next step:")
            print("python3 src/generators/7_youtube_uploader_autonomous.py")
        else:
            print("\n❌ OAuth setup failed")
            print("🔧 Try again or check credentials")
    else:
        print("⏸️  OAuth setup cancelled")


if __name__ == "__main__":
    main()