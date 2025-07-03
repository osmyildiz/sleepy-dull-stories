#!/usr/bin/env python3

import os
from pathlib import Path


def debug_env():
    print("=== ENV DEBUG ===")

    # 1. Dosya varlığı kontrolü
    current_dir = Path.cwd()
    env_file = current_dir / ".env"

    print(f"Current directory: {current_dir}")
    print(f"Looking for .env at: {env_file}")
    print(f".env exists: {env_file.exists()}")

    if env_file.exists():
        print(f".env file size: {env_file.stat().st_size} bytes")

        # Dosya içeriğini oku
        try:
            content = env_file.read_text(encoding='utf-8')
            print("\n=== .env CONTENT ===")
            print(repr(content))  # Raw content with special chars
            print("\n=== .env CONTENT (readable) ===")
            print(content)
        except Exception as e:
            print(f"Error reading .env: {e}")

    # 2. Environment variables kontrolü
    print("\n=== ENVIRONMENT VARIABLES ===")
    claude_key = os.getenv('CLAUDE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')

    print(f"CLAUDE_API_KEY: {claude_key[:10] + '...' if claude_key else 'NOT FOUND'}")
    print(f"OPENAI_API_KEY: {openai_key[:10] + '...' if openai_key else 'NOT FOUND'}")

    # 3. python-dotenv test
    try:
        from dotenv import load_dotenv
        print("\n=== DOTENV TEST ===")
        print("python-dotenv is installed")

        # Manual load
        result = load_dotenv(env_file)
        print(f"load_dotenv result: {result}")

        # Check again after loading
        claude_key_after = os.getenv('CLAUDE_API_KEY')
        print(
            f"CLAUDE_API_KEY after load_dotenv: {claude_key_after[:10] + '...' if claude_key_after else 'STILL NOT FOUND'}")

    except ImportError:
        print("\n❌ python-dotenv NOT INSTALLED")
        print("Run: pip install python-dotenv")
        return False

    # 4. Dosya format kontrolü
    if env_file.exists():
        print("\n=== FILE FORMAT CHECK ===")
        lines = env_file.read_text().strip().split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    print(f"Line {i}: '{key.strip()}' = '{value.strip()}'")

                    # Common issues check
                    if key.strip() != key:
                        print(f"  ⚠️ Key has spaces: '{key}'")
                    if value.startswith('"') and value.endswith('"'):
                        print(f"  ⚠️ Value has quotes: {value}")
                else:
                    print(f"Line {i}: Invalid format: '{line}'")

    return True


def create_correct_env():
    print("\n=== CREATING CORRECT .env ===")

    # Get API keys from user
    claude_key = input("Enter your CLAUDE_API_KEY: ").strip()
    openai_key = input("Enter your OPENAI_API_KEY (optional): ").strip()

    # Create correct format
    env_content = f"""CLAUDE_API_KEY={claude_key}
OPENAI_API_KEY={openai_key}
RUNWAY_API_KEY=your_runway_key_here
ENVIRONMENT=development
"""

    # Write file
    env_file = Path(".env")
    env_file.write_text(env_content, encoding='utf-8')
    print(f"✅ Created {env_file}")

    # Test immediately
    from dotenv import load_dotenv
    load_dotenv(env_file)

    test_key = os.getenv('CLAUDE_API_KEY')
    if test_key:
        print(f"✅ Test successful: {test_key[:10]}...")
        return True
    else:
        print("❌ Still not working")
        return False


def quick_test():
    print("\n=== QUICK TEST ===")

    # Test the config import
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path.cwd()))

        from src.config import CLAUDE_API_KEY
        print(f"✅ Config import works: {CLAUDE_API_KEY[:10] + '...' if CLAUDE_API_KEY else 'EMPTY'}")
        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False


if __name__ == "__main__":
    debug_result = debug_env()

    if not debug_result:
        print("\n🔧 Installing python-dotenv...")
        os.system("pip install python-dotenv")

    # If still problems, recreate .env
    test_key = os.getenv('CLAUDE_API_KEY')
    if not test_key:
        print("\n🔧 Creating new .env file...")
        create_correct_env()

    # Final test
    quick_test()