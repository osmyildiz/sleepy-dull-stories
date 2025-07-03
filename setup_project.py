#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path


def create_minimal_project():
    """Minimal proje yapisini olustur"""

    # Ana klasör
    project_dir = Path("sleepy-dull-stories")
    project_dir.mkdir(exist_ok=True)

    # Temel klasörler
    (project_dir / "src").mkdir(exist_ok=True)
    (project_dir / "data").mkdir(exist_ok=True)
    (project_dir / "logs").mkdir(exist_ok=True)

    # config.py
    config_content = '''import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'

CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

def create_directories():
    DATA_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    print("Directories created")

def validate_config():
    if not CLAUDE_API_KEY:
        raise ValueError("CLAUDE_API_KEY missing")
    print("Config valid")
'''

    # story_generator.py
    story_content = '''import asyncio
from anthropic import Anthropic
from src.config import CLAUDE_API_KEY

class StoryGenerator:
    def __init__(self):
        self.client = Anthropic(api_key=CLAUDE_API_KEY)
        print("Claude client ready")

    async def test_connection(self):
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=100,
                messages=[{"role": "user", "content": "Say: API works"}]
            )
            print("Claude API: OK")
            return True
        except Exception as e:
            print("Claude API error: " + str(e))
            return False

    async def generate_story(self, topic):
        prompt = "Write a boring 2-hour sleep story about " + topic
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"story": response.content[0].text, "topic": topic}

async def test():
    gen = StoryGenerator()
    await gen.test_connection()

if __name__ == "__main__":
    asyncio.run(test())
'''

    # main.py
    main_content = '''#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src import config
from src.story_generator import StoryGenerator

async def main():
    print("Sleepy Dull Stories")

    try:
        config.validate_config()
        config.create_directories()

        gen = StoryGenerator()
        await gen.test_connection()

        story = await gen.generate_story("Ancient Rome")
        print("Story generated: " + str(len(story['story'])) + " characters")

    except Exception as e:
        print("Error: " + str(e))
        print("Please check your .env file")

if __name__ == "__main__":
    asyncio.run(main())
'''

    # .env template
    env_content = '''CLAUDE_API_KEY=your_claude_key_here
OPENAI_API_KEY=your_openai_key_here
RUNWAY_API_KEY=your_runway_key_here
'''

    # requirements.txt
    req_content = '''python-dotenv==1.0.0
anthropic==0.3.11
openai==1.3.8
asyncio
'''

    # test.py
    test_content = '''#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src import config

def test_setup():
    print("Testing setup...")

    try:
        config.validate_config()
        print("Config OK")
    except Exception as e:
        print("Config error: " + str(e))
        return False

    try:
        config.create_directories()
        print("Directories OK")
    except Exception as e:
        print("Directory error: " + str(e))
        return False

    print("Basic setup working")
    return True

if __name__ == "__main__":
    test_setup()
'''

    # Dosyalari yaz
    (project_dir / "src" / "config.py").write_text(config_content, encoding='utf-8')
    (project_dir / "src" / "story_generator.py").write_text(story_content, encoding='utf-8')
    (project_dir / "src" / "__init__.py").write_text("", encoding='utf-8')
    (project_dir / "main.py").write_text(main_content, encoding='utf-8')
    (project_dir / ".env.template").write_text(env_content, encoding='utf-8')
    (project_dir / "requirements.txt").write_text(req_content, encoding='utf-8')
    (project_dir / "test.py").write_text(test_content, encoding='utf-8')

    print("Project created: " + str(project_dir))
    print("\nNext steps:")
    print("1. cd sleepy-dull-stories")
    print("2. pip install -r requirements.txt")
    print("3. cp .env.template .env")
    print("4. Edit .env with your API keys")
    print("5. python test.py")
    print("6. python main.py")


if __name__ == "__main__":
    create_minimal_project()