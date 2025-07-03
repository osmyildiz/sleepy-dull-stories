#!/usr/bin/env python
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import config
from story_generator import StoryGenerator


def main():
    print("Sleepy Dull Stories")
    print("==================")

    try:
        config.validate_config()
        config.create_directories()

        gen = StoryGenerator()
        gen.test_connection()

        story = gen.generate_story("Ancient Rome")
        print("Story generated: " + str(len(story['story'])) + " characters")

        # Save story
        output_file = os.path.join(config.DATA_DIR, "story.txt")
        with open(output_file, 'w') as f:
            f.write(story['story'])
        print("Story saved to: " + output_file)

    except Exception as e:
        print("Error: " + str(e))
        print("Please check your .env file")


if __name__ == "__main__":
    main()