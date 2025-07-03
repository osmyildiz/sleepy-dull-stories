"""
Enhanced Visual Generator for Sleep Stories
ChatGPT gpt-image-1 API Pipeline with Character Reference System

WORKFLOW:
1. Find completed topics (done=1) from CSV
2. For each topic: Load character_profiles.json, scene_plan.json
3. Generate full-body character reference portraits first
4. Generate scene images using character references when characters are present
5. Generate thumbnail using character references from scene 99
6. Save all images in organized folder structure

FOLDER STRUCTURE:
src/output/{row_number}/
  ├── images/
  │   ├── characters/
  │   │   ├── character_name_1.png
  │   │   └── character_name_2.png
  │   ├── scenes/
  │   │   ├── scene_01.png
  │   │   ├── scene_02.png
  │   │   └── ...scene_40.png
  │   └── thumbnail.png
  └── visual_generation_log.json
"""

import os
import json
import pandas as pd
from pathlib import Path
import openai
from dotenv import load_dotenv
import time
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
from PIL import Image
import io

# Load environment variables
load_dotenv()


class VisualGenerator:
    def __init__(self):
        """Initialize Visual Generator with ChatGPT gpt-image-1 API"""

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env file")

        # Initialize OpenAI client
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

        # Configuration
        self.base_dir = Path("src/output")
        self.csv_path = Path("src/data/topics.csv")

        # Generation settings
        self.image_config = {
            "model": "gpt-image-1",
            "size": "1024x1536",     # Landscape for scenes
            "quality": "medium",
            "n": 1
        }

        self.character_config = {
            "model": "gpt-image-1",
            "size": "1024x1536",  # Portrait for full-body character references
            "quality": "medium",
            "n": 1
        }

        self.thumbnail_config = {
            "model": "gpt-image-1",
            "size": "1024x1536",  # YouTube thumbnail ratio
            "quality": "medium",
            "n": 1
        }

        # Statistics tracking
        self.stats = {
            "topics_processed": 0,
            "characters_generated": 0,
            "scenes_generated": 0,
            "thumbnails_generated": 0,
            "api_calls": 0,
            "errors": 0,
            "start_time": datetime.now()
        }

        # Generation log
        self.generation_log = []

        print("✅ Enhanced Visual Generator initialized with gpt-image-1")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📊 CSV path: {self.csv_path}")

    def log_step(self, message: str, status: str = "INFO", metadata: Dict = None):
        """Log generation steps with timestamps"""

        entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "status": status,
            "api_calls": self.stats["api_calls"]
        }

        if metadata:
            entry.update(metadata)

        self.generation_log.append(entry)

        # Print with appropriate icon
        icon = "🔄" if status == "INFO" else "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "⚠️"
        print(f"{icon} {message}")

        if metadata:
            for key, value in metadata.items():
                print(f"   📊 {key}: {value}")

    def get_completed_topics(self) -> List[Tuple[int, str, str]]:
        """Get topics with done=1 from CSV"""

        self.log_step("Loading completed topics from CSV")

        try:
            df = pd.read_csv(self.csv_path)
            completed = df[df["done"] == 1]

            topics = []
            for idx, row in completed.iterrows():
                row_number = idx + 1  # CSV row number (1-based)
                topic = row["topic"]
                description = row.get("description", "")
                topics.append((row_number, topic, description))

            self.log_step("Completed topics loaded", "SUCCESS", {
                "total_topics": len(df),
                "completed_topics": len(topics),
                "topics_list": [f"Row {r}: {t}" for r, t, d in topics]
            })

            return topics

        except Exception as e:
            self.log_step(f"Failed to load CSV: {e}", "ERROR")
            return []

    def load_topic_data(self, row_number: int) -> Dict:
        """Load all JSON files for a specific topic"""

        topic_dir = self.base_dir / str(row_number)

        if not topic_dir.exists():
            raise FileNotFoundError(f"Topic directory not found: {topic_dir}")

        # Required files
        required_files = [
            "character_profiles.json",
            "scene_plan.json"
        ]

        # Optional files
        optional_files = [
            "platform_metadata.json",
            "all_stories.json"
        ]

        data = {}

        # Load required files
        for filename in required_files:
            file_path = topic_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                data[filename.replace('.json', '')] = json.load(f)

        # Load optional files
        for filename in optional_files:
            file_path = topic_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data[filename.replace('.json', '')] = json.load(f)

        self.log_step(f"Topic data loaded", "SUCCESS", {
            "row_number": row_number,
            "files_loaded": list(data.keys()),
            "characters_count": len(data.get('character_profiles', {}).get('main_characters', [])),
            "scenes_count": len(data.get('scene_plan', []))
        })

        return data

    def image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_character_images(self, characters: List[Dict], output_dir: Path) -> Dict[str, str]:
        """Generate full-body portrait reference images for all characters"""

        self.log_step(f"Generating {len(characters)} character reference portraits")

        characters_dir = output_dir / "images" / "characters"
        characters_dir.mkdir(parents=True, exist_ok=True)

        character_files = {}

        for char in characters:
            try:
                char_name = char.get('name', 'Unknown')
                char_desc = char.get('physical_description', 'No description available')
                visual_notes = char.get('visual_notes', '')

                # Build character prompt for full-body portrait
                prompt = f"Full-body character reference portrait: {char_name}. {char_desc}"
                if visual_notes:
                    prompt += f" {visual_notes}"

                prompt += " Full-body standing pose, character reference sheet style, neutral background, professional portrait, high detail, clear lighting showing complete character design from head to toe."

                # Ensure prompt is under 4000 characters
                if len(prompt) > 3900:
                    prompt = prompt[:3900] + "..."

                self.log_step(f"Generating character portrait: {char_name}")

                # Generate image
                response = self.client.images.generate(
                    prompt=prompt,
                    **self.character_config
                )

                self.stats["api_calls"] += 1

                # Download and save image
                image_url = response.data[0].url
                image_response = requests.get(image_url)

                if image_response.status_code == 200:
                    # Clean filename
                    safe_name = "".join(c for c in char_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_name = safe_name.replace(' ', '_')
                    filename = f"{safe_name}.png"

                    file_path = characters_dir / filename

                    with open(file_path, 'wb') as f:
                        f.write(image_response.content)

                    character_files[char_name] = str(file_path)
                    self.stats["characters_generated"] += 1

                    self.log_step(f"Character portrait saved: {filename}", "SUCCESS")
                else:
                    self.log_step(f"Failed to download character image: {char_name}", "ERROR")

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                self.log_step(f"Error generating character {char_name}: {e}", "ERROR")
                self.stats["errors"] += 1
                continue

        self.log_step(f"Character portrait generation complete", "SUCCESS", {
            "characters_generated": len(character_files),
            "files": list(character_files.values())
        })

        return character_files

    def get_scene_characters(self, scene_id: int, scene_plan: List[Dict]) -> List[str]:
        """Get list of characters mentioned in a specific scene"""

        for scene in scene_plan:
            if scene.get('scene_id') == scene_id:
                return scene.get('characters_mentioned', [])
        return []

    def generate_scene_images(self, scene_plan: List[Dict], character_files: Dict[str, str],
                              output_dir: Path) -> List[str]:
        """Generate all scene images using character references when characters are present"""

        self.log_step(f"Generating {len(scene_plan)} scene images with character references")

        scenes_dir = output_dir / "images" / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)

        scene_files = []

        for scene_data in scene_plan:
            try:
                scene_id = scene_data.get('scene_id', 0)
                scene_title = scene_data.get('title', f'Scene {scene_id}')
                scene_description = scene_data.get('description', '')
                characters_mentioned = scene_data.get('characters_mentioned', [])

                self.log_step(f"Generating Scene {scene_id}: {scene_title}")

                # Build scene prompt
                prompt = f"Scene {scene_id}: {scene_title}. {scene_description}"

                # Add atmosphere and style
                location = scene_data.get('location', '')
                emotion = scene_data.get('emotion', 'peaceful')

                prompt += f" Location: {location}. Emotion: {emotion}. Ancient Roman setting, 79 AD Pompeii, cinematic composition, atmospheric lighting, high detail, professional illustration quality."

                # Prepare messages for API call
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]

                # Add character references if characters are present
                reference_images_used = []
                if characters_mentioned:
                    self.log_step(f"  Characters in scene: {', '.join(characters_mentioned)}")

                    for char_name in characters_mentioned:
                        if char_name in character_files:
                            char_image_path = character_files[char_name]
                            if os.path.exists(char_image_path):
                                # Add character reference to the message
                                base64_image = self.image_to_base64(char_image_path)
                                messages[0]["content"].append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                })
                                reference_images_used.append(char_name)

                if reference_images_used:
                    # Update prompt to reference the characters
                    char_refs = ", ".join(reference_images_used)
                    messages[0]["content"][0]["text"] += f" Use the provided character reference images for: {char_refs}. Maintain character consistency and physical appearance as shown in the references."

                # Ensure prompt is under 4000 characters
                if len(messages[0]["content"][0]["text"]) > 3900:
                    messages[0]["content"][0]["text"] = messages[0]["content"][0]["text"][:3900] + "..."

                # Generate image with character references
                response = self.client.images.generate(
                    prompt=messages[0]["content"][0]["text"],
                    **self.image_config
                )

                self.stats["api_calls"] += 1

                # Download and save image
                image_url = response.data[0].url
                image_response = requests.get(image_url)

                if image_response.status_code == 200:
                    filename = f"scene_{scene_id:02d}.png"
                    file_path = scenes_dir / filename

                    with open(file_path, 'wb') as f:
                        f.write(image_response.content)

                    scene_files.append(str(file_path))
                    self.stats["scenes_generated"] += 1

                    success_msg = f"Scene saved: {filename}"
                    if reference_images_used:
                        success_msg += f" (with character refs: {', '.join(reference_images_used)})"

                    self.log_step(success_msg, "SUCCESS")
                else:
                    self.log_step(f"Failed to download scene {scene_id}", "ERROR")

                # Rate limiting
                time.sleep(2)  # Longer delay for scenes

            except Exception as e:
                self.log_step(f"Error generating scene {scene_id}: {e}", "ERROR")
                self.stats["errors"] += 1
                continue

        self.log_step(f"Scene generation complete", "SUCCESS", {
            "scenes_generated": len(scene_files),
            "total_scenes": len(scene_plan)
        })

        return scene_files

    def generate_thumbnail(self, platform_metadata: Dict, topic: str, character_files: Dict[str, str],
                          scene_plan: List[Dict], output_dir: Path) -> Optional[str]:
        """Generate YouTube thumbnail using character references from scene 99 or main characters"""

        self.log_step("Generating YouTube thumbnail with character references")

        thumbnail_dir = output_dir / "images"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Look for scene 99 (thumbnail scene) in scene plan
            thumbnail_scene = None
            for scene in scene_plan:
                if scene.get('scene_id') == 99:
                    thumbnail_scene = scene
                    break

            # Build thumbnail prompt
            if thumbnail_scene:
                thumbnail_title = thumbnail_scene.get('title', topic)
                thumbnail_desc = thumbnail_scene.get('description', '')
                thumbnail_characters = thumbnail_scene.get('characters_mentioned', [])

                prompt = f"YouTube thumbnail: {thumbnail_title}. {thumbnail_desc}"
            else:
                # Fallback to platform metadata or generic
                thumbnail_concept = platform_metadata.get('thumbnail_concept', {})

                if thumbnail_concept:
                    main_char = thumbnail_concept.get('main_character', '')
                    dramatic_scene = thumbnail_concept.get('dramatic_scene', '')
                    text_overlay = thumbnail_concept.get('text_overlay', topic)
                    emotion = thumbnail_concept.get('emotion', 'peaceful')

                    prompt = f"YouTube thumbnail: {dramatic_scene} featuring {main_char}. {emotion} emotion. Large bold text '{text_overlay}'."
                    thumbnail_characters = [main_char] if main_char else []
                else:
                    prompt = f"YouTube thumbnail for '{topic}' sleep story. Peaceful, atmospheric, eye-catching design."
                    thumbnail_characters = []

            # Add thumbnail-specific instructions
            prompt += " Eye-catching YouTube thumbnail style, vibrant colors, high contrast, professional thumbnail design, click-worthy composition."

            # Prepare character references for thumbnail
            reference_images_used = []
            if thumbnail_characters:
                self.log_step(f"  Using character references for thumbnail: {', '.join(thumbnail_characters)}")

                for char_name in thumbnail_characters:
                    if char_name in character_files:
                        char_image_path = character_files[char_name]
                        if os.path.exists(char_image_path):
                            reference_images_used.append(char_name)

            # If we have character references, mention them in prompt
            if reference_images_used:
                char_refs = ", ".join(reference_images_used)
                prompt += f" Use the character appearance from reference images for: {char_refs}. Maintain character consistency."

            # Ensure prompt is under 4000 characters
            if len(prompt) > 3900:
                prompt = prompt[:3900] + "..."

            # Generate thumbnail
            response = self.client.images.generate(
                prompt=prompt,
                **self.thumbnail_config
            )

            self.stats["api_calls"] += 1

            # Download and save
            image_url = response.data[0].url
            image_response = requests.get(image_url)

            if image_response.status_code == 200:
                file_path = thumbnail_dir / "thumbnail.png"

                with open(file_path, 'wb') as f:
                    f.write(image_response.content)

                self.stats["thumbnails_generated"] += 1

                success_msg = "Thumbnail generated successfully"
                if reference_images_used:
                    success_msg += f" (with character refs: {', '.join(reference_images_used)})"

                self.log_step(success_msg, "SUCCESS", {
                    "file_path": str(file_path)
                })

                return str(file_path)
            else:
                self.log_step("Failed to download thumbnail", "ERROR")
                return None

        except Exception as e:
            self.log_step(f"Error generating thumbnail: {e}", "ERROR")
            self.stats["errors"] += 1
            return None

    def save_generation_log(self, output_dir: Path, topic: str, row_number: int):
        """Save detailed generation log"""

        log_data = {
            "topic": topic,
            "row_number": row_number,
            "generation_completed": datetime.now().isoformat(),
            "statistics": self.stats.copy(),
            "generation_log": self.generation_log,
            "file_structure": {
                "characters": list((output_dir / "images" / "characters").glob("*.png")) if (
                            output_dir / "images" / "characters").exists() else [],
                "scenes": list((output_dir / "images" / "scenes").glob("*.png")) if (
                            output_dir / "images" / "scenes").exists() else [],
                "thumbnail": str(output_dir / "images" / "thumbnail.png") if (
                            output_dir / "images" / "thumbnail.png").exists() else None
            }
        }

        # Convert Path objects to strings for JSON serialization
        for category in log_data["file_structure"]:
            if isinstance(log_data["file_structure"][category], list):
                log_data["file_structure"][category] = [str(p) for p in log_data["file_structure"][category]]

        log_path = output_dir / "visual_generation_log.json"

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        self.log_step(f"Generation log saved", "SUCCESS", {
            "log_path": str(log_path)
        })

    def process_topic(self, row_number: int, topic: str, description: str) -> bool:
        """Process a single topic - generate all visuals with character references"""

        self.log_step(f"Processing Topic {row_number}: {topic}", "INFO", {
            "description": description[:100] + "..." if len(description) > 100 else description
        })

        try:
            # Load topic data
            topic_data = self.load_topic_data(row_number)

            # Setup output directory
            output_dir = self.base_dir / str(row_number)

            # Get data
            character_profiles = topic_data.get('character_profiles', {})
            scene_plan = topic_data.get('scene_plan', [])
            platform_metadata = topic_data.get('platform_metadata', {})

            characters = character_profiles.get('main_characters', [])

            # Step 1: Generate character reference portraits
            character_files = {}
            if characters:
                character_files = self.generate_character_images(characters, output_dir)
            else:
                self.log_step("No characters found, skipping character generation")

            # Step 2: Generate scene images with character references
            scene_files = []
            if scene_plan:
                scene_files = self.generate_scene_images(scene_plan, character_files, output_dir)
            else:
                self.log_step("No scene plan found, skipping scene generation", "ERROR")

            # Step 3: Generate thumbnail with character references
            thumbnail_file = None
            thumbnail_file = self.generate_thumbnail(platform_metadata, topic, character_files, scene_plan, output_dir)

            # Step 4: Save generation log
            self.save_generation_log(output_dir, topic, row_number)

            # Update stats
            self.stats["topics_processed"] += 1

            self.log_step(f"Topic {row_number} completed successfully", "SUCCESS", {
                "characters_generated": len(character_files),
                "scenes_generated": len(scene_files),
                "thumbnail_generated": thumbnail_file is not None,
                "total_images": len(character_files) + len(scene_files) + (1 if thumbnail_file else 0)
            })

            return True

        except Exception as e:
            self.log_step(f"Failed to process topic {row_number}: {e}", "ERROR")
            self.stats["errors"] += 1
            return False

    def process_all_completed_topics(self):
        """Process all completed topics from CSV"""

        self.log_step("Starting visual generation for all completed topics")

        # Get completed topics
        completed_topics = self.get_completed_topics()

        if not completed_topics:
            self.log_step("No completed topics found", "ERROR")
            return

        # Process each topic
        successful = 0
        failed = 0

        for row_number, topic, description in completed_topics:
            self.log_step(f"\n{'=' * 60}")
            self.log_step(f"PROCESSING TOPIC {row_number}/{len(completed_topics)}")
            self.log_step(f"{'=' * 60}")

            success = self.process_topic(row_number, topic, description)

            if success:
                successful += 1
            else:
                failed += 1

            # Progress update
            self.log_step(f"Progress: {successful + failed}/{len(completed_topics)} topics processed")

        # Final summary
        elapsed_time = datetime.now() - self.stats["start_time"]

        self.log_step(f"\n🎉 VISUAL GENERATION COMPLETE!", "SUCCESS", {
            "total_topics": len(completed_topics),
            "successful": successful,
            "failed": failed,
            "total_images_generated": self.stats["characters_generated"] + self.stats["scenes_generated"] + self.stats[
                "thumbnails_generated"],
            "characters_generated": self.stats["characters_generated"],
            "scenes_generated": self.stats["scenes_generated"],
            "thumbnails_generated": self.stats["thumbnails_generated"],
            "total_api_calls": self.stats["api_calls"],
            "total_errors": self.stats["errors"],
            "elapsed_time": str(elapsed_time),
            "average_time_per_topic": str(elapsed_time / len(completed_topics)) if completed_topics else "N/A"
        })


def main():
    """Main execution function"""

    print("🎨 ENHANCED VISUAL GENERATOR - gpt-image-1 PIPELINE")
    print("=" * 60)
    print("✅ Character reference portrait generation")
    print("🎬 Scene image generation with character references")
    print("🖼️ YouTube thumbnail with character references")
    print("📁 Organized file structure")
    print("=" * 60)

    try:
        # Initialize generator
        generator = VisualGenerator()

        # Process all completed topics
        generator.process_all_completed_topics()

        print("\n🚀 Enhanced visual generation pipeline complete!")
        print("📁 Check src/output/{row_number}/images/ for generated content")
        print("📊 Review visual_generation_log.json for detailed statistics")

    except Exception as e:
        print(f"\n💥 VISUAL GENERATOR ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()