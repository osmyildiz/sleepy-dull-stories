"""
Independent Thumbnail Generator - Standalone Module
File: independent_thumbnail_generator.py

Generates YouTube thumbnails completely independent of scene generation success.
Can be used standalone or imported by other generators.
"""

import os
import json
import sqlite3
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load environment
load_dotenv()

class IndependentThumbnailGenerator:
    """Standalone thumbnail generator independent of scene generation"""

    def __init__(self, config=None, output_dir=None, api_key=None):
        """Initialize independent thumbnail generator

        Args:
            config: Optional config object (from main generators)
            output_dir: Output directory path
            api_key: PIAPI key for Midjourney
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        self.api_key = api_key or self._get_api_key()

        # Setup
        self.base_url = "https://api.piapi.ai/api/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Tracking
        self.thumbnail_log = []
        self.api_calls_made = 0

        # Directories
        if self.output_dir:
            self.thumbnail_dir = self.output_dir / "thumbnail"
            self.thumbnail_dir.mkdir(exist_ok=True)

        print("🖼️ Independent Thumbnail Generator initialized")

    def _get_api_key(self):
        """Get PIAPI key from environment"""
        api_key = (
            os.getenv('PIAPI_KEY') or
            os.getenv('MIDJOURNEY_API_KEY') or
            os.getenv('PIAPI_API_KEY')
        )

        if not api_key:
            raise ValueError("❌ PIAPI key required! Set PIAPI_KEY environment variable")

        return api_key

    def log_step(self, step: str, status: str = "START", metadata: Dict = None):
        """Log thumbnail generation steps"""
        entry = {
            "step": step,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "api_calls": self.api_calls_made
        }
        if metadata:
            entry.update(metadata)
        self.thumbnail_log.append(entry)

        icon = "🖼️" if status == "START" else "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "🔄"
        print(f"{icon} {step}")

    def get_topic_from_database(self, db_path: str = None) -> Dict:
        """Get current topic information from database"""

        if not db_path:
            # Try to find database
            possible_paths = [
                "data/production.db",
                "../data/production.db",
                "../../data/production.db"
            ]

            for path in possible_paths:
                if Path(path).exists():
                    db_path = path
                    break

        if not db_path or not Path(db_path).exists():
            self.log_step("Database not found, using fallback topic", "ERROR")
            return self._get_fallback_topic()

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get current in-progress topic
            cursor.execute('''
                SELECT topic, description, clickbait_title 
                FROM topics 
                WHERE status = 'in_progress' 
                ORDER BY production_started_at DESC 
                LIMIT 1
            ''')

            result = cursor.fetchone()

            if not result:
                # Get latest topic
                cursor.execute('''
                    SELECT topic, description, clickbait_title 
                    FROM topics 
                    ORDER BY created_at DESC 
                    LIMIT 1
                ''')
                result = cursor.fetchone()

            conn.close()

            if result:
                topic_info = {
                    "topic": result[0],
                    "description": result[1],
                    "clickbait_title": result[2] or f"The Secret History of {result[0]} (2 Hour Sleep Story)"
                }
                self.log_step(f"Topic from database: {topic_info['topic']}", "SUCCESS")
                return topic_info

        except Exception as e:
            self.log_step(f"Database error: {e}", "ERROR")

        return self._get_fallback_topic()

    def _get_fallback_topic(self) -> Dict:
        """Fallback topic when database unavailable"""
        return {
            "topic": "Ancient Historical Library",
            "description": "A peaceful journey through ancient libraries and wisdom",
            "clickbait_title": "Ancient Library Secret That Will Put You to Sleep Instantly"
        }

    def check_existing_characters(self) -> Dict:
        """Check for existing character references in output directory"""

        if not self.output_dir or not self.output_dir.exists():
            return {"has_characters": False, "character_count": 0}

        # Check characters directory
        characters_dir = self.output_dir / "characters"
        if characters_dir.exists():
            for json_file in characters_dir.glob("*.json"):
                if json_file.stem == "thumbnail":
                    continue

                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        char_data = json.load(f)

                    char_name = char_data.get('name')
                    image_url = char_data.get('image_url')

                    if char_name and image_url:
                        self.log_step(f"Found character: {char_name}", "SUCCESS")
                        return {
                            "has_characters": True,
                            "main_character": char_name,
                            "character_url": image_url,
                            "character_count": 1,
                            "character_data": char_data
                        }
                except Exception as e:
                    self.log_step(f"Character file error: {e}", "ERROR")
                    continue

        # Check character_profiles.json
        char_profiles_path = self.output_dir / "character_profiles.json"
        if char_profiles_path.exists():
            try:
                with open(char_profiles_path, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)

                main_characters = profiles_data.get('main_characters', [])
                if main_characters:
                    main_char = main_characters[0]
                    char_name = main_char.get('name')

                    if char_name:
                        self.log_step(f"Found character in profiles: {char_name}", "SUCCESS")
                        return {
                            "has_characters": True,
                            "main_character": char_name,
                            "character_url": None,  # No image URL in profiles
                            "character_count": len(main_characters),
                            "character_data": main_char
                        }
            except Exception as e:
                self.log_step(f"Character profiles error: {e}", "ERROR")

        self.log_step("No characters found - using atmospheric approach", "SUCCESS")
        return {"has_characters": False, "character_count": 0}

    def generate_atmospheric_prompt(self, topic_info: Dict) -> str:
        """Generate atmospheric thumbnail prompt when no characters available"""

        topic = topic_info["topic"]

        prompt = f"""Cinematic atmospheric thumbnail for "{topic}", 
        ancient historical setting positioned RIGHT side of frame (60-70% from left edge),
        dramatic golden hour lighting creating visual depth and mystery,
        LEFT side (30-40%) clear and simple for text overlay space,
        warm amber and gold color palette suggesting peaceful sleep content,
        historical architecture and atmospheric elements on RIGHT side,
        soft shadows and cinematic lighting for visual appeal,
        composition optimized for YouTube thumbnail with RIGHT-side emphasis,
        peaceful but visually compelling atmosphere,
        family-friendly historical educational content,
        warm lighting that suggests comfort and relaxation,
        zoomed out composition ensuring all elements visible in safe area,
        no people or characters focus on setting and atmosphere"""

        return prompt

    def generate_character_prompt(self, topic_info: Dict, character_info: Dict) -> str:
        """Generate character-based thumbnail prompt"""

        topic = topic_info["topic"]
        char_name = character_info["main_character"]
        char_url = character_info.get("character_url", "")

        # If we have character URL, use it as reference
        if char_url:
            prompt = f"""{char_url} 
            Cinematic portrait of {char_name} positioned RIGHT side of frame (60-70% from left edge),
            peaceful but intriguing expression with wisdom and serenity,
            warm golden lighting with atmospheric depth for sleep content appeal,
            LEFT side (30-40%) clear and simple background for text overlay,
            ancient historical setting suggesting "{topic}",
            zoomed out enough so character's head and face fully visible in safe area,
            composition optimized for YouTube thumbnail with engaging visual appeal,
            character shows peaceful concentration and wisdom,
            family-friendly historical educational content,
            warm color palette creating comfort and relaxation feeling"""
        else:
            # No character reference image available
            char_data = character_info.get("character_data", {})
            physical_desc = char_data.get("physical_description", "wise elderly person")

            prompt = f"""Cinematic portrait of {char_name}, {physical_desc},
            positioned RIGHT side of frame (60-70% from left edge),
            peaceful wise expression with gentle eyes and serene demeanor,
            warm golden lighting creating atmospheric depth for sleep content,
            LEFT side (30-40%) clear simple background for text overlay,
            ancient historical setting for "{topic}",
            zoomed out composition with character fully visible in safe area,
            YouTube thumbnail optimized with engaging visual appeal,
            peaceful concentration showing wisdom and tranquility,
            family-friendly historical educational content,
            warm lighting suggesting comfort and relaxation"""

        return prompt

    def submit_midjourney_task(self, prompt: str) -> Optional[str]:
        """Submit thumbnail task to Midjourney"""

        payload = {
            "model": "midjourney",
            "task_type": "imagine",
            "input": {
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "process_mode": "relax"
            }
        }

        try:
            self.api_calls_made += 1
            response = requests.post(
                f"{self.base_url}/task",
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    task_data = result.get("data", {})
                    task_id = task_data.get("task_id")
                    self.log_step(f"Task submitted: {task_id}", "SUCCESS")
                    return task_id
                else:
                    self.log_step(f"API Error: {result.get('message', 'Unknown')}", "ERROR")
            else:
                self.log_step(f"HTTP Error: {response.status_code}", "ERROR")

        except Exception as e:
            self.log_step(f"Submit error: {e}", "ERROR")

        return None

    def check_task_status(self, task_id: str) -> Optional[Dict]:
        """Check task status"""
        try:
            status_url = f"{self.base_url}/task/{task_id}"
            response = requests.get(status_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    task_data = result.get("data", {})
                    status = task_data.get("status", "").lower()
                    output = task_data.get("output", {})

                    if status == "completed":
                        temp_urls = output.get("temporary_image_urls", [])
                        image_url = output.get("image_url", "")

                        if temp_urls:
                            selected_url = temp_urls[1] if len(temp_urls) >= 2 else temp_urls[0]
                            return {"url": selected_url, "source": "temporary_image_urls"}
                        elif image_url:
                            return {"url": image_url, "source": "image_url"}
                        else:
                            return False
                    elif status == "failed":
                        return False
                    else:
                        return None  # Still processing
        except Exception as e:
            self.log_step(f"Status check error: {e}", "ERROR")

        return None

    def download_image(self, result_data: Dict, save_path: str) -> bool:
        """Download thumbnail image"""
        image_url = result_data["url"]

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Referer': 'https://discord.com/',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
            }

            response = requests.get(image_url, headers=headers, timeout=30, stream=True)

            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_size = Path(save_path).stat().st_size
                self.log_step(f"Downloaded successfully ({file_size} bytes)", "SUCCESS")
                return True
            else:
                self.log_step(f"Download failed HTTP {response.status_code}", "ERROR")

        except Exception as e:
            self.log_step(f"Download error: {e}", "ERROR")

        return False

    def create_thumbnail_metadata(self, topic_info: Dict, character_info: Dict,
                                  prompt: str, result_data: Dict, approach: str) -> Dict:
        """Create comprehensive thumbnail metadata"""

        return {
            "scene_number": 99,
            "generation_type": "independent_standalone",
            "generation_approach": approach,
            "character_used": character_info.get("main_character", "None (Atmospheric focus)"),
            "clickbait_title": topic_info["clickbait_title"],
            "font_design": "Bold impact font, uppercase for key words, warm golden color (#d4af37) for main text, contrasted with deep shadows for readability",
            "prompt": prompt,
            "enhanced_prompt": prompt,
            "visual_style": "Cinematic atmospheric lighting with warm golden tones, optimized for sleep content and YouTube clickability",
            "character_positioning": "RIGHT side of frame (60-70% from left), fully visible with head in safe area",
            "text_overlay_strategy": "LEFT side (30-40%) reserved for text overlay with clear, simple background",
            "composition_fix": "ZOOM OUT composition ensures no cropping, positioned RIGHT for optimal text space",
            "emotional_appeal": "Peaceful wisdom and tranquility that appeals to sleep story viewers",
            "target_audience_appeal": "Adults seeking relaxation and sleep content with historical interest",
            "clickability_factors": f"{approach.title()} approach with warm lighting and compelling visual composition",
            "sleep_content_balance": "Maintains calming sleep content feel while being visually engaging for clicks",
            "thumbnail_reasoning": f"Independent generation using {approach} approach - completely standalone",
            "background_scene": f"Atmospheric {topic_info['topic']} setting",
            "lighting_strategy": "Warm golden hour lighting for maximum visual appeal and sleep content appropriateness",
            "composition_notes": "Independent thumbnail generation - no dependencies",

            # Technical details
            "image_url": result_data["url"],
            "url_source": result_data["source"],
            "local_path": str(self.thumbnail_dir / "thumbnail.png") if self.thumbnail_dir else "",
            "generated_at": datetime.now().isoformat(),
            "api_calls_used": self.api_calls_made,

            # Standalone module info
            "standalone_module": True,
            "module_version": "1.0",
            "dependencies": "None - completely independent",
            "can_run_standalone": True,

            # Generation log
            "generation_log": self.thumbnail_log
        }

    def generate_thumbnail(self, topic_info: Dict = None, db_path: str = None,
                          output_dir: str = None) -> bool:
        """Main thumbnail generation method

        Args:
            topic_info: Optional topic info dict
            db_path: Optional database path
            output_dir: Optional output directory

        Returns:
            bool: Success status
        """

        self.log_step("Starting independent thumbnail generation")

        # Setup output directory if provided
        if output_dir:
            self.output_dir = Path(output_dir)
            self.thumbnail_dir = self.output_dir / "thumbnail"
            self.thumbnail_dir.mkdir(parents=True, exist_ok=True)

        if not self.thumbnail_dir:
            self.log_step("No output directory specified", "ERROR")
            return False

        # Get topic information
        if not topic_info:
            topic_info = self.get_topic_from_database(db_path)

        self.log_step(f"Topic: {topic_info['topic']}")

        # Check for existing characters
        character_info = self.check_existing_characters()

        # Generate appropriate prompt
        if character_info["has_characters"]:
            prompt = self.generate_character_prompt(topic_info, character_info)
            approach = "character-based"
            self.log_step(f"Using character approach: {character_info['main_character']}")
        else:
            prompt = self.generate_atmospheric_prompt(topic_info)
            approach = "atmospheric"
            self.log_step("Using atmospheric approach")

        # Submit to Midjourney
        self.log_step("Submitting to Midjourney")
        task_id = self.submit_midjourney_task(prompt)

        if not task_id:
            self.log_step("Failed to submit task", "ERROR")
            return False

        # Monitor generation
        self.log_step(f"Monitoring task: {task_id}")

        for i in range(25):  # 12.5 minutes max wait
            result_data = self.check_task_status(task_id)

            if result_data and isinstance(result_data, dict):
                self.log_step("Generation completed", "SUCCESS")

                # Download thumbnail
                thumbnail_path = self.thumbnail_dir / "thumbnail.png"

                if self.download_image(result_data, str(thumbnail_path)):
                    # Save metadata
                    metadata = self.create_thumbnail_metadata(
                        topic_info, character_info, prompt, result_data, approach
                    )

                    json_path = self.thumbnail_dir / "thumbnail.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                    self.log_step("Metadata saved - Generation complete", "SUCCESS")
                    print(f"✅ Independent thumbnail generated: {thumbnail_path}")
                    return True
                else:
                    self.log_step("Download failed", "ERROR")
                    return False

            elif result_data is False:
                self.log_step("Generation failed - content policy or other error", "ERROR")
                return False
            else:
                print(f"⏳ Processing... ({i + 1}/25)")
                time.sleep(30)

        self.log_step("Generation timeout", "ERROR")
        return False


def main():
    """Standalone execution for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Independent Thumbnail Generator')
    parser.add_argument('--output-dir', required=True, help='Output directory path')
    parser.add_argument('--db-path', help='Database path (optional)')
    parser.add_argument('--topic', help='Topic name (optional)')
    parser.add_argument('--description', help='Topic description (optional)')
    parser.add_argument('--title', help='Clickbait title (optional)')

    args = parser.parse_args()

    # Create topic info if provided
    topic_info = None
    if args.topic:
        topic_info = {
            "topic": args.topic,
            "description": args.description or "A peaceful historical story",
            "clickbait_title": args.title or f"The Secret History of {args.topic} (2 Hour Sleep Story)"
        }

    # Generate thumbnail
    generator = IndependentThumbnailGenerator(output_dir=args.output_dir)
    success = generator.generate_thumbnail(topic_info=topic_info, db_path=args.db_path)

    if success:
        print("🎉 Standalone thumbnail generation successful!")
    else:
        print("❌ Standalone thumbnail generation failed!")
        exit(1)


if __name__ == "__main__":
    main()