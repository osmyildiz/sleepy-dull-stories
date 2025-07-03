"""
Enhanced Midjourney Visual Generator for Sleep Stories
piapi + Midjourney Pipeline with Character Reference System (Economy Mode)

FEATURES:
✅ Character reference system (4-panel auto-select + upscale)
✅ Scene generation with --cref character consistency
✅ Economy mode (relax) - $0.47 for 50 images
✅ Auto-upscaling (U1 selection from 4-panel)
✅ YouTube-optimized dimensions
✅ Integration with scene_plan.json & character_profiles.json

WORKFLOW:
1. Find completed topics (done=1) from CSV
2. Load character_profiles.json & scene_plan.json
3. Generate character reference 4-panels (relax mode)
4. Auto-upscale first panel (U1) for each character
5. Upload character references to get --cref URLs
6. Generate scenes with character references (relax mode)
7. Auto-upscale best panel from each scene
8. Generate thumbnail with character references
9. Save all in organized structure

FOLDER STRUCTURE:
src/output/{row_number}/
  ├── images/
  │   ├── characters/
  │   │   ├── Marcus_Flavius_4panel.png
  │   │   ├── Marcus_Flavius_reference.png (upscaled)
  │   │   └── Livia_reference.png
  │   ├── scenes/
  │   │   ├── scene_01_4panel.png
  │   │   ├── scene_01_final.png (upscaled)
  │   │   └── ...scene_40_final.png
  │   └── thumbnail_final.png
  └── visual_generation_log.json

ECONOMY MODE: ~$0.47 total cost | 8-14 hours processing time
"""

import os
import json
import pandas as pd
from pathlib import Path
import requests
from dotenv import load_dotenv
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import base64

# Load environment variables
load_dotenv()


class MidjourneyVisualGenerator:
    def __init__(self):
        """Initialize Midjourney Visual Generator with piapi (Economy Mode)"""

        # Get API key from environment
        self.api_key = os.getenv("PIAPI_KEY")
        if not self.api_key:
            raise ValueError("❌ PIAPI_KEY not found in .env file")

        # piapi configuration
        self.base_url = "https://api.piapi.ai/api/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Configuration
        self.base_dir = Path("src/output")
        self.csv_path = Path("src/data/topics.csv")

        # 💰 ECONOMY MODE SETTINGS (Relax = $0.01 per image)
        self.character_params = {
            "aspect_ratio": "2:3",       # Portrait for characters
            "version": "6.1",
            "style": "raw",              # Realistic style
            "quality": 1,                # --q 1
            "process_mode": "relax"      # 🐌 Economy mode
        }

        self.scene_params = {
            "aspect_ratio": "16:9",      # YouTube landscape
            "version": "6.1",
            "character_weight": 80,      # --cw 80 for strong consistency
            "quality": 1,
            "process_mode": "relax"      # 🐌 Economy mode
        }

        self.thumbnail_params = {
            "aspect_ratio": "16:9",      # YouTube ratio
            "version": "6.1",
            "character_weight": 90,      # Maximum character consistency
            "quality": 1,
            "process_mode": "relax"      # 🐌 Economy mode
        }

        # Statistics tracking
        self.stats = {
            "topics_processed": 0,
            "characters_generated": 0,
            "scenes_generated": 0,
            "thumbnails_generated": 0,
            "upscales_generated": 0,
            "api_calls": 0,
            "errors": 0,
            "total_cost_estimate": 0.0,
            "start_time": datetime.now()
        }

        # Generation log
        self.generation_log = []

        print("🐌 Midjourney Visual Generator initialized (ECONOMY MODE)")
        print(f"💰 Estimated cost for 50 images: ~$0.47")
        print(f"⏰ Processing time: 8-14 hours")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📊 CSV path: {self.csv_path}")

    def log_step(self, message: str, status: str = "INFO", metadata: Dict = None):
        """Log generation steps with timestamps"""

        entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "status": status,
            "api_calls": self.stats["api_calls"],
            "estimated_cost": self.stats["total_cost_estimate"]
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

    def submit_midjourney_task(self, prompt: str, aspect_ratio: str = "16:9",
                              process_mode: str = "relax") -> Optional[str]:
        """Submit task to Midjourney via piapi"""

        payload = {
            "model": "midjourney",
            "task_type": "imagine",
            "input": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "process_mode": process_mode,
                "skip_prompt_check": False
            }
        }

        try:
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
                    job_id = task_data.get("task_id")
                    self.stats["api_calls"] += 1

                    # Add cost estimate
                    cost_per_task = 0.01  # Relax mode
                    self.stats["total_cost_estimate"] += cost_per_task

                    self.log_step(f"Task submitted: {job_id} (${cost_per_task:.2f})")
                    return job_id
                else:
                    self.log_step(f"API Error: {result}", "ERROR")
                    return None
            else:
                self.log_step(f"HTTP Error: {response.status_code} - {response.text}", "ERROR")
                return None

        except Exception as e:
            self.log_step(f"Error submitting task: {e}", "ERROR")
            return None

    def wait_for_completion(self, task_id: str, timeout: int = 900) -> Optional[str]:
        """Wait for Midjourney task completion (15 min timeout for relax mode)"""

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.base_url}/task/{task_id}",
                    headers=self.headers,
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()

                    if result.get("code") == 200:
                        task_data = result.get("data", {})
                        status = task_data.get("status", "").lower()
                        output = task_data.get("output", {})
                        progress = output.get("progress", 0)

                        if status == "completed":
                            image_url = (output.get("image_url") or
                                       output.get("discord_image_url") or
                                       output.get("image_urls", [None])[0])

                            self.log_step(f"Task completed: {task_id}", "SUCCESS")
                            return image_url
                        elif status == "failed":
                            error = task_data.get("error", {})
                            self.log_step(f"Task failed: {error.get('message', 'Unknown')}", "ERROR")
                            return None
                        else:
                            # Still processing
                            elapsed = int((time.time() - start_time) / 60)
                            print(f"   ⏳ {status} ({progress}%) - {elapsed} min elapsed")
                            time.sleep(30)  # Check every 30 seconds for relax mode

                else:
                    self.log_step(f"Status check error: {response.status_code}", "ERROR")
                    time.sleep(30)

            except Exception as e:
                self.log_step(f"Error checking status: {e}", "ERROR")
                time.sleep(30)

        self.log_step(f"Task timeout: {task_id}", "ERROR")
        return None

    def auto_upscale_task(self, original_task_id: str, panel_number: int = 1) -> Optional[str]:
        """Auto-upscale specific panel (default: panel 1)"""

        upscale_payload = {
            "model": "midjourney",
            "task_type": "upscale",
            "input": {
                "origin_task_id": original_task_id,
                "index": panel_number  # 1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/task",
                headers=self.headers,
                json=upscale_payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                if result.get("code") == 200:
                    upscale_task_id = result.get("data", {}).get("task_id")
                    self.stats["api_calls"] += 1

                    # Add upscale cost
                    self.stats["total_cost_estimate"] += 0.01

                    self.log_step(f"Upscale U{panel_number} submitted: {upscale_task_id}")

                    # Wait for upscale completion
                    upscale_url = self.wait_for_completion(upscale_task_id, timeout=600)

                    if upscale_url:
                        self.stats["upscales_generated"] += 1
                        return upscale_url
                    else:
                        return None

                else:
                    self.log_step(f"Upscale API error: {result}", "ERROR")
                    return None
            else:
                self.log_step(f"Upscale HTTP error: {response.status_code}", "ERROR")
                return None

        except Exception as e:
            self.log_step(f"Upscale error: {e}", "ERROR")
            return None

    def download_and_save_image(self, image_url: str, file_path: Path) -> bool:
        """Download image from URL and save to file"""

        try:
            response = requests.get(image_url, timeout=60)

            if response.status_code == 200:
                # Create directory if needed
                file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(file_path, 'wb') as f:
                    f.write(response.content)

                file_size = file_path.stat().st_size / 1024  # KB
                self.log_step(f"Downloaded: {file_path.name} ({file_size:.1f} KB)", "SUCCESS")
                return True
            else:
                self.log_step(f"Download failed: {response.status_code}", "ERROR")
                return False

        except Exception as e:
            self.log_step(f"Download error: {e}", "ERROR")
            return False

    def upload_reference_image(self, image_path: Path) -> Optional[str]:
        """Upload image to get public URL for --cref parameter"""

        try:
            with open(image_path, 'rb') as f:
                files = {"file": f}

                response = requests.post(
                    f"{self.base_url.replace('/api/v1', '')}/upload",  # Remove /api/v1 for upload
                    headers={"x-api-key": self.api_key},
                    files=files,
                    timeout=60
                )

            if response.status_code == 200:
                result = response.json()
                public_url = result.get("url")
                self.log_step(f"Reference uploaded: {image_path.name}")
                return public_url
            else:
                self.log_step(f"Upload failed: {response.status_code}", "ERROR")
                return None

        except Exception as e:
            self.log_step(f"Upload error: {e}", "ERROR")
            return None

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
                "topics_list": [f"Row {r}: {t}" for r, t, d in topics[:3]]  # Show first 3
            })

            return topics

        except Exception as e:
            self.log_step(f"Failed to load CSV: {e}", "ERROR")
            return []

    def load_topic_data(self, row_number: int) -> Dict:
        """Load character_profiles.json and scene_plan.json for topic"""

        topic_dir = self.base_dir / str(row_number)

        if not topic_dir.exists():
            raise FileNotFoundError(f"Topic directory not found: {topic_dir}")

        # Required files
        required_files = [
            "character_profiles.json",
            "scene_plan.json"
        ]

        data = {}

        # Load required files
        for filename in required_files:
            file_path = topic_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                data[filename.replace('.json', '')] = json.load(f)

        # Check for scene 99 (thumbnail metadata)
        scene_99_file = topic_dir / "scene_99_metadata.json"
        if scene_99_file.exists():
            with open(scene_99_file, 'r', encoding='utf-8') as f:
                data['scene_99_metadata'] = json.load(f)

        self.log_step(f"Topic data loaded", "SUCCESS", {
            "row_number": row_number,
            "files_loaded": list(data.keys()),
            "characters_count": len(data.get('character_profiles', {}).get('main_characters', [])),
            "scenes_count": len(data.get('scene_plan', []))
        })

        return data

    def generate_character_images(self, characters: List[Dict], output_dir: Path) -> Dict[str, str]:
        """Generate character reference images (4-panel + auto-upscale)"""

        self.log_step(f"Generating {len(characters)} character references (ECONOMY MODE)")

        characters_dir = output_dir / "images" / "characters"
        characters_dir.mkdir(parents=True, exist_ok=True)

        character_reference_urls = {}

        for char in characters:
            try:
                char_name = char.get('name', 'Unknown')
                char_desc = char.get('physical_description', 'No description available')
                visual_notes = char.get('visual_notes', '')

                # Build Midjourney character prompt from existing template
                prompt = f"Character reference sheet: {char_name}. {char_desc}"
                if visual_notes:
                    prompt += f" {visual_notes}"

                prompt += f" Ancient Roman setting, 79 AD Pompeii, full body character design, professional character sheet --ar {self.character_params['aspect_ratio']} --v {self.character_params['version']} --style {self.character_params['style']} --q {self.character_params['quality']}"

                self.log_step(f"Generating character: {char_name} (4-panel)")

                # Submit 4-panel generation
                task_id = self.submit_midjourney_task(
                    prompt,
                    self.character_params['aspect_ratio'],
                    self.character_params['process_mode']
                )

                if not task_id:
                    continue

                # Wait for 4-panel completion
                panel_4_url = self.wait_for_completion(task_id)
                if not panel_4_url:
                    continue

                # Save 4-panel version
                safe_name = "".join(c for c in char_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
                panel_4_path = characters_dir / f"{safe_name}_4panel.png"

                if not self.download_and_save_image(panel_4_url, panel_4_path):
                    continue

                # Auto-upscale panel 1 (top-left)
                self.log_step(f"Auto-upscaling {char_name} (U1)")
                upscaled_url = self.auto_upscale_task(task_id, panel_number=1)

                if not upscaled_url:
                    self.log_step(f"Upscale failed for {char_name}, using 4-panel", "ERROR")
                    upscaled_url = panel_4_url

                # Save upscaled reference
                reference_path = characters_dir / f"{safe_name}_reference.png"

                if self.download_and_save_image(upscaled_url, reference_path):
                    # Upload to get --cref URL
                    cref_url = self.upload_reference_image(reference_path)
                    if cref_url:
                        character_reference_urls[char_name] = cref_url
                        self.stats["characters_generated"] += 1
                        self.log_step(f"Character reference ready: {char_name}", "SUCCESS")

            except Exception as e:
                self.log_step(f"Error generating character {char_name}: {e}", "ERROR")
                self.stats["errors"] += 1
                continue

        self.log_step(f"Character generation complete", "SUCCESS", {
            "characters_generated": len(character_reference_urls),
            "reference_urls_ready": len(character_reference_urls)
        })

        return character_reference_urls

    def generate_scene_images(self, scene_plan: List[Dict], character_urls: Dict[str, str],
                              output_dir: Path) -> List[str]:
        """Generate scene images with character references (ECONOMY MODE)"""

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

                # Build scene prompt using existing template
                prompt = f"Scene {scene_id}: {scene_title}. {scene_description}"

                # Add atmosphere and style from scene_data
                location = scene_data.get('location', '')
                emotion = scene_data.get('emotion', 'peaceful')

                prompt += f" Location: {location}. Emotion: {emotion}. Ancient Roman setting, 79 AD Pompeii, cinematic composition, atmospheric lighting, high detail"

                # Add character references if available
                reference_characters = []
                if characters_mentioned:
                    cref_params = []
                    for char_name in characters_mentioned:
                        if char_name in character_urls:
                            cref_params.append(f"--cref {character_urls[char_name]}")
                            reference_characters.append(char_name)

                    if cref_params:
                        prompt += f" {' '.join(cref_params)} --cw {self.scene_params['character_weight']}"

                # Add Midjourney parameters
                prompt += f" --ar {self.scene_params['aspect_ratio']} --v {self.scene_params['version']} --q {self.scene_params['quality']}"

                if reference_characters:
                    self.log_step(f"  Using character refs: {', '.join(reference_characters)}")

                # Submit scene generation
                task_id = self.submit_midjourney_task(
                    prompt,
                    self.scene_params['aspect_ratio'],
                    self.scene_params['process_mode']
                )

                if not task_id:
                    continue

                # Wait for 4-panel completion
                panel_4_url = self.wait_for_completion(task_id)
                if not panel_4_url:
                    continue

                # Save 4-panel version
                panel_4_path = scenes_dir / f"scene_{scene_id:02d}_4panel.png"
                if not self.download_and_save_image(panel_4_url, panel_4_path):
                    continue

                # Auto-upscale panel 1
                self.log_step(f"Auto-upscaling Scene {scene_id} (U1)")
                upscaled_url = self.auto_upscale_task(task_id, panel_number=1)

                if upscaled_url:
                    # Save final upscaled scene
                    final_path = scenes_dir / f"scene_{scene_id:02d}_final.png"

                    if self.download_and_save_image(upscaled_url, final_path):
                        scene_files.append(str(final_path))
                        self.stats["scenes_generated"] += 1

            except Exception as e:
                self.log_step(f"Error generating scene {scene_id}: {e}", "ERROR")
                self.stats["errors"] += 1
                continue

        self.log_step(f"Scene generation complete", "SUCCESS", {
            "scenes_generated": len(scene_files),
            "total_scenes": len(scene_plan)
        })

        return scene_files

    def generate_thumbnail(self, topic_data: Dict, topic: str, character_urls: Dict[str, str],
                          output_dir: Path) -> Optional[str]:
        """Generate YouTube thumbnail with character references"""

        self.log_step("Generating YouTube thumbnail with character references")

        thumbnail_dir = output_dir / "images"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Check for scene 99 metadata
            scene_99_data = topic_data.get('scene_99_metadata', {})

            if scene_99_data:
                # Use scene 99 professional prompt
                prompt = scene_99_data.get('prompt', '')
                thumbnail_characters = [scene_99_data.get('character_used', '')]

                self.log_step("Using Scene 99 professional thumbnail metadata")
            else:
                # Fallback to generic thumbnail
                prompt = f"YouTube thumbnail for '{topic}' sleep story. Ancient Roman setting, warm atmospheric lighting, peaceful yet eye-catching design, cinematic composition"
                thumbnail_characters = list(character_urls.keys())[:1] if character_urls else []

            # Add character references
            reference_characters = []
            if thumbnail_characters:
                cref_params = []
                for char_name in thumbnail_characters:
                    if char_name in character_urls:
                        cref_params.append(f"--cref {character_urls[char_name]}")
                        reference_characters.append(char_name)

                if cref_params:
                    prompt += f" {' '.join(cref_params)} --cw {self.thumbnail_params['character_weight']}"

            # Add thumbnail parameters
            prompt += f" --ar {self.thumbnail_params['aspect_ratio']} --v {self.thumbnail_params['version']} --q {self.thumbnail_params['quality']}"

            if reference_characters:
                self.log_step(f"  Using character refs: {', '.join(reference_characters)}")

            # Submit thumbnail generation
            task_id = self.submit_midjourney_task(
                prompt,
                self.thumbnail_params['aspect_ratio'],
                self.thumbnail_params['process_mode']
            )

            if not task_id:
                return None

            # Wait for 4-panel completion
            panel_4_url = self.wait_for_completion(task_id)
            if not panel_4_url:
                return None

            # Save 4-panel version
            panel_4_path = thumbnail_dir / "thumbnail_4panel.png"
            if not self.download_and_save_image(panel_4_url, panel_4_path):
                return None

            # Auto-upscale panel 1
            self.log_step("Auto-upscaling thumbnail (U1)")
            upscaled_url = self.auto_upscale_task(task_id, panel_number=1)

            if upscaled_url:
                # Save final thumbnail
                final_path = thumbnail_dir / "thumbnail_final.png"

                if self.download_and_save_image(upscaled_url, final_path):
                    self.stats["thumbnails_generated"] += 1
                    return str(final_path)

            return None

        except Exception as e:
            self.log_step(f"Error generating thumbnail: {e}", "ERROR")
            self.stats["errors"] += 1
            return None

    def save_generation_log(self, output_dir: Path, topic: str, row_number: int):
        """Save detailed generation log with economy mode stats"""

        log_data = {
            "topic": topic,
            "row_number": row_number,
            "generation_completed": datetime.now().isoformat(),
            "generator": "Midjourney via piapi (Economy Mode)",
            "mode": "relax",
            "estimated_cost": f"${self.stats['total_cost_estimate']:.2f}",
            "statistics": self.stats.copy(),
            "generation_log": self.generation_log,
            "file_structure": {
                "characters_4panel": list((output_dir / "images" / "characters").glob("*_4panel.png")),
                "characters_reference": list((output_dir / "images" / "characters").glob("*_reference.png")),
                "scenes_4panel": list((output_dir / "images" / "scenes").glob("*_4panel.png")),
                "scenes_final": list((output_dir / "images" / "scenes").glob("*_final.png")),
                "thumbnail_4panel": str(output_dir / "images" / "thumbnail_4panel.png") if (
                            output_dir / "images" / "thumbnail_4panel.png").exists() else None,
                "thumbnail_final": str(output_dir / "images" / "thumbnail_final.png") if (
                            output_dir / "images" / "thumbnail_final.png").exists() else None
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
            "log_path": str(log_path),
            "total_cost": f"${self.stats['total_cost_estimate']:.2f}"
        })

    def process_topic(self, row_number: int, topic: str, description: str) -> bool:
        """Process a single topic with economy mode pipeline"""

        self.log_step(f"Processing Topic {row_number}: {topic} (ECONOMY MODE)", "INFO", {
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

            characters = character_profiles.get('main_characters', [])

            # Step 1: Generate character reference images
            character_urls = {}
            if characters:
                character_urls = self.generate_character_images(characters, output_dir)
            else:
                self.log_step("No characters found, skipping character generation")

            # Step 2: Generate scene images with character references
            scene_files = []
            if scene_plan:
                scene_files = self.generate_scene_images(scene_plan, character_urls, output_dir)
            else:
                self.log_step("No scene plan found, skipping scene generation", "ERROR")

            # Step 3: Generate thumbnail with character references
            thumbnail_file = self.generate_thumbnail(topic_data, topic, character_urls, output_dir)

            # Step 4: Save generation log
            self.save_generation_log(output_dir, topic, row_number)

            # Update stats
            self.stats["topics_processed"] += 1

            elapsed_time = datetime.now() - self.stats["start_time"]

            self.log_step(f"Topic {row_number} completed successfully", "SUCCESS", {
                "characters_generated": len(character_urls),
                "scenes_generated": len(scene_files),
                "thumbnail_generated": thumbnail_file is not None,
                "total_images": self.stats["characters_generated"] + self.stats["scenes_generated"] + self.stats["thumbnails_generated"],
                "upscales_generated": self.stats["upscales_generated"],
                "estimated_cost": f"${self.stats['total_cost_estimate']:.2f}",
                "elapsed_time": str(elapsed_time)
            })

            return True

        except Exception as e:
            self.log_step(f"Failed to process topic {row_number}: {e}", "ERROR")
            self.stats["errors"] += 1
            return False

    def process_all_completed_topics(self):
        """Process all completed topics with economy mode"""

        self.log_step("Starting ECONOMY MODE visual generation for all completed topics")

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
            self.log_step(f"PROCESSING TOPIC {row_number}/{len(completed_topics)} (ECONOMY MODE)")
            self.log_step(f"{'=' * 60}")

            success = self.process_topic(row_number, topic, description)

            if success:
                successful += 1
            else:
                failed += 1

            # Progress update with cost tracking
            self.log_step(f"Progress: {successful + failed}/{len(completed_topics)} topics processed")
            self.log_step(f"Current cost estimate: ${self.stats['total_cost_estimate']:.2f}")

        # Final summary
        elapsed_time = datetime.now() - self.stats["start_time"]

        self.log_step(f"\n🎉 ECONOMY MODE VISUAL GENERATION COMPLETE!", "SUCCESS", {
            "total_topics": len(completed_topics),
            "successful": successful,
            "failed": failed,
            "total_images_generated": self.stats["characters_generated"] + self.stats["scenes_generated"] + self.stats[
                "thumbnails_generated"],
            "characters_generated": self.stats["characters_generated"],
            "scenes_generated": self.stats["scenes_generated"],
            "thumbnails_generated": self.stats["thumbnails_generated"],
            "upscales_generated": self.stats["upscales_generated"],
            "total_api_calls": self.stats["api_calls"],
            "final_cost_estimate": f"${self.stats['total_cost_estimate']:.2f}",
            "total_errors": self.stats["errors"],
            "elapsed_time": str(elapsed_time),
            "average_time_per_topic": str(elapsed_time / len(completed_topics)) if completed_topics else "N/A"
        })


def main():
    """Main execution function"""

    print("🐌 ECONOMY MODE MIDJOURNEY VISUAL GENERATOR")
    print("=" * 60)
    print("💰 Cost-optimized pipeline (~$0.47 for 50 images)")
    print("⏰ Processing time: 8-14 hours")
    print("✅ Character reference system with auto-upscaling")
    print("🎬 Scene generation with --cref character consistency")
    print("🖼️ YouTube-optimized thumbnail generation")
    print("📁 Organized file structure with 4-panel + final versions")
    print("🔄 Integration with scene_plan.json & character_profiles.json")
    print("=" * 60)

    try:
        # Initialize generator
        generator = MidjourneyVisualGenerator()

        # Process all completed topics
        generator.process_all_completed_topics()

        print("\n🚀 Economy mode visual generation pipeline complete!")
        print("📁 Check src/output/{row_number}/images/ for generated content")
        print("📊 Review visual_generation_log.json for detailed statistics and costs")
        print("🎯 Character consistency achieved with --cref references!")
        print("💰 Total estimated cost logged in statistics")

    except Exception as e:
        print(f"\n💥 VISUAL GENERATOR ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()