"""
Manual Audio Files Composer
Converts manually created audio files to the format expected by video generator
Hook + Subscribe + Scene audio files → System format
"""

import os
import json
from pathlib import Path
from pydub import AudioSegment
import shutil
from datetime import datetime

class ManualAudioComposer:
    def __init__(self, project_path: str):
        """
        Initialize composer for a specific project
        project_path: Path to the project output directory (e.g., output/123)
        """
        self.project_path = Path(project_path)
        self.audio_parts_dir = self.project_path / "audio_parts"

        print(f"🎵 Manual Audio Composer initialized")
        print(f"📁 Project: {self.project_path}")
        print(f"📁 Audio parts dir: {self.audio_parts_dir}")

        # Ensure directories exist
        self.audio_parts_dir.mkdir(exist_ok=True)

        if not self.audio_parts_dir.exists():
            print(f"❌ Audio parts directory not found: {self.audio_parts_dir}")
            print(f"💡 Please create it and put your manual MP3 files there")
            print(f"   Expected files: hook_subscribe.mp3, scene_01_1.mp3, scene_01_2.mp3, etc.")

    def scan_manual_files(self):
        """Scan audio_parts directory for manual files to process"""
        if not self.audio_parts_dir.exists():
            return {}, []

        # Look for files that need processing
        all_files = list(self.audio_parts_dir.glob("*.mp3"))

        # Filter out already processed files (system format)
        manual_files = []
        for file in all_files:
            filename = file.stem.lower()
            # Skip if already in system format
            if filename.endswith("_audio"):
                continue
            manual_files.append(file)

        print(f"🔍 Found {len(all_files)} total MP3 files")
        print(f"🔍 Found {len(manual_files)} manual files to process")

        # Categorize manual files
        hook_subscribe_file = None
        scene_files = {}

        for file in manual_files:
            filename = file.stem.lower()

            if "hook" in filename and "subscribe" in filename:
                hook_subscribe_file = file
                print(f"🎬 Found hook+subscribe: {file.name}")

            elif filename.startswith("scene_"):
                # Parse scene number and part
                try:
                    # Expected: scene_01_1.mp3, scene_01_2.mp3, scene_13_1.mp3
                    parts = filename.split("_")
                    if len(parts) >= 3:
                        scene_num = int(parts[1])
                        part_num = int(parts[2])

                        if scene_num not in scene_files:
                            scene_files[scene_num] = {}
                        scene_files[scene_num][part_num] = file

                        print(f"📖 Found scene {scene_num} part {part_num}: {file.name}")
                except ValueError:
                    print(f"⚠️  Couldn't parse scene file: {file.name}")

        return scene_files, hook_subscribe_file

    def split_hook_subscribe(self, hook_subscribe_file: Path):
        """Split hook_subscribe.mp3 into separate hook and subscribe files"""
        print(f"\n✂️  Splitting hook+subscribe file: {hook_subscribe_file.name}")

        try:
            audio = AudioSegment.from_mp3(str(hook_subscribe_file))
            total_duration = len(audio)

            print(f"⏱️  Total duration: {total_duration/1000:.1f} seconds")

            # Assume first 30-45 seconds is hook, rest is subscribe
            # You can adjust these thresholds
            hook_duration = min(45000, total_duration // 2)  # Max 45 seconds or half

            hook_audio = audio[:hook_duration]
            subscribe_audio = audio[hook_duration:]

            # Save hook
            hook_path = self.audio_parts_dir / "hook_audio.mp3"
            hook_audio.export(str(hook_path), format="mp3")
            print(f"🎬 Hook saved: {hook_path.name} ({len(hook_audio)/1000:.1f}s)")

            # Save subscribe
            subscribe_path = self.audio_parts_dir / "subscribe_audio.mp3"
            subscribe_audio.export(str(subscribe_path), format="mp3")
            print(f"📢 Subscribe saved: {subscribe_path.name} ({len(subscribe_audio)/1000:.1f}s)")

            return True

        except Exception as e:
            print(f"❌ Error splitting hook+subscribe: {e}")
            return False

    def combine_scene_parts(self, scene_num: int, parts_dict: dict):
        """Combine multiple parts of a scene into single file"""
        print(f"\n🔗 Combining scene {scene_num} parts...")

        # Sort parts by part number
        sorted_parts = sorted(parts_dict.items())

        combined_audio = AudioSegment.empty()

        for part_num, file_path in sorted_parts:
            print(f"   📄 Adding part {part_num}: {file_path.name}")

            try:
                part_audio = AudioSegment.from_mp3(str(file_path))

                # Add small pause between parts (1 second)
                if len(combined_audio) > 0:
                    pause = AudioSegment.silent(duration=1000)
                    combined_audio += pause

                combined_audio += part_audio

            except Exception as e:
                print(f"⚠️  Error loading part {part_num}: {e}")
                continue

        if len(combined_audio) > 0:
            # Save combined scene
            scene_path = self.audio_parts_dir / f"scene_{scene_num:02d}_audio.mp3"
            combined_audio.export(str(scene_path), format="mp3")

            print(f"✅ Scene {scene_num} combined: {scene_path.name} ({len(combined_audio)/1000:.1f}s)")
            return True
        else:
            print(f"❌ Failed to combine scene {scene_num}")
            return False

    def copy_single_scene(self, scene_num: int, single_file: Path):
        """Copy single scene file to correct location"""
        scene_path = self.audio_parts_dir / f"scene_{scene_num:02d}_audio.mp3"

        try:
            shutil.copy2(str(single_file), str(scene_path))
            print(f"📋 Copied scene {scene_num}: {single_file.name} → {scene_path.name}")
            return True
        except Exception as e:
            print(f"❌ Error copying scene {scene_num}: {e}")
            return False

    def create_timeline_from_manual_files(self, processed_scenes: list):
        """Create timeline JSON from processed manual files"""
        print(f"\n📋 Creating timeline from manual audio files...")

        timeline_data = {
            "total_scenes": len(processed_scenes) + 2,  # +2 for hook and subscribe
            "pause_between_scenes_ms": 2000,
            "scenes": [],
            "total_duration_ms": 0,
            "created_at": datetime.now().isoformat(),
            "source": "manual_audio_composition",
            "voice_used": "Manual Professional Recording",
            "youtube_optimized": True
        }

        current_time_ms = 0

        # Add hook
        hook_file = self.audio_parts_dir / "hook_audio.mp3"
        if hook_file.exists():
            try:
                hook_audio = AudioSegment.from_mp3(str(hook_file))
                hook_duration = len(hook_audio)

                timeline_data["scenes"].append({
                    "type": "youtube_hook",
                    "scene_number": 0,
                    "title": "Golden Hook - Channel Introduction",
                    "start_time_ms": current_time_ms,
                    "end_time_ms": current_time_ms + hook_duration,
                    "duration_ms": hook_duration,
                    "audio_file": "hook_audio.mp3",
                    "image_file": "scene_01.png",
                    "source": "manual_professional"
                })

                current_time_ms += hook_duration + 2000  # +2s pause

            except Exception as e:
                print(f"⚠️  Couldn't process hook for timeline: {e}")

        # Add subscribe
        subscribe_file = self.audio_parts_dir / "subscribe_audio.mp3"
        if subscribe_file.exists():
            try:
                subscribe_audio = AudioSegment.from_mp3(str(subscribe_file))
                subscribe_duration = len(subscribe_audio)

                timeline_data["scenes"].append({
                    "type": "youtube_subscribe",
                    "scene_number": -1,
                    "title": "Subscribe Request - Community Building",
                    "start_time_ms": current_time_ms,
                    "end_time_ms": current_time_ms + subscribe_duration,
                    "duration_ms": subscribe_duration,
                    "audio_file": "subscribe_audio.mp3",
                    "image_file": "scene_01.png",
                    "source": "manual_professional"
                })

                current_time_ms += subscribe_duration + 2000  # +2s pause

            except Exception as e:
                print(f"⚠️  Couldn't process subscribe for timeline: {e}")

        # Add scenes
        for scene_num in sorted(processed_scenes):
            scene_file = self.audio_parts_dir / f"scene_{scene_num:02d}_audio.mp3"

            if scene_file.exists():
                try:
                    scene_audio = AudioSegment.from_mp3(str(scene_file))
                    scene_duration = len(scene_audio)

                    timeline_data["scenes"].append({
                        "type": "story_scene",
                        "scene_number": scene_num,
                        "title": f"Story Scene {scene_num}",
                        "start_time_ms": current_time_ms,
                        "end_time_ms": current_time_ms + scene_duration,
                        "duration_ms": scene_duration,
                        "audio_file": f"scene_{scene_num:02d}_audio.mp3",
                        "image_file": f"scene_{scene_num:02d}.png",
                        "source": "manual_professional"
                    })

                    current_time_ms += scene_duration + 2000  # +2s pause

                except Exception as e:
                    print(f"⚠️  Couldn't process scene {scene_num} for timeline: {e}")

        # Remove last pause
        timeline_data["total_duration_ms"] = current_time_ms - 2000

        # Save timeline
        timeline_path = self.project_path / "story_audio_youtube_timeline.json"

        try:
            with open(timeline_path, 'w', encoding='utf-8') as f:
                json.dump(timeline_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Timeline saved: {timeline_path.name}")
            print(f"⏱️  Total duration: {timeline_data['total_duration_ms']/60000:.1f} minutes")
            return True

        except Exception as e:
            print(f"❌ Error saving timeline: {e}")
            return False

    def process_manual_audio_files(self):
        """Main processing function - processes files directly in audio_parts/"""
        print(f"\n🎵 PROCESSING MANUAL AUDIO FILES IN AUDIO_PARTS/")
        print("=" * 50)

        # Check what's already in audio_parts
        existing_files = list(self.audio_parts_dir.glob("*.mp3"))
        print(f"📁 Found {len(existing_files)} MP3 files in audio_parts/")

        for file in existing_files:
            file_size = file.stat().st_size / 1024
            print(f"   📄 {file.name} ({file_size:.1f}KB)")

        # Scan for manual files that need processing
        scene_files, hook_subscribe_file = self.scan_manual_files()

        if not scene_files and not hook_subscribe_file:
            print(f"\n❌ No manual audio files found to process!")
            print(f"💡 Looking for files like:")
            print(f"   🎬 hook_subscribe.mp3")
            print(f"   📖 scene_01_1.mp3, scene_01_2.mp3")
            print(f"   📖 scene_13_1.mp3")
            print(f"\n📋 System format files (already processed):")
            system_files = [f for f in existing_files if f.stem.endswith('_audio')]
            if system_files:
                for file in system_files:
                    print(f"   ✅ {file.name}")
            else:
                print(f"   📭 None found")
            return False

        processed_scenes = []

        # Process hook+subscribe
        if hook_subscribe_file:
            success = self.split_hook_subscribe(hook_subscribe_file)
            if success:
                # Remove original file after successful split
                try:
                    hook_subscribe_file.unlink()
                    print(f"   🗑️  Removed original: {hook_subscribe_file.name}")
                except:
                    print(f"   ⚠️  Couldn't remove original: {hook_subscribe_file.name}")
            else:
                print(f"❌ Failed to split hook+subscribe")
                return False
        else:
            print(f"⚠️  No hook+subscribe file found")

        # Process scenes
        for scene_num in sorted(scene_files.keys()):
            parts = scene_files[scene_num]

            if len(parts) == 1:
                # Single file - rename to system format
                single_file = list(parts.values())[0]
                success = self.rename_single_scene(scene_num, single_file)
            else:
                # Multiple parts - combine
                success = self.combine_scene_parts(scene_num, parts)

            if success:
                processed_scenes.append(scene_num)
                # Remove original files after successful processing
                for part_file in parts.values():
                    try:
                        part_file.unlink()
                        print(f"   🗑️  Removed original: {part_file.name}")
                    except:
                        print(f"   ⚠️  Couldn't remove: {part_file.name}")

        # Create timeline
        timeline_success = self.create_timeline_from_manual_files(processed_scenes)

        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   🎬 Hook+Subscribe: {'✅' if hook_subscribe_file else '❌'}")
        print(f"   📖 Scenes processed: {len(processed_scenes)}")
        print(f"   📋 Timeline created: {'✅' if timeline_success else '❌'}")
        print(f"   📁 All files now in system format in: {self.audio_parts_dir}")

        if processed_scenes and timeline_success:
            print(f"\n🎉 SUCCESS! Manual audio files converted to system format!")
            print(f"📁 Audio parts ready in: {self.audio_parts_dir}")
            print(f"📋 Timeline ready: story_audio_youtube_timeline.json")
            print(f"🎬 Ready for video generation!")
            return True
        else:
            print(f"\n❌ Some errors occurred during processing")
            return False

    def rename_single_scene(self, scene_num: int, single_file: Path):
        """Rename single scene file to system format"""
        scene_path = self.audio_parts_dir / f"scene_{scene_num:02d}_audio.mp3"

        try:
            single_file.rename(scene_path)
            print(f"📋 Renamed scene {scene_num}: {single_file.name} → {scene_path.name}")
            return True
        except Exception as e:
            print(f"❌ Error renaming scene {scene_num}: {e}")
            return False

def get_project_root():
    """Get project root directory - copied from TTS autonomous"""
    # Detect current file location (same as TTS generator)
    current_file = Path(__file__).resolve()

    # For server: /home/youtube-automation/channels/sleepy-dull-stories/src/generators/
    # Go up to project root
    project_root = current_file.parent.parent.parent

    print(f"📍 Project root detected: {project_root}")
    return project_root

def main():
    """Main function - using TTS autonomous style paths"""
    import sys

    # 🎯 SET YOUR PROJECT ID HERE:
    PROJECT_ID = "1"  # ← Change this to your project ID

    # Use same path setup as TTS autonomous
    project_root = get_project_root()

    paths = {
        'BASE_DIR': str(project_root),
        'DATA_DIR': str(project_root / 'data'),
        'OUTPUT_DIR': str(project_root / 'output'),
        'LOGS_DIR': str(project_root / 'logs'),
        'CONFIG_DIR': str(project_root / 'config')
    }

    output_dir = Path(paths['OUTPUT_DIR'])

    print(f"🎵 MANUAL AUDIO COMPOSER")
    print(f"📁 Project root: {paths['BASE_DIR']}")
    print(f"📁 Output directory: {paths['OUTPUT_DIR']}")

    # Create output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")

    print("=" * 50)

    # Get project ID (priority: command line > hardcoded > interactive)
    if len(sys.argv) == 2:
        project_id = sys.argv[1]
        print(f"🎯 Using command line project ID: {project_id}")
    elif PROJECT_ID:
        project_id = PROJECT_ID
        print(f"🎯 Using hardcoded project ID: {project_id}")
    else:
        # Interactive input with status check
        print("📊 Available projects:")
        if output_dir.exists():
            projects = [d.name for d in output_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if projects:
                for proj_id in sorted(projects, key=int):
                    audio_parts_dir = output_dir / proj_id / 'audio_parts'

                    if audio_parts_dir.exists():
                        audio_files = len(list(audio_parts_dir.glob('*.mp3')))
                        manual_files = len([f for f in audio_parts_dir.glob('*.mp3')
                                          if not f.stem.endswith('_audio')])

                        status = ""
                        if manual_files > 0:
                            status = f" 🟡 ({manual_files} manual files)"
                        elif audio_files > 0:
                            status = f" 🟢 ({audio_files} processed files)"
                        else:
                            status = " 🔴 (no audio files)"
                    else:
                        status = " 📁 (no audio_parts folder)"

                    print(f"   📁 {proj_id}{status}")
            else:
                print("   📭 No projects found")

        print()
        project_id = input("🆔 Enter project ID: ").strip()

    if not project_id or not str(project_id).isdigit():
        print("❌ Project ID must be a number")
        return

    # Build project path using TTS autonomous style
    project_path = output_dir / str(project_id)

    if not project_path.exists():
        print(f"❌ Project directory not found: {project_path}")
        print(f"💡 Creating project directory...")
        project_path.mkdir(parents=True, exist_ok=True)
        audio_parts_dir = project_path / 'audio_parts'
        audio_parts_dir.mkdir(exist_ok=True)
        print(f"📁 Created: {project_path}")
        print(f"📁 Created: {audio_parts_dir}")
        print(f"💡 Put your manual MP3 files in: {audio_parts_dir}")
        return

    print(f"\n🎯 Processing project: {project_id}")
    print(f"📁 Project path: {project_path}")
    print("=" * 50)

    composer = ManualAudioComposer(str(project_path))
    success = composer.process_manual_audio_files()

    if success:
        print("\n🎊 All done! Your manual audio files are now ready for video generation!")
        print(f"📁 Audio parts: {project_path}/audio_parts/")
        print(f"📋 Timeline: {project_path}/story_audio_youtube_timeline.json")
        print("🎬 Ready for video generation!")
    else:
        print("\n⚠️  Processing completed with some issues")

if __name__ == "__main__":
    main()