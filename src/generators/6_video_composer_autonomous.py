"""
Sleepy Dull Stories - ENHANCED 4K Video Composer
✅ 4K Image Priority Detection
✅ Auto PNG to 4K Upscaling
✅ 4K Video Composition
✅ Scene-by-Scene Processing
"""

import pandas as pd
import os
import json
import ffmpeg
import subprocess
import random
from pathlib import Path
from datetime import datetime
import sqlite3
import sys
import signal
import logging
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import time

# Load environment first
load_dotenv()


class DatabaseVideoManager:
    """Video generation topic management using production.db"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_completed_topic_ready_for_video(self) -> Optional[Tuple[int, str, str, str]]:
        """Get completed topic that needs VIDEO generation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, topic, description, output_path 
            FROM topics 
            WHERE status = 'completed' 
            AND scene_generation_status = 'completed'
            AND audio_generation_status = 'completed'
            AND (video_generation_status IS NULL OR video_generation_status = 'pending')
            ORDER BY scene_generation_completed_at ASC 
            LIMIT 1
        ''')

        result = cursor.fetchone()
        conn.close()

        return result if result else None

    def mark_video_generation_started(self, topic_id: int):
        """Mark video generation as started"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET video_generation_status = 'in_progress', 
                video_generation_started_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()

    def mark_video_generation_completed(self, topic_id: int, video_duration: float, file_size_mb: float,
                                        processing_time: float):
        """Mark video generation as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET video_generation_status = 'completed',
                video_generation_completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP,
                video_duration_seconds = ?,
                video_file_size_mb = ?,
                video_processing_time_minutes = ?
            WHERE id = ?
        ''', (video_duration, file_size_mb, processing_time, topic_id))

        conn.commit()
        conn.close()

class Enhanced4KVideoComposer:
    def __init__(self):
        # Setup paths
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent.parent.parent

        self.paths = {
            'BASE_DIR': str(self.project_root),
            'DATA_DIR': str(self.project_root / 'data'),
            'OUTPUT_DIR': str(self.project_root / 'output'),
            'OVERLAY_DIR': str(self.project_root / 'overlays')
        }

        # Current project tracking
        self.current_topic_id = None
        self.current_output_dir = None
        self.current_topic = None

        # Database manager
        db_path = Path(self.paths['DATA_DIR']) / 'production.db'
        self.db_manager = DatabaseVideoManager(str(db_path))

        # Overlay path
        self.overlay_path = Path(self.paths['OVERLAY_DIR'])

        fireplace_file = self.overlay_path / "fireplace.mp4"
        print(f"🔥 Fireplace path: {fireplace_file}")
        print(f"🔥 Fireplace exists: {fireplace_file.exists()}")

        if fireplace_file.exists():
            file_size_mb = os.path.getsize(fireplace_file) / (1024 * 1024)
            print(f"🔥 Fireplace size: {file_size_mb:.1f} MB")
        else:
            print(f"⚠️  WARNING: Fireplace overlay not found!")
            print(f"💥 STOPPING: Cannot proceed without fireplace overlay!")
            sys.exit(1)  # Program durur

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("VideoComposer")

        # 4K Configuration
        self.video_4k_config = {
            "target_resolution": [3840, 2160],  # 4K
            "upscale_method": "lanczos",
            "quality_setting": 2,  # High quality for upscaling
            "auto_upscale": True,
            "force_4k_output": True
        }

        print("🎬 Enhanced 4K Video Composer Initialized")
        print(f"📺 Target Resolution: {self.video_4k_config['target_resolution'][0]}x{self.video_4k_config['target_resolution'][1]}")
        print(f"🔄 Auto Upscale: {'✅ ENABLED' if self.video_4k_config['auto_upscale'] else '❌ DISABLED'}")

    def ensure_4k_scene_image(self, scenes_dir: Path, scene_id: int) -> Optional[Path]:
        """
        4K görsel kontrolü ve otomatik upscale
        1. Önce _4k.png arar
        2. Yoksa .png'yi 4K'ya çevirir
        3. 4K versiyonunu döndürür
        """
        print(f"🔍 Checking 4K image for scene {scene_id}...")

        # Farklı dosya formatlarını dene
        for format_str in [f"scene_{scene_id:02d}", f"scene_{scene_id}"]:
            # 1. Önce 4K versiyonunu ara
            image_4k_file = scenes_dir / f"{format_str}_4k.png"
            if image_4k_file.exists():
                print(f"   ✅ 4K image found: {image_4k_file.name}")
                return image_4k_file

            # 2. Normal PNG'yi ara
            image_normal_file = scenes_dir / f"{format_str}.png"
            if image_normal_file.exists():
                print(f"   📤 Normal PNG found, upscaling to 4K: {image_normal_file.name}")

                if self.video_4k_config["auto_upscale"]:
                    success = self.upscale_image_to_4k(image_normal_file, image_4k_file)
                    if success:
                        print(f"   ✅ Successfully upscaled to: {image_4k_file.name}")
                        return image_4k_file
                    else:
                        print(f"   ⚠️ Upscaling failed, using original: {image_normal_file.name}")
                        return image_normal_file
                else:
                    print(f"   ⚠️ Auto upscale disabled, using original: {image_normal_file.name}")
                    return image_normal_file

        print(f"   ❌ No image found for scene {scene_id}")
        return None

    def upscale_image_to_4k(self, input_image: Path, output_image: Path) -> bool:
        """
        PNG'yi 4K'ya upscale et
        """
        try:
            print(f"      🔄 Upscaling {input_image.name} to 4K...")

            ffmpeg_cmd = [
                'ffmpeg',
                '-i', str(input_image),
                '-vf', f'scale=3840:2160:flags={self.video_4k_config["upscale_method"]}',
                '-q:v', str(self.video_4k_config["quality_setting"]),
                '-y',  # Overwrite output file
                str(output_image)
            ]

            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            if result.returncode == 0:
                # Dosya boyutunu kontrol et
                if output_image.exists():
                    file_size_mb = os.path.getsize(output_image) / (1024 * 1024)
                    print(f"      ✅ Upscaling successful! Size: {file_size_mb:.1f} MB")
                    return True
                else:
                    print(f"      ❌ Output file not created")
                    return False
            else:
                print(f"      ❌ FFmpeg upscaling failed:")
                print(f"         {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"      ❌ Upscaling timeout (>2 minutes)")
            return False
        except Exception as e:
            print(f"      ❌ Upscaling error: {e}")
            return False

    def batch_upscale_scenes_directory(self, scenes_dir: Path):
        """
        Tüm scenes dizinindeki PNG'leri 4K'ya çevir
        """
        print(f"\n🔄 BATCH 4K UPSCALING:")
        print(f"   📁 Directory: {scenes_dir}")

        if not scenes_dir.exists():
            print(f"   ❌ Directory not found!")
            return

        # Tüm PNG dosyalarını bul (4K olmayanları)
        png_files = []
        for png_file in scenes_dir.glob("*.png"):
            if not png_file.name.endswith("_4k.png"):
                png_files.append(png_file)

        if not png_files:
            print(f"   ✅ No PNG files need upscaling")
            return

        print(f"   📊 Found {len(png_files)} PNG files to upscale")

        success_count = 0
        total_start_time = time.time()

        for i, png_file in enumerate(png_files, 1):
            output_file = scenes_dir / f"{png_file.stem}_4k.png"

            # Skip if 4K version already exists
            if output_file.exists():
                print(f"   ⏭️  [{i}/{len(png_files)}] {png_file.name} - 4K version already exists")
                success_count += 1
                continue

            print(f"   🔄 [{i}/{len(png_files)}] Processing: {png_file.name}")
            start_time = time.time()

            success = self.upscale_image_to_4k(png_file, output_file)

            processing_time = time.time() - start_time
            print(f"      ⏱️  Processing time: {processing_time:.1f}s")

            if success:
                success_count += 1

        total_time = time.time() - total_start_time
        print(f"\n   📊 UPSCALING SUMMARY:")
        print(f"      ✅ Successful: {success_count}/{len(png_files)}")
        print(f"      ⏱️  Total time: {total_time / 60:.1f} minutes")
        print(f"      📊 Average time per image: {total_time / len(png_files):.1f}s")

    def find_scene_files_4k(self, audio_dir: Path, scenes_dir: Path, scene_id: int) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Enhanced scene file finder with 4K priority and auto upscaling
        """
        # Audio dosyası
        audio_file = None
        for format_str in [f"scene_{scene_id:02d}_audio", f"scene_{scene_id}_audio"]:
            test_file = audio_dir / f"{format_str}.mp3"
            if test_file.exists():
                audio_file = test_file
                break

        # 4K Image dosyası (enhanced)
        image_file = self.ensure_4k_scene_image(scenes_dir, scene_id)

        return audio_file, image_file

    def check_existing_scene_video(self, scene_videos_dir: Path, scene_data, scene_type="story") -> Optional[Path]:
        """
        Mevcut scene videosunu kontrol et ve 4K olup olmadığını doğrula
        """
        if scene_type == "hook":
            scene_file = scene_videos_dir / "00_hook.mp4"
            scene_file_4k = scene_videos_dir / "00_hook_4k.mp4"
        elif scene_type == "subscribe":
            scene_file = scene_videos_dir / "01_subscribe.mp4"
            scene_file_4k = scene_videos_dir / "01_subscribe_4k.mp4"
        else:
            scene_id = scene_data['scene_id']
            # Farklı naming formatlarını kontrol et
            possible_names = [
                f"{scene_id:03d}_scene_{scene_id}.mp4",
                f"{scene_id:03d}_scene_{scene_id}_4k.mp4",
                f"scene_{scene_id}.mp4",
                f"scene_{scene_id}_4k.mp4"
            ]

            scene_file = None
            scene_file_4k = None

            for name in possible_names:
                test_file = scene_videos_dir / name
                if test_file.exists():
                    if "_4k" in name:
                        scene_file_4k = test_file
                    else:
                        scene_file = test_file
                    break

        # Önce 4K versiyonunu kontrol et
        for test_file in [scene_file_4k, scene_file]:
            if test_file and test_file.exists():
                try:
                    # FFmpeg ile video bilgilerini al
                    probe = ffmpeg.probe(str(test_file))
                    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

                    if video_stream:
                        width = int(video_stream['width'])
                        height = int(video_stream['height'])
                        duration = float(video_stream.get('duration', 0))

                        print(f"      📹 Found: {test_file.name} ({width}x{height}, {duration:.1f}s)")

                        # 4K kontrolü (3840x2160 veya yakın)
                        if width >= 3840 and height >= 2160:
                            print(f"      ✅ 4K video exists, skipping render")
                            return test_file
                        else:
                            print(f"      ⚠️  Video not 4K ({width}x{height}), will re-render")
                            return None

                except Exception as e:
                    print(f"      ⚠️  Could not check video properties: {e}")
                    return None

        return None

    def create_4k_video_scene_by_scene_style(self, story_scenes, hook_subscribe_data, row_index, total_duration):
        """
        Enhanced 4K Scene-by-Scene Video Creation
        ✅ Automatic 4K image detection/upscaling
        ✅ Skip existing 4K scene videos
        ✅ 4K video output
        ✅ Perfect audio sync
        """
        fireplace_video = self.overlay_path / "fireplace.mp4"
        final_video = Path(self.current_output_dir) / "final_video_4k.mp4"
        scene_videos_dir = Path(self.current_output_dir) / "scene_videos"  # Mevcut dizin adını kullan
        audio_dir = Path(self.current_output_dir) / "audio_parts"
        scenes_dir = Path(self.current_output_dir) / "scenes"

        print(f"\n🎬 4K SCENE-BY-SCENE MOVIEPY PROCESSING:")
        print(f"   📺 Target Resolution: 3840x2160 (4K)")
        print(f"   🔄 Auto 4K Upscaling: {'✅ ENABLED' if self.video_4k_config['auto_upscale'] else '❌ DISABLED'}")
        print(f"   ♻️  Skip Existing 4K Videos: ✅ ENABLED")
        print(f"   📊 Total scenes: {len(story_scenes)}")
        print(f"   ⏱️  Total duration: {total_duration / 60:.1f} minutes")

        # Mevcut scene videolarını kontrol et
        print(f"\n🔍 CHECKING EXISTING SCENE VIDEOS:")
        print(f"   📁 Scene videos directory: {scene_videos_dir}")

        if scene_videos_dir.exists():
            existing_videos = list(scene_videos_dir.glob("*.mp4"))
            print(f"   📊 Found {len(existing_videos)} existing scene videos")
            for video in existing_videos:
                print(f"      📹 {video.name}")
        else:
            print(f"   📁 Creating scene videos directory...")
            scene_videos_dir.mkdir(exist_ok=True)

        # Batch upscale all scenes first (if needed)
        print(f"\n🔄 PRE-PROCESSING: Ensuring all scenes are 4K...")
        self.batch_upscale_scenes_directory(scenes_dir)

        scene_videos_dir.mkdir(exist_ok=True)

        try:
            from moviepy.editor import ImageClip, VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips
            import gc

            # Load fireplace overlay
            fireplace_overlay_base = None
            if fireplace_video.exists():
                print("🔥 Loading fireplace overlay...")
                fireplace_overlay_base = VideoFileClip(str(fireplace_video))
                # Upscale fireplace to 4K if needed
                if fireplace_overlay_base.size != (3840, 2160):
                    print("   📤 Upscaling fireplace overlay to 4K...")
                    fireplace_overlay_base = fireplace_overlay_base.resize((3840, 2160))

            hook_scene, subscribe_scene = hook_subscribe_data
            scene_video_files = []

            def render_4k_scene(scene_data, scene_type="story"):
                """Enhanced 4K scene renderer"""
                clips_to_cleanup = []

                try:
                    if scene_type in ["hook", "subscribe"]:
                        print(f"\n🎬 RENDERING 4K {scene_type.upper()} SCENE:")
                        audio_file = audio_dir / scene_data['audio_file']
                        scene_duration = self.get_audio_duration(audio_file)

                        # Select random scenes for hook/subscribe
                        if scene_type == "hook":
                            available_scenes = [s for s in story_scenes if s['scene_id'] >= 10][:10]
                            selected_scenes = random.sample(available_scenes, min(5, len(available_scenes)))
                        else:  # subscribe
                            available_scenes = [s for s in story_scenes if s['scene_id'] <= 15][:8]
                            selected_scenes = random.sample(available_scenes, min(3, len(available_scenes)))

                        image_duration = scene_duration / len(selected_scenes)
                        image_clips = []

                        for scene_info in selected_scenes:
                            _, image_file = self.find_scene_files_4k(audio_dir, scenes_dir, scene_info['scene_id'])
                            if image_file:
                                # Force 4K resolution
                                img_clip = ImageClip(str(image_file))
                                img_clip = img_clip.resize((3840, 2160))  # Force 4K
                                img_clip = img_clip.set_duration(image_duration)
                                image_clips.append(img_clip)
                                clips_to_cleanup.append(img_clip)

                        if image_clips:
                            main_video = concatenate_videoclips(image_clips, method="compose")
                            clips_to_cleanup.append(main_video)
                        else:
                            print(f"   ❌ No 4K images found")
                            return None

                    else:  # story scene
                        scene_id = scene_data['scene_id']
                        print(f"\n📖 RENDERING 4K STORY SCENE {scene_id}:")

                        # Find 4K files
                        audio_file, image_file = self.find_scene_files_4k(audio_dir, scenes_dir, scene_id)

                        if not audio_file or not image_file:
                            print(f"   ❌ Missing files for scene {scene_id}")
                            return None

                        scene_duration = self.get_audio_duration(audio_file)

                        # Create 4K main video
                        main_video = ImageClip(str(image_file))
                        main_video = main_video.resize((3840, 2160))  # Force 4K
                        main_video = main_video.set_duration(scene_duration)
                        clips_to_cleanup.append(main_video)

                    print(f"   ✅ 4K main video created: {scene_duration:.1f}s @ 3840x2160")

                    # Add 4K fireplace overlay
                    if fireplace_overlay_base:
                        fireplace_duration = fireplace_overlay_base.duration

                        if scene_duration <= fireplace_duration:
                            scene_fireplace = fireplace_overlay_base.subclip(0, scene_duration)
                        else:
                            loops_needed = int(scene_duration / fireplace_duration) + 1
                            fireplace_clips = [fireplace_overlay_base.copy() for _ in range(loops_needed)]
                            scene_fireplace_looped = concatenate_videoclips(fireplace_clips)
                            scene_fireplace = scene_fireplace_looped.subclip(0, scene_duration)
                            clips_to_cleanup.extend(fireplace_clips)
                            clips_to_cleanup.append(scene_fireplace_looped)

                        scene_fireplace = scene_fireplace.resize((3840, 2160))  # Ensure 4K
                        scene_fireplace = scene_fireplace.set_opacity(0.3)
                        scene_fireplace = scene_fireplace.without_audio()

                        scene_final = CompositeVideoClip([main_video, scene_fireplace])
                        clips_to_cleanup.append(scene_fireplace)
                        clips_to_cleanup.append(scene_final)
                    else:
                        scene_final = main_video

                    # Add audio
                    scene_audio = AudioFileClip(str(audio_file))
                    scene_final = scene_final.set_audio(scene_audio)
                    clips_to_cleanup.append(scene_audio)

                    # Render 4K scene
                    if scene_type == "hook":
                        scene_file = scene_videos_dir / "00_hook_4k.mp4"
                    elif scene_type == "subscribe":
                        scene_file = scene_videos_dir / "01_subscribe_4k.mp4"
                    else:
                        scene_file = scene_videos_dir / f"{scene_id:03d}_scene_{scene_id}_4k.mp4"

                    print(f"   🚀 Rendering 4K scene...")
                    print(f"      📁 Output: {scene_file.name}")
                    print(f"      📺 Resolution: 3840x2160 (4K)")

                    start_render_time = time.time()

                    # Enhanced 4K render settings
                    scene_final.write_videofile(
                        str(scene_file),
                        fps=30,
                        codec="libx264",
                        preset="medium",  # Better quality for 4K
                        audio_codec="aac",
                        bitrate="20M",    # Higher bitrate for 4K
                        temp_audiofile=f'temp-audio-4k-{scene_file.stem}.m4a',
                        remove_temp=True,
                        verbose=False,
                        logger='bar'
                    )

                    render_time = time.time() - start_render_time
                    file_size_mb = os.path.getsize(scene_file) / (1024 * 1024)

                    print(f"      ⏱️  Render time: {render_time / 60:.1f} minutes")
                    print(f"      📊 Render speed: {scene_duration / render_time:.1f}x realtime")
                    print(f"      📦 4K file size: {file_size_mb:.1f} MB")

                    # Cleanup
                    for clip in clips_to_cleanup:
                        if clip is not None:
                            try:
                                clip.close()
                            except:
                                pass
                    gc.collect()

                    print(f"   ✅ 4K scene rendered successfully!")
                    return str(scene_file)

                except Exception as e:
                    print(f"   ❌ 4K scene rendering failed: {e}")
                    for clip in clips_to_cleanup:
                        if clip is not None:
                            try:
                                clip.close()
                            except:
                                pass
                    return None

            # Process all scenes with 4K (with existing video check)
            scene_counter = 0
            total_scenes = len(story_scenes) + (1 if hook_scene else 0) + (1 if subscribe_scene else 0)
            skipped_count = 0
            rendered_count = 0

            print(f"\n🎬 STARTING 4K SCENE-BY-SCENE PROCESSING:")
            print(f"   📊 Total scenes: {total_scenes}")

            # Hook scene
            if hook_scene:
                scene_counter += 1
                print(f"\n📊 4K PROCESSING [{scene_counter}/{total_scenes}] Hook Scene")

                # Check if hook video already exists and is 4K
                existing_hook = self.check_existing_scene_video(scene_videos_dir, hook_scene, "hook")
                if existing_hook:
                    scene_video_files.append(str(existing_hook))
                    skipped_count += 1
                    print(f"   ⏭️  Using existing 4K hook video")
                else:
                    hook_video = render_4k_scene(hook_scene, "hook")
                    if hook_video:
                        scene_video_files.append(hook_video)
                        rendered_count += 1

            # Subscribe scene
            if subscribe_scene:
                scene_counter += 1
                print(f"\n📊 4K PROCESSING [{scene_counter}/{total_scenes}] Subscribe Scene")

                # Check if subscribe video already exists and is 4K
                existing_subscribe = self.check_existing_scene_video(scene_videos_dir, subscribe_scene, "subscribe")
                if existing_subscribe:
                    scene_video_files.append(str(existing_subscribe))
                    skipped_count += 1
                    print(f"   ⏭️  Using existing 4K subscribe video")
                else:
                    subscribe_video = render_4k_scene(subscribe_scene, "subscribe")
                    if subscribe_video:
                        scene_video_files.append(subscribe_video)
                        rendered_count += 1

            # Story scenes
            for scene_data in story_scenes:
                scene_counter += 1
                print(f"\n📊 4K PROCESSING [{scene_counter}/{total_scenes}] Story Scene {scene_data['scene_id']}")

                # Check if scene video already exists and is 4K
                existing_scene = self.check_existing_scene_video(scene_videos_dir, scene_data, "story")
                if existing_scene:
                    scene_video_files.append(str(existing_scene))
                    skipped_count += 1
                    print(f"   ⏭️  Using existing 4K scene video")
                else:
                    scene_video = render_4k_scene(scene_data, "story")
                    if scene_video:
                        scene_video_files.append(scene_video)
                        rendered_count += 1

            print(f"\n📊 SCENE PROCESSING SUMMARY:")
            print(f"   ✅ Total scenes processed: {scene_counter}")
            print(f"   ♻️  Existing 4K videos used: {skipped_count}")
            print(f"   🔄 New videos rendered: {rendered_count}")
            print(f"   📦 Total videos for combination: {len(scene_video_files)}")

            # Cleanup fireplace
            if fireplace_overlay_base:
                fireplace_overlay_base.close()

            print(f"\n🔗 COMBINING 4K SCENE VIDEOS:")
            print(f"   📦 Total 4K scenes: {len(scene_video_files)}")

            if not scene_video_files:
                print("   ❌ No 4K scene videos created")
                return None

            # Combine with FFmpeg (4K settings) - DON'T delete existing videos
            scene_list_file = scene_videos_dir / "scene_list_4k.txt"
            with open(scene_list_file, 'w') as f:
                for scene_file in scene_video_files:
                    f.write(f"file '{scene_file}'\n")

            # Enhanced 4K FFmpeg combination
            ffmpeg_cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(scene_list_file),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',      # High quality for 4K
                '-c:a', 'aac',
                '-b:a', '192k',    # High quality audio
                '-movflags', '+faststart',  # Web optimization
                '-y',
                str(final_video)
            ]

            print(f"   🔄 Running enhanced 4K FFmpeg combination...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"   ✅ 4K videos combined successfully!")

                # Only cleanup scene list file (keep scene videos for future use)
                try:
                    os.remove(scene_list_file)
                    print(f"   🗑️  Scene list file cleaned up")
                    print(f"   📦 Scene videos kept for future use")
                except:
                    pass

                print(f"   ✅ Combination completed!")
            else:
                print(f"   ❌ 4K FFmpeg combination failed:")
                print(f"      {result.stderr}")
                return None

            if final_video.exists():
                file_size_mb = os.path.getsize(final_video) / (1024 * 1024)
                print(f"\n✅ 4K SCENE-BY-SCENE PROCESSING COMPLETED!")
                print(f"   📁 File: {final_video.name}")
                print(f"   📺 Resolution: 3840x2160 (4K)")
                print(f"   📦 File size: {file_size_mb:.1f} MB")
                print(f"   🎬 Scenes: {len(scene_video_files)}")
                print(f"   🔥 Fireplace: 5.25min optimized")
                print(f"   🎵 Audio: Perfect sync")
                return final_video
            else:
                print("❌ 4K final video not created")
                return None

        except ImportError:
            print("❌ MoviePy not installed")
            return None
        except Exception as e:
            print(f"❌ 4K scene-by-scene processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_next_project_from_database(self) -> Tuple[bool, Optional[Dict]]:
            """Get next completed project that needs VIDEO generation"""
            print("🔍 Finding completed project for video generation")

            result = self.db_manager.get_completed_topic_ready_for_video()

            if not result:
                print("✅ No completed projects ready for video generation")
                return False, None

            topic_id, topic, description, output_path = result

            # Setup project paths
            self.current_topic_id = topic_id
            self.current_output_dir = output_path
            self.current_topic = topic

            project_info = {
                "topic_id": topic_id,
                "topic": topic,
                "description": description,
                "output_dir": output_path
            }

            # Mark as started in database
            self.db_manager.mark_video_generation_started(topic_id)

            print(f"✅ Found project: {topic}")
            return True, project_info

    def get_audio_duration(self, audio_file: Path) -> float:
        """Get audio file duration in seconds"""
        try:
            probe = ffmpeg.probe(str(audio_file))
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            print(f"⚠️ Could not get duration for {audio_file}: {e}")
            return 0.0

    def run_video_generation(self) -> bool:
        """Main video generation process"""
        print("🎬" * 50)
        print("ENHANCED 4K VIDEO COMPOSER")
        print("🎬" * 50)

        # Get next project from database
        found, project_info = self.get_next_project_from_database()
        if not found:
            return False

        print(f"✅ Project found: {project_info['topic']}")
        print(f"📁 Output directory: {project_info['output_dir']}")

        try:
            # Load story data
            output_dir = Path(self.current_output_dir)
            output_dir = Path(self.current_output_dir)
            timeline_file = output_dir / "story_audio_youtube_timeline.json"  # ← DOĞRU

            if not timeline_file.exists():
                print(f"❌ Timeline not found: {timeline_file}")
                return False

            with open(timeline_file, 'r', encoding='utf-8') as f:
                timeline_data = json.load(f)

            # Extract scene data from timeline
            all_scenes = timeline_data.get('scenes', [])
            story_scenes = []
            hook_scene = None
            subscribe_scene = None

            for scene in all_scenes:
                if scene['type'] == 'story_scene':
                    story_scenes.append({
                        'scene_id': scene['scene_number'],
                        'title': scene['title'],
                        'audio_file': scene['audio_file'],
                        'duration_seconds': scene.get('duration_ms', 0) / 1000.0
                    })
                elif scene['type'] == 'youtube_hook':
                    hook_scene = scene
                elif scene['type'] == 'youtube_subscribe':
                    subscribe_scene = scene

            hook_subscribe_data = (hook_scene, subscribe_scene)
            total_duration = sum(scene.get('duration_seconds', 0) for scene in story_scenes)

            print(f"📊 Story loaded: {len(story_scenes)} scenes, {total_duration / 60:.1f} minutes")

            # Create 4K video
            start_time = time.time()
            final_video = self.create_4k_video_scene_by_scene_style(
                story_scenes, hook_subscribe_data, 0, total_duration
            )
            processing_time = (time.time() - start_time) / 60  # minutes

            if final_video and final_video.exists():
                # Get video stats
                file_size_mb = os.path.getsize(final_video) / (1024 * 1024)
                video_duration = self.get_audio_duration(final_video)

                # Update database
                self.db_manager.mark_video_generation_completed(
                    self.current_topic_id, video_duration, file_size_mb, processing_time
                )

                print(f"\n🎉 VIDEO GENERATION COMPLETED!")
                print(f"📁 File: {final_video}")
                print(f"📦 Size: {file_size_mb:.1f} MB")
                print(f"⏱️ Processing time: {processing_time:.1f} minutes")
                return True
            else:
                print("❌ Video generation failed")
                return False

        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    try:
        print("🚀 ENHANCED 4K VIDEO COMPOSER")
        print("🔗 Database integration")
        print("📺 4K Video Generation")
        print("♻️ Scene video caching")
        print("=" * 60)

        composer = Enhanced4KVideoComposer()
        success = composer.run_video_generation()

        if success:
            print("🎊 4K Video generation completed successfully!")
        else:
            print("⚠️ Video generation failed or no projects ready")

    except KeyboardInterrupt:
        print("\n⏹️ Video generation stopped by user")
    except Exception as e:
        print(f"💥 Video generation failed: {e}")
        import traceback
        traceback.print_exc()