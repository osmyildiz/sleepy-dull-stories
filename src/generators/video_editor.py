# src/generators/video_editor.py
import os
import json
import pandas as pd
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# .env dosyasını yükle
load_dotenv()

# MoviePy for video editing
from moviepy import (
    VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip,
    CompositeAudioClip, ColorClip, concatenate_videoclips, concatenate_audioclips
)

print("✅ MoviePy available for video editing")

# Pillow for image handling
try:
    from PIL import Image

    PIL_AVAILABLE = True
    print("✅ PIL available for image processing")
except ImportError:
    print("❌ PIL not found!")
    print("💡 Install with: pip install Pillow")
    PIL_AVAILABLE = False


def find_correct_paths():
    """Doğru dosya yollarını bul"""
    print("🔍 Path detection starting...")

    possible_csv_paths = [
        "src/data/topics.csv",
        "data/topics.csv",
        "../data/topics.csv",
        "../../src/data/topics.csv",
        "../../data/topics.csv"
    ]

    possible_output_paths = [
        "../../output",
        "../output",
        "output",
        "src/output",
        "../src/output"
    ]

    csv_path = None
    output_path = None

    # CSV path bul
    for path in possible_csv_paths:
        if os.path.exists(path):
            csv_path = path
            print(f"   ✅ CSV found: {path}")
            break

    # Output path bul
    for path in possible_output_paths:
        if os.path.exists(path):
            test_prompt_path = os.path.join(path, "1", "prompt.json")
            if os.path.exists(test_prompt_path):
                output_path = path
                print(f"   ✅ Output dir found: {path}")
                break

    return csv_path, output_path


# Doğru yolları bul
TOPIC_CSV_PATH, OUTPUT_BASE_PATH = find_correct_paths()

# Video settings - YouTube Optimized
VIDEO_SETTINGS = {
    'fps': 30,
    'resolution': (1920, 1080),  # Full HD for YouTube
    'segment_duration': 240,  # 4 minutes per segment in seconds
    'max_segments': 10,
    'black_screen_after_segments': True,
    'audio_fadeout_duration': 3.0,  # 3 seconds fadeout
    'background_music_volume': 0.3,  # 30% volume for fireplace
    'story_audio_volume': 1.0,  # 100% volume for narration
    'youtube_optimized': True,  # YouTube specific optimizations
    'skip_cover_in_video': True,  # Cover will be used as separate thumbnail
    'create_thumbnail': True,  # Create separate 1280x720 thumbnail
}


def print_step(step_num: str, description: str):
    """Adımları yazdır"""
    print(f"\n🔄 Adım {step_num}: {description}")
    print("-" * 60)


def check_csv_for_editing_tasks():
    """CSV'den video editing'e hazır hikayeleri bul"""
    try:
        df = pd.read_csv(TOPIC_CSV_PATH)

        # editing_completed kolonu yoksa ekle
        if 'editing_completed' not in df.columns:
            df['editing_completed'] = 0
            df.to_csv(TOPIC_CSV_PATH, index=False)

        # Video editing'e hazır olan hikayeler
        ready_for_editing = df[
            (df['done'] == 1) &
            (df['audio_generated'] == 1) &
            (df['cover_image_created'] == 1) &
            (df['images_generated'] == 1) &
            (df['editing_completed'] == 0)
            ]

        return df, ready_for_editing

    except Exception as e:
        print(f"❌ CSV okuma hatası: {e}")
        return None, None


def update_csv_editing_status(csv_path: str, row_index: int, completed: bool = True):
    """CSV'de editing_completed durumunu güncelle"""
    try:
        df = pd.read_csv(csv_path)
        df.at[row_index, 'editing_completed'] = 1 if completed else 0
        df.to_csv(csv_path, index=False)
        print(f"   ✅ editing_completed = {1 if completed else 0} güncellendi")
        return True, None
    except Exception as e:
        return False, f"CSV güncelleme hatası: {e}"


def get_story_files(story_id: int) -> Dict[str, str]:
    """Hikaye için gerekli dosya yollarını topla"""
    story_dir = os.path.join(OUTPUT_BASE_PATH, str(story_id))
    runway_dir = os.path.join(story_dir, "runway_visuals")

    files = {
        'story_dir': story_dir,
        'runway_dir': runway_dir,
        'cover_image': os.path.join(runway_dir, "topic_cover_final.jpg"),
        'story_audio': os.path.join(story_dir, "story_audio.mp3"),
        'background_music': os.path.join(story_dir, "fireplace.mp3"),
        'story_txt': os.path.join(story_dir, "story.txt"),
        'segments': []
    }

    # Segment images'ları bul
    for i in range(1, VIDEO_SETTINGS['max_segments'] + 1):
        segment_path = os.path.join(runway_dir, f"segment_{i:02d}.jpg")
        if os.path.exists(segment_path):
            files['segments'].append(segment_path)

    return files


def validate_story_files(files: Dict[str, str]) -> Tuple[bool, List[str]]:
    """Hikaye dosyalarının varlığını kontrol et"""
    missing_files = []

    # Zorunlu dosyalar
    required_files = ['cover_image', 'story_audio', 'background_music']

    for file_key in required_files:
        if not os.path.exists(files[file_key]):
            missing_files.append(f"{file_key}: {files[file_key]}")

    # En az 3 segment olmalı
    if len(files['segments']) < 3:
        missing_files.append(f"segments: only {len(files['segments'])} found, minimum 3 required")

    return len(missing_files) == 0, missing_files


def create_youtube_thumbnail(cover_image_path: str, output_dir: str) -> str:
    """YouTube için ayrı thumbnail oluştur (1280x720)"""
    try:
        from PIL import Image

        thumbnail_path = os.path.join(output_dir, "youtube_thumbnail.jpg")

        # Cover image'ı YouTube thumbnail boyutuna getir
        with Image.open(cover_image_path) as img:
            # YouTube thumbnail size: 1280x720
            youtube_size = (1280, 720)

            # Resize while maintaining aspect ratio
            img.thumbnail(youtube_size, Image.Resampling.LANCZOS)

            # Create a black background and paste the resized image centered
            background = Image.new('RGB', youtube_size, (0, 0, 0))

            # Calculate position to center the image
            x = (youtube_size[0] - img.width) // 2
            y = (youtube_size[1] - img.height) // 2

            background.paste(img, (x, y))
            background.save(thumbnail_path, 'JPEG', quality=95)

        print(f"   ✅ YouTube thumbnail created: {os.path.basename(thumbnail_path)}")
        return thumbnail_path

    except Exception as e:
        print(f"   ❌ Thumbnail creation error: {e}")
        return None
    """Hikaye için gerekli dosya yollarını topla"""
    story_dir = os.path.join(OUTPUT_BASE_PATH, str(story_id))
    runway_dir = os.path.join(story_dir, "runway_visuals")

    files = {
        'story_dir': story_dir,
        'runway_dir': runway_dir,
        'cover_image': os.path.join(runway_dir, "topic_cover_final.jpg"),
        'story_audio': os.path.join(story_dir, "story_audio.mp3"),
        'background_music': os.path.join(story_dir, "fireplace.mp3"),
        'story_txt': os.path.join(story_dir, "story.txt"),
        'segments': []
    }

    # Segment images'ları bul
    for i in range(1, VIDEO_SETTINGS['max_segments'] + 1):
        segment_path = os.path.join(runway_dir, f"segment_{i:02d}.jpg")
        if os.path.exists(segment_path):
            files['segments'].append(segment_path)

    return files


def validate_story_files(files: Dict[str, str]) -> Tuple[bool, List[str]]:
    """Hikaye dosyalarının varlığını kontrol et"""
    missing_files = []

    # Zorunlu dosyalar
    required_files = ['cover_image', 'story_audio', 'background_music']

    for file_key in required_files:
        if not os.path.exists(files[file_key]):
            missing_files.append(f"{file_key}: {files[file_key]}")

    # En az 3 segment olmalı
    if len(files['segments']) < 3:
        missing_files.append(f"segments: only {len(files['segments'])} found, minimum 3 required")

    return len(missing_files) == 0, missing_files


def get_audio_duration(audio_path: str) -> float:
    """Audio dosyasının süresini al"""
    try:
        audio = AudioFileClip(audio_path)
        duration = audio.duration
        audio.close()
        return duration
    except Exception as e:
        print(f"❌ Audio duration error: {e}")
        return 0


def create_segment_video(image_path: str, duration: float, resolution: Tuple[int, int]):
    """Tek bir segment için video clip oluştur"""
    try:
        # MoviePy v2.0: ImageClip duration parameter may need with_duration
        try:
            clip = ImageClip(image_path, duration=duration)
        except:
            clip = ImageClip(image_path).with_duration(duration)

        # MoviePy v2.0 API: resize method still exists but may need different parameters
        # First try to resize to fit height
        clip = clip.resized(height=resolution[1])

        # If wider than target resolution, resize to width
        if clip.w > resolution[0]:
            clip = clip.resized(width=resolution[0])

        # Center the clip in the frame using with_position (v2.0 API)
        clip = clip.with_position('center')

        return clip
    except Exception as e:
        print(f"❌ Segment creation error for {image_path}: {e}")
        # Try alternative resize method if resized() doesn't work
        try:
            try:
                clip = ImageClip(image_path, duration=duration)
            except:
                clip = ImageClip(image_path).with_duration(duration)
            # Alternative: direct resize with tuple
            target_size = (resolution[0], resolution[1])
            clip = clip.resized(target_size)
            clip = clip.with_position('center')
            return clip
        except Exception as e2:
            print(f"❌ Alternative resize also failed: {e2}")
            return None


def create_black_screen(duration: float, resolution: Tuple[int, int]):
    """Siyah ekran oluştur"""
    try:
        # MoviePy v2.0: ColorClip parameters may have changed
        black_clip = ColorClip(size=resolution, color=(0, 0, 0), duration=duration)
        return black_clip
    except Exception as e:
        try:
            # Alternative: with_duration method
            black_clip = ColorClip(size=resolution, color=(0, 0, 0)).with_duration(duration)
            return black_clip
        except Exception as e2:
            print(f"❌ Black screen creation error: {e}, {e2}")
            return None


def create_story_video(story_id: int, files: Dict[str, str]) -> Tuple[bool, str, Dict]:
    """Ana video oluşturma fonksiyonu - YouTube Optimized"""
    try:
        print(f"🎬 Creating YouTube video for story {story_id}...")

        # Audio süresini al
        story_audio_duration = get_audio_duration(files['story_audio'])
        if story_audio_duration <= 0:
            return False, "Invalid story audio duration", {}

        print(
            f"   📊 Story audio duration: {story_audio_duration:.1f} seconds ({story_audio_duration / 60:.1f} minutes)")

        # Video clips listesi
        video_clips = []

        # YouTube için: Cover image'ı video başında kullanma, direkt segment'lerle başla
        if not VIDEO_SETTINGS.get('skip_cover_in_video', False):
            # Normal mode: Cover image ekle
            cover_duration = min(5.0, story_audio_duration)
            cover_clip = create_segment_video(files['cover_image'], cover_duration, VIDEO_SETTINGS['resolution'])
            if cover_clip:
                video_clips.append(cover_clip)
                print(f"   ✅ Cover added: {cover_duration:.1f}s")
            remaining_duration = story_audio_duration - cover_duration
        else:
            # YouTube mode: Direkt segment'lerle başla
            print(f"   🎬 YouTube mode: Skipping cover in video (will be used as thumbnail)")
            remaining_duration = story_audio_duration

        # Segment images - TOPLAM SESİN SÜRES İNE GÖRE DİNAMİK DAĞITIM
        segment_count = len(files['segments'])
        if segment_count > 0:
            # Her segment'in süresi = toplam süre / segment sayısı
            dynamic_segment_duration = remaining_duration / segment_count
            print(
                f"   📊 Dynamic segment duration: {dynamic_segment_duration:.1f}s ({dynamic_segment_duration / 60:.1f} minutes) per segment")

            for i, segment_path in enumerate(files['segments']):
                segment_clip = create_segment_video(segment_path, dynamic_segment_duration,
                                                    VIDEO_SETTINGS['resolution'])

                if segment_clip:
                    video_clips.append(segment_clip)
                    print(f"   ✅ Segment {i + 1}/{segment_count} added: {dynamic_segment_duration:.1f}s")
        else:
            print("   ❌ No segments found!")
            return False, "No segment images found", {}

        if not video_clips:
            return False, "No video clips created", {}

        # Video clips'leri birleştir
        print(f"   🔄 Concatenating {len(video_clips)} video clips...")
        final_video = concatenate_videoclips(video_clips, method="compose")

        # Audio yükle ve işle
        print(f"   🔊 Loading audio files...")

        try:
            # Story audio (ana ses) - MoviePy v2.0 API: with_volume_scaled instead of volumex
            story_audio = AudioFileClip(files['story_audio'])
            print(f"       ✅ Story audio loaded: {story_audio.duration:.1f}s")

            # Volume ayarla
            story_audio = story_audio.with_volume_scaled(VIDEO_SETTINGS['story_audio_volume'])
            print(f"       ✅ Story audio volume set to {VIDEO_SETTINGS['story_audio_volume']}")
        except Exception as e:
            print(f"       ❌ Story audio error: {e}")
            return False, f"Story audio loading failed: {e}", {}

        try:
            # Background music (fireplace) - MoviePy v2.0 API
            background_audio = AudioFileClip(files['background_music'])
            print(f"       ✅ Background audio loaded: {background_audio.duration:.1f}s")

            # Volume ayarla
            background_audio = background_audio.with_volume_scaled(VIDEO_SETTINGS['background_music_volume'])
            print(f"       ✅ Background audio volume set to {VIDEO_SETTINGS['background_music_volume']}")
        except Exception as e:
            print(f"       ❌ Background audio error: {e}")
            return False, f"Background audio loading failed: {e}", {}

        # Background music'i loop yap
        print(f"   🔄 Processing background music loop...")
        try:
            if background_audio.duration < story_audio_duration:
                loop_count = int(story_audio_duration / background_audio.duration) + 1
                print(f"       📊 Need {loop_count} loops for background music")

                # MoviePy v2.0: loop method may be different, try looped() first
                try:
                    background_audio = background_audio.looped(duration=story_audio_duration)
                    print(f"       ✅ Background looped using .looped() method")
                except AttributeError:
                    # Fallback to manual loop if looped() doesn't exist
                    print(f"       🔄 Using manual loop method...")
                    clips_to_loop = [background_audio] * loop_count
                    try:
                        background_audio = concatenate_audioclips(clips_to_loop).subclipped(0, story_audio_duration)
                        print(f"       ✅ Background looped using manual method")
                    except AttributeError:
                        # Try old subclip method
                        background_audio = concatenate_audioclips(clips_to_loop).subclip(0, story_audio_duration)
                        print(f"       ✅ Background looped using old subclip method")
            else:
                try:
                    background_audio = background_audio.subclipped(0, story_audio_duration)
                    print(f"       ✅ Background trimmed using .subclipped()")
                except AttributeError:
                    background_audio = background_audio.subclip(0, story_audio_duration)
                    print(f"       ✅ Background trimmed using .subclip()")
        except Exception as e:
            print(f"       ❌ Background loop error: {e}")
            # Continue with original background audio
            print(f"       ⚠️ Using original background audio without loop")

        # Audio'ları birleştir
        print(f"   🎵 Mixing audio tracks...")
        try:
            final_audio = CompositeAudioClip([story_audio, background_audio])
            print(f"       ✅ Audio tracks mixed successfully")
            print(f"       📊 Final audio duration: {final_audio.duration:.1f}s")
        except Exception as e:
            print(f"       ❌ Audio mixing error: {e}")
            # Fallback to story audio only
            final_audio = story_audio
            print(f"       ⚠️ Using story audio only (no background music)")

        # Video'ya audio ekle - MoviePy v2.0 API: with_audio instead of set_audio
        print(f"   🎬 Adding audio to video...")
        try:
            final_video = final_video.with_audio(final_audio)
            print(f"       ✅ Audio added to video using .with_audio()")
        except AttributeError:
            # Fallback to old method if with_audio doesn't exist
            try:
                final_video = final_video.set_audio(final_audio)
                print(f"       ✅ Audio added to video using .set_audio()")
            except Exception as e:
                print(f"       ❌ Audio attachment error: {e}")
                return False, f"Failed to attach audio to video: {e}", {}
        except Exception as e:
            print(f"       ❌ Audio attachment error: {e}")
            return False, f"Failed to attach audio to video: {e}", {}

        # YouTube için dosya adı
        output_video_path = os.path.join(files['story_dir'], f"youtube_video_{story_id}.mp4")

        print(f"   💾 Rendering YouTube video: {output_video_path}")
        print(f"   ⚡ This may take several minutes...")

        # Final video info before rendering
        print(f"       📊 Video duration: {final_video.duration:.1f}s")
        print(f"       📊 Audio duration: {final_audio.duration:.1f}s")
        print(f"       📊 Video has audio: {final_video.audio is not None}")
        print(f"       📊 Audio channels: {final_audio.nchannels if hasattr(final_audio, 'nchannels') else 'Unknown'}")

        start_time = time.time()

        # MoviePy v2.0: write_videofile parameters may have changed
        try:
            # QuickTime uyumlu encoding için özel parametreler
            final_video.write_videofile(
                output_video_path,
                fps=VIDEO_SETTINGS['fps'],
                audio_codec='aac',
                codec='libx264',
                audio_bitrate='128k',  # Explicit audio bitrate
                audio_fps=44100,  # Standard audio sample rate
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None,
                preset='medium',  # x264 preset for better compatibility
                audio_nbytes=4  # Audio byte depth
            )
            print(f"       ✅ Video rendered with QuickTime-compatible settings")
        except Exception as e:
            print(f"❌ Video writing error with full parameters: {e}")
            print(f"   🔄 Trying with simplified parameters...")
            # Try with minimal but essential parameters for QuickTime compatibility
            try:
                final_video.write_videofile(
                    output_video_path,
                    fps=VIDEO_SETTINGS['fps'],
                    audio_codec='aac',
                    codec='libx264',
                    audio_bitrate='128k',
                    audio_fps=44100
                )
                print(f"       ✅ Video rendered with simplified QuickTime-compatible settings")
            except Exception as e2:
                print(f"❌ Simplified rendering also failed: {e2}")
                # Last resort: very basic parameters
                final_video.write_videofile(
                    output_video_path,
                    fps=VIDEO_SETTINGS['fps']
                )
                print(f"       ⚠️ Video rendered with basic settings (audio compatibility not guaranteed)")

        render_time = time.time() - start_time

        # Dosya boyutunu al
        file_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)

        # Cleanup - MoviePy v2.0 may have different close methods
        try:
            final_video.close()
            story_audio.close()
            background_audio.close()
            for clip in video_clips:
                if hasattr(clip, 'close'):
                    clip.close()
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
            # Continue anyway as this is not critical

        # Sonuç bilgileri
        result_info = {
            'output_path': output_video_path,
            'file_size_mb': file_size_mb,
            'duration_seconds': story_audio_duration,
            'duration_minutes': story_audio_duration / 60,
            'render_time_seconds': render_time,
            'render_time_minutes': render_time / 60,
            'segments_used': len(files['segments']),
            'video_clips_count': len(video_clips),
            'fps': VIDEO_SETTINGS['fps'],
            'resolution': VIDEO_SETTINGS['resolution'],
            'youtube_optimized': VIDEO_SETTINGS.get('youtube_optimized', False),
            'cover_in_video': not VIDEO_SETTINGS.get('skip_cover_in_video', False)
        }

        print(f"   ✅ YouTube video created successfully!")
        print(f"   📁 File: {os.path.basename(output_video_path)}")
        print(f"   💾 Size: {file_size_mb:.1f} MB")
        print(f"   ⏱️ Duration: {story_audio_duration / 60:.1f} minutes")
        print(f"   🚀 Render time: {render_time / 60:.1f} minutes")

        return True, "YouTube video created successfully", result_info

    except Exception as e:
        print(f"❌ Video creation error: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Video creation failed: {e}", {}


def create_video_summary(story_id: int, topic: str, success: bool, message: str, result_info: Dict):
    """Video oluşturma sonuçlarını kaydet"""
    output_dir = os.path.join(OUTPUT_BASE_PATH, str(story_id))
    summary_path = os.path.join(output_dir, "video_editing_results.json")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "story_id": story_id,
        "topic": topic,
        "success": success,
        "message": message,
        "result_info": result_info,
        "video_settings": VIDEO_SETTINGS,
        "libraries_used": {
            "moviepy": True,
            "pil": PIL_AVAILABLE
        }
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def process_video_editing():
    """Ana video editing işlemi - YouTube Optimized"""
    print("🎬 SLEEPY DULL STORIES - YouTube Video Editor v2.0")
    print("📺 YouTube Optimized Video Assembly System")
    print("=" * 70)

    # Path check
    if not TOPIC_CSV_PATH or not OUTPUT_BASE_PATH:
        print("❌ Required paths not found!")
        return

    print(f"📁 CSV Path: {TOPIC_CSV_PATH}")
    print(f"📁 Output Path: {OUTPUT_BASE_PATH}")
    print(
        f"🎬 Video Settings: {VIDEO_SETTINGS['fps']}fps, {VIDEO_SETTINGS['resolution'][0]}x{VIDEO_SETTINGS['resolution'][1]}")
    print(f"📺 YouTube Mode: {'✅ Enabled' if VIDEO_SETTINGS.get('youtube_optimized') else '❌ Disabled'}")
    print(f"🖼️ Thumbnail Creation: {'✅ Enabled' if VIDEO_SETTINGS.get('create_thumbnail') else '❌ Disabled'}")
    print(f"🎭 Cover in Video: {'❌ Skipped (YouTube)' if VIDEO_SETTINGS.get('skip_cover_in_video') else '✅ Included'}")

    print_step("1", "Checking stories ready for editing")

    df, ready_for_editing = check_csv_for_editing_tasks()
    if df is None:
        return

    if ready_for_editing.empty:
        print("✅ No stories ready for video editing!")
        return

    print(f"🎯 Stories ready for editing: {len(ready_for_editing)}")

    # Process each story
    total_stories = len(ready_for_editing)
    success_count = 0
    error_count = 0

    for idx, (csv_index, row) in enumerate(ready_for_editing.iterrows(), 1):
        story_id = csv_index + 1
        topic = row['topic']

        print_step(f"2.{idx}", f"Processing Story {story_id}/{total_stories}")
        print(f"📚 Topic: {topic}")

        # Get story files
        files = get_story_files(story_id)

        # Validate files
        files_valid, missing_files = validate_story_files(files)

        if not files_valid:
            print(f"❌ Missing files for story {story_id}:")
            for missing in missing_files:
                print(f"   - {missing}")
            error_count += 1
            continue

        print(f"✅ All required files found:")
        print(f"   📸 Segments: {len(files['segments'])}")
        print(f"   🖼️ Cover: {os.path.basename(files['cover_image'])}")
        print(f"   🔊 Audio: {os.path.basename(files['story_audio'])}")
        print(f"   🎵 Music: {os.path.basename(files['background_music'])}")

        # Create video
        start_time = time.time()
        success, message, result_info = create_story_video(story_id, files)
        processing_time = time.time() - start_time

        # YouTube specific: Create thumbnail and metadata
        if success and VIDEO_SETTINGS.get('create_thumbnail', False):
            print(f"   🖼️ Creating YouTube thumbnail...")
            thumbnail_path = create_youtube_thumbnail(files['cover_image'], files['story_dir'])
            if thumbnail_path:
                result_info['thumbnail_path'] = thumbnail_path

            print(f"   📝 Creating YouTube metadata...")
            try:
                # story.txt'den açıklama al
                story_txt_path = os.path.join(files['story_dir'], "story.txt")
                description = f"🎙️ SLEEPY DULL STORIES\n\n📚 {topic}\n\n"

                if os.path.exists(story_txt_path):
                    with open(story_txt_path, 'r', encoding='utf-8') as f:
                        story_content = f.read()
                        # İlk 200 karakteri al
                        description += story_content[:200].strip() + "...\n\n"

                description += "🔥 Background: Cozy fireplace sounds\n"
                description += "😴 Perfect for: Sleep, relaxation, background listening\n"
                description += "⏰ Duration: Long-form storytelling\n\n"
                description += "#SleepyStories #Relaxation #Bedtime #HistoricalStories"

                metadata = {
                    'title': f"{topic} | SLEEPY DULL STORIES",
                    'description': description,
                    'tags': ['sleepy stories', 'bedtime stories', 'relaxation', 'historical stories', 'sleep aid'],
                    'category': 'Entertainment',
                    'privacy': 'public'
                }

                # Metadata'yı dosyaya kaydet
                metadata_path = os.path.join(files['story_dir'], "youtube_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                result_info['metadata'] = metadata
                print(f"   ✅ YouTube metadata created")

            except Exception as e:
                print(f"   ❌ Metadata creation error: {e}")
                result_info['metadata'] = {
                    'title': f"{topic} | SLEEPY DULL STORIES",
                    'description': f"🎙️ A calming story about {topic}",
                    'tags': ['sleepy stories'],
                    'category': 'Entertainment'
                }

        # Save results
        summary = create_video_summary(story_id, topic, success, message, result_info)

        if success:
            # Update CSV
            update_success, update_error = update_csv_editing_status(TOPIC_CSV_PATH, csv_index, True)
            if update_success:
                success_count += 1
                print(f"✅ Story {story_id} completed successfully!")
            else:
                print(f"⚠️ Video created but CSV update failed: {update_error}")
                success_count += 1
        else:
            error_count += 1
            print(f"❌ Story {story_id} failed: {message}")

        print(f"⏱️ Processing time: {processing_time / 60:.1f} minutes")

    # Final report
    print_step("3", "YouTube video editing process completed")

    print(f"📊 FINAL REPORT:")
    print(f"  ✅ Successful: {success_count} videos")
    print(f"  ❌ Failed: {error_count} videos")
    print(f"  📁 Total processed: {total_stories} stories")
    print(
        f"  🎬 Settings: {VIDEO_SETTINGS['fps']}fps, {VIDEO_SETTINGS['resolution'][0]}x{VIDEO_SETTINGS['resolution'][1]}")
    print(f"  📺 YouTube Optimized: {'✅' if VIDEO_SETTINGS.get('youtube_optimized') else '❌'}")

    if success_count > 0:
        print(f"\n🎉 Successfully created {success_count} YouTube videos!")
        print(f"📁 Videos location: output/*/youtube_video_*.mp4")
        print(f"🖼️ Thumbnails location: output/*/youtube_thumbnail.jpg")
        print(f"📝 Metadata location: output/*/youtube_metadata.json")
        print(f"\n📺 Ready for YouTube upload!")
        print(f"   1. Use youtube_video_*.mp4 as main video file")
        print(f"   2. Use youtube_thumbnail.jpg as custom thumbnail")
        print(f"   3. Copy title/description from youtube_metadata.json")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        process_video_editing()
    except KeyboardInterrupt:
        print("\n⚠️ YouTube video processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error in YouTube video processing: {e}")
        import traceback

        traceback.print_exc()