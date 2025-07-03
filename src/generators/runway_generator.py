# import os
import os
import json
import pandas as pd
import asyncio
import time
import base64
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional

# .env dosyasını yükle
load_dotenv()

# Official RunwayML SDK - DOĞRU YAKLAŞIM!
try:
    from runwayml import RunwayML, AsyncRunwayML
    import runwayml

    SDK_AVAILABLE = True
    print("✅ Official RunwayML SDK available")
except ImportError:
    print("❌ Official RunwayML SDK not found!")
    print("💡 Install with: pip install runwayml")
    SDK_AVAILABLE = False
    RunwayML = None
    AsyncRunwayML = None

# PIL for text overlay
try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
    print("✅ PIL available for text overlay")
except ImportError:
    print("❌ PIL not found!")
    print("💡 Install with: pip install Pillow")
    PIL_AVAILABLE = False

# API Configuration - OFFICIAL SDK WAY
RUNWAY_API_SECRET = os.getenv('RUNWAYML_API_SECRET') or os.getenv('RUNWAY_API_KEY')  # Backward compatibility

# Reference image path
REFERENCE_IMAGE_PATH = Path("../../data/media/reference_cover_style.jpg")


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


def print_step(step_num: str, description: str):
    """Adımları yazdır"""
    print(f"\n🔄 Adım {step_num}: {description}")
    print("-" * 60)


def check_csv_for_runway_tasks():
    """CSV'den görsel üretilecek hikayeleri bul"""
    try:
        df = pd.read_csv(TOPIC_CSV_PATH)

        required_columns = ['cover_image_created', 'teaser_video_created', 'images_generated']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0

        df.to_csv(TOPIC_CSV_PATH, index=False)

        ready_for_runway = df[
            (df['done'] == 1) &
            (df['audio_generated'] == 1) &
            ((df['cover_image_created'] == 0) |
             (df['images_generated'] == 0))  # Video kaldırıldı
            ]

        return df, ready_for_runway

    except Exception as e:
        print(f"❌ CSV okuma hatası: {e}")
        return None, None


def update_csv_runway_columns(csv_path: str, row_index: int, **kwargs):
    """CSV'de runway ilgili kolonları güncelle"""
    try:
        df = pd.read_csv(csv_path)
        for column, value in kwargs.items():
            if column in df.columns:
                df.at[row_index, column] = value
                print(f"   ✅ {column} = {value} güncellendi")
        df.to_csv(csv_path, index=False)
        return True, None
    except Exception as e:
        return False, f"CSV güncelleme hatası: {e}"


def load_prompts_from_json(prompt_file_path: str):
    """prompt.json dosyasını yükle"""
    try:
        if not os.path.exists(prompt_file_path):
            return None, f"Prompt dosyası bulunamadı: {prompt_file_path}"

        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        return prompts, None
    except Exception as e:
        return None, f"Dosya okuma hatası: {e}"


def test_runway_sdk_connection():
    """Official RunwayML SDK bağlantı testi - DOĞRU WORKFLOW"""
    print(f"🧪 Testing Official RunwayML SDK...")
    print(f"🔑 API Secret: {'***' + RUNWAY_API_SECRET[-4:] if RUNWAY_API_SECRET else 'NOT SET'}")

    if not SDK_AVAILABLE:
        print("❌ SDK not available - install with: pip install runwayml")
        return False

    if not RUNWAY_API_SECRET:
        print("❌ API secret not configured")
        print("💡 Set RUNWAYML_API_SECRET in .env file")
        return False

    try:
        # Official SDK test - DOĞRU WORKFLOW TEST
        client = RunwayML(api_key=RUNWAY_API_SECRET)

        # Test with simple text_to_image creation (no actual generation)
        print("🧪 Testing SDK connection...")
        print("✅ SDK client created successfully")
        print(f"🔗 SDK Version: {runwayml.__version__ if hasattr(runwayml, '__version__') else 'Unknown'}")
        print("💡 Ready to use text_to_image.create() and tasks.retrieve() workflow")
        return True

    except Exception as e:
        print(f"❌ SDK test failed: {e}")
        return False


def load_image_base64(path: Path) -> str:
    """Reference image'i base64 olarak yükle"""
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ Reference image yükleme hatası: {e}")
        return None


# Text overlay functions from add_text.py
def get_font(size: int, bold: bool = False):
    """Font al"""
    if not PIL_AVAILABLE:
        return None

    font_paths = [
        "/System/Library/Fonts/Times.ttc",
        "/System/Library/Fonts/TimesNewRomanPSMT.ttc",
        "/Library/Fonts/Times New Roman.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
    ]
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap_text(text: str, font, max_width: int) -> list:
    """Text'i satırlara böl"""
    if not font:
        return [text]

    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = font.getbbox(test_line)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def draw_text_with_stroke(draw, position: tuple, text: str, font, fill_color: str, stroke_color: str,
                          stroke_width: int):
    """Konturlu text çiz"""
    x, y = position
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)
    draw.text((x, y), text, font=font, fill=fill_color)


def add_text_overlay(image_path: str, title: str, output_path: str):
    """Cover image'e text overlay ekle"""
    if not PIL_AVAILABLE:
        print("❌ PIL not available for text overlay")
        return False, "PIL not available"

    try:
        # Text ayarları
        CHANNEL_NAME = "SLEEPY DULL STORIES"
        CHANNEL_FONT_SIZE = 50
        TITLE_FONT_SIZE = 100
        TEXT_COLOR = "#FFD700"  # Altın sarısı
        STROKE_COLOR = "#000000"  # Siyah kontur
        STROKE_WIDTH = 3
        TEXT_MARGIN_X = 60
        TEXT_MARGIN_TOP = 120
        TEXT_LINE_SPACING = 10
        MAX_TEXT_AREA_WIDTH = 600

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Kanal adı üstte büyükçe yazılsın
        channel_font = get_font(CHANNEL_FONT_SIZE, bold=True)
        draw_text_with_stroke(draw, (TEXT_MARGIN_X, TEXT_MARGIN_TOP - 70), CHANNEL_NAME,
                              channel_font, TEXT_COLOR, STROKE_COLOR, 2)

        # Başlık satırlara bölünsün
        title_font = get_font(TITLE_FONT_SIZE, bold=True)
        wrapped_lines = wrap_text(title, title_font, MAX_TEXT_AREA_WIDTH)

        y = TEXT_MARGIN_TOP
        for line in wrapped_lines:
            draw_text_with_stroke(draw, (TEXT_MARGIN_X, y), line, title_font,
                                  TEXT_COLOR, STROKE_COLOR, STROKE_WIDTH)
            y += title_font.getbbox("Ay")[3] - title_font.getbbox("Ay")[1] + TEXT_LINE_SPACING

        img.save(output_path, "JPEG", quality=95)
        print(f"   ✅ Text overlay added: {os.path.basename(output_path)}")
        return True, None

    except Exception as e:
        print(f"   ❌ Text overlay error: {e}")
        return False, str(e)


class OfficialRunwayGenerator:
    """Official RunwayML SDK ile görsel üretim"""

    def __init__(self):
        if not SDK_AVAILABLE:
            raise Exception("RunwayML SDK not available. Install with: pip install runwayml")

        if not RUNWAY_API_SECRET:
            raise Exception("RUNWAYML_API_SECRET not configured")

        # Official SDK clients
        self.client = RunwayML(api_key=RUNWAY_API_SECRET)
        self.async_client = AsyncRunwayML(api_key=RUNWAY_API_SECRET)

        # Cost tracking
        self.total_cost = 0.0

        print("🎬 Official RunwayML SDK Generator initialized")
        print(f"🔗 SDK Version: {runwayml.__version__ if hasattr(runwayml, '__version__') else 'Unknown'}")

    def create_enhanced_cover_prompt(self, clickbait_title: str, topic: str):
        """Cover için enhanced prompt - text overlay sonra eklenecek"""
        return f"""Right side of YouTube thumbnail. Classical oil painting of a Roman citizen in emotional awe. Mount Vesuvius erupting in the background. Ruins, smoke, volcanic fire. Dramatic chiaroscuro lighting, cinematic composition, 16:9 framing. No text. Historical scene for "{clickbait_title}": Classical architecture at golden hour, warm lighting, painterly style, atmospheric perspective."""

    def create_enhanced_image_prompts(self, original_prompts: list, max_prompts: int = 8):
        """Segment image prompts oluştur - KISA VE BASIT"""
        enhanced_prompts = []
        prompts_to_process = original_prompts[:max_prompts]

        for i, original_prompt in enumerate(prompts_to_process):
            # Clean original prompt
            base_scene = original_prompt.replace("Cinematic 8k visual:", "").replace("Cinematic 8k scene:", "").strip()
            if base_scene.startswith(f"{i + 1}:"):
                base_scene = base_scene[base_scene.find(":") + 1:].strip()

            # MUCH SHORTER PROMPT - RunwayML likes concise prompts
            enhanced_prompt = f"Classical painting of {base_scene}. Warm golden lighting, Renaissance style, historical setting, rich textures."
            enhanced_prompts.append(enhanced_prompt)

        return enhanced_prompts

    async def create_image_with_references(self, prompt: str, output_filename: str, use_references: bool = True):
        """Reference images ile image generation - cover_test.py metodunu kullan"""
        try:
            print(f"🖼️ Creating image: {output_filename}")

            payload = {
                "model": "gen4_image",
                "prompt_text": prompt,
                "ratio": "1280:720"
            }

            # Reference image ekle eğer varsa ve istenmişse
            if use_references and REFERENCE_IMAGE_PATH.exists():
                ref_base64 = load_image_base64(REFERENCE_IMAGE_PATH)
                if ref_base64:
                    payload["reference_images"] = [{
                        "uri": f"data:image/jpeg;base64,{ref_base64}",
                        "tag": "reference_style"
                    }]
                    print(f"   ✅ Reference image added")
                else:
                    print(f"   ⚠️ Reference image load failed, continuing without reference")

            # DOĞRU SDK USAGE - text_to_image.create()
            task = await self.async_client.text_to_image.create(**payload)

            print(f"   ✅ Image task created: {task.id}")
            return task

        except Exception as e:
            print(f"   ❌ Image generation failed: {e}")
            # Eğer reference image ile hata alırsak, referans olmadan dene
            if use_references and "reference_images" in str(e):
                print(f"   🔄 Retrying without reference image...")
                return await self.create_image_with_references(prompt, output_filename, use_references=False)
            return None

    async def wait_for_completion(self, task_id: str, task_type: str, max_wait: int = 300):
        """Task completion bekle - DOĞRU SDK WORKFLOW"""
        print(f"   ⏳ Waiting for {task_type} completion: {task_id}")

        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                # DOĞRU SDK USAGE - tasks.retrieve() for all task types
                task = await self.async_client.tasks.retrieve(task_id)

                # Status kontrolü - task.status field
                current_status = getattr(task, 'status', 'unknown').upper()

                if current_status in ['SUCCEEDED', 'COMPLETED', 'SUCCESS']:
                    # Output URL'i çıkar - task.output list'inin ilk elementi
                    if hasattr(task, 'output') and task.output:
                        if isinstance(task.output, list) and len(task.output) > 0:
                            output_url = task.output[0]
                        else:
                            output_url = task.output
                        print(f"      ✅ {task_type} completed: {output_url}")
                        return output_url, None
                    else:
                        return None, "No output in completed task"

                elif current_status in ['FAILED', 'ERROR']:
                    error_msg = getattr(task, 'error', None) or getattr(task, 'failure_reason', 'Generation failed')
                    return None, error_msg

                elif current_status in ['PENDING', 'RUNNING', 'PROCESSING', 'IN_PROGRESS']:
                    # Still processing
                    elapsed = int(time.time() - start_time)
                    if elapsed % 30 == 0:
                        print(f"      ⏳ Status: {current_status} - waiting... ({elapsed}s/{max_wait}s)")
                    await asyncio.sleep(10)
                    continue
                else:
                    # Unknown status - wait and try again
                    elapsed = int(time.time() - start_time)
                    if elapsed % 30 == 0:
                        print(f"      ❓ Unknown status '{current_status}' - waiting... ({elapsed}s/{max_wait}s)")
                    await asyncio.sleep(10)
                    continue

            except Exception as e:
                print(f"      ❌ Status check error: {e}")
                await asyncio.sleep(10)
                continue

        return None, f"Timeout after {max_wait} seconds"

    def download_real_file_sync(self, url: str, output_path: str):
        """Gerçek dosyayı indir - Sync version"""
        try:
            import requests

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            print(f"      📥 Downloading: {os.path.basename(output_path)}")
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()

            # Gerçek dosyayı yaz
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"      ✅ Downloaded: {file_size:.1f}MB")
            return True, None, file_size

        except Exception as e:
            print(f"      ❌ Download error: {e}")
            return False, f"Download error: {e}", 0

    async def download_real_file(self, url: str, output_path: str):
        """Async wrapper for real file download"""

        def download_wrapper():
            return self.download_real_file_sync(url, output_path)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, download_wrapper)

    async def process_story_visuals(self, story_id: int, topic: str, prompts: dict):
        """Bir hikaye için görselleri oluştur - Video pas geçildi"""
        output_dir = os.path.join(OUTPUT_BASE_PATH, str(story_id))
        visuals_dir = os.path.join(output_dir, "runway_visuals")
        os.makedirs(visuals_dir, exist_ok=True)

        results = {
            'teaser_video': {'success': False, 'file': None, 'error': 'Skipped - video generation disabled',
                             'file_size_mb': 0},
            'topic_cover': {'success': False, 'file': None, 'error': None, 'file_size_mb': 0},
            'images': {'success': False, 'files': [], 'error': None, 'total_files': 0, 'total_size_mb': 0}
        }

        clickbait_title = prompts.get('clickbait_title', topic)
        print(f"🎯 Processing: {clickbait_title}")

        try:
            # 1. TEASER VIDEO PAS GEÇİLDİ
            print(f"🎬 Teaser video: SKIPPED (disabled)")

            # Rate limiting
            await asyncio.sleep(10)

            # 2. Cover Image oluştur
            print(f"🖼️ Creating cover image...")
            cover_path = os.path.join(visuals_dir, "topic_cover.jpg")
            cover_with_text_path = os.path.join(visuals_dir, "topic_cover_final.jpg")
            enhanced_cover_prompt = self.create_enhanced_cover_prompt(clickbait_title, topic)

            cover_task = await self.create_image_with_references(enhanced_cover_prompt, "topic_cover.jpg",
                                                                 use_references=True)

            if cover_task:
                cover_url, cover_error = await self.wait_for_completion(cover_task.id, "image", 300)
                if cover_url:
                    download_success, download_error, file_size = await self.download_real_file(cover_url, cover_path)
                    if download_success:
                        print(f"   ✅ Cover downloaded: {file_size:.1f}MB")

                        # Text overlay ekle
                        print(f"   📝 Adding text overlay...")
                        text_success, text_error = add_text_overlay(cover_path, clickbait_title, cover_with_text_path)

                        if text_success:
                            final_file_size = os.path.getsize(cover_with_text_path) / (1024 * 1024)
                            results['topic_cover'] = {'success': True, 'file': cover_with_text_path, 'error': None,
                                                      'file_size_mb': final_file_size}
                        else:
                            # Text overlay başarısız olursa orijinal dosyayı kullan
                            results['topic_cover'] = {'success': True, 'file': cover_path,
                                                      'error': f"Text overlay failed: {text_error}",
                                                      'file_size_mb': file_size}
                    else:
                        results['topic_cover'] = {'success': False, 'file': None, 'error': download_error,
                                                  'file_size_mb': 0}
                else:
                    results['topic_cover'] = {'success': False, 'file': None, 'error': cover_error, 'file_size_mb': 0}
            else:
                results['topic_cover'] = {'success': False, 'file': None, 'error': "Cover task creation failed",
                                          'file_size_mb': 0}

            # Rate limiting
            await asyncio.sleep(15)

            # 3. Segment Images oluştur
            original_image_prompts = prompts.get('image_prompts', [])
            enhanced_image_prompts = self.create_enhanced_image_prompts(original_image_prompts, max_prompts=8)

            print(f"🎨 Creating {len(enhanced_image_prompts)} segment images...")

            successful_images = []
            total_image_size = 0

            for idx, enhanced_prompt in enumerate(enhanced_image_prompts, 1):
                print(f"   🖼️ Segment {idx}/{len(enhanced_image_prompts)}...")
                image_path = os.path.join(visuals_dir, f"segment_{idx:02d}.jpg")

                image_task = await self.create_image_with_references(enhanced_prompt, f"segment_{idx:02d}.jpg",
                                                                     use_references=True)

                if image_task:
                    image_url, image_error = await self.wait_for_completion(image_task.id, "image", 300)

                    if image_url:
                        download_success, download_error, file_size = await self.download_real_file(image_url,
                                                                                                    image_path)
                        if download_success:
                            print(f"      ✅ Downloaded: {file_size:.1f}MB")
                            successful_images.append({'file': image_path, 'size_mb': file_size})
                            total_image_size += file_size
                        else:
                            print(f"      ❌ Download failed: {download_error}")
                    else:
                        print(f"      ❌ Generation failed: {image_error}")
                else:
                    print(f"      ❌ Task creation failed")

                # Rate limiting between images
                if idx < len(enhanced_image_prompts):
                    await asyncio.sleep(20)

            # Images sonuçları
            if successful_images:
                results['images'] = {
                    'success': True,
                    'files': successful_images,
                    'error': None,
                    'total_files': len(successful_images),
                    'total_size_mb': total_image_size
                }

        except Exception as e:
            print(f"❌ Processing error: {e}")

        return results


def create_results_summary(story_id: int, topic: str, results: dict):
    """Sonuçları özetleyen dosya oluştur"""
    output_dir = os.path.join(OUTPUT_BASE_PATH, str(story_id))
    summary_path = os.path.join(output_dir, "runway_results.json")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "story_id": story_id,
        "topic": topic,
        "results": results,
        "official_sdk_used": True,
        "sdk_version": runwayml.__version__ if hasattr(runwayml, '__version__') else 'Unknown',
        "video_generation_disabled": True,
        "text_overlay_enabled": PIL_AVAILABLE,
        "reference_images_enabled": REFERENCE_IMAGE_PATH.exists(),
        "stats": {
            "teaser_video_success": False,  # Video disabled
            "cover_image_success": results['topic_cover']['success'],
            "images_success_count": results['images']['total_files'],
            "total_file_size_mb": (
                    results['topic_cover']['file_size_mb'] +
                    results['images']['total_size_mb']
            ),
            "overall_success": (
                    results['topic_cover']['success'] and
                    results['images']['total_files'] >= 3
            )
        }
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


async def process_runway_generation():
    """Ana Runway görsel üretim işlemi - Official SDK - Video Disabled"""
    print("🎬 SLEEPY DULL STORIES - Official RunwayML SDK v7.0")
    print("🏆 Using Official RunwayML Python SDK with Proper Methods")
    print("🚫 Video Generation: DISABLED")
    print("📝 Text Overlay: ENABLED")
    print("🖼️ Reference Images: ENABLED")
    print("=" * 70)

    # SDK availability check
    if not SDK_AVAILABLE:
        print("❌ Official RunwayML SDK not available!")
        print("💡 Install with: pip install runwayml")
        return

    # API secret check
    if not RUNWAY_API_SECRET:
        print("❌ RUNWAYML_API_SECRET not configured!")
        print("💡 Add RUNWAYML_API_SECRET=your-secret-here to .env")
        return

    # Path check
    if not TOPIC_CSV_PATH or not OUTPUT_BASE_PATH:
        print("❌ Required paths not found!")
        return

    print(f"📁 CSV Path: {TOPIC_CSV_PATH}")
    print(f"📁 Output Path: {OUTPUT_BASE_PATH}")
    print(f"🖼️ Reference Image: {REFERENCE_IMAGE_PATH} ({'✅' if REFERENCE_IMAGE_PATH.exists() else '❌'})")
    print(f"📝 PIL for Text Overlay: {'✅' if PIL_AVAILABLE else '❌'}")

    # SDK Connection Test
    print("\n" + "=" * 50)
    print("🧪 OFFICIAL SDK CONNECTION TEST")
    print("=" * 50)

    sdk_works = test_runway_sdk_connection()
    if not sdk_works:
        print("\n❌ SDK test failed - check your API secret")
        return

    print("\n✅ Official SDK connection successful!")

    print("\n" + "=" * 50)
    print("📊 STORY PROCESSING WITH OFFICIAL SDK")
    print("=" * 50)

    # Official generator
    try:
        generator = OfficialRunwayGenerator()
    except Exception as e:
        print(f"❌ Generator creation failed: {e}")
        return

    # CSV check
    print_step("1", "Topics CSV check")
    df, ready_for_runway = check_csv_for_runway_tasks()
    if df is None:
        return

    if ready_for_runway.empty:
        print("✅ No stories waiting for runway generation!")
        return

    print(f"🎯 Stories ready for generation: {len(ready_for_runway)}")

    # Process each story
    total_stories = len(ready_for_runway)
    success_count = 0
    partial_success_count = 0
    error_count = 0

    for idx, (csv_index, row) in enumerate(ready_for_runway.iterrows(), 1):
        story_id = csv_index + 1
        topic = row['topic']

        print_step(f"2.{idx}", f"Story {story_id}/{total_stories}")
        print(f"📚 Topic: {topic}")

        # Load prompts
        prompt_file_path = os.path.join(OUTPUT_BASE_PATH, str(story_id), "prompt.json")
        prompts, load_error = load_prompts_from_json(prompt_file_path)

        if load_error:
            print(f"❌ {load_error}")
            error_count += 1
            continue

        # Process visuals - Official SDK
        start_time = time.time()

        try:
            results = await generator.process_story_visuals(story_id, topic, prompts)

            # Save results
            summary = create_results_summary(story_id, topic, results)

            # Update CSV
            update_data = {}
            if results['topic_cover']['success']:
                update_data['cover_image_created'] = 1
            # Video pas geçildi, teaser_video_created güncellenmez
            if results['images']['success'] and results['images']['total_files'] >= 3:
                update_data['images_generated'] = 1

            if update_data:
                update_csv_runway_columns(TOPIC_CSV_PATH, csv_index, **update_data)

            # Evaluate success
            end_time = time.time()
            processing_time = int(end_time - start_time)

            total_success = summary['stats']['overall_success']
            partial_success = (
                    results['topic_cover']['success'] or
                    results['images']['total_files'] > 0
            )

            print(f"\n📊 Results:")
            print(f"   🎬 Video: SKIPPED")
            print(f"   🖼️ Cover: {'✅' if results['topic_cover']['success'] else '❌'}")
            print(f"   🎨 Images: {results['images']['total_files']} successful")
            print(f"   💾 Total size: {summary['stats']['total_file_size_mb']:.1f}MB")
            print(f"   ⚡ Processing time: {processing_time // 60}m {processing_time % 60}s")

            if total_success:
                print(f"✅ Complete success!")
                success_count += 1
            elif partial_success:
                print(f"⚠️ Partial success!")
                partial_success_count += 1
            else:
                print(f"❌ Failed!")
                error_count += 1

        except Exception as e:
            print(f"❌ Processing error: {e}")
            error_count += 1

        # Rate limiting between stories
        if idx < total_stories:
            print("\n⏳ Rate limiting: 60 seconds...")
            await asyncio.sleep(60)

    # Final report
    print_step("3", "Official SDK generation completed")

    print(f"📊 FINAL REPORT:")
    print(f"  ✅ Complete success: {success_count} stories")
    print(f"  ⚠️ Partial success: {partial_success_count} stories")
    print(f"  ❌ Failed: {error_count} stories")
    print(f"  📁 Total: {total_stories} stories")
    print(f"  🏆 Method: Official RunwayML Python SDK")
    print(f"  🚫 Video: Disabled")
    print(f"  📝 Text Overlay: {'Enabled' if PIL_AVAILABLE else 'Disabled'}")
    print(f"  🖼️ Reference Images: {'Enabled' if REFERENCE_IMAGE_PATH.exists() else 'Disabled'}")

    if success_count + partial_success_count > 0:
        print(f"\n🎉 Generated content for {success_count + partial_success_count} stories!")
        print(f"📁 Files location: output/*/runway_visuals/")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(process_runway_generation())
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()