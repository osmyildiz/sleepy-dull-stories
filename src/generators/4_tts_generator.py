import os
import re
import time
import json
import pandas as pd
from google.cloud import texttospeech
from pydub import AudioSegment
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# VOICE CONFIGURATION - Enceladus Voice
VOICE_NAME = "en-US-Chirp3-HD-Enceladus"
LANGUAGE_CODE = "en-US"


class ProgressTracker:
    """Scene processing progress tracking ve resume functionality"""

    def __init__(self, story_id: int, output_base_path: str):
        self.story_id = story_id
        self.output_dir = os.path.join(output_base_path, str(story_id))
        self.audio_parts_dir = os.path.join(self.output_dir, "audio_parts")
        self.progress_file = os.path.join(self.output_dir, "audio_progress.json")

        # Ensure directories exist
        os.makedirs(self.audio_parts_dir, exist_ok=True)

        # Load existing progress
        self.progress_data = self.load_progress()

    def load_progress(self):
        """Existing progress'i yükle"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"📂 Progress loaded: {len(data.get('completed_chunks', []))} chunks completed")
                return data
            except Exception as e:
                print(f"⚠️  Progress file corrupted, starting fresh: {e}")

        return {
            "story_id": self.story_id,
            "completed_chunks": [],
            "failed_chunks": [],
            "blacklisted_chunks": [],  # New: chunks that failed too many times
            "chunk_attempt_count": {},  # New: track attempts per chunk
            "total_cost_so_far": 0.0,
            "total_characters_so_far": 0,
            "last_update": datetime.now().isoformat(),
            "session_start": datetime.now().isoformat()
        }

    def save_progress(self):
        """Progress'i kaydet"""
        self.progress_data["last_update"] = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Progress save warning: {e}")

    def is_chunk_completed(self, chunk_name: str) -> bool:
        """Chunk tamamlanmış mı kontrol et"""
        return chunk_name in self.progress_data.get("completed_chunks", [])

    def is_chunk_blacklisted(self, chunk_name: str) -> bool:
        """Chunk blacklist'te mi kontrol et"""
        return chunk_name in self.progress_data.get("blacklisted_chunks", [])

    def get_chunk_audio_path(self, chunk_name: str) -> str:
        """Chunk için audio dosya path'i"""
        return os.path.join(self.audio_parts_dir, f"{chunk_name}_audio.mp3")

    def increment_chunk_attempts(self, chunk_name: str) -> int:
        """Chunk attempt sayısını artır ve döndür"""
        if "chunk_attempt_count" not in self.progress_data:
            self.progress_data["chunk_attempt_count"] = {}

        current_attempts = self.progress_data["chunk_attempt_count"].get(chunk_name, 0)
        self.progress_data["chunk_attempt_count"][chunk_name] = current_attempts + 1
        self.save_progress()
        return current_attempts + 1

    def blacklist_chunk(self, chunk_name: str, reason: str):
        """Chunk'ı blacklist'e ekle"""
        if "blacklisted_chunks" not in self.progress_data:
            self.progress_data["blacklisted_chunks"] = []

        if chunk_name not in self.progress_data["blacklisted_chunks"]:
            self.progress_data["blacklisted_chunks"].append(chunk_name)
            attempts = self.progress_data.get("chunk_attempt_count", {}).get(chunk_name, 0)
            print(f"⚫ {chunk_name}: Blacklisted after {attempts} failed attempts - {reason}")
            self.save_progress()

    def mark_chunk_completed(self, chunk_name: str, char_count: int, cost: float):
        """Chunk'ı tamamlandı olarak işaretle"""
        if chunk_name not in self.progress_data["completed_chunks"]:
            self.progress_data["completed_chunks"].append(chunk_name)
            self.progress_data["total_cost_so_far"] += cost
            self.progress_data["total_characters_so_far"] += char_count
            self.save_progress()
            print(f"      💾 Progress saved: {chunk_name} completed")

    def mark_chunk_failed(self, chunk_name: str, error: str):
        """Chunk'ı failed olarak işaretle"""
        if "failed_chunks" not in self.progress_data:
            self.progress_data["failed_chunks"] = []

        # Remove existing entry for this chunk if exists
        self.progress_data["failed_chunks"] = [
            f for f in self.progress_data["failed_chunks"]
            if not (isinstance(f, dict) and f.get("chunk_name") == chunk_name)
        ]

        # Add new failure record
        self.progress_data["failed_chunks"].append({
            "chunk_name": chunk_name,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self.save_progress()

    def get_resume_summary(self):
        """Resume özeti"""
        completed = len(self.progress_data.get("completed_chunks", []))
        blacklisted = len(self.progress_data.get("blacklisted_chunks", []))
        cost = self.progress_data.get("total_cost_so_far", 0.0)
        chars = self.progress_data.get("total_characters_so_far", 0)

        return {
            "completed_chunks": completed,
            "blacklisted_chunks": blacklisted,
            "total_cost_so_far": cost,
            "total_characters_so_far": chars,
            "can_resume": completed > 0
        }

    def cleanup_on_success(self):
        """Başarılı tamamlandığında progress dosyasını temizle"""
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
                print(f"🗑️  Progress file cleaned up (successful completion)")
        except Exception as e:
            print(f"⚠️  Progress cleanup warning: {e}")


class UsageTracker:
    """Google Cloud TTS API usage ve cost tracking"""

    def __init__(self):
        self.session_start = datetime.now()
        self.total_characters = 0
        self.total_requests = 0
        self.requests_log = []
        # Chirp3-HD pricing (premium voice)
        self.cost_per_million_chars = 16.0  # $16 per 1M characters

    def add_request(self, char_count, chunk_name, duration_seconds=0):
        """TTS request'i log'a ekle"""
        self.total_characters += char_count
        self.total_requests += 1

        request_cost = (char_count / 1_000_000) * self.cost_per_million_chars

        request_log = {
            "request_number": self.total_requests,
            "chunk_name": chunk_name,
            "characters": char_count,
            "cost": request_cost,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat()
        }

        self.requests_log.append(request_log)

        # Progress update
        total_cost = self.get_total_cost()
        print(f"      💰 Cost: ${request_cost:.6f} | Total: ${total_cost:.4f} | Chars: {self.total_characters:,}")

        return request_cost  # Return individual cost for progress tracking

    def get_total_cost(self):
        """Toplam maliyeti hesapla"""
        return (self.total_characters / 1_000_000) * self.cost_per_million_chars

    def get_session_duration(self):
        """Session süresini hesapla"""
        return (datetime.now() - self.session_start).total_seconds()

    def print_progress_summary(self, current_step, total_steps):
        """İlerleme özeti yazdır"""
        progress_percent = (current_step / total_steps) * 100
        total_cost = self.get_total_cost()
        session_duration = self.get_session_duration()

        print(f"\n📊 PROGRESS UPDATE ({current_step}/{total_steps} - {progress_percent:.1f}%)")
        print(f"   📝 Characters processed: {self.total_characters:,}")
        print(f"   🔄 Requests made: {self.total_requests}")
        print(f"   💰 Session cost: ${total_cost:.4f}")
        print(f"   ⏱️  Session time: {session_duration / 60:.1f} minutes")
        if current_step > 0:
            avg_cost_per_request = total_cost / self.total_requests
            print(f"   📈 Avg cost/request: ${avg_cost_per_request:.6f}")

    def print_final_summary(self):
        """Final session özeti"""
        total_cost = self.get_total_cost()
        session_duration = self.get_session_duration()
        free_tier_usage = (self.total_characters / 1_000_000) * 100

        print(f"\n💰 FINAL USAGE SUMMARY")
        print(f"=" * 50)
        print(f"📝 Total characters: {self.total_characters:,}")
        print(f"🔄 Total requests: {self.total_requests}")
        print(f"💰 Total cost: ${total_cost:.4f}")
        print(f"🆓 Free tier usage: {free_tier_usage:.2f}%")
        print(f"⏱️  Session duration: {session_duration / 60:.1f} minutes")

        if self.total_requests > 0:
            avg_chars_per_request = self.total_characters / self.total_requests
            avg_cost_per_request = total_cost / self.total_requests
            chars_per_second = self.total_characters / session_duration if session_duration > 0 else 0

            print(f"📊 Averages:")
            print(f"   📝 {avg_chars_per_request:.0f} chars/request")
            print(f"   💰 ${avg_cost_per_request:.6f} cost/request")
            print(f"   🚀 {chars_per_second:.1f} chars/second")

        # Free tier remaining
        free_tier_limit = 1_000_000  # 1M characters free per month
        remaining_chars = free_tier_limit - self.total_characters
        print(f"🎁 Free tier remaining: {remaining_chars:,} characters")

        return {
            "total_characters": self.total_characters,
            "total_requests": self.total_requests,
            "total_cost": total_cost,
            "session_duration_minutes": session_duration / 60,
            "free_tier_usage_percent": free_tier_usage
        }


def find_correct_paths():
    """Doğru dosya yollarını bul"""
    print("🔍 Path detection starting...")
    print(f"📂 Current working directory: {os.getcwd()}")

    # Show current directory contents
    try:
        files = os.listdir(".")
        print(f"📁 Current directory contents: {files}")
    except:
        pass

    possible_csv_paths = [
        "../data/topics.csv",  # src/generators/ → src/data/
        "../../data/topics.csv",  # Backup path
        "data/topics.csv",
        "src/data/topics.csv"
    ]

    possible_output_paths = [
        "../output",  # src/generators/ → src/output/
        "../../output",  # Backup path
        "output",
        "src/output"
    ]

    csv_path = None
    output_path = None

    # CSV path bul
    print("🔍 Looking for topics.csv...")
    for path in possible_csv_paths:
        abs_path = os.path.abspath(path)
        print(f"   Trying: {path} → {abs_path}")
        if os.path.exists(path):
            csv_path = path
            print(f"   ✅ CSV found: {path}")
            break
        else:
            print(f"   ❌ Not found")

    # Output path bul
    print("🔍 Looking for output directory...")
    for path in possible_output_paths:
        abs_path = os.path.abspath(path)
        print(f"   Trying: {path} → {abs_path}")
        if os.path.exists(path):
            output_path = path
            print(f"   ✅ Output dir found: {path}")
            break
        elif not os.path.exists(path):
            # Create if doesn't exist
            try:
                os.makedirs(path, exist_ok=True)
                output_path = path
                print(f"   ✅ Output dir created: {path}")
                break
            except Exception as e:
                print(f"   ❌ Could not create: {e}")
                continue

    return csv_path, output_path


# Doğru yolları bul
TOPIC_CSV_PATH, OUTPUT_BASE_PATH = find_correct_paths()

if not TOPIC_CSV_PATH:
    print("❌ topics.csv bulunamadı! Şu lokasyonları deneyin:")
    print("   - ../data/topics.csv (from src/generators/)")
    print("   - src/data/topics.csv")
if not OUTPUT_BASE_PATH:
    print("❌ output klasörü oluşturulamadı!")
    print("   - ../output/ (from src/generators/)")


def print_step(step_num: int, description: str):
    """Adımları yazdır"""
    print(f"\n🔄 Adım {step_num}: {description}")
    print("-" * 60)


def check_csv_for_audio_tasks():
    """CSV'den audio oluşturulacak hikayeleri bul"""
    try:
        df = pd.read_csv(TOPIC_CSV_PATH)

        # audio_generated kolunu kontrol et/oluştur
        if 'audio_generated' not in df.columns:
            df['audio_generated'] = 0
            df.to_csv(TOPIC_CSV_PATH, index=False)
            print("✅ audio_generated kolonu oluşturuldu")

        # done=1 VE images_generated=1 VE audio_generated=0 olan satırları bul
        ready_for_audio = df[(df['done'] == 1) & (df['images_generated'] == 1) & (df['audio_generated'] == 0)]

        return df, ready_for_audio

    except Exception as e:
        print(f"❌ CSV okuma hatası: {e}")
        return None, None


def update_csv_audio_status(csv_path: str, row_index: int, status: int):
    """CSV'de audio_generated kolumunu güncelle"""
    try:
        df = pd.read_csv(csv_path)
        df.at[row_index, 'audio_generated'] = status
        df.to_csv(csv_path, index=False)
        return True, None
    except Exception as e:
        return False, f"CSV güncelleme hatası: {e}"


def extract_hook_and_subscribe(complete_text):
    """Complete story'den Hook ve Subscribe kısımlarını çıkar"""
    hook_pattern = r'=== GOLDEN HOOK \(0-30 seconds\) ===(.*?)=== SUBSCRIBE REQUEST'
    subscribe_pattern = r'=== SUBSCRIBE REQUEST \(30-60 seconds\) ===(.*?)=== MAIN STORY'

    hook_match = re.search(hook_pattern, complete_text, re.DOTALL)
    subscribe_match = re.search(subscribe_pattern, complete_text, re.DOTALL)

    hook_text = hook_match.group(1).strip() if hook_match else None
    subscribe_text = subscribe_match.group(1).strip() if subscribe_match else None

    if hook_text and subscribe_text:
        return {
            "hook": hook_text,
            "subscribe": subscribe_text
        }
    return None


def load_stories_and_directions(story_id: int):
    """All stories, complete story ve voice directions dosyalarını yükle"""
    story_dir = os.path.join(OUTPUT_BASE_PATH, str(story_id))

    # all_stories.json dosyasını yükle
    stories_path = os.path.join(story_dir, "all_stories.json")
    if not os.path.exists(stories_path):
        return None, None, None, f"all_stories.json bulunamadı: {stories_path}"

    try:
        with open(stories_path, 'r', encoding='utf-8') as f:
            stories_data = json.load(f)
    except Exception as e:
        return None, None, None, f"all_stories.json okuma hatası: {e}"

    # complete_story.txt dosyasını yükle (Hook ve Subscribe için)
    complete_story_path = os.path.join(story_dir, "complete_story.txt")
    hook_and_subscribe = None
    if os.path.exists(complete_story_path):
        try:
            with open(complete_story_path, 'r', encoding='utf-8') as f:
                complete_text = f.read()
            hook_and_subscribe = extract_hook_and_subscribe(complete_text)
        except Exception as e:
            print(f"⚠️  complete_story.txt okuma hatası: {e}")

    # voice_directions.json dosyasını yükle
    voice_directions_path = os.path.join(story_dir, "voice_directions.json")
    if not os.path.exists(voice_directions_path):
        return stories_data, hook_and_subscribe, None, f"voice_directions.json bulunamadı: {voice_directions_path}"

    try:
        with open(voice_directions_path, 'r', encoding='utf-8') as f:
            voice_directions = json.load(f)
    except Exception as e:
        return stories_data, hook_and_subscribe, None, f"voice_directions.json okuma hatası: {e}"

    return stories_data, hook_and_subscribe, voice_directions, None


def check_scene_images(story_id: int, scene_count: int, has_hook_subscribe: bool = False):
    """Scene image'larının varlığını kontrol et - Hook/Subscribe scene_01.png kullanır"""
    story_dir = os.path.join(OUTPUT_BASE_PATH, str(story_id))
    scenes_dir = os.path.join(story_dir, "scenes")

    if not os.path.exists(scenes_dir):
        return [], f"Scenes klasörü bulunamadı: {scenes_dir}"

    available_scenes = []
    missing_scenes = []

    # Story scene'leri için direct mapping:
    # Hook + Subscribe + Story Scene 1 → scene_01.png
    # Story Scene 2 → scene_02.png
    # Story Scene 3 → scene_03.png
    # ...
    # Story Scene 40 → scene_40.png
    for story_scene_num in range(1, scene_count + 1):
        scene_filename = f"scene_{story_scene_num:02d}.png"
        scene_path = os.path.join(scenes_dir, scene_filename)

        if os.path.exists(scene_path):
            available_scenes.append(story_scene_num)
        else:
            missing_scenes.append(story_scene_num)

    print(f"🖼️  Scene Images Status:")
    if has_hook_subscribe:
        scene_01_exists = os.path.exists(os.path.join(scenes_dir, "scene_01.png"))
        print(f"   🎬 Hook + Subscribe + Scene 1: scene_01.png {'✅' if scene_01_exists else '❌'}")
    print(f"   ✅ Available story scenes: {len(available_scenes)}")
    print(f"   ❌ Missing story scenes: {len(missing_scenes)}")

    if missing_scenes:
        print(f"   📝 Missing story scenes: {missing_scenes}")
        print(f"   📝 Missing image files: {[f'scene_{i:02d}.png' for i in missing_scenes]}")

    return available_scenes, None


def get_youtube_intro_directions():
    """YouTube Hook ve Subscribe için özel voice directions"""
    return {
        "hook": {
            "scene_number": 0,
            "title": "Golden Hook - Channel Introduction",
            "direction": "Captivating, mysterious, cinematic buildup. Start with intrigue and wonder. Build anticipation like a movie trailer. Dramatic but engaging pace. Create that 'I must keep watching' feeling.",
            "template": "youtube_hook",
            "style": "cinematic_trailer",
            "speaking_rate": 0.8,  # Natural engaging pace (was 0.75 - too slow)
            "pitch": 0.1  # Slightly higher for engagement (reduced from 0.2)
        },
        "subscribe": {
            "scene_number": -1,
            "title": "Subscribe Request - Community Building",
            "direction": "Warm, personal, community-focused like MrBeast. Genuine connection with audience. Not salesy but genuinely inviting. Create feeling of joining a special community of dreamers and history lovers. Friendly but passionate.",
            "template": "youtube_subscribe",
            "style": "community_building",
            "speaking_rate": 0.8,  # Natural conversational pace
            "pitch": 0.0  # Natural pitch
        }
    }


def get_voice_direction_for_scene(voice_directions, scene_num):
    """Belirli bir scene için voice direction'ı bul"""
    if not voice_directions:
        return {}

    for direction in voice_directions:
        if direction.get('scene_number') == scene_num:
            return direction

    return {}


def format_time_ms(ms):
    """Millisecond'i MM:SS.mmm formatına çevir"""
    total_seconds = ms / 1000
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(ms % 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def is_retryable_error(error_str: str) -> bool:
    """API error'ının retry edilebilir olup olmadığını kontrol et"""
    retryable_patterns = [
        "503",  # Service unavailable
        "502",  # Bad gateway
        "500",  # Internal server error
        "429",  # Too many requests
        "timeout",
        "connection",
        "network",
        "unavailable",
        "bad gateway",
        "service unavailable"
    ]

    error_lower = error_str.lower()
    return any(pattern in error_lower for pattern in retryable_patterns)


def apply_scene_content_filter(scene_text, scene_num):
    """Apply content filtering for specific problematic scenes"""

    # Scene-specific filters
    if scene_num == 21:
        # Oracle scene filtering
        replacements = {
            "breathes in the mystical fumes": "receives sacred visions",
            "breathing in the sacred vapors": "receiving sacred visions",
            "sacred vapors": "mystical energy",
            "mystical fumes": "divine energy",
            "pupils dilated from": "eyes affected by",
            "consciousness drifting": "awareness expanding"
        }
        for old, new in replacements.items():
            scene_text = scene_text.replace(old, new)

    elif scene_num == 23:
        # Mother-child scene filtering
        replacements = {
            "rocks her youngest child": "holds her youngest child",
            "traces gentle circles on her child's back": "gently comforts her child",
            "baby stirs slightly in her arms": "baby rests peacefully",
            "tiny fist uncurling against Juno's chest": "tiny hand relaxing peacefully",
            "baby's eyelids flutter": "baby grows sleepy",
            "Mother Juno rocks": "Mother Juno holds",
            "chest as trust": "heart as trust"
        }
        for old, new in replacements.items():
            scene_text = scene_text.replace(old, new)

        # Add safety qualifier
        if "historical family scene" not in scene_text.lower():
            scene_text = "Historical family scene in ancient Rome: " + scene_text

    # Universal content safety measures
    general_replacements = {
        "children playing": "young people enjoying activities",
        "intimate": "peaceful",
        "tender": "gentle",
        "embrace": "peaceful moment"
    }

    for old, new in general_replacements.items():
        scene_text = scene_text.replace(old, new)

    return scene_text


def create_scene_audio_with_robust_retry(scene_text, voice_direction, chunk_name, progress_tracker, tracker=None,
                                         max_retries=5):
    """
    TTS generation with robust retry mechanism similar to visual generator
    Returns: (success: bool, error: str or None, file_path: str or None)
    """

    # Extract scene number for content filtering
    scene_num = None
    if chunk_name.startswith("scene_"):
        try:
            scene_num = int(chunk_name.split("_")[1])
        except:
            pass

    # Apply content filtering for problematic scenes
    if scene_num:
        original_length = len(scene_text)
        scene_text = apply_scene_content_filter(scene_text, scene_num)
        if len(scene_text) != original_length:
            print(
                f"      🛡️  Content filter applied to scene {scene_num} ({original_length} → {len(scene_text)} chars)")


def create_single_audio_chunk(scene_text, voice_direction, chunk_name, progress_tracker, tracker=None):
    """Create single audio chunk without retry - for internal use"""
    try:
        client = texttospeech.TextToSpeechClient()
        voice_name = VOICE_NAME
        language_code = LANGUAGE_CODE

        # Process text
        if "Chirp3-HD" in voice_name:
            processed_text = scene_text.replace("[PAUSE]", "... ")
            synthesis_input = texttospeech.SynthesisInput(text=processed_text)
        else:
            processed_text = scene_text.replace("[PAUSE]", '<break time="2s"/>')
            ssml_text = f'<speak>{processed_text}</speak>'
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

        # Voice settings
        direction_text = voice_direction.get('direction', '')
        speaking_rate = voice_direction.get('speaking_rate', 0.8)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )

        audio_config_params = {
            "audio_encoding": texttospeech.AudioEncoding.MP3,
            "speaking_rate": speaking_rate,
            "sample_rate_hertz": 44100
        }

        if "Chirp3-HD" not in voice_name:
            audio_config_params["pitch"] = voice_direction.get('pitch', 0.0)

        audio_config = texttospeech.AudioConfig(**audio_config_params)

        # Make request
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Save file
        file_path = progress_tracker.get_chunk_audio_path(chunk_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as out:
            out.write(response.audio_content)

        return True, None, file_path

    except Exception as e:
        return False, str(e), None


def create_scene_audio_with_robust_retry(scene_text, voice_direction, chunk_name, progress_tracker, tracker=None,
                                         max_retries=5):
    """
    TTS generation with robust retry mechanism similar to visual generator
    Returns: (success: bool, error: str or None, file_path: str or None)
    """

    # Extract scene number for content filtering
    scene_num = None
    if chunk_name.startswith("scene_"):
        try:
            scene_num = int(chunk_name.split("_")[1])
        except:
            pass

    # Apply content filtering for problematic scenes
    if scene_num:
        original_length = len(scene_text)
        scene_text = apply_scene_content_filter(scene_text, scene_num)
        if len(scene_text) != original_length:
            print(
                f"      🛡️  Content filter applied to scene {scene_num} ({original_length} → {len(scene_text)} chars)")

    # Handle overly long scenes (AUTO-SPLIT if >3000 chars)
    if len(scene_text) > 3000:
        print(f"      ⚠️  Scene too long ({len(scene_text)} chars) - AUTO-SPLITTING")

        # Find good split point
        split_point = len(scene_text) // 2
        search_start = max(0, split_point - 200)
        search_end = min(len(scene_text), split_point + 200)
        search_text = scene_text[search_start:search_end]

        sentence_ends = [i for i, char in enumerate(search_text) if char in '.!?']
        if sentence_ends:
            middle_offset = len(search_text) // 2
            best_split = min(sentence_ends, key=lambda x: abs(x - middle_offset))
            actual_split = search_start + best_split + 1
        else:
            space_pos = scene_text.rfind(' ', split_point - 100, split_point + 100)
            actual_split = space_pos if space_pos != -1 else split_point

        part1 = scene_text[:actual_split].strip()
        part2 = scene_text[actual_split:].strip()

        print(f"      ✂️  Split: Part 1 ({len(part1)} chars) + Part 2 ({len(part2)} chars)")

        # Process parts with simple retry
        success1, error1, file1 = None, None, None
        success2, error2, file2 = None, None, None

        for attempt in range(3):  # 3 attempts for each part
            if not success1:
                success1, error1, file1 = create_single_audio_chunk(
                    part1, voice_direction, f"{chunk_name}_part1", progress_tracker, tracker
                )
                if success1:
                    print(f"      ✅ Part 1 completed")
                else:
                    print(f"      🔄 Part 1 retry {attempt + 1}/3: {error1}")
                    time.sleep(5)

            if not success2:
                success2, error2, file2 = create_single_audio_chunk(
                    part2, voice_direction, f"{chunk_name}_part2", progress_tracker, tracker
                )
                if success2:
                    print(f"      ✅ Part 2 completed")
                else:
                    print(f"      🔄 Part 2 retry {attempt + 1}/3: {error2}")
                    time.sleep(5)

            if success1 and success2:
                break

        if not (success1 and success2):
            return False, f"Split parts failed: Part1={error1}, Part2={error2}", None

        # Combine parts
        try:
            from pydub import AudioSegment
            audio1 = AudioSegment.from_mp3(file1)
            audio2 = AudioSegment.from_mp3(file2)
            pause = AudioSegment.silent(duration=1000)  # 1 second pause
            combined = audio1 + pause + audio2

            combined_file = progress_tracker.get_chunk_audio_path(chunk_name)
            combined.export(combined_file, format="mp3")

            print(f"      🔗 Combined: {os.path.getsize(combined_file) / 1024:.1f} KB")

            # Mark as completed
            total_chars = len(part1) + len(part2)
            estimated_cost = (total_chars / 1_000_000) * 16.0
            progress_tracker.mark_chunk_completed(chunk_name, total_chars, estimated_cost)

            return True, None, combined_file

        except Exception as combine_error:
            return False, f"Combine failed: {combine_error}", None

    # NORMAL LENGTH PROCESSING (≤3000 chars)
    if len(scene_text) > 2500:
        print(f"      ⚠️  Scene quite long ({len(scene_text)} chars), processing as single chunk")

    # Check if already completed

    # Check if already completed
    if progress_tracker.is_chunk_completed(chunk_name):
        permanent_file = progress_tracker.get_chunk_audio_path(chunk_name)
        if os.path.exists(permanent_file):
            print(f"      ⏭️  Skipping {chunk_name} (already completed)")
            file_size = os.path.getsize(permanent_file) / 1024
            print(f"      ✅ Restored: {file_size:.1f} KB (from progress)")
            return True, None, permanent_file
        else:
            print(f"      ⚠️  Progress shows completed but file missing, regenerating: {chunk_name}")

    # Check if blacklisted
    if progress_tracker.is_chunk_blacklisted(chunk_name):
        print(f"      ⚫ Skipping {chunk_name} (blacklisted after too many failures)")
        return False, "Blacklisted due to repeated failures", None

    # Retry delays: 10s, 20s, 30s, 60s, 120s
    retry_delays = [10, 20, 30, 60, 120]

    for attempt in range(max_retries + 1):  # 0-5 (6 total attempts)
        try:
            # Increment attempt count
            attempt_count = progress_tracker.increment_chunk_attempts(chunk_name)

            print(f"      🎯 {chunk_name} - Attempt {attempt + 1}/{max_retries + 1} (Total attempts: {attempt_count})")

            # Blacklist if too many attempts
            if attempt_count > 8:  # After 8 total attempts across all sessions
                progress_tracker.blacklist_chunk(chunk_name, f"Exceeded {attempt_count} total attempts")
                return False, f"Blacklisted after {attempt_count} attempts", None

            client = texttospeech.TextToSpeechClient()

            # User specified Enceladus voice
            voice_name = VOICE_NAME
            language_code = LANGUAGE_CODE

            # Text'i hazırla - Chirp3-HD voices SSML desteklemiyor
            if "Chirp3-HD" in voice_name:
                processed_text = scene_text.replace("[PAUSE]", "... ")
                synthesis_input = texttospeech.SynthesisInput(text=processed_text)
                if attempt == 0:  # Only print once
                    print(f"      ⚠️  Chirp3-HD voice detected: using plain text (no SSML)")
            else:
                processed_text = scene_text.replace("[PAUSE]", '<break time="2s"/>')
                ssml_text = f'<speak>{processed_text}</speak>'
                synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

            # Direction'dan ses özelliklerini belirle
            direction_text = voice_direction.get('direction', '')
            speaking_rate = voice_direction.get('speaking_rate', 0.8)

            # Direction'a göre speaking rate ayarla
            if 'speaking_rate' not in voice_direction:
                if 'slow' in direction_text.lower() or 'meditative' in direction_text.lower():
                    speaking_rate = 0.7
                elif 'rhythmic' in direction_text.lower() or 'flowing' in direction_text.lower():
                    speaking_rate = 0.85
                elif 'gentle' in direction_text.lower() or 'tender' in direction_text.lower():
                    speaking_rate = 0.75
                elif 'alert' in direction_text.lower() or 'business' in direction_text.lower():
                    speaking_rate = 0.9

            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )

            # Audio config - Chirp3-HD doesn't support pitch
            audio_config_params = {
                "audio_encoding": texttospeech.AudioEncoding.MP3,
                "speaking_rate": speaking_rate,
                "sample_rate_hertz": 44100
            }

            if "Chirp3-HD" not in voice_name:
                audio_config_params["pitch"] = voice_direction.get('pitch', 0.0)
            elif attempt == 0:
                print(f"      ⚠️  Chirp3-HD voice detected: pitch adjustment disabled")

            audio_config = texttospeech.AudioConfig(**audio_config_params)

            if attempt == 0:  # Only print details on first attempt
                print(f"      📤 TTS request ({len(scene_text)} chars)")
                print(f"      🎭 Direction: {direction_text[:50]}...")
                print(f"      🎙️ Voice: {voice_name}")
                print(f"      ⚡ Speaking rate: {speaking_rate}")

                # CONTENT ANALYSIS FOR DEBUGGING
                print(f"      🔍 CONTENT ANALYSIS:")
                # Check for potentially problematic words
                problematic_words = [
                    "child", "children", "baby", "infant", "mother", "father",
                    "embrace", "kiss", "intimate", "tender", "chest", "arms",
                    "rocks", "holds", "touches", "caress", "love", "family"
                ]
                found_words = [word for word in problematic_words if word.lower() in scene_text.lower()]
                if found_words:
                    print(f"         ├─ Flagged words: {found_words[:10]}...")  # First 10

                # Check text length categories
                if len(scene_text) > 3000:
                    print(f"         ├─ WARNING: Very long text ({len(scene_text)} chars)")
                elif len(scene_text) > 2000:
                    print(f"         ├─ CAUTION: Long text ({len(scene_text)} chars)")

                # Show first and last 100 chars for debugging
                text_start = scene_text[:100].replace('\n', ' ')
                text_end = scene_text[-100:].replace('\n', ' ')
                print(f"         ├─ Text start: {text_start}...")
                print(f"         └─ Text end: ...{text_end}")

            # TTS request timing
            start_time = time.time()

            try:
                response = client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                generation_time = time.time() - start_time

            except Exception as tts_error:
                generation_time = time.time() - start_time

                # CAPTURE DETAILED TTS API ERROR
                print(f"      🚨 TTS API ERROR DETAILS:")
                print(f"         ├─ TTS Error Type: {type(tts_error).__name__}")
                print(f"         ├─ TTS Error: {str(tts_error)}")
                print(f"         ├─ Request Duration: {generation_time:.2f}s")

                # Check for Google Cloud specific errors
                if hasattr(tts_error, 'code'):
                    print(f"         ├─ Google Cloud Error Code: {tts_error.code}")
                if hasattr(tts_error, 'message'):
                    print(f"         ├─ Google Cloud Message: {tts_error.message}")
                if hasattr(tts_error, 'details'):
                    print(f"         ├─ Google Cloud Details: {tts_error.details}")

                # Re-raise for outer exception handler
                raise tts_error

            # Save to permanent location immediately
            permanent_file = progress_tracker.get_chunk_audio_path(chunk_name)
            os.makedirs(os.path.dirname(permanent_file), exist_ok=True)

            with open(permanent_file, "wb") as out:
                out.write(response.audio_content)

            file_size = os.path.getsize(permanent_file) / 1024

            if attempt > 0:
                print(f"      🎉 Success on attempt {attempt + 1}: {file_size:.1f} KB in {generation_time:.2f}s")
            else:
                print(f"      ✅ Success: {file_size:.1f} KB in {generation_time:.2f}s")

            # Usage tracking
            cost = 0
            if tracker:
                cost = tracker.add_request(len(scene_text), chunk_name, generation_time)
            else:
                cost = (len(scene_text) / 1_000_000) * 16.0  # Estimate cost

            # Mark as completed
            progress_tracker.mark_chunk_completed(chunk_name, len(scene_text), cost)

            return True, None, permanent_file

        except Exception as e:
            error_str = str(e)

            # DETAILED ERROR DEBUGGING
            print(f"      🐛 DETAILED ERROR DEBUG:")
            print(f"         ├─ Error Type: {type(e).__name__}")
            print(f"         ├─ Error Message: {error_str}")
            print(f"         ├─ Scene Number: {scene_num}")
            print(f"         ├─ Text Length: {len(scene_text)} chars")
            print(f"         ├─ Voice: {voice_name}")
            print(f"         ├─ Speaking Rate: {speaking_rate}")

            # Check for specific error patterns
            if hasattr(e, 'response') and e.response is not None:
                try:
                    print(f"         ├─ HTTP Status: {e.response.status_code}")
                    response_text = e.response.text[:500] if hasattr(e.response, 'text') else str(e.response)
                    print(f"         ├─ Response Text: {response_text}")
                except Exception as debug_error:
                    print(f"         ├─ Response: [Could not read - {debug_error}]")

            # Handle gRPC errors (Google Cloud)
            if hasattr(e, 'code') and hasattr(e, 'details'):
                try:
                    print(f"         ├─ gRPC Code: {e.code()}")
                    print(f"         ├─ gRPC Details: {e.details()}")
                except Exception as grpc_error:
                    print(f"         ├─ gRPC Info: [Could not read - {grpc_error}]")

            # Show first 200 chars of problematic text
            text_preview = scene_text[:200].replace('\n', ' ')
            print(f"         ├─ Text Preview: {text_preview}...")

            # Check for common content policy indicators
            policy_indicators = [
                "content", "policy", "violation", "inappropriate",
                "safety", "harmful", "blocked", "filtered", "denied"
            ]
            error_lower = error_str.lower()
            found_indicators = [ind for ind in policy_indicators if ind in error_lower]
            if found_indicators:
                print(f"         ├─ Policy Indicators: {found_indicators}")

            # Check for rate limiting indicators
            rate_indicators = ["rate", "limit", "quota", "throttle", "too many"]
            found_rate = [ind for ind in rate_indicators if ind in error_lower]
            if found_rate:
                print(f"         ├─ Rate Limit Indicators: {found_rate}")

            print(f"         └─ Attempt: {attempt + 1}/{max_retries + 1}")

            # Mark as failed
            progress_tracker.mark_chunk_failed(chunk_name, error_str)

            if attempt < max_retries and is_retryable_error(error_str):
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                print(f"      🔄 Retry {attempt + 1}/{max_retries} in {delay}s (Error: {error_str[:100]}...)")
                time.sleep(delay)
                continue
            else:
                print(f"      ❌ Final failure after {attempt + 1} attempts: {error_str}")

                # Consider blacklisting if too many failures
                if attempt_count >= 5:
                    progress_tracker.blacklist_chunk(chunk_name, f"Failed {attempt_count} times: {error_str[:100]}")

                return False, error_str, None

    return False, "Max retries exceeded", None


def process_scene_based_audio_generation_with_retry(stories_data, hook_and_subscribe, voice_directions,
                                                    available_scenes,
                                                    output_file, story_id, quality="youtube", max_retry_rounds=5):
    """Scene-based audio generation with robust retry system"""
    print("🎵 SCENE-BASED AUDIO GENERATION WITH ROBUST RETRY")
    print("🎬 YouTube Production Quality + Hook & Subscribe")
    print(f"🎙️ Voice: {VOICE_NAME}")
    print(f"🔄 Retry system: {max_retry_rounds} rounds with exponential backoff")
    print("=" * 70)

    # Initialize trackers
    progress_tracker = ProgressTracker(story_id, OUTPUT_BASE_PATH)
    tracker = UsageTracker()

    # Show resume info
    resume_info = progress_tracker.get_resume_summary()
    if resume_info["can_resume"]:
        print(f"📂 RESUMING: {resume_info['completed_chunks']} chunks already completed")
        print(f"   💰 Previous cost: ${resume_info['total_cost_so_far']:.4f}")
        print(f"   📝 Previous chars: {resume_info['total_characters_so_far']:,}")
        if resume_info['blacklisted_chunks'] > 0:
            print(f"   ⚫ Blacklisted: {resume_info['blacklisted_chunks']} chunks")

    # Kalite ayarları
    quality_settings = {
        "youtube": {"bitrate": "192k", "description": "YouTube High Quality", "sample_rate": 44100},
        "podcast": {"bitrate": "128k", "description": "Podcast Quality", "sample_rate": 44100},
        "balanced": {"bitrate": "96k", "description": "Balanced Quality", "sample_rate": 22050}
    }

    settings = quality_settings.get(quality, quality_settings["youtube"])
    print(f"🎛️  Quality: {quality} ({settings['description']}) - {settings['bitrate']}")

    # YouTube intro directions
    intro_directions = get_youtube_intro_directions()

    # Determine all chunks to process
    all_chunks = []

    # Add hook if available
    if hook_and_subscribe and hook_and_subscribe.get('hook'):
        all_chunks.append(("hook", hook_and_subscribe['hook'], intro_directions['hook']))

    # Add subscribe if available
    if hook_and_subscribe and hook_and_subscribe.get('subscribe'):
        all_chunks.append(("subscribe", hook_and_subscribe['subscribe'], intro_directions['subscribe']))

    # Add story scenes
    for scene_num in available_scenes:
        scene_key = str(scene_num)
        if scene_key in stories_data:
            scene_text = stories_data[scene_key]
            voice_direction = get_voice_direction_for_scene(voice_directions, scene_num)
            all_chunks.append((f"scene_{scene_num}", scene_text, voice_direction))

    print(f"📊 Total chunks to process: {len(all_chunks)}")

    # Process chunks with retry rounds
    for retry_round in range(max_retry_rounds):
        print(f"\n{'🔄 RETRY ROUND ' + str(retry_round) if retry_round > 0 else '🚀 INITIAL ROUND'}")
        print("=" * 50)

        # Get missing chunks (not completed and not blacklisted)
        missing_chunks = []
        for chunk_name, chunk_text, chunk_direction in all_chunks:
            if not progress_tracker.is_chunk_completed(chunk_name) and not progress_tracker.is_chunk_blacklisted(
                    chunk_name):
                missing_chunks.append((chunk_name, chunk_text, chunk_direction))

        if not missing_chunks:
            print("✅ All chunks completed!")
            break

        print(f"📝 Missing chunks: {len(missing_chunks)}")

        # Show blacklisted chunks if any
        blacklisted_count = len(progress_tracker.progress_data.get("blacklisted_chunks", []))
        if blacklisted_count > 0:
            print(f"⚫ Blacklisted chunks: {blacklisted_count}")

        # Wait between retry rounds (except first)
        if retry_round > 0:
            wait_time = 30 + (retry_round * 15)  # 45s, 60s, 75s, 90s
            print(f"⏳ Waiting {wait_time}s before retry round...")
            time.sleep(wait_time)

        # Process missing chunks
        successful_in_round = 0
        failed_in_round = 0

        for i, (chunk_name, chunk_text, chunk_direction) in enumerate(missing_chunks):
            print(f"\n📄 Processing {chunk_name} ({i + 1}/{len(missing_chunks)})")

            success, error, file_path = create_scene_audio_with_robust_retry(
                chunk_text, chunk_direction, chunk_name, progress_tracker, tracker
            )

            if success:
                successful_in_round += 1
                print(f"      ✅ {chunk_name} completed and saved to audio_parts/")
            else:
                failed_in_round += 1
                print(f"      ❌ {chunk_name} failed: {error}")

            # Rate limiting between chunks
            if i < len(missing_chunks) - 1:
                base_wait = 3 if retry_round == 0 else 5 + retry_round
                time.sleep(base_wait)

        print(f"\n📊 Round {retry_round + 1} Results:")
        print(f"   ✅ Successful: {successful_in_round}")
        print(f"   ❌ Failed: {failed_in_round}")

        # Print progress summary
        if successful_in_round > 0:
            completed_total = len(progress_tracker.progress_data.get("completed_chunks", []))
            tracker.print_progress_summary(completed_total, len(all_chunks))

    # Final check and combine
    print(f"\n🔍 FINAL COMBINATION CHECK:")
    print("=" * 50)

    final_missing = []
    completed_chunks = []

    for chunk_name, chunk_text, chunk_direction in all_chunks:
        if progress_tracker.is_chunk_completed(chunk_name):
            file_path = progress_tracker.get_chunk_audio_path(chunk_name)
            if os.path.exists(file_path):
                completed_chunks.append((chunk_name, file_path, chunk_direction))
                print(f"   ✅ {chunk_name}: {os.path.getsize(file_path) / 1024:.1f} KB")
            else:
                final_missing.append(chunk_name)
                print(f"   ❌ {chunk_name}: marked complete but file missing")
        else:
            final_missing.append(chunk_name)
            print(f"   ⏳ {chunk_name}: not completed")

    if final_missing:
        blacklisted_chunks = progress_tracker.progress_data.get("blacklisted_chunks", [])
        missing_not_blacklisted = [c for c in final_missing if c not in blacklisted_chunks]

        if missing_not_blacklisted:
            print(f"\n❌ Still missing after {max_retry_rounds} rounds: {missing_not_blacklisted}")
            return False, f"Missing chunks: {missing_not_blacklisted}", None
        else:
            print(f"\n⚠️  Some chunks blacklisted but continuing with {len(completed_chunks)} completed chunks")

    if not completed_chunks:
        return False, "No chunks were successfully generated", None

    # Combine completed chunks
    print(f"\n🔗 Combining {len(completed_chunks)} audio chunks...")

    try:
        combined = AudioSegment.empty()
        current_time_ms = 0
        timeline_data = {
            "total_scenes": len(completed_chunks),
            "pause_between_scenes_ms": 2000,
            "scenes": [],
            "total_duration_ms": 0,
            "total_duration_formatted": "",
            "created_at": datetime.now().isoformat(),
            "youtube_optimized": True,
            "voice_used": VOICE_NAME,
            "story_id": story_id,
            "retry_rounds_used": max_retry_rounds,
            "final_missing_count": len(final_missing),
            "blacklisted_count": len(progress_tracker.progress_data.get("blacklisted_chunks", []))
        }

        for i, (chunk_name, file_path, voice_direction) in enumerate(completed_chunks):
            print(f"   🔗 Adding {chunk_name}")

            audio = AudioSegment.from_mp3(file_path)
            audio_duration_ms = len(audio)

            # Timeline data
            chunk_start_ms = current_time_ms
            chunk_end_ms = current_time_ms + audio_duration_ms

            # Determine chunk type
            if chunk_name == "hook":
                chunk_type = "youtube_hook"
                scene_num = 0
                image_file = "scene_01.png"
            elif chunk_name == "subscribe":
                chunk_type = "youtube_subscribe"
                scene_num = -1
                image_file = "scene_01.png"
            else:
                chunk_type = "story_scene"
                scene_num = int(chunk_name.split('_')[1])
                image_file = f"scene_{scene_num:02d}.png"

            chunk_timeline = {
                "type": chunk_type,
                "scene_number": scene_num,
                "title": voice_direction.get('title', chunk_name),
                "direction": voice_direction.get('direction', ''),
                "start_time_ms": chunk_start_ms,
                "end_time_ms": chunk_end_ms,
                "duration_ms": audio_duration_ms,
                "start_time_formatted": format_time_ms(chunk_start_ms),
                "end_time_formatted": format_time_ms(chunk_end_ms),
                "duration_formatted": format_time_ms(audio_duration_ms),
                "audio_file": f"{chunk_name}_audio.mp3",
                "image_file": image_file
            }

            timeline_data["scenes"].append(chunk_timeline)

            # Add audio
            combined += audio
            current_time_ms += audio_duration_ms

            # Add pause (except last)
            if i < len(completed_chunks) - 1:
                pause = AudioSegment.silent(duration=2000)
                combined += pause
                current_time_ms += 2000

        # Finalize timeline
        timeline_data["total_duration_ms"] = current_time_ms
        timeline_data["total_duration_formatted"] = format_time_ms(current_time_ms)

        # Add usage summary
        usage_summary = tracker.print_final_summary()
        timeline_data["usage_summary"] = usage_summary

        # Export final audio
        print(f"\n💾 Exporting final YouTube audio...")
        combined.export(
            output_file,
            format="mp3",
            bitrate=settings['bitrate'],
            parameters=[
                "-ac", "2",  # Stereo
                "-ar", str(settings['sample_rate']),  # Sample rate
                "-q:a", "0"  # Highest quality
            ]
        )

        # Save timeline
        timeline_file = output_file.replace('.mp3', '_timeline.json')
        with open(timeline_file, 'w', encoding='utf-8') as f:
            json.dump(timeline_data, f, indent=2, ensure_ascii=False)

        # Stats
        duration_min = len(combined) / 60000
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

        print(f"\n🎉 SUCCESS!")
        print(f"   📁 Audio: {output_file}")
        print(f"   📋 Timeline: {timeline_file}")
        print(f"   ⏱️  Duration: {duration_min:.1f} minutes")
        print(f"   📦 Size: {file_size_mb:.1f} MB")
        print(f"   🎭 Chunks used: {len(completed_chunks)}/{len(all_chunks)}")

        # Cleanup progress on success
        progress_tracker.cleanup_on_success()

        return True, None, timeline_data

    except Exception as e:
        return False, f"Audio combining failed: {e}", None


def create_audio_summary(story_id: int, topic: str, success: bool, scenes_processed: int = 0, error: str = None,
                         usage_data: dict = None):
    """Audio oluşturma özeti - Usage tracking ile"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "story_id": story_id,
        "topic": topic,
        "audio_generation_success": success,
        "scenes_processed": scenes_processed,
        "voice_used": VOICE_NAME,
        "error": error,
        "approach": "robust_retry_system_with_progress_tracking",
        "quality": "youtube_production",
        "retry_system": "enabled_with_blacklisting"
    }

    # Usage data ekle
    if usage_data:
        summary["usage_data"] = usage_data

    output_dir = os.path.join(OUTPUT_BASE_PATH, str(story_id))
    summary_path = os.path.join(output_dir, "audio_summary.json")

    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ Audio summary saved: {summary_path}")
    except Exception as e:
        print(f"⚠️  Summary save warning: {e}")

    return summary


def process_audio_pipeline(quality="youtube"):
    """Ana audio generation pipeline - Robust retry system"""

    print("🎵 SLEEPY DULL STORIES - YouTube Audio Pipeline v10.0")
    print("🎬 Hook + Subscribe + Voice Directions + Enceladus Voice")
    print(f"🎙️ Voice: {VOICE_NAME}")
    print("🔄 ROBUST RETRY SYSTEM - Inspired by Visual Generator")
    print("💾 IMMEDIATE SAVE - Each chunk saved to audio_parts/ immediately")
    print("⚫ BLACKLIST SYSTEM - Failed chunks get blacklisted after too many attempts")
    print("=" * 80)

    # Path kontrolü
    if not TOPIC_CSV_PATH or not OUTPUT_BASE_PATH:
        print("❌ Gerekli dosya yolları bulunamadı!")
        print("💡 Lütfen script'i doğru dizinde çalıştırdığınızdan emin olun.")
        return

    print(f"📁 CSV Path: {TOPIC_CSV_PATH}")
    print(f"📁 Output Path: {OUTPUT_BASE_PATH}")

    # Adım 1: CSV'yi kontrol et
    print_step(1, "CSV dosyası kontrol ediliyor")

    df, ready_for_audio = check_csv_for_audio_tasks()
    if df is None:
        return

    if ready_for_audio.empty:
        print("✅ Audio oluşturulacak hikaye bulunamadı!")
        print("💡 Koşullar: done=1 AND images_generated=1 AND audio_generated=0")
        print("💡 Hikayelerin önce image generation'ı tamamlanmalı.")
        return

    print(f"🎯 Audio oluşturulacak hikaye sayısı: {len(ready_for_audio)}")

    # Adım 2: Her hikaye için audio oluştur
    total_stories = len(ready_for_audio)
    success_count = 0
    error_count = 0

    for idx, (csv_index, row) in enumerate(ready_for_audio.iterrows(), 1):

        story_id = csv_index + 1  # CSV satır numarası (1-based)
        topic = row['topic']

        print_step(f"2.{idx}", f"Hikaye {story_id}/{total_stories} işleniyor - ROBUST RETRY")
        print(f"📚 Konu: {topic}")
        print(f"📄 CSV Index: {csv_index}")

        # Stories, hook/subscribe ve voice directions dosyalarını yükle
        stories_data, hook_and_subscribe, voice_directions, load_error = load_stories_and_directions(story_id)

        if load_error:
            print(f"❌ {load_error}")
            create_audio_summary(story_id, topic, False, 0, load_error, None)
            error_count += 1
            continue

        scene_count = len(stories_data)
        has_hook = bool(hook_and_subscribe and hook_and_subscribe.get('hook'))
        has_subscribe = bool(hook_and_subscribe and hook_and_subscribe.get('subscribe'))

        print(f"📊 Total scenes in stories: {scene_count}")
        print(f"🎬 YouTube Hook: {'✅' if has_hook else '❌'}")
        print(f"📢 Subscribe Request: {'✅' if has_subscribe else '❌'}")
        print(f"🎭 Voice directions loaded: {'Yes' if voice_directions else 'No'}")

        # Scene image'larını kontrol et
        available_scenes, image_error = check_scene_images(story_id, scene_count, has_hook or has_subscribe)

        if image_error:
            print(f"❌ {image_error}")
            create_audio_summary(story_id, topic, False, 0, image_error, None)
            error_count += 1
            continue

        if not available_scenes:
            error_msg = "Scene image'ları bulunamadı. Audio oluşturma atlanıyor."
            print(f"❌ {error_msg}")
            create_audio_summary(story_id, topic, False, 0, error_msg, None)
            error_count += 1
            continue

        # Audio output path
        audio_output = os.path.join(OUTPUT_BASE_PATH, str(story_id), "story_audio_youtube.mp3")

        # Audio oluştur with robust retry
        start_time = time.time()

        print(f"🔧 Starting robust retry audio generation for story_id={story_id}")
        print(f"🔧 Audio output path: {audio_output}")
        print(f"🔧 Audio parts will be saved to: {OUTPUT_BASE_PATH}/{story_id}/audio_parts/")

        success, error, timeline_data = process_scene_based_audio_generation_with_retry(
            stories_data=stories_data,
            hook_and_subscribe=hook_and_subscribe,
            voice_directions=voice_directions,
            available_scenes=available_scenes,
            output_file=audio_output,
            story_id=story_id,
            quality=quality,
            max_retry_rounds=5
        )

        end_time = time.time()
        processing_time = int(end_time - start_time)

        if success:
            print(f"✅ Robust retry audio generation başarılı!")
            print(f"⚡ İşlem süresi: {processing_time // 60}m {processing_time % 60}s")
            print(f"🎬 YouTube Hook: {'✅' if has_hook else '❌'}")
            print(f"📢 Subscribe Request: {'✅' if has_subscribe else '❌'}")
            print(f"💾 Audio parts preserved in audio_parts/ directory")

            if timeline_data:
                hook_count = sum(1 for scene in timeline_data['scenes'] if scene.get('type') == 'youtube_hook')
                subscribe_count = sum(
                    1 for scene in timeline_data['scenes'] if scene.get('type') == 'youtube_subscribe')
                story_count = sum(1 for scene in timeline_data['scenes'] if scene.get('type') == 'story_scene')

                print(f"📋 Timeline: {hook_count} hook + {subscribe_count} subscribe + {story_count} scenes")
                print(f"⏱️  Total duration: {timeline_data['total_duration_formatted']}")

                if timeline_data.get('blacklisted_count', 0) > 0:
                    print(f"⚫ Blacklisted chunks: {timeline_data['blacklisted_count']}")

            # CSV'yi güncelle
            update_success, update_error = update_csv_audio_status(TOPIC_CSV_PATH, csv_index, 1)

            if update_success:
                print(f"✅ CSV güncellendi: audio_generated = 1")
                success_count += 1
            else:
                print(f"⚠️  CSV güncelleme hatası: {update_error}")

            # Summary oluştur
            usage_data = timeline_data.get('usage_summary') if timeline_data else None
            create_audio_summary(story_id, topic, True, len(available_scenes), None, usage_data)

        else:
            print(f"❌ Robust retry audio generation başarısız: {error}")
            print(f"💾 Partial progress saved in audio_parts/ - can resume later")
            create_audio_summary(story_id, topic, False, len(available_scenes), error, None)
            error_count += 1

    # Adım 3: Özet rapor
    print_step(3, "YouTube audio pipeline tamamlandı")

    print(f"📊 SONUÇ RAPORU:")
    print(f"  ✅ Başarılı: {success_count} hikaye")
    print(f"  ❌ Hatalı: {error_count} hikaye")
    print(f"  📁 Toplam: {total_stories} hikaye")
    print(f"  🎙️ Voice: {VOICE_NAME}")
    print(f"  🎬 Quality: {quality} (YouTube Production)")
    print(f"  🔄 Retry System: Robust with blacklisting")

    if success_count > 0:
        print(f"\n🎉 {success_count} hikaye için YouTube-optimized audio başarıyla oluşturuldu!")
        print(f"📁 Audio dosyaları: src/output/*/story_audio_youtube.mp3")
        print(f"📋 Timeline dosyaları: src/output/*/story_audio_youtube_timeline.json")
        print(f"📊 Usage summaries: src/output/*/audio_summary.json")
        print(f"💾 Audio parts saved: src/output/*/audio_parts/*.mp3 (resume protection)")
        print(f"🎬 YouTube Hook & Subscribe dahil edildi")
        print(f"🎭 Voice directions kullanıldı")
        print(f"🖼️  Scene images kontrol edildi")
        print(f"🎙️ Enceladus voice quality")
        print(f"🎛️  YouTube production kalitesi (192kbps)")
        print(f"💰 Real-time cost tracking")
        print(f"🛡️  ROBUST API error protection with exponential backoff")
        print(f"⚫ Automatic blacklisting of repeatedly failed chunks")
        print(f"⏸️  Perfect timing for video editing")
        print(f"🔄 Resume capability - restart anytime to continue")

    print("\n" + "=" * 80)


# Kalite seçenekleri
def show_quality_options():
    """Kalite seçeneklerini göster"""
    print("🎛️  QUALITY OPTIONS:")
    print("   🎬 youtube: 192kbps, 44.1kHz (~150MB for 2h) - YouTube optimized")
    print("   📻 podcast: 128kbps, 44.1kHz (~100MB for 2h) - Podcast quality")
    print("   📊 balanced: 96kbps, 22kHz (~75MB for 2h) - Balanced")


if __name__ == "__main__":
    try:
        print("🚀 YouTube Audio Generation Pipeline Starting...")
        print(f"📂 Current directory: {os.getcwd()}")
        print(f"🎙️ Voice: {VOICE_NAME}")
        print(f"💰 Real-time usage tracking enabled")
        print(f"🛡️  ROBUST RETRY SYSTEM - Visual Generator Inspired")
        print(f"💾 IMMEDIATE SAVE - Each chunk saved immediately")
        print(f"⚫ BLACKLIST SYSTEM - Auto-blacklist failed chunks")
        print(f"🔄 RESUME SYSTEM - Continue from any interruption")

        show_quality_options()

        # Kalite seçimi - YouTube production
        quality = "youtube"  # 192kbps, 44.1kHz - YouTube optimized

        print(f"🎯 Selected quality: {quality} (192kbps, 44.1kHz - YouTube Production)")

        process_audio_pipeline(quality)

    except KeyboardInterrupt:
        print("\n⚠️  İşlem kullanıcı tarafından durduruldu")
        print("🛡️  Progress saved! Restart to resume from last completed scene.")
        print("💾 All completed chunks saved in audio_parts/ directory")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        print("🛡️  Progress saved! Check audio_progress.json for resume info.")
        print("🔄 Robust retry system will attempt recovery on restart.")
        print("💾 Completed chunks preserved in audio_parts/ directory")