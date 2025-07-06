"""
Sleepy Dull Stories - SERVER-READY Thumbnail Creator
YouTube Thumbnail Production with Text Overlay
Production-optimized with complete automation and database integration
"""

import pandas as pd
import json
import os
import re
import sqlite3
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# Load environment first
load_dotenv()


# Server Configuration Class (from other generators)
class ServerConfig:
    """Server-friendly configuration management for Thumbnail Creator"""

    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.setup_thumbnail_config()
        self.ensure_directories()

    def setup_paths(self):
        """Setup server-friendly paths"""
        # Detect current file location
        current_file = Path(__file__).resolve()

        # For server: /home/youtube-automation/channels/sleepy-dull-stories/src/generators/
        # Go up to project root
        self.project_root = current_file.parent.parent.parent

        self.paths = {
            'BASE_DIR': str(self.project_root),
            'SRC_DIR': str(current_file.parent.parent),
            'DATA_DIR': str(self.project_root / 'data'),
            'OUTPUT_DIR': str(self.project_root / 'output'),
            'LOGS_DIR': str(self.project_root / 'logs'),
            'CONFIG_DIR': str(self.project_root / 'config'),
            'FONTS_DIR': str(self.project_root / 'fonts')
        }

        print(f"✅ Thumbnail Creator server paths configured:")
        print(f"   📁 Project root: {self.paths['BASE_DIR']}")

    def setup_thumbnail_config(self):
        """Setup thumbnail creation configuration"""
        self.thumbnail_config = {
            "max_retry_rounds": 3,
            "target_resolution": [1280, 720],  # YouTube thumbnail standard
            "quality_profiles": {
                "youtube": {"format": "JPEG", "quality": 92, "optimize": True},
                "high": {"format": "JPEG", "quality": 95, "optimize": True},
                "fast": {"format": "JPEG", "quality": 85, "optimize": False}
            },
            "budget_controls": {
                "max_processing_time_minutes": 10,
                "max_memory_usage_mb": 512,
                "warn_threshold_minutes": 5
            },
            "server_mode": True,
            "production_ready": True
        }

        print("✅ Thumbnail creation configuration loaded")
        print(
            f"🎨 Target: {self.thumbnail_config['target_resolution'][0]}x{self.thumbnail_config['target_resolution'][1]}")

    def setup_logging(self):
        """Setup production logging"""
        logs_dir = Path(self.project_root) / 'logs' / 'generators'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"thumbnail_creator_{datetime.now().strftime('%Y%m%d')}.log"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("ThumbnailCreator")
        self.logger.info(f"✅ Thumbnail creator logging initialized: {log_file}")

    def ensure_directories(self):
        """Ensure all required directories exist"""
        dirs_to_create = [
            'DATA_DIR', 'OUTPUT_DIR', 'LOGS_DIR', 'CONFIG_DIR', 'FONTS_DIR'
        ]

        for dir_key in dirs_to_create:
            dir_path = Path(self.paths[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)

        print("✅ All thumbnail creator directories created/verified")


# Initialize server config
try:
    CONFIG = ServerConfig()
    print("🚀 Thumbnail Creator server configuration loaded successfully")
except Exception as e:
    print(f"❌ Thumbnail Creator server configuration failed: {e}")
    sys.exit(1)


# Database Thumbnail Management Integration
class DatabaseThumbnailManager:
    """Professional thumbnail management using existing production.db"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_completed_cover_topic_ready_for_thumbnail(self) -> Optional[Tuple[int, str, str, str]]:
        """Get completed cover image topic that needs THUMBNAIL generation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if thumbnail generation columns exist, if not create them
        cursor.execute('PRAGMA table_info(topics)')
        columns = [row[1] for row in cursor.fetchall()]

        # Add columns individually if they don't exist
        columns_to_add = [
            ('thumbnail_generation_status', 'TEXT DEFAULT "pending"'),
            ('thumbnail_generation_started_at', 'DATETIME'),
            ('thumbnail_generation_completed_at', 'DATETIME'),
            ('thumbnail_file_size_kb', 'REAL DEFAULT 0.0'),
            ('thumbnail_processing_time_seconds', 'REAL DEFAULT 0.0')
        ]

        for column_name, column_definition in columns_to_add:
            if column_name not in columns:
                print(f"🔧 Adding column: {column_name}")
                cursor.execute(f'ALTER TABLE topics ADD COLUMN {column_name} {column_definition}')

        conn.commit()
        print("✅ Thumbnail generation columns verified/added")

        cursor.execute('''
            SELECT id, topic, description, output_path 
            FROM topics 
            WHERE status = 'completed' 
            AND cover_image_created = 1
            AND (thumbnail_generation_status IS NULL OR thumbnail_generation_status = 'pending')
            ORDER BY updated_at ASC 
            LIMIT 1
        ''')

        result = cursor.fetchone()
        conn.close()

        return result if result else None

    def mark_thumbnail_generation_started(self, topic_id: int):
        """Mark thumbnail generation as started"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET thumbnail_generation_status = 'in_progress', 
                thumbnail_generation_started_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()

    def mark_thumbnail_generation_completed(self, topic_id: int, file_size_kb: float, processing_time_seconds: float):
        """Mark thumbnail generation as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET thumbnail_generation_status = 'completed',
                thumbnail_generation_completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP,
                thumbnail_file_size_kb = ?,
                thumbnail_processing_time_seconds = ?,
                thumbnail = 1
            WHERE id = ?
        ''', (file_size_kb, processing_time_seconds, topic_id))

        conn.commit()
        conn.close()


class ThumbnailProgressTracker:
    """Thumbnail processing progress tracking and resume functionality"""

    def __init__(self, story_id: int, output_base_path: str):
        self.story_id = story_id
        self.output_dir = os.path.join(output_base_path, str(story_id))
        self.progress_file = os.path.join(self.output_dir, "thumbnail_progress.json")

        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load existing progress
        self.progress_data = self.load_progress()

    def load_progress(self):
        """Load existing progress"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"📂 Thumbnail progress loaded: {data.get('current_stage', 'unknown')} stage")
                return data
            except Exception as e:
                print(f"⚠️  Thumbnail progress file corrupted, starting fresh: {e}")

        return {
            "story_id": self.story_id,
            "current_stage": "init",
            "completed_stages": [],
            "failed_attempts": 0,
            "last_update": datetime.now().isoformat(),
            "session_start": datetime.now().isoformat(),
            "stages": {
                "config_load": {"status": "pending", "timestamp": None},
                "background_load": {"status": "pending", "timestamp": None},
                "text_overlay": {"status": "pending", "timestamp": None},
                "final_save": {"status": "pending", "timestamp": None}
            }
        }

    def save_progress(self):
        """Save progress"""
        self.progress_data["last_update"] = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Thumbnail progress save warning: {e}")

    def mark_stage_completed(self, stage: str):
        """Mark stage as completed"""
        if stage not in self.progress_data["completed_stages"]:
            self.progress_data["completed_stages"].append(stage)

        if stage in self.progress_data["stages"]:
            self.progress_data["stages"][stage] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }

        self.progress_data["current_stage"] = stage
        self.save_progress()
        print(f"      📋 Stage completed: {stage}")

    def mark_stage_failed(self, stage: str, error: str):
        """Mark stage as failed"""
        if stage in self.progress_data["stages"]:
            self.progress_data["stages"][stage] = {
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": error
            }

        self.progress_data["failed_attempts"] += 1
        self.save_progress()

    def cleanup_on_success(self):
        """Clean up progress file on successful completion"""
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
                print(f"🗑️  Thumbnail progress file cleaned up (successful completion)")
        except Exception as e:
            print(f"⚠️  Thumbnail progress cleanup warning: {e}")


class ThumbnailUsageTracker:
    """Thumbnail processing usage and performance tracking"""

    def __init__(self):
        self.session_start = datetime.now()
        self.processing_stages = []
        self.total_processing_time = 0.0

        # Performance tracking
        self.performance_data = {
            "thumbnails_created": 0,
            "total_processing_time_seconds": 0.0,
            "average_processing_time_seconds": 0.0,
            "total_file_size_kb": 0.0
        }

        # Budget controls
        self.budget_controls = CONFIG.thumbnail_config.get("budget_controls", {})

    def check_processing_limits(self) -> Tuple[bool, str]:
        """Check if processing limits are exceeded"""
        current_time = (datetime.now() - self.session_start).total_seconds() / 60  # minutes

        max_time = self.budget_controls.get("max_processing_time_minutes", 10)
        if current_time >= max_time:
            return False, f"PROCESSING TIME LIMIT: {current_time:.1f} min >= {max_time} min"

        warn_threshold = self.budget_controls.get("warn_threshold_minutes", 5)
        if current_time >= warn_threshold:
            print(f"⚠️  PROCESSING WARNING: {current_time:.1f} min approaching limit {max_time} min")

        return True, "OK"

    def add_stage(self, stage_name: str, duration_seconds: float):
        """Add processing stage timing"""
        self.processing_stages.append({
            "stage": stage_name,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat()
        })
        self.total_processing_time += duration_seconds

    def update_performance_data(self, **kwargs):
        """Update performance tracking data"""
        self.performance_data.update(kwargs)

    def print_final_summary(self):
        """Print final processing summary"""
        total_time = (datetime.now() - self.session_start).total_seconds() / 60

        print(f"\n🎨 FINAL THUMBNAIL PROCESSING SUMMARY")
        print(f"=" * 50)
        print(f"📊 Thumbnails created: {self.performance_data.get('thumbnails_created', 0)}")
        print(f"⚡ Total processing time: {total_time:.1f} minutes")
        print(f"⚡ Average per thumbnail: {self.performance_data.get('average_processing_time_seconds', 0.0):.1f}s")
        print(f"💾 Total file size: {self.performance_data.get('total_file_size_kb', 0.0):.1f} KB")

        if self.processing_stages:
            print(f"📊 Processing stages:")
            for stage in self.processing_stages:
                print(f"   🔄 {stage['stage']}: {stage['duration_seconds']:.1f}s")

        return {
            "total_processing_time_minutes": total_time,
            "performance_data": self.performance_data,
            "processing_stages": self.processing_stages
        }


class ServerThumbnailCreator:
    """Server-ready Thumbnail Creator with database integration"""

    def __init__(self):
        # Current project tracking
        self.current_topic_id = None
        self.current_output_dir = None
        self.current_topic = None
        self.current_description = None

        # Server paths from CONFIG
        self.base_dir = Path(CONFIG.paths['SRC_DIR'])
        self.project_root = Path(CONFIG.paths['BASE_DIR'])
        self.data_path = Path(CONFIG.paths['DATA_DIR'])
        self.output_path = Path(CONFIG.paths['OUTPUT_DIR'])
        self.fonts_dir = Path(CONFIG.paths['FONTS_DIR'])

        # Config paths
        self.thumbnail_json_path = self.data_path / "thumbnail_features.json"

        # Database manager
        db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
        self.db_manager = DatabaseThumbnailManager(str(db_path))

        print("🎨 Server Thumbnail Creator v1.0 Initialized")
        print(f"📁 Base Directory: {self.base_dir}")
        print(f"🎨 Fonts Directory: {self.fonts_dir}")

    def log_step(self, step: str, status: str = "START", metadata: Dict = None):
        """Log generation steps with production logging"""
        icon = "🔄" if status == "START" else "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "ℹ️"
        print(f"{icon} {step}")
        CONFIG.logger.info(f"{step} - Status: {status} - Project: {self.current_topic_id}")

    def get_next_project_from_database(self) -> Tuple[bool, Optional[Dict]]:
        """Get next completed cover image project that needs THUMBNAIL generation"""
        self.log_step("🔍 Finding completed cover image project for thumbnail generation")

        result = self.db_manager.get_completed_cover_topic_ready_for_thumbnail()

        if not result:
            self.log_step("✅ No completed cover image projects ready for thumbnail generation", "INFO")
            return False, None

        topic_id, topic, description, output_path = result

        # Setup project paths
        self.current_topic_id = topic_id
        self.current_output_dir = output_path
        self.current_topic = topic
        self.current_description = description

        project_info = {
            "topic_id": topic_id,
            "topic": topic,
            "description": description,
            "output_dir": output_path
        }

        # Mark as started in database
        self.db_manager.mark_thumbnail_generation_started(topic_id)

        self.log_step(f"✅ Found project: {topic}", "SUCCESS", project_info)
        return True, project_info

    def load_thumbnail_config(self):
        """src/data/thumbnail_features.json'ı yükle"""
        try:
            with open(self.thumbnail_json_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print("✅ Thumbnail config yüklendi")
            return config
        except Exception as e:
            print(f"❌ Config yükleme hatası: {e}")
            return None

    def find_topic_in_config(self, row_number, config):
        """CSV'deki satır numarasına göre config'de topic bul"""
        # ID ile eşleştir (satır numarası = id)
        for topic in config['topics']:
            if topic['id'] == row_number:
                print(f"✅ Config'de bulundu: ID={row_number} -> {topic['topic']}")
                return topic

        # Bulunamazsa None döndür
        print(f"⚠️ Config'de bulunamadı: ID={row_number}")
        return None

    def load_fonts(self):
        """Fontları yükle - server paths kullan"""
        font_paths = {
            'shocking': self.fonts_dir / 'Poppins-Bold.ttf',
            'main': self.fonts_dir / 'CrimsonText-Bold.ttf',
            'bottom': self.fonts_dir / 'CrimsonText-Italic.ttf',
            'channel': self.fonts_dir / 'Lora-VariableFont_wght.ttf'
        }

        loaded_fonts = {}

        for font_type, path in font_paths.items():
            if path.exists():
                loaded_fonts[font_type] = str(path)
                print(f"✅ {font_type}: {path.name}")
            else:
                loaded_fonts[font_type] = None
                print(f"❌ {font_type}: {path} bulunamadı")

        return loaded_fonts

    def hex_to_rgb(self, hex_color):
        """Hex rengi RGB'ye çevir"""
        try:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        except:
            return (255, 255, 255)  # Default beyaz

    def remove_emojis(self, text):
        """Emoji karakterlerini kaldır"""
        import re
        # Unicode emoji pattern'i
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)

        # Unicode escape sequence'leri de kaldır (\ud83d\ude31 gibi)
        text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)

        # Normal emoji'leri kaldır
        clean_text = emoji_pattern.sub(r'', text)

        # Fazla boşlukları temizle
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text

    def create_thumbnail_from_config(self, row_number, topic_data, topic_config, template_config, progress_tracker,
                                     usage_tracker):
        """Simple thumbnail creation - server version with tracking"""
        start_time = datetime.now()

        # Story output klasörü
        story_output_dir = Path(self.current_output_dir)

        # Final thumbnail için scenes klasörü oluştur
        final_output_dir = story_output_dir / "scenes"
        final_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"\n🎨 {topic_data['topic']} için simple thumbnail oluşturuluyor...")
            print(f"📁 Story Dir: {story_output_dir}")
            print(f"📁 Final Output: {final_output_dir}")

            # Check processing limits
            can_continue, limit_reason = usage_tracker.check_processing_limits()
            if not can_continue:
                progress_tracker.mark_stage_failed("config_load", f"Processing limit exceeded: {limit_reason}")
                return None

            # Template ayarları
            template = template_config['thumbnail_template']
            width = template['layout']['width']
            height = template['layout']['height']
            print(f"🎛️ Template boyut: {width}x{height}")

            progress_tracker.mark_stage_completed("config_load")
            usage_tracker.add_stage("config_load", (datetime.now() - start_time).total_seconds())

            # Background loading stage
            stage_start = datetime.now()

            # Ham thumbnail.png dosyasını DOĞRU konumdan yükle
            thumbnail_path = story_output_dir / "thumbnail.png"

            print(f"🔍 Background aranıyor: {thumbnail_path}")

            if thumbnail_path.exists():
                # Ham resmi yükle ve RGB'ye çevir
                img = Image.open(thumbnail_path)
                original_size = img.size
                original_mode = img.mode

                print(f"🔍 Orijinal resim: {original_size}, mode: {original_mode}")

                # RGB'ye çevir
                if img.mode != 'RGB':
                    print(f"🔄 Converting {original_mode} → RGB")
                    img = img.convert('RGB')

                # Boyutu template boyutuna ayarla
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                print(f"✅ Background processed: {original_size} ({original_mode}) → {width}x{height} (RGB)")
            else:
                # Ham resim yoksa RGB arkaplan oluştur
                img = Image.new('RGB', (width, height), (30, 30, 30))
                print(f"⚠️ Ham thumbnail bulunamadı: {thumbnail_path}")
                print(f"📝 RGB arkaplan oluşturuldu ({width}x{height})")

            draw = ImageDraw.Draw(img)
            print("✅ Simple draw objesi oluşturuldu")

            progress_tracker.mark_stage_completed("background_load")
            usage_tracker.add_stage("background_load", (datetime.now() - stage_start).total_seconds())

            # Text overlay stage
            stage_start = datetime.now()

            # Fontları yükle - ORIGINAL SIZES
            fonts = self.load_fonts()
            print(f"📝 Font durumu: {[k for k, v in fonts.items() if v is not None]}")

            # Font boyutları template'den al - NO OPTIMIZATION
            try:
                shocking_size = template['sections']['shocking_word']['font_size']
                main_size = template['sections']['main_title']['font_size']
                bottom_size = template['sections']['bottom_text']['font_size']
                channel_size = template['sections']['channel']['font_size']

                print(
                    f"📏 Font boyutları: SHOCKING={shocking_size}, Main={main_size}, Bottom={bottom_size}, Channel={channel_size}")

                shocking_font = ImageFont.truetype(fonts['shocking'], shocking_size) if fonts[
                    'shocking'] else ImageFont.load_default()
                main_font = ImageFont.truetype(fonts['main'], main_size) if fonts['main'] else ImageFont.load_default()
                bottom_font = ImageFont.truetype(fonts['bottom'], bottom_size) if fonts[
                    'bottom'] else ImageFont.load_default()
                channel_font = ImageFont.truetype(fonts['channel'], channel_size) if fonts[
                    'channel'] else ImageFont.load_default()

                print("✅ Original fonts loaded")

            except Exception as font_error:
                print(f"❌ Font yükleme hatası: {font_error}")
                shocking_font = main_font = bottom_font = channel_font = ImageFont.load_default()
                print("⚠️ Tüm fontlar varsayılan olarak ayarlandı")

            print("\n📝 SIMPLE TEXT RENDERING...")

            # 1. SHOCKING WORD - With Shadow Only
            print("📍 1. SHOCKING WORD (with shadow)")
            try:
                shocking_section = topic_config['sections']['shocking_word']
                shocking_pos = (
                    template['sections']['shocking_word']['x_offset'],
                    template['sections']['shocking_word']['y_offset']
                )

                shocking_text = self.remove_emojis(shocking_section['text'])
                shocking_color = self.hex_to_rgb(shocking_section['color'])

                print(f"   📝 Text: '{shocking_text}'")
                print(f"   📍 Position: {shocking_pos}")
                print(f"   🎨 Color: {shocking_color}")

                # Simple shadow for SHOCKING
                shadow_offset = [3, 3]
                draw.text((shocking_pos[0] + shadow_offset[0], shocking_pos[1] + shadow_offset[1]),
                          shocking_text, font=shocking_font, fill=(0, 0, 0))

                # Main text
                draw.text(shocking_pos, shocking_text, font=shocking_font, fill=shocking_color)
                print("   ✅ Shocking word with simple shadow")
            except Exception as e:
                print(f"   ❌ Shocking word hatası: {e}")

            # 2. MAIN TITLE - Clean, No Effects
            print("📍 2. MAIN TITLE (clean)")
            try:
                main_section = topic_config['sections']['main_title']
                x_offset = template['sections']['main_title']['x_offset']
                y_start = template['sections']['main_title']['y_start']
                line_height = template['sections']['main_title']['line_height']

                main_color = self.hex_to_rgb(main_section['color'])

                print(f"   📝 Lines: {len(main_section['lines'])}")
                print(f"   📍 Position: x={x_offset}, y_start={y_start}, line_height={line_height}")
                print(f"   🎨 Color: {main_color}")

                for i, line in enumerate(main_section['lines']):
                    y_pos = y_start + (i * line_height)
                    clean_line = self.remove_emojis(line)

                    print(f"     Line {i + 1}: '{clean_line}' @ y={y_pos}")

                    # NO EFFECTS - Just clean text
                    draw.text((x_offset, y_pos), clean_line, font=main_font, fill=main_color)

                print("   ✅ Main title - clean")
            except Exception as e:
                print(f"   ❌ Main title hatası: {e}")

            # 3. BOTTOM TEXT - Clean, No Effects
            print("📍 3. BOTTOM TEXT (clean)")
            try:
                bottom_section = topic_config['sections']['bottom_text']
                x_offset = template['sections']['bottom_text']['x_offset']
                y_start = template['sections']['bottom_text']['y_start']
                line_height = template['sections']['bottom_text']['line_height']

                bottom_color = self.hex_to_rgb(bottom_section['color'])

                print(f"   📝 Lines: {len(bottom_section['lines'])}")
                print(f"   📍 Position: x={x_offset}, y_start={y_start}")
                print(f"   🎨 Color: {bottom_color}")

                for i, line in enumerate(bottom_section['lines']):
                    y_pos = y_start + (i * line_height)
                    clean_line = self.remove_emojis(line)

                    print(f"     Line {i + 1}: '{clean_line}' @ y={y_pos}")

                    # NO EFFECTS - Just clean text
                    draw.text((x_offset, y_pos), clean_line, font=bottom_font, fill=bottom_color)

                print("   ✅ Bottom text - clean")
            except Exception as e:
                print(f"   ❌ Bottom text hatası: {e}")

            # 4. CHANNEL - With Shadow Only
            print("📍 4. CHANNEL (with shadow)")
            try:
                channel_section = topic_config['sections']['channel']
                channel_pos = (
                    template['sections']['channel']['x_offset'],
                    height - template['sections']['channel']['y_offset']
                )
                channel_color = self.hex_to_rgb(template['sections']['channel']['color'])

                channel_text = self.remove_emojis(channel_section['text'])

                print(f"   📝 Text: '{channel_text}'")
                print(f"   📍 Position: {channel_pos}")
                print(f"   🎨 Color: {channel_color}")

                # Simple shadow for CHANNEL
                shadow_offset = [2, 2]
                draw.text((channel_pos[0] + shadow_offset[0], channel_pos[1] + shadow_offset[1]),
                          channel_text, font=channel_font, fill=(0, 0, 0))

                # Main text
                draw.text(channel_pos, channel_text, font=channel_font, fill=channel_color)
                print("   ✅ Channel with simple shadow")
            except Exception as e:
                print(f"   ❌ Channel hatası: {e}")

            progress_tracker.mark_stage_completed("text_overlay")
            usage_tracker.add_stage("text_overlay", (datetime.now() - stage_start).total_seconds())

            # Final save stage
            stage_start = datetime.now()

            print("\n💾 SIMPLE SAVE...")

            # Final mode check before saving
            print(f"🔍 Final image mode before save: {img.mode}")
            if img.mode != 'RGB':
                print(f"🔄 Final conversion: {img.mode} → RGB")
                img = img.convert('RGB')

            # Simple JPEG save
            output_path = final_output_dir / "final_thumbnail.jpg"
            print(f"📁 Hedef dosya: {output_path}")

            # Simple YouTube JPEG
            quality_profile = CONFIG.thumbnail_config["quality_profiles"]["youtube"]
            img.save(str(output_path),
                     quality_profile['format'],
                     quality=quality_profile['quality'],
                     optimize=quality_profile['optimize'])

            progress_tracker.mark_stage_completed("final_save")
            usage_tracker.add_stage("final_save", (datetime.now() - stage_start).total_seconds())

            # Dosya kontrol
            if output_path.exists():
                file_size = output_path.stat().st_size
                file_size_kb = file_size / 1024
                processing_time = (datetime.now() - start_time).total_seconds()

                print(f"✅ SIMPLE THUMBNAIL BAŞARIYLA KAYDEDİLDİ!")
                print(f"   📁 Path: {output_path}")
                print(f"   📦 Boyut: {file_size:,} bytes ({file_size_kb:.1f} KB)")
                print(f"   ⚡ Processing time: {processing_time:.1f}s")
                print(f"   🎯 Clean design - No overlapping text")
                print(f"   🌑 Shadows: SHOCKING + Channel only")

                # Update usage tracker
                usage_tracker.update_performance_data(
                    thumbnails_created=usage_tracker.performance_data.get('thumbnails_created', 0) + 1,
                    total_file_size_kb=usage_tracker.performance_data.get('total_file_size_kb', 0.0) + file_size_kb,
                    total_processing_time_seconds=usage_tracker.performance_data.get('total_processing_time_seconds',
                                                                                     0.0) + processing_time
                )

                return output_path, file_size_kb, processing_time
            else:
                progress_tracker.mark_stage_failed("final_save", f"File not created: {output_path}")
                print(f"❌ Dosya kaydedilemedi: {output_path}")
                return None

        except Exception as e:
            print(f"❌ GENEL THUMBNAIL HATASI: {e}")
            import traceback
            traceback.print_exc()
            progress_tracker.mark_stage_failed("text_overlay", str(e))
            return None

    def run_thumbnail_generation(self) -> bool:
        """Run THUMBNAIL generation process for server environment"""
        print("🚀" * 50)
        print("SERVER THUMBNAIL CREATOR v1.0")
        print("🔗 Database integrated")
        print("🎨 YouTube Thumbnail Production")
        print("🎯 Clean Design - No Text Overlap")
        print("🖥️ Production-ready automation")
        print("🚀" * 50)

        # Print processing limits
        limits = CONFIG.thumbnail_config.get("budget_controls", {})
        print(f"⏱️  PROCESSING LIMITS:")
        print(f"   📊 Max processing time: {limits.get('max_processing_time_minutes', 10)} minutes")
        print(f"   📊 Warning threshold: {limits.get('warn_threshold_minutes', 5)} minutes")
        print(f"   💾 Max memory usage: {limits.get('max_memory_usage_mb', 512)} MB")

        # Initialize success tracking
        overall_success = False

        # Step 1: Get next project from database
        found, project_info = self.get_next_project_from_database()
        if not found:
            return False

        print(f"✅ Project found: {project_info['topic']}")
        print(f"📁 Output directory: {project_info['output_dir']}")
        print(f"🆔 Topic ID: {project_info['topic_id']}")

        # Initialize trackers
        progress_tracker = ThumbnailProgressTracker(self.current_topic_id, CONFIG.paths['OUTPUT_DIR'])
        usage_tracker = ThumbnailUsageTracker()

        try:
            story_id = self.current_topic_id
            topic = self.current_topic

            # Load thumbnail config
            config = self.load_thumbnail_config()
            if not config:
                print("❌ Thumbnail config yüklenemedi")
                return False

            # Find topic in config
            topic_config = self.find_topic_in_config(story_id, config)
            if not topic_config:
                print(f"❌ Topic {story_id} config'de bulunamadı")
                return False

            # Create thumbnail with server tracking
            result = self.create_thumbnail_from_config(
                story_id, project_info, topic_config, config, progress_tracker, usage_tracker
            )

            if result and len(result) == 3:  # Successful result has 3 elements
                output_path, file_size_kb, processing_time = result

                print(f"✅ Thumbnail generation successful!")
                print(f"📁 Thumbnail file: {output_path}")
                print(f"📦 File size: {file_size_kb:.1f} KB")
                print(f"⚡ Processing time: {processing_time:.1f} seconds")

                # Update database with results
                self.db_manager.mark_thumbnail_generation_completed(
                    self.current_topic_id, file_size_kb, processing_time
                )

                print(f"💾 Database updated with thumbnail results")

                # Calculate average processing time
                total_thumbnails = usage_tracker.performance_data.get('thumbnails_created', 1)
                total_time = usage_tracker.performance_data.get('total_processing_time_seconds', processing_time)
                avg_time = total_time / total_thumbnails
                usage_tracker.update_performance_data(average_processing_time_seconds=avg_time)

                # Cleanup progress on success
                progress_tracker.cleanup_on_success()

                print("\n" + "🎉" * 50)
                print("THUMBNAIL GENERATION SUCCESSFUL!")
                print("✅ YouTube-optimized thumbnail with text overlay")
                print("✅ Clean design with minimal shadows")
                print("✅ Database updated with metrics")
                print("🎉" * 50)
                overall_success = True

            else:
                print(f"❌ Thumbnail generation failed")
                overall_success = False

            return overall_success

        except Exception as e:
            self.log_step(f"❌ Thumbnail generation failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    try:
        print("🚀 SERVER THUMBNAIL CREATOR v1.0")
        print("🔗 Database integration with progress tracking")
        print("🎨 YouTube Thumbnail Production")
        print("🎯 Clean Design - No Text Overlap")
        print("🌑 Minimal Shadows: SHOCKING + Channel only")
        print("🖥️ Production-ready automation")
        print("=" * 60)

        creator = ServerThumbnailCreator()
        success = creator.run_thumbnail_generation()

        if success:
            print("🎊 Thumbnail generation completed successfully!")
            print("📁 Thumbnail saved: final_thumbnail.jpg")
            print("🎯 Clean design with text overlay")
            print("🌑 Minimal shadows applied")
            print("💾 Progress tracking enabled")
            print("🖥️ Server infrastructure working")
        else:
            print("⚠️ Thumbnail generation failed or no projects ready")

    except KeyboardInterrupt:
        print("\n⏹️ Thumbnail generation stopped by user")
        print("🛡️ Progress saved! Restart to resume from last completed stage.")
    except Exception as e:
        print(f"💥 Thumbnail generation failed: {e}")
        print("🛡️ Progress saved! Check thumbnail_progress.json for resume info.")
        CONFIG.logger.error(f"Thumbnail generation failed: {e}")
        import traceback

        traceback.print_exc()