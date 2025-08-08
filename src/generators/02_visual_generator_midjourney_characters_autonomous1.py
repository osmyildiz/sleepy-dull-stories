"""
Sleepy Dull Stories - ROBUST Midjourney Visual Generator with Multi-Variant Support
✅ Smart debug (only on errors)
✅ Skip existing characters
✅ Individual character error handling
✅ Partial success acceptance
✅ Multi-variant character poses (headshot, bust, profile, full_body, action)
"""

import requests
import os
import json
import pandas as pd
import time
import sys
import sqlite3
import signal
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
import urllib.request
from pathlib import Path
import logging

# Load environment first
load_dotenv()


# Server Configuration Class (from story generator)
class ServerConfig:
    """Server-friendly configuration management"""

    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.setup_visual_config()
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
            'CONFIG_DIR': str(self.project_root / 'config')
        }

        print(f"✅ Visual Generator server paths configured:")
        print(f"   📁 Project root: {self.paths['BASE_DIR']}")

    def setup_visual_config(self):
        """Setup Midjourney visual generation configuration - V6.1 for --cref support"""
        self.visual_config = {
            "api_base_url": "https://api.piapi.ai/api/v1",
            "max_concurrent_tasks": 10,
            "max_wait_cycles": 30,
            "wait_interval_seconds": 30,
            "default_aspect_ratios": {
                "characters": "2:3",
                "scenes": "16:9",
                "thumbnail": "16:9"
            },
            "default_version": "6.1",  # ✅ V7 → V6.1 DEĞİŞTİRİLDİ (--cref desteği için)
            "quality": "2",  # ✅ YENİ: Yüksek kalite
            "stylize": "100",  # ✅ YENİ: Orta stylization
            "process_mode": "relax",
            "character_generation": True,
            "scene_generation": True,
            "thumbnail_generation": True,
            "server_mode": True,
            "production_ready": True,
            "v6_optimized": True  # ✅ V6 optimization flag
        }
        self.api_key = self.get_midjourney_api_key()

    def get_midjourney_api_key(self):
        """Get Midjourney API key from multiple sources"""
        # Try different environment variable names
        api_key = (
                os.getenv('PIAPI_KEY') or
                os.getenv('MIDJOURNEY_API_KEY') or
                os.getenv('PIAPI_API_KEY') or
                os.getenv('MIDJOURNEY_KEY')
        )

        if not api_key:
            # Check .env file
            env_files = [
                Path('.env'),
                Path('../../.env'),
                self.project_root / '.env'
            ]

            for env_file in env_files:
                if env_file.exists():
                    load_dotenv(env_file)
                    api_key = os.getenv('PIAPI_KEY')
                    if api_key:
                        print(f"✅ Midjourney API key loaded from: {env_file}")
                        break

        if not api_key:
            raise ValueError(
                "❌ Midjourney API key required!\n"
                "Set in .env file:\n"
                "PIAPI_KEY=your_api_key_here\n"
                "Or environment variable: PIAPI_KEY"
            )

        print("✅ Midjourney API key loaded successfully")
        return api_key

    def setup_logging(self):
        """Setup production logging"""
        logs_dir = Path(self.project_root) / 'logs' / 'generators'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"visual_gen_{datetime.now().strftime('%Y%m%d')}.log"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("VisualGenerator")
        self.logger.info(f"✅ Visual generator logging initialized: {log_file}")

    def ensure_directories(self):
        """Ensure all required directories exist"""
        dirs_to_create = [
            'DATA_DIR', 'OUTPUT_DIR', 'LOGS_DIR', 'CONFIG_DIR'
        ]

        for dir_key in dirs_to_create:
            dir_path = Path(self.paths[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)

        print("✅ All visual generator directories created/verified")


# Initialize server config
try:
    CONFIG = ServerConfig()
    print("🚀 Visual Generator server configuration loaded successfully")
except Exception as e:
    print(f"❌ Visual Generator server configuration failed: {e}")
    sys.exit(1)


# Database Topic Management Integration (from story generator)
class DatabaseTopicManager:
    """Professional topic management using existing production.db"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_completed_topic_ready_for_characters(self) -> Optional[Tuple[int, str, str, str]]:
        """Get completed story topic that needs CHARACTER generation only"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, topic, description, output_path 
            FROM topics 
            WHERE status = 'completed' 
            AND (character_generation_status IS NULL OR character_generation_status = 'pending')
            ORDER BY production_completed_at ASC 
            LIMIT 1
        ''')

        result = cursor.fetchone()
        conn.close()

        return result if result else None

    def mark_character_generation_started(self, topic_id: int):
        """Mark character generation as started"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Add column if it doesn't exist
        cursor.execute('PRAGMA table_info(topics)')
        columns = [row[1] for row in cursor.fetchall()]

        if 'character_generation_status' not in columns:
            cursor.execute('ALTER TABLE topics ADD COLUMN character_generation_status TEXT DEFAULT "pending"')
            cursor.execute('ALTER TABLE topics ADD COLUMN character_generation_started_at DATETIME')
            cursor.execute('ALTER TABLE topics ADD COLUMN character_generation_completed_at DATETIME')
            cursor.execute('ALTER TABLE topics ADD COLUMN characters_generated INTEGER DEFAULT 0')

        cursor.execute('''
            UPDATE topics 
            SET character_generation_status = 'in_progress', 
                character_generation_started_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()

    def mark_character_generation_completed(self, topic_id: int, characters_count: int):
        """Mark character generation as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET character_generation_status = 'completed',
                character_generation_completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP,
                characters_generated = ?
            WHERE id = ?
        ''', (characters_count, topic_id))

        conn.commit()
        conn.close()

    def mark_character_generation_failed(self, topic_id: int, error_message: str):
        """Mark character generation as failed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET character_generation_status = 'failed',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()

    def get_pipeline_status(self) -> Dict:
        """Get overall pipeline status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Add columns if they don't exist
        cursor.execute('PRAGMA table_info(topics)')
        columns = [row[1] for row in cursor.fetchall()]

        if 'character_generation_status' not in columns:
            cursor.execute('ALTER TABLE topics ADD COLUMN character_generation_status TEXT DEFAULT "pending"')

        # Count character generation queue
        cursor.execute('''
            SELECT COUNT(*) FROM topics 
            WHERE status = 'completed' 
            AND (character_generation_status IS NULL OR character_generation_status = 'pending')
        ''')
        character_queue = cursor.fetchone()[0]

        # Count active character generation
        cursor.execute('''
            SELECT COUNT(*) FROM topics 
            WHERE character_generation_status = 'in_progress'
        ''')
        character_active = cursor.fetchone()[0]

        conn.close()

        return {
            'character_generation_queue': character_queue,
            'character_generation_active': character_active
        }


class ServerMidjourneyVisualGenerator:
    """Server-ready Midjourney visual generator with robust error handling and multi-variant support"""

    def __init__(self):
        self.api_key = CONFIG.api_key
        self.base_url = CONFIG.visual_config["api_base_url"]

        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Current project tracking
        self.current_topic_id = None
        self.current_output_dir = None
        self.current_topic = None
        self.current_description = None
        self.current_historical_period = None

        # Generation tracking
        self.generation_log = []
        self.character_references = {}
        self.api_calls_made = 0
        self.successful_downloads = 0

        # Database manager
        db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
        self.db_manager = DatabaseTopicManager(str(db_path))

        print("🚀 Server Midjourney Visual Generator v2.0 Initialized (Multi-Variant)")
        print(f"🔑 API Key: {self.api_key[:8]}...")
        print(f"🌐 Base URL: {self.base_url}")

    def debug_log_on_error(self, method: str, url: str, headers: dict, payload: dict = None, response: requests.Response = None):
        """Only log debug info when there's an error"""
        if response and response.status_code != 200:
            timestamp = datetime.now().isoformat()
            print(f"\n🔍 ERROR DEBUG [{timestamp}]")
            print(f"📡 Method: {method} | Status: {response.status_code}")
            print(f"🌐 URL: {url}")

            if payload:
                print(f"📦 Request Payload:")
                print(json.dumps(payload, indent=2))

            print(f"📄 Error Response:")
            try:
                error_json = response.json()
                print(json.dumps(error_json, indent=2))
            except:
                print(f"Raw error: {response.text}")
            print("🔍" + "="*60)

    def check_existing_character(self, char_name: str) -> bool:
        """Check if character already exists to avoid regeneration"""
        if not hasattr(self, 'characters_dir'):
            return False

        safe_name = char_name.lower().replace(" ", "_").replace(".", "")
        image_path = self.characters_dir / f"{safe_name}.png"
        json_path = self.characters_dir / f"{safe_name}.json"

        if image_path.exists() and json_path.exists():
            # Verify file is not empty and valid
            try:
                if os.path.getsize(image_path) > 1000:  # At least 1KB
                    with open(json_path, 'r') as f:
                        json.load(f)  # Test if JSON is valid
                    print(f"   ✅ {char_name} already exists, skipping")
                    return True
            except:
                # File corrupted, will regenerate
                pass

        return False

    def log_step(self, step: str, status: str = "START", metadata: Dict = None):
        """Log generation steps with production logging"""
        entry = {
            "step": step,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "project": self.current_topic_id,
            "api_calls": self.api_calls_made,
            "successful_downloads": self.successful_downloads,
            "metadata": metadata or {}
        }
        self.generation_log.append(entry)

        icon = "🔄" if status == "START" else "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "ℹ️"
        print(f"{icon} {step} [Calls: {self.api_calls_made}] [Downloads: {self.successful_downloads}]")
        CONFIG.logger.info(f"{step} - Status: {status} - Project: {self.current_topic_id}")

    def get_next_project_from_database(self) -> Tuple[bool, Optional[Dict]]:
        """Get next completed story project that needs CHARACTER generation"""
        self.log_step("🔍 Finding completed story project for character generation")

        result = self.db_manager.get_completed_topic_ready_for_characters()

        if not result:
            self.log_step("✅ No completed stories ready for character generation", "INFO")
            return False, None

        topic_id, topic, description, output_path = result

        # Setup project paths
        self.current_topic_id = topic_id
        self.current_output_dir = output_path
        self.current_topic = topic
        self.current_description = description

        # Try to detect historical period from existing files
        try:
            character_path = Path(output_path) / "character_profiles.json"
            if character_path.exists():
                with open(character_path, 'r', encoding='utf-8') as f:
                    character_data = json.load(f)
                    visual_style = character_data.get('visual_style_notes', {})
                    self.current_historical_period = visual_style.get('period_accuracy', 'ancient times')
            else:
                self.current_historical_period = "ancient times"
        except:
            self.current_historical_period = "ancient times"

        project_info = {
            "topic_id": topic_id,
            "topic": topic,
            "description": description,
            "output_dir": output_path,
            "historical_period": self.current_historical_period
        }

        # Mark as started in database
        self.db_manager.mark_character_generation_started(topic_id)

        self.log_step(f"✅ Found project: {topic}", "SUCCESS", project_info)
        return True, project_info

    def load_character_profiles_only(self) -> Dict:
        """Load character profiles only from story generator output"""
        self.log_step("📂 Loading character profiles only")

        output_dir = Path(self.current_output_dir)

        # Load character profiles
        char_profiles_path = output_dir / "character_profiles.json"
        if not char_profiles_path.exists():
            raise FileNotFoundError(f"Character profiles not found: {char_profiles_path}")

        with open(char_profiles_path, 'r', encoding='utf-8') as f:
            character_profiles = json.load(f)

        # Validate data
        main_characters = character_profiles.get("main_characters", [])

        self.log_step("✅ Character profiles loaded", "SUCCESS", {
            "characters_count": len(main_characters),
            "marketing_characters": len([c for c in main_characters if c.get("use_in_marketing", False)])
        })

        return character_profiles

    def setup_character_directories(self):
        """Create necessary directories for character generation only"""
        output_dir = Path(self.current_output_dir)

        self.characters_dir = output_dir / "characters"
        self.characters_dir.mkdir(exist_ok=True)

        self.log_step("📁 Character generation directory created", "SUCCESS")

    def save_character_generation_report(self):
        """Save character generation report"""
        output_dir = Path(self.current_output_dir)

        report = {
            "character_generation_completed": datetime.now().isoformat(),
            "topic_id": self.current_topic_id,
            "topic": self.current_topic,
            "api_calls_made": self.api_calls_made,
            "successful_downloads": self.successful_downloads,
            "character_references_created": len(self.character_references),
            "characters_dir": str(self.characters_dir),
            "historical_period": self.current_historical_period,
            "generation_log": self.generation_log,
            "server_optimized": True,
            "character_only_mode": True,
            "robust_error_handling": True,
            "multi_variant_support": True
        }

        report_path = output_dir / "character_generation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.log_step(f"✅ Character generation report saved: {report_path}", "SUCCESS")

    def extract_character_gender_and_age(self, character: Dict) -> Tuple[str, str]:
        """Karakterin cinsiyeti ve yaş grubunu tespit et"""
        name = character.get('name', '').lower()
        description = character.get('physical_description', '').lower()

        # Cinsiyet tespiti
        gender = "unknown"

        # İsimden cinsiyet tespiti
        female_names = ['livia', 'claudia', 'helena', 'julia', 'antonia']
        male_names = ['marcus', 'gaius', 'quintus', 'lucius', 'titus', 'brutus', 'cassius']

        for fname in female_names:
            if fname in name:
                gender = "female"
                break

        for mname in male_names:
            if mname in name:
                gender = "male"
                break

        # İsimden anahtar kelimelerle cinsiyet tespiti
        if any(word in name for word in ['wife', 'mother', 'woman']):
            gender = "female"
        elif any(word in name for word in ['husband', 'father', 'man']):
            gender = "male"

        # Fiziksel tanımlamadan cinsiyet tespiti
        if 'mother' in description or 'maternal' in description or 'she ' in description:
            gender = "female"
        elif 'father' in description or 'paternal' in description or 'he ' in description:
            gender = "male"

        # Yaş grubu tespiti
        age_group = "adult"
        if any(word in description for word in ['child', 'young boy', 'young girl', '6 years old']):
            age_group = "child"
        elif any(word in description for word in ['old', 'aged', 'elderly', 'gray hair', 'grey hair']):
            age_group = "elderly"
        elif any(word in description for word in ['young', 'youthful', 'late twenties']):
            age_group = "young_adult"

        return gender, age_group

    def extract_character_role(self, character: Dict) -> str:
        """Cinsiyet ve yaş dikkate alınarak karakter rolü çıkarma - İYİLEŞTİRİLMİŞ"""
        description = character.get('physical_description', '').lower()
        name = character.get('name', '').lower()
        historical_period = self.current_historical_period

        # Cinsiyet ve yaş tespiti
        gender, age_group = self.extract_character_gender_and_age(character)

        # Meslek/rol tespiti
        role_keywords = {
            'baker': ['flour', 'bread', 'kneading', 'oven', 'dough', 'bakery', 'baker'],
            'weaver': ['weaver', 'needle', 'thread', 'fabric', 'sewing', 'dress'],
            'soldier': ['soldier', 'military', 'armor', 'sword', 'warrior', 'battle'],
            'merchant': ['merchant', 'trade', 'business', 'wealthy', 'scrolls', 'commerce'],
            'gardener': ['garden', 'plants', 'cultivation', 'growing'],
            'mother': ['mother', 'maternal', 'child', 'caring'],
            'scholar': ['scholar', 'scrolls', 'manuscripts', 'learning']
        }

        detected_role = None
        for role, keywords in role_keywords.items():
            if any(keyword in description or keyword in name for keyword in keywords):
                detected_role = role
                break

        # Cinsiyet ve yaş bazlı rol düzeltmeleri
        if detected_role == 'baker' and gender == 'female':
            base_role = "baker woman" if age_group != "elderly" else "elderly baker woman"
        elif detected_role == 'weaver' and gender == 'female':
            base_role = "weaver woman"
        elif detected_role == 'soldier' and gender == 'male':
            base_role = "soldier" if age_group != "elderly" else "retired soldier"
        elif detected_role == 'merchant' and gender == 'male':
            base_role = "merchant"
        elif detected_role == 'mother' and gender == 'female':
            base_role = "mother"
        elif age_group == 'child':
            if gender == 'male':
                base_role = "young boy"
            elif gender == 'female':
                base_role = "young girl"
            else:
                base_role = "child"
        else:
            # Fallback - cinsiyet ve yaş bazlı genel roller
            if gender == 'female':
                if age_group == 'elderly':
                    base_role = "elderly woman"
                elif age_group == 'young_adult':
                    base_role = "young woman"
                else:
                    base_role = "woman"
            elif gender == 'male':
                if age_group == 'elderly':
                    base_role = "elderly man"
                elif age_group == 'young_adult':
                    base_role = "young man"
                else:
                    base_role = "man"
            else:
                base_role = "person"

        # Tarihi dönem ekleme
        if 'roman' in historical_period.lower() or '79 ad' in historical_period.lower():
            role_prefix = "ancient Roman"
        elif 'medieval' in historical_period.lower():
            role_prefix = "medieval"
        elif 'egyptian' in historical_period.lower():
            role_prefix = "ancient Egyptian"
        elif 'alexandria' in self.current_topic.lower() or 'library' in self.current_topic.lower():
            role_prefix = "ancient Hellenistic"
        else:
            role_prefix = "historical"

        return f"{role_prefix} {base_role}"

    def clean_prompt_for_midjourney(self, prompt: str) -> str:
        """Midjourney yasaklı kelimelerini kapsamlı temizleme"""

        banned_words_replacements = {
            # Şiddet ve savaş
            "violence": "conflict",
            "violent": "intense",
            "war": "service",
            "battle": "duty",
            "fight": "struggle",
            "fighting": "working",
            "blood": "heritage",
            "bloody": "intense",
            "death": "loss",
            "dead": "still",
            "kill": "overcome",
            "killing": "overcoming",
            "murder": "mystery",
            "weapon": "tool",
            "weapons": "tools",
            "sword": "blade",
            "gun": "implement",
            "knife": "blade",
            "arrow": "dart",
            "spear": "staff",
            "memories of violence": "memories of past struggles",
            "shadowed by memories of violence": "marked by experiences of duty and service",
            "haunted by": "shaped by",
            "trauma": "experience",
            "traumatic": "challenging",
            "brutal": "harsh",
            "savage": "wild",
            "destruction": "change",
            "destroy": "transform",
            "attack": "approach",
            "attacking": "approaching",
            "enemy": "opponent",
            "enemies": "opponents",

            # Cinsellik ve vücut
            "nude": "unclothed",
            "naked": "bare",
            "breast": "chest",
            "breasts": "chest area",
            "nipple": "chest",
            "nipples": "chest",
            "intimate": "close",
            "explicit": "detailed",
            "sexual": "romantic",
            "sexy": "attractive",
            "erotic": "artistic",
            "seductive": "charming",
            "sensual": "graceful",
            "provocative": "striking",
            "suggestive": "expressive",
            "arousing": "inspiring",
            "lust": "desire",
            "passion": "enthusiasm",
            "orgasm": "climax",
            "genitals": "anatomy",
            "penis": "anatomy",
            "vagina": "anatomy",
            "ass": "posterior",
            "butt": "posterior",
            "thong": "underwear",
            "lingerie": "undergarments",
            "bikini": "swimwear",
            "cleavage": "neckline",
            "revealing": "open",
            "tight": "fitted",
            "skin-tight": "form-fitting",
            "see-through": "translucent",
            "transparent": "clear",
            "sheer": "light",

            # Uyuşturucu ve alkol
            "drug": "medicine",
            "drugs": "medicines",
            "cocaine": "powder",
            "heroin": "substance",
            "marijuana": "herb",
            "cannabis": "plant",
            "weed": "plant",
            "smoking": "breathing",
            "drunk": "dizzy",
            "alcohol": "beverage",
            "beer": "drink",
            "wine": "grape juice",
            "intoxicated": "relaxed",
            "overdose": "excess",

            # Yaş ve çocuk içeriği
            "child": "young person",
            "children": "young people",
            "kid": "youth",
            "kids": "youths",
            "baby": "infant",
            "infant": "small child",
            "toddler": "small child",
            "minor": "young person",
            "underage": "young",
            "teenage": "young adult",
            "teen": "young adult",
            "young girl": "young woman",
            "young boy": "young man",
            "little girl": "small person",
            "little boy": "small person",

            # Irkçılık ve nefret
            "nazi": "authoritarian",
            "hitler": "dictator",
            "slave": "worker",
            "slavery": "forced labor",
            "racist": "prejudiced",
            "racism": "prejudice",
            "hate": "dislike",
            "hatred": "dislike",

            # Dinî ve kutsal
            "god": "deity",
            "jesus": "religious figure",
            "christ": "religious figure",
            "buddha": "spiritual figure",
            "prophet": "spiritual teacher",
            "sacred": "special",
            "holy": "blessed",

            # Siyasi figürler
            "trump": "leader",
            "biden": "official",
            "putin": "leader",
            "politician": "official",

            # Korku ve dehşet
            "horror": "mystery",
            "scary": "mysterious",
            "terrifying": "impressive",
            "nightmare": "dream",
            "demon": "spirit",
            "devil": "dark figure",
            "hell": "underworld",
            "torture": "hardship",
            "pain": "discomfort",
            "suffering": "difficulty",
            "agony": "struggle",

            # Hastalık ve yaralanma
            "sick": "unwell",
            "disease": "condition",
            "cancer": "illness",
            "virus": "infection",
            "wound": "mark",
            "injury": "mark",
            "scar": "mark",
            "bruise": "mark",
            "cut": "line",

            # Suç ve illegal
            "crime": "incident",
            "criminal": "person",
            "illegal": "forbidden",
            "steal": "take",
            "theft": "taking",
            "rob": "take from",
            "robbery": "taking",
            "fraud": "deception",

            # Teknoloji markaları (bazen sorun olabiliyor)
            "iphone": "phone",
            "android": "device",
            "samsung": "device",
            "apple": "fruit",
            "google": "search",
            "facebook": "social media",
            "instagram": "social platform",
            "twitter": "social platform",

            # Diğer hassas konular
            "suicide": "ending",
            "depression": "sadness",
            "mental illness": "mental state",
            "crazy": "unusual",
            "insane": "unusual",
            "mad": "unusual",
            "stupid": "simple",
            "idiot": "person",
            "ugly": "plain",
            "fat": "large",
            "skinny": "thin",
            "anorexia": "thinness",
            "bulimia": "eating issue"
        }

        cleaned_prompt = prompt.lower()  # Küçük harfe çevir

        # Kelime değiştirme
        for banned, replacement in banned_words_replacements.items():
            cleaned_prompt = cleaned_prompt.replace(banned.lower(), replacement)

        # İlk harfi büyük yap
        cleaned_prompt = cleaned_prompt[0].upper() + cleaned_prompt[1:] if cleaned_prompt else ""

        return cleaned_prompt

    def generate_safe_emotional_expression(self, psychology: Dict) -> str:
        """Güvenli emotional expression oluştur"""

        emotional_complexity = psychology.get("emotional_complexity", "peaceful contemplative expression")

        # Yasaklı ifadeleri temizle
        safe_expressions = {
            "shadowed by memories of violence": "marked by years of dedicated service",
            "memories of violence": "memories of duty and honor",
            "haunted by war": "shaped by military experience",
            "traumatic memories": "challenging experiences",
            "painful past": "difficult history",
            "dark thoughts": "deep thoughts",
            "inner demons": "inner struggles",
            "emotional scars": "emotional depth",
            "wounded soul": "experienced spirit",
            "broken heart": "tender heart",
            "tortured by": "shaped by",
            "consumed by": "influenced by",
            "obsessed with": "focused on",
            "addicted to": "devoted to"
        }

        safe_expression = emotional_complexity
        for risky, safe in safe_expressions.items():
            safe_expression = safe_expression.replace(risky, safe)

        # Final temizlik
        safe_expression = self.clean_prompt_for_midjourney(safe_expression)

        # Uzunsa kısalt
        if len(safe_expression) > 100:
            safe_expression = safe_expression[:97] + "..."

        return safe_expression

    def generate_character_prompt(self, character: Dict) -> str:
        """JSON verilerinden güvenli Midjourney prompt - UPDATED"""

        # Temel bilgiler
        name = character.get("name", "Unknown")
        gender = character.get("gender", "unknown")
        role = character.get("role", "character")
        physical = character.get("physical_description", "")
        importance_score = character.get("importance_score", 5)
        use_in_marketing = character.get("use_in_marketing", False)

        # Emotional core - GÜVENLİ versiyon
        psychology = character.get("tóibín_psychology", {})
        emotional_core = self.generate_safe_emotional_expression(psychology)

        # Context bilgileri
        topic = self.current_topic
        period = self.current_historical_period

        # Pose seçimi
        if use_in_marketing or importance_score >= 8:
            pose = "front facing portrait"
        elif importance_score >= 7:
            pose = "three-quarter view portrait"
        else:
            pose = "profile portrait"

        # Prompt oluştur
        prompt = f"{topic} {period} character named {name}, {gender} {role}, {physical}, emotional expression: {emotional_core}, {pose}, portrait photography style, highly detailed, soft golden hour lighting, 2:3 aspect ratio"

        # Final güvenlik temizliği
        clean_prompt = self.clean_prompt_for_midjourney(prompt)

        print(f"   🎭 Generated safe prompt for {name}: {clean_prompt}")
        return clean_prompt

    def generate_detailed_character_prompt(self, character: Dict, is_base: bool = False) -> str:
        """
        Çok detaylı karakter prompt'u - consistency için
        """
        # JSON'dan detaylı visual signature kullan
        visual_sig = character.get("visual_signature", {})

        # Temel bilgiler
        name = character.get("name", "Unknown")
        physical = character.get("physical_description", "")
        age_range = character.get("age_range", "adult")

        # Visual signature detayları
        hair = visual_sig.get("hair", "dark hair")
        face = visual_sig.get("face", "strong features")
        body = visual_sig.get("body", "average build")
        skin = visual_sig.get("skin", "olive skin")
        clothing = visual_sig.get("clothing", "simple tunic")

        # Detailed prompt for consistency
        detailed_prompt = f"""
        {physical}, {age_range}, {hair}, {face}, {body}, {skin}, {clothing},
        ancient Roman times, historical accuracy, portrait photography style,
        highly detailed facial features, cinematic lighting, professional quality
        """

        if is_base:
            detailed_prompt += ", front-facing portrait, direct eye contact, neutral expression"

        # Temizlik
        clean_prompt = self.clean_prompt_for_midjourney(detailed_prompt)

        print(f"   🎭 Detailed prompt for {name}: {clean_prompt}")
        return clean_prompt

    def get_character_variants_for_consistency(self) -> List[Dict]:
        """
        Consistency odaklı varyant listesi - güvenli kelimeler
        """
        return [
            {"key": "profile", "pose": "side profile portrait", "aspect_ratio": "2:3", "framing": "clean profile view"},
            {"key": "three_quarter", "pose": "three-quarter view portrait", "aspect_ratio": "2:3", "framing": "slight turn, both eyes visible"},
            {"key": "upper_body", "pose": "upper body portrait, shoulders visible", "aspect_ratio": "2:3", "framing": "from waist up, shoulders relaxed"},
            {"key": "full_body", "pose": "full-body standing pose", "aspect_ratio": "3:4", "framing": "full height, natural stance"}
        ]

    def generate_variant_prompt_with_reference(self, character: Dict, variant: Dict, reference_url: str) -> str:
        """
        Reference URL kullanarak tutarlı varyant prompt'u oluştur - DETAYLI VERSİYON
        """
        # Temel bilgileri JSON'dan al
        name = character.get("name", "character")
        gender = character.get("gender", "person")
        age_range = character.get("age_range", "adult")
        physical_desc = character.get("physical_description", "")

        # Visual signature detayları
        visual_sig = character.get("visual_signature", {})
        hair = visual_sig.get("hair", "dark hair")
        face = visual_sig.get("face", "strong features")
        skin = visual_sig.get("skin", "olive skin")

        # Gender düzeltme
        gender_word = "man" if gender == "male" else "woman" if gender == "female" else "person"

        # Yaş ile gender birleştir
        if "year" in age_range or "40s" in age_range or "30s" in age_range or "60s" in age_range:
            age_description = f"{age_range} {gender_word}"
        else:
            age_description = f"{gender_word}"

        # Detaylı prompt oluştur
        detailed_prompt = f"""
        ancient Roman {age_description}, {hair}, {face}, {skin}, 
        {variant['pose']}, {variant['framing']}, 
        historical accuracy, portrait photography style, 
        highly detailed, professional quality
        """

        # Temizlik
        clean_prompt = self.clean_prompt_for_midjourney(detailed_prompt)

        return clean_prompt

    def wait_for_task_completion(self, task_id: str, char_name: str, max_wait: int = 600) -> Optional[Dict]:
        """
        Task tamamlanana kadar bekle (max 10 dakika)
        """
        print(f"   ⏳ Waiting for {char_name} completion...")

        wait_cycles = max_wait // 10  # 10 saniye aralıklarla kontrol

        for cycle in range(wait_cycles):
            result_data = self.check_task_status(task_id)

            if result_data and isinstance(result_data, dict):
                print(f"   ✅ {char_name} completed after {cycle * 10} seconds")
                return result_data
            elif result_data is False:
                print(f"   ❌ {char_name} failed after {cycle * 10} seconds")
                return None

            # Progress indicator
            if cycle % 6 == 0:  # Her dakika
                print(f"   ⏳ Still waiting for {char_name}... ({cycle * 10}s)")

            time.sleep(10)

        print(f"   ⏰ {char_name} timed out after {max_wait} seconds")
        return None

    def clean_prompt_for_piapi_v7(self, prompt: str) -> str:
        """V7 parametrelerini koruyarak temizleme - ESKİ clean_prompt_for_piapi'nin YENİ VERSİYONU"""
        import re

        # Eski --ar ve eski --v parametrelerini kaldır (ama V7'yi değil)
        prompt = re.sub(r'--ar\s+\d+:\d+', '', prompt)
        prompt = re.sub(r'--v\s+[1-6](?:\.\d+)?', '', prompt)  # Sadece V6 ve altını kaldır

        # V7 parametrelerini KORUYACAK şekilde genel temizlik
        # Tireleri sadece prompt içeriğinde değiştir, parametrelerde değil
        parts = prompt.split(' --')
        if len(parts) > 1:
            # İlk kısım prompt, geri kalanı parametreler
            content = parts[0]
            params = [' --' + p for p in parts[1:]]

            # Sadece content kısmındaki tireleri temizle
            content = content.replace(' - ', ' ').replace('-', ' ')
            content = re.sub(r'\s+', ' ', content).strip()

            # Parametreleri geri ekle
            prompt = content + ''.join(params)
        else:
            # Parametre yok, sadece content temizle
            prompt = prompt.replace(' - ', ' ').replace('-', ' ')
            prompt = re.sub(r'\s+', ' ', prompt).strip()

        return prompt

    def test_api_connection(self) -> bool:
        """Test PIAPI connection"""
        self.log_step("🔍 Testing PIAPI connection")

        test_prompt = "red apple on white table --ar 1:1 --v 6.1"

        payload = {
            "model": "midjourney",
            "task_type": "imagine",
            "input": {
                "prompt": test_prompt,
                "aspect_ratio": "1:1",
                "process_mode": "relax"
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/task",
                headers=self.headers,
                json=payload,
                timeout=10
            )

            print(f"📡 Test Response: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                self.log_step("✅ API Connection OK", "SUCCESS", {"response": result})
                return True
            else:
                self.debug_log_on_error("POST", f"{self.base_url}/task", self.headers, payload, response)
                self.log_step(f"❌ API Error: {response.status_code}", "ERROR")
                return False

        except Exception as e:
            self.log_step(f"❌ Connection Test Failed: {e}", "ERROR")
            return False

    def submit_midjourney_task_v6(self, prompt: str, aspect_ratio: str = "16:9", char_name: str = "") -> Optional[str]:
        """V6.1 parametreleriyle task submission - --cref desteği için"""

        # V6.1 parametrelerini prompt'a ekle
        v6_params = f" --ar {aspect_ratio} --v 6.1 --stylize 100"
        full_prompt = prompt + v6_params

        # V6 parametrelerini koruyarak temizle
        clean_prompt = self.clean_prompt_for_piapi_v7(full_prompt)

        payload = {
            "model": "midjourney",
            "task_type": "imagine",
            "input": {
                "prompt": clean_prompt,
                "aspect_ratio": aspect_ratio,
                "process_mode": CONFIG.visual_config["process_mode"],
                "skip_prompt_check": False
            }
        }

        url = f"{self.base_url}/task"

        try:
            self.api_calls_made += 1
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    task_data = result.get("data", {})
                    task_id = task_data.get("task_id")
                    print(f"✅ V6.1 Task submitted: {task_id} ({char_name})")
                    print(f"   📝 V6.1 Prompt: {clean_prompt}")
                    return task_id
                else:
                    print(f"❌ API Error for {char_name}: {result.get('message', 'Unknown')}")
                    return None
            else:
                print(f"\n🔍 HTTP ERROR DEBUG for {char_name}:")
                print(f"Status Code: {response.status_code}")
                print(f"URL: {url}")
                print(f"Request Payload: {json.dumps(payload, indent=2)}")
                print(f"Response Headers: {dict(response.headers)}")
                print(f"Response Text: {response.text}")
                try:
                    error_json = response.json()
                    print(f"Error JSON: {json.dumps(error_json, indent=2)}")
                except:
                    print("Could not parse response as JSON")
                return None

        except Exception as e:
            print(f"❌ EXCEPTION DEBUG for {char_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def check_task_status(self, task_id: str) -> Optional[Dict]:
        """Check single task status"""
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
                        # Get image URLs with priority: temporary_image_urls > image_url
                        temp_urls = output.get("temporary_image_urls", [])
                        image_url = output.get("image_url", "")

                        if temp_urls and len(temp_urls) > 0:
                            selected_url = temp_urls[1] if len(temp_urls) >= 2 else temp_urls[0]
                            return {"url": selected_url, "source": "temporary_image_urls"}
                        elif image_url:
                            return {"url": image_url, "source": "image_url"}
                        else:
                            return False  # Completed but no images

                    elif status == "failed":
                        return False  # Failed
                    else:
                        return None  # Still processing

        except Exception as e:
            return None  # Error, treat as still processing

        return None

    def download_image(self, result_data: Dict, save_path: str) -> bool:
        """Download image with proper headers"""
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

                file_size = os.path.getsize(save_path)
                self.successful_downloads += 1
                self.log_step(f"✅ Downloaded: {os.path.basename(save_path)} ({file_size} bytes)", "SUCCESS")
                return True
            else:
                self.log_step(f"❌ HTTP {response.status_code}", "ERROR")
                return False

        except Exception as e:
            self.log_step(f"❌ Download failed: {e}", "ERROR")
            return False

    def generate_all_characters_parallel(self, character_profiles: Dict):
        """
        CHARACTER CONSISTENCY odaklı generation
        1. Önce her karakter için base portrait (reference)
        2. Reference kullanarak consistent variants
        """
        self.log_step("🎭 Starting Character Consistency Generation")

        main_characters = character_profiles.get("main_characters", [])

        # Marketing karakterleri + önemli karakterler (importance_score >= 7)
        marketing_characters = [char for char in main_characters if char.get("use_in_marketing", False)]
        important_characters = [char for char in main_characters if char.get("importance_score", 0) >= 7 and not char.get("use_in_marketing", False)]

        # Birleştir
        characters_to_process = marketing_characters + important_characters

        if not characters_to_process:
            self.log_step("❌ No marketing or important characters found", "ERROR")
            return False

        print(f"📊 Total characters to process:")
        print(f"   🎯 Marketing characters: {len(marketing_characters)} - {[c['name'] for c in marketing_characters]}")
        print(f"   ⭐ Important characters (score≥7): {len(important_characters)} - {[c['name'] for c in important_characters]}")
        print(f"   📊 Total: {len(characters_to_process)}")

        # PHASE 1: Generate base portraits (references) for each character
        print(f"\n🎯 PHASE 1: Generating base portraits for character consistency")
        base_portraits = {}

        for character in characters_to_process:
            char_name = character["name"]
            safe_name = char_name.lower().replace(" ", "_").replace(".", "")
            base_path = self.characters_dir / f"{safe_name}_base_portrait.png"

            # Check if base portrait already exists
            if base_path.exists() and base_path.stat().st_size > 1024:
                print(f"   ✅ {char_name} base portrait already exists")
                # Load existing reference URL from JSON
                json_path = self.characters_dir / f"{safe_name}_base_portrait.json"
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            metadata = json.load(f)
                            base_portraits[char_name] = metadata.get('image_url', '')
                        continue
                    except:
                        pass

            # Generate base portrait
            print(f"   🎨 Generating base portrait: {char_name}")
            base_prompt = self.generate_detailed_character_prompt(character, is_base=True)

            task_id = self.submit_midjourney_task_v7(
                base_prompt,
                aspect_ratio="2:3",
                char_name=f"{char_name} (BASE)"
            )

            if task_id:
                # Wait for completion and download
                result_data = self.wait_for_task_completion(task_id, char_name)
                if result_data:
                    if self.download_image(result_data, str(base_path)):
                        base_portraits[char_name] = result_data["url"]

                        # Save base portrait metadata
                        metadata = {
                            "name": char_name,
                            "variant": "base_portrait",
                            "role": self.extract_character_role(character),
                            "prompt": base_prompt,
                            "image_url": result_data["url"],
                            "local_path": str(base_path),
                            "generated_at": datetime.now().isoformat(),
                            "is_reference": True
                        }

                        json_path = self.characters_dir / f"{safe_name}_base_portrait.json"
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)

                        print(f"   ✅ Base portrait completed: {char_name}")
                    else:
                        print(f"   ❌ Base portrait download failed: {char_name}")
                else:
                    print(f"   ❌ Base portrait generation failed: {char_name}")

            time.sleep(2)  # Brief pause between characters

        if not base_portraits:
            self.log_step("❌ No base portraits generated", "ERROR")
            return False

        print(f"\n📊 Base portraits completed: {len(base_portraits)}/{len(characters_to_process)}")

        # PHASE 2: Generate consistent variants using base portraits as reference
        print(f"\n🎯 PHASE 2: Generating consistent variants using references")

        character_tasks = {}
        existing_assets = 0
        failed_submissions = []

        for character in characters_to_process:
            char_name = character["name"]
            safe_name = char_name.lower().replace(" ", "_").replace(".", "")

            # Skip if no base portrait
            if char_name not in base_portraits:
                print(f"   ⚠️ Skipping {char_name} - no base portrait")
                continue

            base_reference_url = base_portraits[char_name]
            variants = self.get_character_variants_for_consistency()

            print(f"\n🎭 Processing variants for: {char_name}")
            print(f"   📋 Variants: {[v['key'] for v in variants]}")

            for variant in variants:
                variant_key = variant["key"]
                image_path = self.characters_dir / f"{safe_name}_{variant_key}.png"
                json_path = self.characters_dir / f"{safe_name}_{variant_key}.json"

                # Check existing
                if image_path.exists() and json_path.exists():
                    try:
                        if image_path.stat().st_size > 1024:
                            with open(json_path, "r", encoding="utf-8") as f:
                                json.load(f)
                            print(f"   ✅ {variant_key} already exists")
                            existing_assets += 1
                            continue
                    except Exception:
                        print(f"   🔄 {variant_key} corrupted, will regenerate")

                # Generate variant with reference
                variant_prompt = self.generate_variant_prompt_with_reference(
                    character, variant, base_reference_url
                )

                # Reference sistemini AKTİF hale getir - ama syntax düzelt
                variant_prompt += f" --cref {base_reference_url}"

                aspect_ratio = variant["aspect_ratio"]

                print(f"   📝 Generating {variant_key}: {variant['pose']}")
                print(f"   🔗 Reference URL: {base_reference_url}")
                print(f"   📝 Final prompt: {variant_prompt}")

                task_id = self.submit_midjourney_task_v7(
                    variant_prompt,
                    aspect_ratio=aspect_ratio,
                    char_name=f"{char_name} ({variant_key})"
                )

                if task_id:
                    character_tasks[(char_name, variant_key)] = {
                        "task_id": task_id,
                        "prompt": variant_prompt,
                        "aspect_ratio": aspect_ratio,
                        "character_data": character,
                        "variant": variant,
                        "reference_url": base_reference_url
                    }
                    print(f"   ✅ Task submitted: {task_id}")
                else:
                    failed_submissions.append((char_name, variant_key))
                    print(f"   ❌ Task failed: {variant_key}")

                time.sleep(0.5)

        print(f"\n📊 VARIANT SUBMISSION SUMMARY:")
        print(f"   📦 Existing variant assets: {existing_assets}")
        print(f"   ✅ New variant tasks submitted: {len(character_tasks)}")
        print(f"   ❌ Failed variant submissions: {len(failed_submissions)}")

        if not character_tasks and existing_assets == 0:
            self.log_step("❌ No variant tasks submitted and no existing assets", "ERROR")
            return False

        # Monitor variant tasks
        completed = {}
        max_cycles = CONFIG.visual_config["max_wait_cycles"]

        for cycle in range(max_cycles):
            if not character_tasks:
                break

            pending_count = len(character_tasks)
            completed_count = len(completed) + existing_assets

            self.log_step(f"📊 Cycle {cycle + 1}: {completed_count} completed, {pending_count} pending")

            to_remove = []
            for (char_name, variant_key), task in list(character_tasks.items()):
                result_data = self.check_task_status(task["task_id"])

                if result_data and isinstance(result_data, dict):
                    print(f"   ✅ Completed: {char_name} ({variant_key})")
                    completed[(char_name, variant_key)] = {
                        "result": result_data,
                        "task": task
                    }
                    to_remove.append((char_name, variant_key))

                elif result_data is False:
                    print(f"   ❌ Failed: {char_name} ({variant_key})")
                    to_remove.append((char_name, variant_key))

            for item in to_remove:
                character_tasks.pop(item, None)

            if not character_tasks:
                break

            time.sleep(CONFIG.visual_config["wait_interval_seconds"])

        # Download variants and save metadata
        successful_downloads = 0

        for (char_name, variant_key), data in completed.items():
            result_data = data["result"]
            task = data["task"]
            safe_name = char_name.lower().replace(" ", "_").replace(".", "")
            image_path = self.characters_dir / f"{safe_name}_{variant_key}.png"

            if self.download_image(result_data, str(image_path)):
                successful_downloads += 1

                # Update character references (use base portrait as main reference)
                if char_name not in self.character_references:
                    self.character_references[char_name] = task["reference_url"]

                # Save variant metadata
                metadata = {
                    "name": char_name,
                    "variant": variant_key,
                    "variant_info": task["variant"],
                    "role": self.extract_character_role(task["character_data"]),
                    "prompt": task["prompt"],
                    "image_url": result_data["url"],
                    "url_source": result_data["source"],
                    "aspect_ratio": task["aspect_ratio"],
                    "local_path": str(image_path),
                    "generated_at": datetime.now().isoformat(),
                    "reference_image": task["reference_url"],
                    "importance_score": task["character_data"].get("importance_score", 0),
                    "use_in_marketing": task["character_data"].get("use_in_marketing", False)
                }

                json_path = self.characters_dir / f"{safe_name}_{variant_key}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Final assessment
        total_base_portraits = len(base_portraits)
        total_variants = successful_downloads + existing_assets

        character_variant_count = {}
        for (char_name, variant_key) in completed.keys():
            character_variant_count[char_name] = character_variant_count.get(char_name, 0) + 1

        characters_with_variants = len(character_variant_count)
        success_rate = characters_with_variants / len(characters_to_process) if characters_to_process else 0

        is_successful = total_base_portraits >= 1 and success_rate >= 0.5

        print(f"\n📊 FINAL CHARACTER CONSISTENCY SUMMARY:")
        print(f"   👥 Total characters processed: {len(characters_to_process)}")
        print(f"   🎯 Marketing characters: {len(marketing_characters)}")
        print(f"   ⭐ Important characters: {len(important_characters)}")
        print(f"   🎨 Base portraits generated: {total_base_portraits}")
        print(f"   ✅ Characters with variants: {characters_with_variants}")
        print(f"   📸 Total variant assets: {total_variants}")
        print(f"   🔄 Newly generated variants: {successful_downloads}")
        print(f"   📦 Previously existing variants: {existing_assets}")
        print(f"   📈 Success rate: {success_rate:.1%}")

        self.log_step(
            f"✅ Consistent Character Generation {'SUCCESS' if is_successful else 'PARTIAL'}: "
            f"{total_base_portraits} base + {characters_with_variants}/{len(characters_to_process)} chars with variants",
            "SUCCESS" if is_successful else "ERROR"
        )

        return is_successful

    def clean_prompt_for_piapi(self, prompt: str) -> str:
        """Remove all characters that might confuse PiAPI's prompt parser"""
        import re

        # Remove all --ar parameters
        prompt = re.sub(r'--ar\s+\d+:\d+', '', prompt)

        # Remove all --v parameters
        prompt = re.sub(r'--v\s+[\d.]+', '', prompt)

        # Remove any other -- parameters
        prompt = re.sub(r'--\w+(?:\s+[\w:.]+)?', '', prompt)

        # KRITIK: Tüm tire karakterlerini çıkar veya değiştir
        prompt = prompt.replace(' - ', ' ')  # " - " -> " "
        prompt = prompt.replace('-', ' ')  # Tüm tireleri boşlukla değiştir

        # Extra spaces ve temizlik
        prompt = re.sub(r'\s+', ' ', prompt)  # Multiple spaces to single
        prompt = prompt.strip()

        return prompt

    def run_character_only_generation(self) -> bool:
        """Run CHARACTER-ONLY generation process (multi-variant compatible)"""
        print("🚀" * 50)
        print("SERVER MIDJOURNEY CHARACTER GENERATOR v2.0 (multi-variant)")
        print("🔗 Database integrated")
        print("🎭 ROBUST CHARACTER GENERATION")
        print("✅ Skip existing | ✅ Individual error handling | ✅ Smart debug")
        print("🖥️ Production-ready automation")
        print("🚀" * 50)

        # 0) API test
        if not self.test_api_connection():
            self.log_step("❌ API connection failed - aborting", "ERROR")
            return False

        # 1) Sıradaki projeyi al
        found, project_info = self.get_next_project_from_database()
        if not found:
            return False

        print(f"✅ Project found: {project_info['topic']}")
        print(f"📁 Output directory: {project_info['output_dir']}")
        print(f"🏛️ Historical period: {project_info['historical_period']}")

        try:
            # 2) Klasörleri hazırla
            self.setup_character_directories()

            # 3) Sadece karakter profillerini yükle
            character_profiles = self.load_character_profiles_only()

            # 4) Karakterleri (çoklu varyant) üret
            characters_success = self.generate_all_characters_parallel(character_profiles)

            # 5) Raporu kaydet
            self.save_character_generation_report()

            # 6) DB güncelle
            characters_count = len(self.character_references)

            if characters_success:
                self.db_manager.mark_character_generation_completed(self.current_topic_id, characters_count)
            else:
                self.db_manager.mark_character_generation_failed(self.current_topic_id, "Character generation failed")

            # Son durum
            if characters_success:
                print("\n" + "🎉" * 50)
                print("ROBUST CHARACTER GENERATION SUCCESSFUL!")
                print(f"✅ Characters Generated: {characters_count}")
                print(f"✅ API Calls: {self.api_calls_made}")
                print(f"✅ Downloads: {self.successful_downloads}")
                print(f"📁 Saved to: {self.characters_dir}")
                print("🎉" * 50)
            else:
                print("\n" + "❌" * 50)
                print("CHARACTER GENERATION FAILED!")
                print("Check logs for details")
                print("❌" * 50)

            return characters_success

        except Exception as e:
            self.log_step(f"❌ Character generation failed: {e}", "ERROR")
            self.db_manager.mark_character_generation_failed(self.current_topic_id, str(e))
            import traceback;
            traceback.print_exc()
            return False


def run_autonomous_mode():
    """Run autonomous mode - continuously process completed story topics for character generation"""
    print("🤖 AUTONOMOUS ROBUST CHARACTER GENERATION MODE STARTED")
    print("🔄 Will process all completed story topics continuously")
    print("⏹️ Press Ctrl+C to stop gracefully")

    # Setup graceful shutdown
    running = True
    processed_count = 0
    cycles_count = 0
    start_time = time.time()
    last_activity_time = time.time()

    def signal_handler(signum, frame):
        nonlocal running
        print(f"\n⏹️ Received shutdown signal ({signum})")
        print("🔄 Finishing current character generation and shutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while running:
        try:
            # Initialize character generator to check for work
            character_generator = ServerMidjourneyVisualGenerator()
            found, project_info = character_generator.get_next_project_from_database()

            if found:
                cycle_had_work = False
                cycle_start_time = time.time()

                print(f"\n🔄 CYCLE {cycles_count + 1}: Found completed story topics ready for character generation")

                # Process ALL available topics in this cycle
                while running and found:
                    # Process one topic (generator already has the project loaded)
                    success = character_generator.run_character_only_generation()

                    if success:
                        processed_count += 1
                        cycle_had_work = True
                        last_activity_time = time.time()
                        print(f"✅ Character generation completed! (Topic {processed_count})")
                    else:
                        print(f"⚠️ Character generation failed")
                        break

                    # Short pause between topics in same cycle
                    if running:
                        time.sleep(2)

                    # Check for more work with fresh generator instance
                    character_generator = ServerMidjourneyVisualGenerator()
                    found, project_info = character_generator.get_next_project_from_database()

                # Cycle completed
                cycles_count += 1
                cycle_time = time.time() - cycle_start_time

                if cycle_had_work:
                    print(f"\n📊 CYCLE {cycles_count} COMPLETED:")
                    print(f"   ✅ Topics processed this cycle: {processed_count}")
                    print(f"   ⏱️ Cycle time: {cycle_time:.1f} seconds")
                    print(f"   📈 Total topics processed: {processed_count}")

                # Short pause between cycles
                if running:
                    print("⏳ Pausing 10 seconds before next cycle...")
                    time.sleep(10)

            else:
                # No topics ready - smart waiting
                time_since_activity = time.time() - last_activity_time

                if time_since_activity < 300:  # Less than 5 minutes since last activity
                    wait_time = 60  # Wait 1 minute
                    print("😴 No topics ready. Recent activity detected - waiting 60s...")
                else:
                    wait_time = 3600  # Wait 1 hour
                    print("😴 No topics ready for extended period - waiting 1 hour...")
                    print(f"⏰ Last activity: {time_since_activity / 60:.1f} minutes ago")

                # Wait with interrupt capability
                for i in range(wait_time):
                    if not running:
                        break
                    if i > 0 and i % 300 == 0:  # Show progress every 5 minutes
                        remaining = (wait_time - i) / 60
                        print(f"⏳ Still waiting... {remaining:.1f} minutes remaining")
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n⏹️ Keyboard interrupt received")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("⏳ Waiting 30 seconds before retry...")
            time.sleep(30)

    # Shutdown summary
    runtime = time.time() - start_time
    print(f"\n🏁 AUTONOMOUS CHARACTER GENERATION SHUTDOWN")
    print(f"⏱️ Total runtime: {runtime / 3600:.1f} hours")
    print(f"🔄 Total cycles: {cycles_count}")
    print(f"✅ Topics processed: {processed_count}")
    if processed_count > 0:
        print(f"📈 Average topics per cycle: {processed_count / cycles_count:.1f}")
    print("👋 Goodbye!")


if __name__ == "__main__":
    # Check for autonomous mode
    if len(sys.argv) > 1 and sys.argv[1] == '--autonomous':
        run_autonomous_mode()
    else:
        # Original single topic mode
        try:
            print("🚀 SERVER MIDJOURNEY CHARACTER GENERATOR v2.0")
            print("🔗 Database integration with story generator")
            print("🎭 ROBUST CHARACTER GENERATION")
            print("✅ Skip existing | ✅ Individual error handling | ✅ Smart debug")
            print("🖥️ Production-ready automation")
            print("=" * 60)

            generator = ServerMidjourneyVisualGenerator()
            success = generator.run_character_only_generation()

            if success:
                print("🎊 Robust character generation completed successfully!")
            else:
                print("⚠️ Character generation failed or no projects ready")

        except KeyboardInterrupt:
            print("\n⏹️ Character generation stopped by user")
        except Exception as e:
            print(f"💥 Character generation failed: {e}")
            CONFIG.logger.error(f"Character generation failed: {e}")
            import traceback
            traceback.print_exc()