"""
Sleepy Dull Stories - Storage Manager
Dosya yönetimi, CSV okuma, data storage ve retrieval
"""

import json
import shutil
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import re
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src import config


class StorageManager:
    """Dosya yönetimi ve data storage"""

    def __init__(self):
        """Initialize storage manager"""
        self.ensure_directories()
        print("✅ Storage Manager başlatıldı")

    def ensure_directories(self):
        """Gerekli klasörleri oluştur"""
        try:
            config.create_directories()
            print("📁 Storage klasörleri kontrol edildi")
        except Exception as e:
            print(f"❌ Klasör oluşturma hatası: {e}")
            raise

    # ==================== TOPIC MANAGEMENT ====================

    def get_today_topic(self, target_date: date = None) -> str:
        """Bugünün konusunu CSV'den al"""

        if not target_date:
            target_date = date.today()

        try:
            topics_df = pd.read_csv(config.TOPICS_CSV_PATH)

            # Tarihi parse et
            topics_df['date'] = pd.to_datetime(topics_df['date']).dt.date

            # Bugünün konusunu bul
            today_row = topics_df[topics_df['date'] == target_date]

            if not today_row.empty:
                topic = today_row.iloc[0]['topic']
                print(f"📅 {target_date} konusu: {topic}")
                return topic
            else:
                # Eğer tarih bulunamazsa, CSV'deki ilk konuyu al
                fallback_topic = topics_df.iloc[0]['topic']
                print(f"⚠️ {target_date} için konu bulunamadı, fallback: {fallback_topic}")
                return fallback_topic

        except Exception as e:
            print(f"❌ Topic okuma hatası: {e}")
            return "Ancient Roman Daily Life"  # Default fallback

    def get_topic_details(self, topic: str) -> Dict[str, Any]:
        """Konu detaylarını CSV'den al"""

        try:
            topics_df = pd.read_csv(config.TOPICS_CSV_PATH)
            topic_row = topics_df[topics_df['topic'] == topic]

            if not topic_row.empty:
                details = topic_row.iloc[0].to_dict()
                print(f"📖 Konu detayları alındı: {topic}")
                return details
            else:
                print(f"⚠️ Konu detayı bulunamadı: {topic}")
                return {'topic': topic, 'category': 'Unknown', 'priority': 'medium'}

        except Exception as e:
            print(f"❌ Topic detay okuma hatası: {e}")
            return {'topic': topic, 'category': 'Unknown', 'priority': 'medium'}

    def get_upcoming_topics(self, days: int = 7) -> List[Dict[str, Any]]:
        """Gelecek konuları al"""

        try:
            topics_df = pd.read_csv(config.TOPICS_CSV_PATH)
            topics_df['date'] = pd.to_datetime(topics_df['date']).dt.date

            today = date.today()
            upcoming = topics_df[topics_df['date'] >= today].head(days)

            return upcoming.to_dict('records')

        except Exception as e:
            print(f"❌ Upcoming topics okuma hatası: {e}")
            return []

    # ==================== STORY DATA MANAGEMENT ====================

    def save_story_data(self, story_data: Dict[str, Any]) -> Path:
        """Hikaye verisini kaydet"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        topic_clean = self._clean_filename(story_data.get('topic', 'unknown'))
        filename = f"story_{topic_clean}_{timestamp}.json"

        file_path = config.RAW_DATA_DIR / filename

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(story_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Hikaye verisi kaydedildi: {file_path}")
            return file_path

        except Exception as e:
            print(f"❌ Hikaye kaydetme hatası: {e}")
            raise

    def get_latest_story(self, topic: str = None) -> Dict[str, Any]:
        """En son hikaye verisini al"""

        try:
            story_files = list(config.RAW_DATA_DIR.glob("story_*.json"))

            if topic:
                topic_clean = self._clean_filename(topic)
                story_files = [f for f in story_files if topic_clean in f.name]

            if not story_files:
                raise FileNotFoundError("Hikaye dosyası bulunamadı")

            # En son dosyayı al
            latest_file = max(story_files, key=lambda x: x.stat().st_mtime)

            with open(latest_file, 'r', encoding='utf-8') as f:
                story_data = json.load(f)

            print(f"📖 En son hikaye alındı: {latest_file.name}")
            return story_data

        except Exception as e:
            print(f"❌ Hikaye okuma hatası: {e}")
            raise

    # ==================== ENHANCED SCRIPT MANAGEMENT ====================

    def save_enhanced_script(self, enhanced_data: Dict[str, Any]) -> Path:
        """Enhanced script'i kaydet"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        topic_clean = self._clean_filename(enhanced_data.get('topic', 'unknown'))
        filename = f"enhanced_{topic_clean}_{timestamp}.json"

        file_path = config.PROCESSED_DATA_DIR / filename

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Enhanced script kaydedildi: {file_path}")
            return file_path

        except Exception as e:
            print(f"❌ Enhanced script kaydetme hatası: {e}")
            raise

    def get_latest_script(self, topic: str = None) -> Dict[str, Any]:
        """En son enhanced script'i al"""

        try:
            script_files = list(config.PROCESSED_DATA_DIR.glob("enhanced_*.json"))

            if topic:
                topic_clean = self._clean_filename(topic)
                script_files = [f for f in script_files if topic_clean in f.name]

            if not script_files:
                raise FileNotFoundError("Enhanced script dosyası bulunamadı")

            # En son dosyayı al
            latest_file = max(script_files, key=lambda x: x.stat().st_mtime)

            with open(latest_file, 'r', encoding='utf-8') as f:
                script_data = json.load(f)

            print(f"📝 En son enhanced script alındı: {latest_file.name}")
            return script_data

        except Exception as e:
            print(f"❌ Enhanced script okuma hatası: {e}")
            raise

    # ==================== AUDIO MANAGEMENT ====================

    def get_latest_audio(self, topic: str = None) -> Path:
        """En son audio dosyasını al"""

        try:
            audio_files = list(config.AUDIO_DATA_DIR.glob("processed_*.mp3"))

            if topic:
                topic_clean = self._clean_filename(topic)
                audio_files = [f for f in audio_files if topic_clean in f.name]

            if not audio_files:
                raise FileNotFoundError("Audio dosyası bulunamadı")

            # En son dosyayı al
            latest_file = max(audio_files, key=lambda x: x.stat().st_mtime)

            print(f"🎵 En son audio alındı: {latest_file.name}")
            return latest_file

        except Exception as e:
            print(f"❌ Audio dosyası okuma hatası: {e}")
            raise

    def get_audio_metadata(self, audio_path: Path) -> Dict[str, Any]:
        """Audio metadata'sını al"""

        metadata_path = audio_path.parent / f"{audio_path.stem}_metadata.json"

        try:
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"📊 Audio metadata alındı: {metadata_path.name}")
                return metadata
            else:
                print(f"⚠️ Audio metadata bulunamadı: {metadata_path}")
                return {}

        except Exception as e:
            print(f"❌ Audio metadata okuma hatası: {e}")
            return {}

    # ==================== MEDIA MANAGEMENT ====================

    def get_latest_media(self, topic: str = None) -> Dict[str, Any]:
        """En son media dosyalarını al"""

        try:
            media_files = {
                'images': list(config.MEDIA_DATA_DIR.glob("image_*.jpg")) +
                          list(config.MEDIA_DATA_DIR.glob("image_*.png")),
                'teaser_video': list(config.MEDIA_DATA_DIR.glob("teaser_*.mp4"))
            }

            if topic:
                topic_clean = self._clean_filename(topic)
                media_files['images'] = [f for f in media_files['images'] if topic_clean in f.name]
                media_files['teaser_video'] = [f for f in media_files['teaser_video'] if topic_clean in f.name]

            # En son dosyaları al
            if media_files['images']:
                media_files['images'] = sorted(media_files['images'], key=lambda x: x.stat().st_mtime)[
                                        -10:]  # Son 10 resim

            if media_files['teaser_video']:
                media_files['teaser_video'] = max(media_files['teaser_video'], key=lambda x: x.stat().st_mtime)
            else:
                media_files['teaser_video'] = None

            print(f"🎬 Media dosyaları alındı: {len(media_files['images'])} resim")
            return media_files

        except Exception as e:
            print(f"❌ Media dosyaları okuma hatası: {e}")
            return {'images': [], 'teaser_video': None}

    # ==================== FINAL VIDEO MANAGEMENT ====================

    def get_latest_final_video(self, topic: str = None) -> Path:
        """En son final video dosyasını al"""

        try:
            video_files = list(config.FINAL_DATA_DIR.glob("final_*.mp4"))

            if topic:
                topic_clean = self._clean_filename(topic)
                video_files = [f for f in video_files if topic_clean in f.name]

            if not video_files:
                raise FileNotFoundError("Final video dosyası bulunamadı")

            # En son dosyayı al
            latest_file = max(video_files, key=lambda x: x.stat().st_mtime)

            print(f"🎥 En son final video alındı: {latest_file.name}")
            return latest_file

        except Exception as e:
            print(f"❌ Final video okuma hatası: {e}")
            raise

    # ==================== UTILITY METHODS ====================

    def _clean_filename(self, text: str) -> str:
        """Dosya adı için güvenli text oluştur"""

        # Özel karakterleri kaldır
        cleaned = re.sub(r'[^\w\s-]', '', text).strip()
        # Boşlukları alt çizgi yap
        cleaned = re.sub(r'[-\s]+', '_', cleaned)
        # Küçük harfe çevir
        cleaned = cleaned.lower()

        return cleaned

    def get_storage_stats(self) -> Dict[str, Any]:
        """Storage istatistiklerini al"""

        stats = {
            'directories': {},
            'total_files': 0,
            'total_size_mb': 0
        }

        directories = {
            'raw': config.RAW_DATA_DIR,
            'processed': config.PROCESSED_DATA_DIR,
            'audio': config.AUDIO_DATA_DIR,
            'media': config.MEDIA_DATA_DIR,
            'final': config.FINAL_DATA_DIR,
            'logs': config.LOGS_DIR
        }

        try:
            for name, directory in directories.items():
                if directory.exists():
                    files = list(directory.rglob('*'))
                    files = [f for f in files if f.is_file()]

                    total_size = sum(f.stat().st_size for f in files)

                    stats['directories'][name] = {
                        'file_count': len(files),
                        'size_mb': total_size / (1024 * 1024)
                    }

                    stats['total_files'] += len(files)
                    stats['total_size_mb'] += total_size / (1024 * 1024)
                else:
                    stats['directories'][name] = {
                        'file_count': 0,
                        'size_mb': 0
                    }

            print(f"📊 Storage Stats: {stats['total_files']} dosya, {stats['total_size_mb']:.1f} MB")
            return stats

        except Exception as e:
            print(f"❌ Storage stats hatası: {e}")
            return stats

    def cleanup_old_files(self, days_old: int = 30) -> int:
        """Eski dosyaları temizle"""

        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        deleted_count = 0

        try:
            directories = [
                config.RAW_DATA_DIR,
                config.PROCESSED_DATA_DIR,
                config.AUDIO_DATA_DIR,
                config.MEDIA_DATA_DIR,
                config.LOGS_DIR
            ]

            for directory in directories:
                if directory.exists():
                    for file_path in directory.rglob('*'):
                        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                            file_path.unlink()
                            deleted_count += 1

            print(f"🧹 {deleted_count} eski dosya temizlendi ({days_old} günden eski)")
            return deleted_count

        except Exception as e:
            print(f"❌ Cleanup hatası: {e}")
            return 0

    def backup_data(self, backup_dir: Path) -> bool:
        """Data backup oluştur"""

        try:
            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"sleepy_dull_stories_backup_{timestamp}"

            # Data dizinini kopyala
            shutil.copytree(config.DATA_DIR, backup_dir / backup_name)

            print(f"💾 Backup oluşturuldu: {backup_dir / backup_name}")
            return True

        except Exception as e:
            print(f"❌ Backup hatası: {e}")
            return False

    def create_workflow_summary(self, date_str: str = None) -> Dict[str, Any]:
        """Günlük workflow özetini oluştur"""

        if not date_str:
            date_str = datetime.now().strftime('%Y%m%d')

        summary = {
            'date': date_str,
            'topic': None,
            'story_generated': False,
            'script_enhanced': False,
            'audio_generated': False,
            'media_generated': False,
            'final_video_created': False,
            'files': {}
        }

        try:
            # Bugünün konusunu al
            today_date = datetime.strptime(date_str, '%Y%m%d').date()
            topic = self.get_today_topic(today_date)
            summary['topic'] = topic

            topic_clean = self._clean_filename(topic)

            # Dosyaları kontrol et
            story_files = list(config.RAW_DATA_DIR.glob(f"story_{topic_clean}_{date_str}*.json"))
            summary['story_generated'] = len(story_files) > 0
            if story_files:
                summary['files']['story'] = str(story_files[-1])

            enhanced_files = list(config.PROCESSED_DATA_DIR.glob(f"enhanced_{topic_clean}_{date_str}*.json"))
            summary['script_enhanced'] = len(enhanced_files) > 0
            if enhanced_files:
                summary['files']['enhanced_script'] = str(enhanced_files[-1])

            audio_files = list(config.AUDIO_DATA_DIR.glob(f"*{topic_clean}_{date_str}*.mp3"))
            summary['audio_generated'] = len(audio_files) > 0
            if audio_files:
                summary['files']['audio'] = str(audio_files[-1])

            final_files = list(config.FINAL_DATA_DIR.glob(f"final_{topic_clean}_{date_str}*.mp4"))
            summary['final_video_created'] = len(final_files) > 0
            if final_files:
                summary['files']['final_video'] = str(final_files[-1])

            print(f"📋 Workflow summary oluşturuldu: {date_str}")
            return summary

        except Exception as e:
            print(f"❌ Workflow summary hatası: {e}")
            return summary


# Test fonksiyonu
def test_storage_manager():
    """Storage Manager test fonksiyonu"""

    try:
        print("📁 Storage Manager Test Başlatılıyor...")
        print("=" * 50)

        # Storage manager oluştur
        storage = StorageManager()

        # Bugünün konusunu al
        today_topic = storage.get_today_topic()
        print(f"📅 Bugünün konusu: {today_topic}")

        # Konu detaylarını al
        topic_details = storage.get_topic_details(today_topic)
        print(f"📖 Konu kategorisi: {topic_details.get('category', 'Unknown')}")

        # Gelecek konular
        upcoming = storage.get_upcoming_topics(3)
        print(f"📊 Gelecek 3 konu: {len(upcoming)} adet")

        # Storage stats
        stats = storage.get_storage_stats()
        print(f"💾 Toplam dosya: {stats['total_files']}")
        print(f"📊 Toplam boyut: {stats['total_size_mb']:.1f} MB")

        # Test data kaydet
        test_story = {
            'topic': 'Test Story',
            'story': 'This is a test story for storage manager.',
            'word_count': 10,
            'generated_at': datetime.now().isoformat()
        }

        saved_path = storage.save_story_data(test_story)
        print(f"💾 Test story kaydedildi: {saved_path.name}")

        # Test data oku
        retrieved_story = storage.get_latest_story('Test Story')
        print(f"📖 Test story okundu: {retrieved_story['topic']}")

        # Workflow summary
        summary = storage.create_workflow_summary()
        print(f"📋 Workflow summary: {summary['topic']}")

        print("=" * 50)
        print("✅ Storage Manager Test Başarılı!")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"❌ Storage test hatası: {e}")
        return False


if __name__ == "__main__":
    test_storage_manager()