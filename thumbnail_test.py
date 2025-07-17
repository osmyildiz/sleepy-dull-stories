import requests
import time
import os
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment
load_dotenv()


class ImprovedPiapiThumbnail:
    def __init__(self):
        self.api_key = os.getenv('PIAPI_KEY')
        self.base_url = "https://api.piapi.ai/api/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # GÜÇLÜ SİNEMATİK TEMPLATE'LER - PARAMETRELER ÇIKARILDI
        self.templates = {
            # Pompeii - Daha kısa ve güçlü
            0: "Mount Vesuvius erupting with volcanic lightning, terrified Roman screaming with ash-covered face, burning Pompeii below, dramatic volcanic glow, disaster movie poster",

            # Alternatif Pompeii - Daha duygusal
            1: "Horrified Roman watching Vesuvius explode, tears on ash-covered face, massive eruption behind, burning ancient city, dramatic fire lighting",

            # Library of Alexandria - Odaklı
            2: "Library of Alexandria burning with massive flames, desperate scholar holding scrolls crying, flying burning manuscripts, dramatic fire lighting",

            # Constantinople - Güçlü
            3: "Ottoman cannons firing at Constantinople walls, terrified Byzantine defender screaming, exploding architecture, dramatic cannon fire glow",

            # Titanic - Duygusal
            4: "Titanic sinking into dark Atlantic, terrified passenger in 1912 clothing screaming, massive ship tilting, icy waves, moonlight",

            # Hindenburg - Şok edici
            5: "Hindenburg exploding in massive fireball, horrified witness with shocked expression, huge flames against dark sky, 1937 disaster",

            # Atlantis - Mistik
            6: "Atlantis being swallowed by tsunami waves, desperate Atlantean reaching upward, golden underwater city sinking, dramatic lighting",

            # Black Death - Korkunç
            7: "Medieval plague doctor with black beak mask, burning medieval city behind, plague victims in streets, dark stormy sky",

            # Chernobyl - Modern korku
            8: "Chernobyl nuclear plant exploding with radioactive glow, Soviet engineer with horrified expression, massive nuclear explosion, 1986",

            # Great Chicago Fire - Klasik
            9: "Great Chicago Fire with massive flames consuming city, terrified Victorian woman screaming, burning wooden buildings, 1871 disaster"
        }

        # KISA SİNEMATİK BOOSTER'LAR
        self.cinematic_boosters = [
            "epic movie poster",
            "blockbuster film style",
            "award winning photo",
            "IMAX quality",
            "Hollywood style"
        ]

        # KISA LIGHTING MODIFIERS
        self.lighting_enhancers = [
            "dramatic lighting",
            "cinematic glow",
            "moody atmosphere",
            "epic backlighting",
            "intense contrast"
        ]

    def enhance_prompt(self, base_prompt, topic_id):
        """Prompt'u DOĞRU SIRAYLA güçlendirici elementlerle zenginleştir"""
        import random

        # Random booster ve lighting seç
        booster = random.choice(self.cinematic_boosters)
        lighting = random.choice(self.lighting_enhancers)

        # DOĞRU SIRA: base_prompt + qualifiers + parameters
        enhanced = f"{base_prompt}, {booster}, {lighting}, photorealistic, professional photography, hyper detailed --ar 16:9 --v 7.0"

        print(f"🎬 Enhanced prompt for topic {topic_id}:")
        print(f"📝 {enhanced}")
        print(f"📊 Length: {len(enhanced)} characters")

        # 280 karakterden uzunsa kısalt
        if len(enhanced) > 280:
            enhanced = f"{base_prompt}, {booster}, photorealistic --ar 16:9 --v 7.0"
            print(f"⚠️ Shortened to: {len(enhanced)} characters")

        return enhanced

    def submit_task_with_retry(self, prompt, max_retries=3):
        """PIAPI'ye retry logic ile task gönder"""

        for attempt in range(max_retries):
            print(f"🚀 Submitting task to PIAPI (attempt {attempt + 1}/{max_retries})...")

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
                response = requests.post(
                    f"{self.base_url}/task",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("code") == 200:
                        task_id = result["data"]["task_id"]
                        print(f"✅ Task submitted: {task_id}")
                        return task_id
                    else:
                        print(f"❌ API Error: {result.get('message')}")

                elif response.status_code == 500:
                    # Rate limiting - exponential backoff
                    wait_time = (attempt + 1) * 30  # 30, 60, 90 seconds
                    print(f"⚠️ HTTP 500 (Rate limit) - Waiting {wait_time}s before retry...")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ Max retries reached for HTTP 500")
                        return None

                else:
                    print(f"❌ HTTP Error: {response.status_code}")
                    if attempt < max_retries - 1:
                        print(f"⚠️ Retrying in 15 seconds...")
                        time.sleep(15)
                        continue

            except Exception as e:
                print(f"❌ Request failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⚠️ Retrying in 10 seconds...")
                    time.sleep(10)
                    continue

        print(f"❌ All {max_retries} attempts failed")
        return None
        """PIAPI'ye task gönder"""
        print(f"🚀 Submitting task to PIAPI...")

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
            response = requests.post(
                f"{self.base_url}/task",
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    task_id = result["data"]["task_id"]
                    print(f"✅ Task submitted: {task_id}")
                    return task_id
                else:
                    print(f"❌ API Error: {result.get('message')}")
                    return None
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None

    def check_status(self, task_id):
        """Task durumunu kontrol et"""
        try:
            response = requests.get(
                f"{self.base_url}/task/{task_id}",
                headers=self.headers,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    task_data = result["data"]
                    status = task_data.get("status", "").lower()

                    if status == "completed":
                        output = task_data.get("output", {})
                        temp_urls = output.get("temporary_image_urls", [])
                        image_url = output.get("image_url", "")

                        if temp_urls and len(temp_urls) >= 4:
                            return {"status": "completed", "urls": temp_urls[:4]}
                        elif image_url:
                            return {"status": "completed", "urls": [image_url]}
                        else:
                            return {"status": "completed", "urls": []}

                    elif status == "failed":
                        return {"status": "failed"}
                    else:
                        return {"status": "processing"}

            return {"status": "error"}

        except Exception as e:
            print(f"❌ Status check error: {e}")
            return {"status": "error"}

    def download_image(self, url, filename):
        """Resmi indir"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Referer': 'https://discord.com/'
            }

            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded: {filename}")
                return True
            else:
                print(f"❌ Download failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Download error: {e}")
            return False

    def generate_multiple_versions(self, topic_id, versions=3, output_dir="improved_thumbnails"):
        """Aynı konudan birden fazla versiyon oluştur"""

        Path(output_dir).mkdir(exist_ok=True)

        if topic_id not in self.templates:
            print(f"❌ Topic {topic_id} template not found!")
            return False

        base_prompt = self.templates[topic_id]
        print(f"🎨 Generating {versions} versions for topic {topic_id}")

        successful_downloads = 0

        for version in range(versions):
            print(f"\n🔄 VERSION {version + 1}/{versions}")
            print("=" * 40)

            # Enhanced prompt oluştur
            enhanced_prompt = self.enhance_prompt(base_prompt, topic_id)

            # Task gönder (retry logic ile)
            task_id = self.submit_task_with_retry(enhanced_prompt, max_retries=3)
            if not task_id:
                print(f"❌ Version {version + 1} submission failed")
                continue

            # Status takibi - daha az sık kontrol
            print(f"⏳ Monitoring version {version + 1}: {task_id}")
            max_cycles = 30  # 15 dakika (30 * 30 saniye)

            for cycle in range(max_cycles):
                result = self.check_status(task_id)

                if result["status"] == "completed":
                    urls = result["urls"]
                    print(f"✅ Version {version + 1} completed! Found {len(urls)} images")

                    # İlk resmi indir (en iyi kalite genelde ilk)
                    if urls:
                        url = urls[0]  # İlk resim genelde en kaliteli
                        filename = f"{output_dir}/topic_{topic_id}_version_{version + 1}.png"
                        if self.download_image(url, filename):
                            successful_downloads += 1
                    break

                elif result["status"] == "failed":
                    print(f"❌ Version {version + 1} failed!")
                    break

                else:
                    if cycle % 3 == 0:  # Her 90 saniyede bir rapor
                        print(f"⏳ Version {version + 1} processing... ({cycle + 1}/{max_cycles})")
                    time.sleep(30)

            # Versiyonlar arası bekleme - HTTP 500'den kaçınmak için artırıldı
            if version < versions - 1:
                print(f"⏳ Waiting 90 seconds before next version (rate limiting)...")
                time.sleep(90)

        print(f"\n🎉 Generated {successful_downloads}/{versions} successful versions!")
        return successful_downloads > 0


def test_improved_thumbnails():
    """İyileştirilmiş thumbnail testleri - TÜM KONULAR SIRYLA"""

    generator = ImprovedPiapiThumbnail()

    if not generator.api_key:
        print("❌ PIAPI_KEY environment variable not found!")
        return

    print("🚀 IMPROVED CINEMATIC THUMBNAIL GENERATOR")
    print("🎬 Using shorter, more powerful prompts")
    print("🔄 Processing ALL topics in sequence")
    print("=" * 60)

    # TÜM KONULARI SIRYLA İŞLE (0-9)
    all_topics = list(generator.templates.keys())  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    total_topics = len(all_topics)

    print(f"📋 Found {total_topics} topics to process:")
    topic_names = {
        0: "Pompeii's Final Night",
        1: "Alternative Pompeii",
        2: "Library of Alexandria",
        3: "Fall of Constantinople",
        4: "Titanic Disaster",
        5: "Hindenburg Explosion",
        6: "Atlantis Destruction",
        7: "Black Death Plague",
        8: "Chernobyl Meltdown",
        9: "Great Chicago Fire"
    }

    for topic_id in all_topics:
        topic_name = topic_names.get(topic_id, f"Topic {topic_id}")
        print(f"   🎭 Topic {topic_id}: {topic_name}")

    print("=" * 60)

    successful_topics = 0
    failed_topics = 0

    for i, topic_id in enumerate(all_topics):
        topic_name = topic_names.get(topic_id, f"Topic {topic_id}")

        print(f"\n🎬 PROCESSING TOPIC {topic_id}/{total_topics}: {topic_name}")
        print(f"📊 Progress: {i + 1}/{total_topics} ({((i + 1) / total_topics) * 100:.1f}%)")
        print("=" * 50)

        success = generator.generate_multiple_versions(
            topic_id=topic_id,
            versions=2,  # Her konu için 2 versiyon
            output_dir=f"improved_thumbnails"
        )

        if success:
            print(f"✅ Topic {topic_id} ({topic_name}) successful!")
            successful_topics += 1
        else:
            print(f"❌ Topic {topic_id} ({topic_name}) failed!")
            failed_topics += 1

        # Konular arası bekleme (son konu değilse) - HTTP 500'den kaçınmak için artırıldı
        if i < len(all_topics) - 1:
            print(f"⏳ Waiting 120 seconds before next topic (rate limiting)...")
            print(f"📊 Remaining: {len(all_topics) - i - 1} topics")
            time.sleep(120)

    # Final results
    print(f"\n🎉 ALL TOPICS PROCESSING COMPLETED!")
    print("=" * 60)
    print(f"✅ Successful: {successful_topics}/{total_topics}")
    print(f"❌ Failed: {failed_topics}/{total_topics}")
    print(f"📊 Success rate: {(successful_topics / total_topics) * 100:.1f}%")
    print(f"📁 Check 'improved_thumbnails' folder for all results")
    print(f"🎬 Total images generated: ~{successful_topics * 2} (2 per successful topic)")

    if successful_topics > 0:
        print(f"\n🎯 SUCCESSFUL TOPICS:")
        for topic_id in range(total_topics):
            if topic_id in all_topics:
                topic_name = topic_names.get(topic_id, f"Topic {topic_id}")
                # Check if files exist
                version1 = Path(f"improved_thumbnails/topic_{topic_id}_version_1.png")
                version2 = Path(f"improved_thumbnails/topic_{topic_id}_version_2.png")
                if version1.exists() or version2.exists():
                    files_found = []
                    if version1.exists(): files_found.append("v1")
                    if version2.exists(): files_found.append("v2")
                    print(f"   ✅ Topic {topic_id}: {topic_name} [{', '.join(files_found)}]")


def test_single_topic():
    """Tek bir konuyu test etmek için"""

    generator = ImprovedPiapiThumbnail()

    if not generator.api_key:
        print("❌ PIAPI_KEY environment variable not found!")
        return

    print("🎯 SINGLE TOPIC TEST")
    print("=" * 40)

    # Kullanıcıdan topic seçimi al
    topic_names = {
        0: "Pompeii's Final Night",
        1: "Alternative Pompeii",
        2: "Library of Alexandria",
        3: "Fall of Constantinople",
        4: "Titanic Disaster",
        5: "Hindenburg Explosion",
        6: "Atlantis Destruction",
        7: "Black Death Plague",
        8: "Chernobyl Meltdown",
        9: "Great Chicago Fire"
    }

    print("Available topics:")
    for topic_id, name in topic_names.items():
        print(f"   {topic_id}: {name}")

    try:
        selected = int(input("\nEnter topic ID (0-9): "))
        if selected not in topic_names:
            print("❌ Invalid topic ID!")
            return

        topic_name = topic_names[selected]
        print(f"\n🎬 Testing Topic {selected}: {topic_name}")

        success = generator.generate_multiple_versions(
            topic_id=selected,
            versions=2,  # Tek konu için 2 versiyon (3'ten azaltıldı - rate limiting için)
            output_dir="single_topic_test"
        )

        if success:
            print(f"✅ {topic_name} test successful!")
        else:
            print(f"❌ {topic_name} test failed!")

    except ValueError:
        print("❌ Please enter a valid number!")
    except KeyboardInterrupt:
        print("\n⏹️ Test cancelled by user")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # python thumbnail_improved.py single
        test_single_topic()
    else:
        # python thumbnail_improved.py (default - all topics)
        test_improved_thumbnails()
EOF

echo
"🎬 Improved cinematic thumbnail generator created!"
echo
""
echo
"🚀 To test improved prompts:"
echo
"   python thumbnail_improved.py"
echo
""
echo
"⚠️ IMPROVED with HTTP 500 fixes:"
echo
"   - Shorter prompts (150-200 chars max)"
echo
"   - Retry logic for HTTP 500 errors"
echo
"   - Longer wait times (90s between versions, 120s between topics)"
echo
"   - Less frequent status checks"
echo
""
echo
"⏱️ Estimated time: 3-4 hours for all 10 topics"
EOF
"""
Database Check Script - Check current database status and structure
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime


def find_database():
    """Find the database file"""
    possible_paths = [
        "production.db",
        "data/production.db",
        "../data/production.db",
        "../../data/production.db",
        "../../../data/production.db"
    ]

    for path in possible_paths:
        if Path(path).exists():
            return Path(path)

    return None


def check_database_structure():
    """Check database structure and current status"""

    db_path = find_database()

    if not db_path:
        print("❌ Database not found!")
        print("Checked paths:")
        possible_paths = [
            "production.db",
            "data/production.db",
            "../data/production.db",
            "../../data/production.db",
            "../../../data/production.db"
        ]
        for path in possible_paths:
            print(f"   ❌ {path}")
        return False

    print(f"✅ Database found: {db_path}")
    print(f"📁 File size: {db_path.stat().st_size / 1024:.1f} KB")
    print(f"📅 Modified: {datetime.fromtimestamp(db_path.stat().st_mtime)}")

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check tables
        print(f"\n📋 DATABASE TABLES:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table in tables:
            print(f"   📄 {table[0]}")

        # Focus on topics table
        if ('topics',) in tables:
            print(f"\n📊 TOPICS TABLE STRUCTURE:")
            cursor.execute("PRAGMA table_info(topics);")
            columns = cursor.fetchall()

            for col in columns:
                col_name = col[1]
                col_type = col[2]
                not_null = "NOT NULL" if col[3] else "NULL OK"
                default_val = f"DEFAULT: {col[4]}" if col[4] else "NO DEFAULT"
                primary_key = "PRIMARY KEY" if col[5] else ""

                print(f"   📝 {col_name:<30} {col_type:<10} {not_null:<10} {default_val:<15} {primary_key}")

            # Check record counts
            print(f"\n📈 RECORD COUNTS:")
            cursor.execute("SELECT COUNT(*) FROM topics;")
            total = cursor.fetchone()[0]
            print(f"   📊 Total topics: {total}")

            # Check status distribution
            print(f"\n📊 STATUS DISTRIBUTION:")
            try:
                cursor.execute("SELECT status, COUNT(*) FROM topics GROUP BY status ORDER BY COUNT(*) DESC;")
                status_dist = cursor.fetchall()
                for status, count in status_dist:
                    percentage = (count / total * 100) if total > 0 else 0
                    print(f"   🔹 {status or 'NULL':<20} {count:>3} ({percentage:>5.1f}%)")
            except Exception as e:
                print(f"   ⚠️ Status check failed: {e}")

            # Check completion fields
            completion_fields = [
                'production_started_at',
                'production_completed_at',
                'scene_generation_status',
                'scene_generation_started_at',
                'scene_generation_completed_at',
                'character_generation_status',
                'character_generation_started_at',
                'character_generation_completed_at',
                'api_calls_used',
                'total_cost',
                'scenes_generated',
                'thumbnail_generated'
            ]

            print(f"\n🔍 COMPLETION FIELDS STATUS:")
            for field in completion_fields:
                try:
                    # Check if field exists and has non-null values
                    cursor.execute(f"SELECT COUNT(*) FROM topics WHERE {field} IS NOT NULL;")
                    non_null_count = cursor.fetchone()[0]

                    cursor.execute(f"SELECT COUNT(*) FROM topics WHERE {field} IS NULL;")
                    null_count = cursor.fetchone()[0]

                    if non_null_count > 0:
                        # Get some sample values
                        cursor.execute(f"SELECT {field} FROM topics WHERE {field} IS NOT NULL LIMIT 3;")
                        samples = [str(row[0]) for row in cursor.fetchall()]
                        sample_text = ", ".join(samples[:3])
                        print(
                            f"   ✅ {field:<35} NULL: {null_count:>3} | Non-NULL: {non_null_count:>3} | Samples: {sample_text[:50]}")
                    else:
                        print(f"   ⚪ {field:<35} NULL: {null_count:>3} | Non-NULL: {non_null_count:>3} | All empty")

                except sqlite3.OperationalError:
                    print(f"   ❌ {field:<35} Column doesn't exist")
                except Exception as e:
                    print(f"   ⚠️ {field:<35} Error: {e}")

            # Show some sample records
            print(f"\n📝 SAMPLE RECORDS:")
            try:
                cursor.execute("""
                    SELECT id, topic, status, 
                           production_started_at, scene_generation_status,
                           api_calls_used, total_cost
                    FROM topics 
                    LIMIT 5
                """)

                records = cursor.fetchall()
                for record in records:
                    topic_id, topic, status, prod_start, scene_status, api_calls, cost = record
                    print(
                        f"   📌 ID {topic_id}: {topic[:40]:<40} | Status: {status or 'NULL':<12} | Scene: {scene_status or 'NULL':<12} | API: {api_calls or 0:>3} | Cost: ${cost or 0:.2f}")

            except Exception as e:
                print(f"   ⚠️ Sample records failed: {e}")

        else:
            print(f"❌ Topics table not found!")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False


def check_ready_for_generation():
    """Check if database is ready for story generation"""

    db_path = find_database()
    if not db_path:
        return False

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        print(f"\n🚀 GENERATION READINESS CHECK:")

        # Check for pending topics
        cursor.execute("SELECT COUNT(*) FROM topics WHERE status = 'pending';")
        pending_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM topics WHERE status = 'completed';")
        completed_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM topics WHERE status = 'in_progress';")
        in_progress_count = cursor.fetchone()[0]

        print(f"   📋 Pending topics: {pending_count}")
        print(f"   ✅ Completed topics: {completed_count}")
        print(f"   🔄 In progress: {in_progress_count}")

        if pending_count > 0:
            print(f"   ✅ Ready for story generation!")

            # Show next topic
            cursor.execute("""
                SELECT id, topic, description 
                FROM topics 
                WHERE status = 'pending' 
                ORDER BY priority ASC, created_at ASC 
                LIMIT 1
            """)
            next_topic = cursor.fetchone()
            if next_topic:
                print(f"   🎯 Next topic: ID {next_topic[0]} - {next_topic[1]}")
        else:
            print(f"   ⚠️ No pending topics found!")

        # Check for topics ready for scene generation
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM topics 
                WHERE status = 'completed' 
                AND (scene_generation_status IS NULL OR scene_generation_status = 'pending')
            """)
            ready_for_scenes = cursor.fetchone()[0]
            print(f"   🎬 Ready for scene generation: {ready_for_scenes}")
        except:
            print(f"   ⚠️ Scene generation status check failed")

        conn.close()

    except Exception as e:
        print(f"❌ Readiness check failed: {e}")


if __name__ == "__main__":
    print("🔍" * 60)
    print("DATABASE STATUS CHECK")
    print("🔍" * 60)

    success = check_database_structure()

    if success:
        check_ready_for_generation()

        print(f"\n💡 NEXT STEPS:")
        print(f"   🔄 To reset database: python reset_database.py")
        print(f"   🚀 To generate stories: python 1_story_generator_claude_server.py")
        print(f"   🎬 To generate scenes: python 3_visual_generator_midjourney_scene_server.py")
    else:
        print(f"\n❌ Database check failed!")

    print("🔍" * 60)
