"""
Disaster-Focused Topic Import System
Import existing CSV topics + thumbnail features to professional database
Channel Concept: "Final Days Before Disasters"
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


class DisasterTopicImporter:
    """Import disaster-focused topics from CSV + thumbnail features"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.db_path = project_root / "data" / "production.db"
        else:
            self.db_path = Path(db_path)

        self.setup_disaster_tables()
        print("✅ Disaster Topic Importer initialized")

    def setup_disaster_tables(self):
        """Create disaster-focused topic management tables"""

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute('PRAGMA journal_mode=WAL;')

            # Enhanced Topic Queue for Disaster Concept
            conn.execute('''
                CREATE TABLE IF NOT EXISTS disaster_topic_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    description TEXT NOT NULL,
                    disaster_type TEXT NOT NULL,
                    historical_period TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    estimated_duration INTEGER DEFAULT 120,
                    clickbait_title TEXT NOT NULL,
                    font_design TEXT NOT NULL,
                    scheduled_date TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    status TEXT DEFAULT 'pending',
                    priority TEXT DEFAULT 'medium',
                    estimated_performance INTEGER DEFAULT 5,
                    target_demographics TEXT DEFAULT '[]',

                    -- CSV Import Fields
                    csv_id INTEGER,
                    audio_generated INTEGER DEFAULT 0,
                    cover_image_created INTEGER DEFAULT 0,
                    teaser_video_created INTEGER DEFAULT 0,
                    images_generated INTEGER DEFAULT 0,
                    background_music_added INTEGER DEFAULT 0,
                    editing_completed INTEGER DEFAULT 0,
                    published INTEGER DEFAULT 0,
                    thumbnail_created INTEGER DEFAULT 0,

                    -- Production tracking
                    produced_at TEXT,
                    video_id INTEGER,
                    performance_actual INTEGER
                )
            ''')

            # Thumbnail Template System
            conn.execute('''
                CREATE TABLE IF NOT EXISTS thumbnail_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id INTEGER,
                    shocking_word TEXT NOT NULL,
                    shocking_word_color TEXT NOT NULL,
                    main_title_lines TEXT NOT NULL,
                    bottom_text_lines TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (topic_id) REFERENCES disaster_topic_queue (id)
                )
            ''')

            # Disaster Categories
            conn.execute('''
                CREATE TABLE IF NOT EXISTS disaster_categories (
                    category TEXT PRIMARY KEY,
                    description TEXT,
                    example_disasters TEXT,
                    emotional_impact TEXT,
                    target_frequency INTEGER DEFAULT 7,
                    last_used TEXT,
                    usage_count INTEGER DEFAULT 0
                )
            ''')

            conn.commit()
            print("✅ Disaster topic tables created/verified")

    def import_csv_topics(self, csv_path: str = "topics.csv") -> List[Dict]:
        """Import topics from existing CSV"""

        print(f"📊 Importing disaster topics from {csv_path}...")

        try:
            # Read CSV
            if not Path(csv_path).exists():
                csv_path = f"data/{csv_path}"

            df = pd.read_csv(csv_path)
            print(f"✅ Found {len(df)} topics in CSV")

            imported_topics = []

            with sqlite3.connect(self.db_path, timeout=30) as conn:
                for index, row in df.iterrows():
                    # Determine disaster type and period
                    disaster_type = self._classify_disaster_type(row['topic'])

                    # Generate clickbait title if not present
                    clickbait_title = self._generate_disaster_clickbait(row['topic'])

                    # Generate font design
                    font_design = self._generate_disaster_font_design(disaster_type)

                    # Schedule topics starting from today
                    scheduled_date = datetime.now() + timedelta(days=index)

                    # Insert into database
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO disaster_topic_queue 
                        (topic, description, disaster_type, historical_period, keywords, 
                         estimated_duration, clickbait_title, font_design, scheduled_date,
                         csv_id, audio_generated, cover_image_created, teaser_video_created,
                         images_generated, background_music_added, editing_completed, 
                         published, thumbnail_created)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['topic'],
                        row['description'],
                        disaster_type,
                        row['historical_period'],
                        row['keywords'],
                        row['estimated_duration'],
                        clickbait_title,
                        font_design,
                        scheduled_date.isoformat(),
                        row.get('Unnamed: 0', index),
                        row.get('audio_generated', 0),
                        row.get('cover_image_created', 0),
                        row.get('teaser_video_created', 0),
                        row.get('images_generated', 0),
                        row.get('background_music_added', 0),
                        row.get('editing_completed', 0),
                        row.get('published', 0),
                        row.get('thumbnail', 0)
                    ))

                    topic_id = cursor.lastrowid

                    imported_topics.append({
                        'id': topic_id,
                        'topic': row['topic'],
                        'disaster_type': disaster_type,
                        'historical_period': row['historical_period']
                    })

                conn.commit()

            print(f"✅ Imported {len(imported_topics)} disaster topics")
            return imported_topics

        except Exception as e:
            print(f"❌ CSV import error: {e}")
            return []

    def import_thumbnail_features(self, json_path: str = "thumbnail_features.json") -> bool:
        """Import thumbnail templates from JSON"""

        print(f"🖼️ Importing thumbnail features from {json_path}...")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                thumbnail_data = json.load(f)

            topics = thumbnail_data.get('topics', [])
            print(f"✅ Found {len(topics)} thumbnail templates")

            with sqlite3.connect(self.db_path, timeout=30) as conn:
                for topic_template in topics:
                    # Find matching topic in database
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT id FROM disaster_topic_queue 
                        WHERE topic LIKE ? OR topic LIKE ?
                    ''', (f"%{topic_template['topic']}%", f"%{topic_template['topic'].split()[-1]}%"))

                    result = cursor.fetchone()
                    if result:
                        topic_id = result[0]

                        # Insert thumbnail template
                        sections = topic_template['sections']
                        cursor.execute('''
                            INSERT INTO thumbnail_templates 
                            (topic_id, shocking_word, shocking_word_color, 
                             main_title_lines, bottom_text_lines)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            topic_id,
                            sections['shocking_word']['text'],
                            sections['shocking_word']['color'],
                            json.dumps(sections['main_title']['lines']),
                            json.dumps(sections['bottom_text']['lines'])
                        ))

                        print(f"✅ Thumbnail template for: {topic_template['topic']}")
                    else:
                        print(f"⚠️ No matching topic for thumbnail: {topic_template['topic']}")

                conn.commit()

            print(f"✅ Thumbnail templates imported successfully")
            return True

        except Exception as e:
            print(f"❌ Thumbnail import error: {e}")
            return False

    def setup_disaster_categories(self):
        """Setup disaster category system"""

        disaster_categories = [
            {
                'category': 'natural_disasters',
                'description': 'Volcanic eruptions, earthquakes, floods, storms',
                'example_disasters': 'Pompeii, Lisbon Earthquake, Great Flood',
                'emotional_impact': 'Tragic inevitability, human vs nature',
                'target_frequency': 5
            },
            {
                'category': 'war_and_siege',
                'description': 'Final days of cities, battles, empires falling',
                'example_disasters': 'Fall of Constantinople, Siege of Troy, Fall of Rome',
                'emotional_impact': 'Heroic last stands, civilization collapse',
                'target_frequency': 6
            },
            {
                'category': 'maritime_disasters',
                'description': 'Ship sinkings, lost expeditions, sea tragedies',
                'example_disasters': 'Titanic, Lusitania, Lost Franklin Expedition',
                'emotional_impact': 'Isolation, hubris, final moments',
                'target_frequency': 4
            },
            {
                'category': 'political_collapse',
                'description': 'Revolutions, regime changes, political assassinations',
                'example_disasters': 'French Revolution, Fall of Czar, Roman Republic End',
                'emotional_impact': 'Power shifts, final luxury before chaos',
                'target_frequency': 3
            },
            {
                'category': 'plague_and_disease',
                'description': 'Pandemic final days, medical disasters',
                'example_disasters': 'Black Death, 1918 Flu, Plague of Justinian',
                'emotional_impact': 'Social breakdown, medical helplessness',
                'target_frequency': 3
            },
            {
                'category': 'fire_disasters',
                'description': 'Great fires, city burning, building destruction',
                'example_disasters': 'Great Fire of London, Chicago Fire, Library of Alexandria',
                'emotional_impact': 'Cultural loss, spreading destruction',
                'target_frequency': 3
            },
            {
                'category': 'technological_disasters',
                'description': 'Industrial accidents, engineering failures',
                'example_disasters': 'Hindenburg, Bridge Collapses, Factory Disasters',
                'emotional_impact': 'Modern hubris, technology failure',
                'target_frequency': 2
            }
        ]

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            for category in disaster_categories:
                conn.execute('''
                    INSERT OR REPLACE INTO disaster_categories 
                    (category, description, example_disasters, emotional_impact, target_frequency)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    category['category'],
                    category['description'],
                    category['example_disasters'],
                    category['emotional_impact'],
                    category['target_frequency']
                ))
            conn.commit()

        print("✅ Disaster categories setup complete")

    def _classify_disaster_type(self, topic: str) -> str:
        """Classify topic into disaster type"""
        topic_lower = topic.lower()

        if any(word in topic_lower for word in ['pompeii', 'volcano', 'earthquake', 'flood', 'storm']):
            return 'natural_disasters'
        elif any(word in topic_lower for word in ['titanic', 'ship', 'ocean', 'sea', 'maritime']):
            return 'maritime_disasters'
        elif any(word in topic_lower for word in ['war', 'siege', 'battle', 'fall', 'empire', 'conquest']):
            return 'war_and_siege'
        elif any(word in topic_lower for word in ['plague', 'disease', 'pandemic', 'illness']):
            return 'plague_and_disease'
        elif any(word in topic_lower for word in ['fire', 'burn', 'flame']):
            return 'fire_disasters'
        elif any(word in topic_lower for word in ['revolution', 'execution', 'political', 'queen', 'king']):
            return 'political_collapse'
        elif any(word in topic_lower for word in ['machine', 'invention', 'technology', 'factory']):
            return 'technological_disasters'
        else:
            return 'natural_disasters'  # Default

    def _generate_disaster_clickbait(self, topic: str) -> str:
        """Generate disaster-focused clickbait title"""
        templates = [
            f"The FINAL HOURS of {topic} - What Really Happened",
            f"The SHOCKING Truth About {topic}'s Last Day",
            f"What {topic} Did in Their FINAL MOMENTS Will STUN You",
            f"The TERRIFYING Last Hours of {topic} - Never Before Told",
            f"EXCLUSIVE: The Final Day of {topic} - Heartbreaking Details",
            f"The UNTOLD Story of {topic}'s Final Hours",
            f"REVEALED: What Really Happened in {topic}'s Last Day",
            f"The DEVASTATING Final Moments of {topic} - 2 Hour Story"
        ]

        import random
        return random.choice(templates)

    def _generate_disaster_font_design(self, disaster_type: str) -> str:
        """Generate font design based on disaster type"""
        font_designs = {
            'natural_disasters': "Bold impact font with orange-red gradient, ash texture, dramatic shadows",
            'maritime_disasters': "Classic serif font with deep blue and white, water texture effects",
            'war_and_siege': "Medieval bold font with dark red and gold, battle-worn texture",
            'political_collapse': "Elegant royal font with purple and gold, crown motifs",
            'plague_and_disease': "Gothic font with dark green and black, aged parchment texture",
            'fire_disasters': "Flame-style font with orange-yellow gradient, fire glow effects",
            'technological_disasters': "Industrial font with metallic silver and red, mechanical texture"
        }

        return font_designs.get(disaster_type, "Bold dramatic font with warm colors and shadow effects")

    def get_next_disaster_topic(self) -> tuple:
        """Get next disaster topic from queue"""

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get next pending topic
            cursor.execute('''
                SELECT * FROM disaster_topic_queue 
                WHERE status = 'pending'
                ORDER BY scheduled_date ASC 
                LIMIT 1
            ''')

            row = cursor.fetchone()

            if not row:
                print("⚠️ No disaster topics in queue!")
                return None

            # Mark as in production
            cursor.execute('''
                UPDATE disaster_topic_queue 
                SET status = 'in_production', produced_at = datetime('now')
                WHERE id = ?
            ''', (row['id'],))

            # Get thumbnail template if available
            cursor.execute('''
                SELECT * FROM thumbnail_templates 
                WHERE topic_id = ?
            ''', (row['id'],))

            thumbnail = cursor.fetchone()

            conn.commit()

            result = {
                'id': row['id'],
                'topic': row['topic'],
                'description': row['description'],
                'disaster_type': row['disaster_type'],
                'historical_period': row['historical_period'],
                'keywords': row['keywords'],
                'clickbait_title': row['clickbait_title'],
                'font_design': row['font_design'],
                'thumbnail_template': dict(thumbnail) if thumbnail else None
            }

            print(f"📚 Next Disaster Topic: {row['topic']}")
            print(f"🔥 Disaster Type: {row['disaster_type']}")
            print(f"⏰ Historical Period: {row['historical_period']}")

            return result

    def get_queue_status(self) -> Dict:
        """Get disaster queue status"""

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cursor = conn.cursor()

            # Status distribution
            cursor.execute('''
                SELECT status, COUNT(*) as count 
                FROM disaster_topic_queue 
                GROUP BY status
            ''')
            status_stats = dict(cursor.fetchall())

            # Disaster type distribution
            cursor.execute('''
                SELECT disaster_type, COUNT(*) as count 
                FROM disaster_topic_queue 
                WHERE status = 'pending'
                GROUP BY disaster_type
            ''')
            disaster_distribution = dict(cursor.fetchall())

            return {
                'status_distribution': status_stats,
                'disaster_distribution': disaster_distribution,
                'total_pending': status_stats.get('pending', 0)
            }


def main():
    """Import disaster topics and setup system"""

    print("🔥 DISASTER-FOCUSED TOPIC IMPORT SYSTEM")
    print("Channel Concept: Final Days Before Disasters")
    print("=" * 60)

    # Initialize importer
    importer = DisasterTopicImporter()

    # Setup disaster categories
    importer.setup_disaster_categories()

    # Import CSV topics
    topics = importer.import_csv_topics("topics.csv")

    # Import thumbnail features
    importer.import_thumbnail_features("thumbnail_features.json")

    # Show status
    status = importer.get_queue_status()
    print(f"\n📊 Import Summary:")
    print(f"📝 Total Topics: {sum(status['status_distribution'].values())}")
    print(f"⏳ Pending: {status['total_pending']}")

    print(f"\n🔥 Disaster Type Distribution:")
    for disaster_type, count in status['disaster_distribution'].items():
        print(f"   {disaster_type}: {count} topics")

    # Test getting next topic
    print(f"\n🎯 Next Topic Test:")
    next_topic = importer.get_next_disaster_topic()
    if next_topic:
        print(f"   Topic: {next_topic['topic']}")
        print(f"   Type: {next_topic['disaster_type']}")
        print(f"   Clickbait: {next_topic['clickbait_title']}")
        if next_topic['thumbnail_template']:
            print(f"   Thumbnail: {next_topic['thumbnail_template']['shocking_word']}")

    print(f"\n✅ Disaster Topic System Ready!")


if __name__ == "__main__":
    main()