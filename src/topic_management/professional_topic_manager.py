"""
Professional Topic Management System for YouTube Automation
Replaces amateur CSV approach with production-grade topic scheduling
Location: src/topic_management/professional_topic_manager.py
"""

import sqlite3
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TopicCategory(Enum):
    """Topic categories for variety control"""
    ANCIENT_CIVILIZATIONS = "ancient_civilizations"
    MEDIEVAL_HISTORY = "medieval_history"
    NATURE_LANDSCAPES = "nature_landscapes"
    ARCHITECTURE = "architecture"
    CULTURAL_HERITAGE = "cultural_heritage"
    MYSTICAL_PLACES = "mystical_places"
    SEASONAL_THEMES = "seasonal_themes"
    TRAVEL_DESTINATIONS = "travel_destinations"


class TopicPriority(Enum):
    """Topic priority levels"""
    URGENT = "urgent"  # Must be done today
    HIGH = "high"  # High engagement potential
    MEDIUM = "medium"  # Standard content
    LOW = "low"  # Filler content
    SEASONAL = "seasonal"  # Time-sensitive


@dataclass
class TopicData:
    """Professional topic data structure"""
    id: int
    topic: str
    description: str
    category: TopicCategory
    priority: TopicPriority
    clickbait_title: str
    font_design: str
    scheduled_date: datetime
    created_at: datetime
    estimated_performance: int  # 1-10 score
    keywords: List[str]
    target_demographics: List[str]
    seasonal_relevance: Optional[str] = None
    competition_analysis: Optional[Dict] = None


class ProfessionalTopicManager:
    """Production-grade topic management and scheduling system"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.db_path = project_root / "data" / "production.db"
        else:
            self.db_path = Path(db_path)

        self.setup_topic_tables()
        print("✅ Professional Topic Manager initialized")

    def setup_topic_tables(self):
        """Create professional topic management tables"""

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute('PRAGMA journal_mode=WAL;')

            # Topic Queue Table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS topic_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    clickbait_title TEXT NOT NULL,
                    font_design TEXT NOT NULL,
                    scheduled_date TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    status TEXT DEFAULT 'pending',
                    estimated_performance INTEGER DEFAULT 5,
                    keywords TEXT DEFAULT '[]',
                    target_demographics TEXT DEFAULT '[]',
                    seasonal_relevance TEXT,
                    competition_analysis TEXT DEFAULT '{}',
                    produced_at TEXT,
                    video_id INTEGER,
                    performance_actual INTEGER
                )
            ''')

            # Topic Categories Table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS topic_categories (
                    category TEXT PRIMARY KEY,
                    description TEXT,
                    last_used TEXT,
                    usage_count INTEGER DEFAULT 0,
                    performance_avg REAL DEFAULT 5.0,
                    target_frequency INTEGER DEFAULT 7
                )
            ''')

            # Topic Performance Analytics
            conn.execute('''
                CREATE TABLE IF NOT EXISTS topic_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id INTEGER,
                    metric_name TEXT,
                    metric_value REAL,
                    recorded_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (topic_id) REFERENCES topic_queue (id)
                )
            ''')

            conn.commit()
            print("✅ Topic management tables created/verified")

    def initialize_category_data(self):
        """Initialize category rotation data"""

        categories_data = [
            ("ancient_civilizations", "Ancient Roman, Greek, Egyptian themes", 7),
            ("medieval_history", "Medieval castles, monasteries, villages", 7),
            ("nature_landscapes", "Gardens, forests, mountains, rivers", 5),
            ("architecture", "Historic buildings, palaces, temples", 6),
            ("cultural_heritage", "Traditional customs and places", 8),
            ("mystical_places", "Mysterious and atmospheric locations", 4),
            ("seasonal_themes", "Holiday and season-specific content", 14),
            ("travel_destinations", "Famous historic travel locations", 10)
        ]

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            for category, description, frequency in categories_data:
                conn.execute('''
                    INSERT OR IGNORE INTO topic_categories 
                    (category, description, target_frequency)
                    VALUES (?, ?, ?)
                ''', (category, description, frequency))
            conn.commit()

        print("✅ Category rotation data initialized")

    def generate_intelligent_topic_queue(self, days_ahead: int = 30) -> List[TopicData]:
        """Generate intelligent topic queue with variety control"""

        print(f"🧠 Generating {days_ahead}-day intelligent topic queue...")

        # Get category rotation schedule
        category_schedule = self._plan_category_rotation(days_ahead)

        # Generate topics for each scheduled category
        topic_queue = []

        for i, (date, category) in enumerate(category_schedule):
            # Generate topic based on category and date
            topic_data = self._generate_category_topic(category, date, i + 1)
            topic_queue.append(topic_data)

        # Save to database
        self._save_topic_queue(topic_queue)

        print(f"✅ Generated {len(topic_queue)} intelligent topics")
        return topic_queue

    def _plan_category_rotation(self, days: int) -> List[Tuple[datetime, TopicCategory]]:
        """Plan intelligent category rotation"""

        schedule = []
        current_date = datetime.now()

        # Category weights (higher = more frequent)
        category_weights = {
            TopicCategory.ANCIENT_CIVILIZATIONS: 3,
            TopicCategory.MEDIEVAL_HISTORY: 3,
            TopicCategory.NATURE_LANDSCAPES: 2,
            TopicCategory.ARCHITECTURE: 2,
            TopicCategory.CULTURAL_HERITAGE: 2,
            TopicCategory.MYSTICAL_PLACES: 1,
            TopicCategory.SEASONAL_THEMES: 1,
            TopicCategory.TRAVEL_DESTINATIONS: 2
        }

        # Create weighted rotation
        weighted_categories = []
        for category, weight in category_weights.items():
            weighted_categories.extend([category] * weight)

        # Shuffle for variety
        random.shuffle(weighted_categories)

        # Assign to dates
        for day in range(days):
            date = current_date + timedelta(days=day)
            category = weighted_categories[day % len(weighted_categories)]

            # Seasonal adjustments
            if date.month == 12:  # December
                if day % 7 == 0:  # Every 7th day in December
                    category = TopicCategory.SEASONAL_THEMES

            schedule.append((date, category))

        return schedule

    def _generate_category_topic(self, category: TopicCategory, date: datetime, day_number: int) -> TopicData:
        """Generate topic for specific category"""

        # Topic templates by category
        topic_templates = {
            TopicCategory.ANCIENT_CIVILIZATIONS: [
                ("Roman Villa in {region}", "A luxurious Roman villa during the height of the empire"),
                ("Greek Temple of {deity}", "An ancient Greek temple dedicated to the gods"),
                ("Egyptian Palace of {pharaoh}", "The opulent palace of an ancient Egyptian ruler"),
                ("Phoenician Trading Post", "A bustling Phoenician merchant settlement"),
                ("Byzantine Monastery", "A peaceful Byzantine monastery in the mountains")
            ],
            TopicCategory.MEDIEVAL_HISTORY: [
                ("Medieval Castle Library", "A vast library in a medieval fortress"),
                ("Monastery Scriptorium", "Monks copying manuscripts in candlelight"),
                ("Medieval Market Square", "A busy medieval town on market day"),
                ("Abbey Garden", "The peaceful gardens of a medieval abbey"),
                ("Knights' Hall", "The great hall of a medieval castle")
            ],
            TopicCategory.NATURE_LANDSCAPES: [
                ("Japanese Tea Garden", "A traditional Japanese garden in spring"),
                ("English Countryside Manor", "A peaceful manor house in rolling hills"),
                ("Mediterranean Olive Grove", "Ancient olive trees under warm sun"),
                ("Alpine Meadow", "Wildflowers in a high mountain meadow"),
                ("Forest Monastery", "A hidden monastery deep in ancient woods")
            ],
            TopicCategory.ARCHITECTURE: [
                ("Gothic Cathedral", "Soaring stone arches and stained glass"),
                ("Moorish Palace", "Intricate Islamic architecture and fountains"),
                ("Renaissance Villa", "Italian villa with classical proportions"),
                ("Art Nouveau Mansion", "Elegant curves and natural motifs"),
                ("Baroque Palace", "Ornate palace with golden decorations")
            ]
        }

        # Get templates for category
        templates = topic_templates.get(category, [
            ("Historic {place}", "A peaceful historical location")
        ])

        # Select template
        template = random.choice(templates)
        topic_title = template[0]
        base_description = template[1]

        # Add variety to topic
        regions = ["Tuscany", "Provence", "Andalusia", "Crete", "Sicily"]
        deities = ["Apollo", "Athena", "Artemis", "Demeter", "Dionysus"]
        pharaohs = ["Ramesses II", "Hatshepsut", "Akhenaten", "Tutankhamun"]

        if "{region}" in topic_title:
            topic_title = topic_title.format(region=random.choice(regions))
        elif "{deity}" in topic_title:
            topic_title = topic_title.format(deity=random.choice(deities))
        elif "{pharaoh}" in topic_title:
            topic_title = topic_title.format(pharaoh=random.choice(pharaohs))
        elif "{place}" in topic_title:
            places = ["Villa", "Garden", "Palace", "Temple", "Monastery"]
            topic_title = topic_title.format(place=random.choice(places))

        # Generate clickbait title
        clickbait_templates = [
            "The Secret History of {topic} (2 Hour Sleep Story)",
            "Ancient Secrets Hidden in {topic} - Deep Sleep Story",
            "What Historians Don't Tell You About {topic}",
            "The Lost Mysteries of {topic} - Sleep Meditation",
            "Discover the Hidden Truth of {topic} - 2 Hours"
        ]

        clickbait_title = random.choice(clickbait_templates).format(topic=topic_title)

        # Generate font design
        font_designs = [
            "Bold golden serif font with warm amber shadows",
            "Elegant script font with soft blue-gray coloring",
            "Classic Roman font with deep red and gold accents",
            "Medieval Gothic font with rich brown and cream tones",
            "Modern serif with gentle green and earth tones"
        ]

        font_design = random.choice(font_designs)

        # Determine priority
        priority = TopicPriority.MEDIUM
        if day_number <= 7:  # First week higher priority
            priority = TopicPriority.HIGH
        elif category == TopicCategory.SEASONAL_THEMES:
            priority = TopicPriority.SEASONAL

        # Generate keywords
        category_keywords = {
            TopicCategory.ANCIENT_CIVILIZATIONS: ["ancient", "history", "civilization", "empire"],
            TopicCategory.MEDIEVAL_HISTORY: ["medieval", "castle", "monastery", "knights"],
            TopicCategory.NATURE_LANDSCAPES: ["nature", "garden", "peaceful", "serene"],
            TopicCategory.ARCHITECTURE: ["architecture", "building", "design", "historic"]
        }

        keywords = category_keywords.get(category, ["sleep", "relaxation", "story"])
        keywords.extend(["sleep story", "relaxation", "meditation", "2 hours"])

        return TopicData(
            id=0,  # Will be set by database
            topic=topic_title,
            description=base_description,
            category=category,
            priority=priority,
            clickbait_title=clickbait_title,
            font_design=font_design,
            scheduled_date=date,
            created_at=datetime.now(),
            estimated_performance=random.randint(6, 9),  # Optimistic estimates
            keywords=keywords,
            target_demographics=["sleep", "relaxation", "history lovers", "meditation"]
        )

    def _save_topic_queue(self, topics: List[TopicData]):
        """Save topic queue to database"""

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            for topic in topics:
                conn.execute('''
                    INSERT INTO topic_queue 
                    (topic, description, category, priority, clickbait_title, 
                     font_design, scheduled_date, estimated_performance, keywords, target_demographics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    topic.topic,
                    topic.description,
                    topic.category.value,
                    topic.priority.value,
                    topic.clickbait_title,
                    topic.font_design,
                    topic.scheduled_date.isoformat(),
                    topic.estimated_performance,
                    json.dumps(topic.keywords),
                    json.dumps(topic.target_demographics)
                ))
            conn.commit()

        print(f"✅ Saved {len(topics)} topics to queue")

    def get_next_scheduled_topic(self) -> Tuple[int, str, str, str, str]:
        """Get next topic from professional queue"""

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row

            # Get next pending topic scheduled for today or past
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM topic_queue 
                WHERE status = 'pending' 
                AND date(scheduled_date) <= date('now')
                ORDER BY priority DESC, scheduled_date ASC 
                LIMIT 1
            ''')

            row = cursor.fetchone()

            if not row:
                # No scheduled topics - get next pending
                cursor.execute('''
                    SELECT * FROM topic_queue 
                    WHERE status = 'pending'
                    ORDER BY priority DESC, scheduled_date ASC 
                    LIMIT 1
                ''')
                row = cursor.fetchone()

            if not row:
                raise ValueError("No topics in queue! Generate topics first.")

            # Mark as in production
            cursor.execute('''
                UPDATE topic_queue 
                SET status = 'in_production', produced_at = datetime('now')
                WHERE id = ?
            ''', (row['id'],))

            conn.commit()

            print(f"📚 Selected topic: {row['topic']}")
            print(f"🎯 Category: {row['category']}")
            print(f"⭐ Priority: {row['priority']}")

            return (
                row['id'],
                row['topic'],
                row['description'],
                row['clickbait_title'],
                row['font_design']
            )

    def get_queue_status(self) -> Dict:
        """Get current queue status"""

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cursor = conn.cursor()

            # Queue statistics
            cursor.execute('''
                SELECT 
                    status,
                    COUNT(*) as count,
                    MIN(scheduled_date) as earliest,
                    MAX(scheduled_date) as latest
                FROM topic_queue 
                GROUP BY status
            ''')

            status_stats = {}
            for row in cursor.fetchall():
                status_stats[row[0]] = {
                    'count': row[1],
                    'earliest': row[2],
                    'latest': row[3]
                }

            # Category distribution
            cursor.execute('''
                SELECT category, COUNT(*) as count 
                FROM topic_queue 
                WHERE status = 'pending'
                GROUP BY category
            ''')

            category_stats = dict(cursor.fetchall())

            return {
                'status_distribution': status_stats,
                'category_distribution': category_stats,
                'total_pending': status_stats.get('pending', {}).get('count', 0),
                'next_scheduled': self._get_next_scheduled_date()
            }

    def _get_next_scheduled_date(self) -> str:
        """Get next scheduled topic date"""

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT scheduled_date FROM topic_queue 
                WHERE status = 'pending'
                ORDER BY scheduled_date ASC 
                LIMIT 1
            ''')

            row = cursor.fetchone()
            return row[0] if row else "No topics scheduled"


def main():
    """Initialize professional topic management"""

    print("🚀 Professional Topic Management System")
    print("=" * 50)

    # Initialize manager
    manager = ProfessionalTopicManager()

    # Initialize category data
    manager.initialize_category_data()

    # Generate intelligent queue
    topics = manager.generate_intelligent_topic_queue(days_ahead=30)

    # Show queue status
    status = manager.get_queue_status()

    print(f"\n📊 Queue Status:")
    print(f"📝 Total Pending: {status['total_pending']}")
    print(f"📅 Next Scheduled: {status['next_scheduled']}")

    print(f"\n📈 Category Distribution:")
    for category, count in status['category_distribution'].items():
        print(f"   {category}: {count} topics")

    # Get next topic
    print(f"\n🎯 Next Topic to Produce:")
    try:
        topic_id, topic, description, clickbait, font = manager.get_next_scheduled_topic()
        print(f"   ID: {topic_id}")
        print(f"   Topic: {topic}")
        print(f"   Description: {description}")
        print(f"   Clickbait: {clickbait}")
    except Exception as e:
        print(f"   Error: {e}")

    print(f"\n✅ Professional Topic Management Ready!")
    print(f"🎬 Ready for automated production pipeline")


if __name__ == "__main__":
    main()