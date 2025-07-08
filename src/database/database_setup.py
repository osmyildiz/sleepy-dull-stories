"""
Production Database Setup - Professional Status Tracking + Topics Management
Location: src/database/database_setup.py
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
import json
import csv


class ProductionDatabase:
    """Professional production status tracking database"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to current directory/production.db (your existing setup)
            current_dir = Path(__file__).parent
            self.db_path = current_dir / "production.db"
        else:
            self.db_path = Path(db_path)

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"📁 Database path: {self.db_path}")

    def create_tables(self):
        """Create all production tracking tables"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # TOPICS TABLE - CSV verileriniz için (keywords ve historical_period DAHİL)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    description TEXT NOT NULL,
                    clickbait_title TEXT DEFAULT '',
                    font_design TEXT DEFAULT '',
                    category TEXT DEFAULT 'sleep_story',
                    priority INTEGER DEFAULT 1,
                    target_duration_minutes INTEGER DEFAULT 135,
                    status TEXT DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    production_started_at DATETIME,
                    production_completed_at DATETIME,
                    scene_count INTEGER,
                    total_duration_minutes REAL,
                    api_calls_used INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    output_path TEXT,
                    
                    -- Visual Generation Stages
                    visual_generation_status TEXT DEFAULT "pending",
                    visual_generation_started_at DATETIME,
                    visual_generation_completed_at DATETIME,
                    characters_generated INTEGER DEFAULT 0,
                    scenes_generated INTEGER DEFAULT 0,
                    
                    -- Character Generation
                    character_generation_status TEXT DEFAULT "pending",
                    character_generation_started_at DATETIME,
                    character_generation_completed_at DATETIME,
                    
                    -- Scene Generation
                    scene_generation_status TEXT DEFAULT "pending",
                    scene_generation_started_at DATETIME,
                    scene_generation_completed_at DATETIME,
                    
                    -- Audio Generation
                    audio_generation_status TEXT DEFAULT "pending",
                    audio_generation_started_at DATETIME,
                    audio_generation_completed_at DATETIME,
                    audio_chunks_generated INTEGER DEFAULT 0,
                    audio_duration_seconds REAL DEFAULT 0.0,
                    audio_cost_usd REAL DEFAULT 0.0,
                    
                    -- Thumbnail Generation (CSV'den)
                    thumbnail_generation_status TEXT DEFAULT "pending",
                    thumbnail_generation_started_at DATETIME,
                    thumbnail_generation_completed_at DATETIME,
                    thumbnail_generated INTEGER DEFAULT 0,
                    thumbnail_file_size_kb REAL DEFAULT 0.0,
                    thumbnail_processing_time_seconds REAL DEFAULT 0.0,
                    
                    -- Cover Image Creation (CSV'den)
                    cover_image_creation_status TEXT DEFAULT "pending",
                    cover_image_created INTEGER DEFAULT 0,
                    
                    -- Teaser Video Creation (CSV'den)
                    teaser_video_creation_status TEXT DEFAULT "pending",
                    teaser_video_created INTEGER DEFAULT 0,
                    
                    -- Images Generation (CSV'den)
                    images_generation_status TEXT DEFAULT "pending",
                    images_generated INTEGER DEFAULT 0,
                    
                    -- Background Music (CSV'den)
                    background_music_status TEXT DEFAULT "pending",
                    background_music_added INTEGER DEFAULT 0,
                    
                    -- Video Editing/Combining (CSV'den)
                    editing_status TEXT DEFAULT "pending",
                    editing_completed INTEGER DEFAULT 0,
                    
                    -- Video Generation
                    video_generation_status TEXT DEFAULT "pending",
                    video_generation_started_at DATETIME,
                    video_generation_completed_at DATETIME,
                    video_duration_seconds REAL DEFAULT 0.0,
                    video_file_size_mb REAL DEFAULT 0.0,
                    video_processing_time_minutes REAL DEFAULT 0.0,
                    
                    -- Publishing (CSV'den)
                    publishing_status TEXT DEFAULT "pending",
                    published INTEGER DEFAULT 0,
                    
                    -- CSV verilerinizden eklenen ÖNEMLI kolonlar
                    keywords TEXT,                    -- "Vesuvius, eruption, Roman Empire, Campania, tragedy"
                    historical_period TEXT,          -- "1st Century CE"
                    
                    -- CSV production tracking
                    done INTEGER DEFAULT 0,                    -- CSV'deki done field
                    
                    UNIQUE(topic, description)
                );
            """)

            # Main video production table (GELIŞMIŞ STAGE TRACKING)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_production (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id INTEGER,  -- topics tablosuna referans
                    topic TEXT NOT NULL,
                    description TEXT,
                    clickbait_title TEXT,
                    font_design TEXT,

                    -- Overall Status
                    status TEXT DEFAULT 'pending',  -- pending, in_progress, completed, failed, retry
                    current_stage TEXT DEFAULT 'story_generation',

                    -- COMPLETE STAGE STATUS TRACKING (CSV alanlarına uygun)
                    story_generation_status TEXT DEFAULT 'pending',
                    character_extraction_status TEXT DEFAULT 'pending',
                    visual_generation_status TEXT DEFAULT 'pending',
                    audio_generation_status TEXT DEFAULT 'pending',
                    thumbnail_generation_status TEXT DEFAULT 'pending',      -- CSV: thumbnail
                    cover_image_creation_status TEXT DEFAULT 'pending',     -- CSV: cover_image_created
                    teaser_video_creation_status TEXT DEFAULT 'pending',    -- CSV: teaser_video_created
                    images_generation_status TEXT DEFAULT 'pending',        -- CSV: images_generated
                    background_music_status TEXT DEFAULT 'pending',         -- CSV: background_music_added
                    video_editing_status TEXT DEFAULT 'pending',            -- CSV: editing_completed
                    video_combination_status TEXT DEFAULT 'pending',        -- Combine all parts
                    final_video_generation_status TEXT DEFAULT 'pending',
                    publishing_status TEXT DEFAULT 'pending',               -- CSV: published
                    youtube_upload_status TEXT DEFAULT 'pending',

                    -- Quality Metrics
                    story_completion_rate REAL DEFAULT 0.0,
                    character_count INTEGER DEFAULT 0,
                    visual_prompts_count INTEGER DEFAULT 0,
                    thumbnail_generated BOOLEAN DEFAULT 0,
                    intro_visuals_generated BOOLEAN DEFAULT 0,

                    -- Performance Metrics
                    api_calls_used INTEGER DEFAULT 0,
                    total_duration_seconds INTEGER DEFAULT 0,

                    -- File Tracking
                    output_directory TEXT,
                    files_generated TEXT,  -- JSON array of filenames

                    -- Error Handling
                    error_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    retry_count INTEGER DEFAULT 0,

                    -- Resource Usage
                    cpu_usage_avg REAL DEFAULT 0.0,
                    memory_usage_peak REAL DEFAULT 0.0,
                    disk_usage_mb REAL DEFAULT 0.0,

                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (topic_id) REFERENCES topics (id)
                )
            """)

            # Stage execution details table (EXPANDED STAGES)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stage_execution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    stage_name TEXT NOT NULL,  -- story_generation, character_extraction, visual_generation, 
                                               -- audio_generation, thumbnail_generation, cover_image_creation,
                                               -- teaser_video_creation, images_generation, background_music,
                                               -- video_editing, video_combination, publishing, youtube_upload
                    status TEXT NOT NULL,      -- pending, in_progress, completed, failed

                    -- Execution Details
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    duration_seconds REAL DEFAULT 0.0,
                    api_calls INTEGER DEFAULT 0,

                    -- Quality Metrics
                    quality_score REAL DEFAULT 0.0,
                    quality_details TEXT,  -- JSON

                    -- Output Files
                    output_files TEXT,  -- JSON array

                    -- Error Information
                    error_message TEXT,
                    error_details TEXT,  -- JSON

                    -- Resource Usage
                    cpu_usage REAL DEFAULT 0.0,
                    memory_usage REAL DEFAULT 0.0,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (video_id) REFERENCES video_production (id)
                )
            """)

            # System metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- System Resources
                    cpu_usage_percent REAL,
                    memory_usage_percent REAL,
                    disk_usage_percent REAL,

                    -- Production Metrics
                    active_productions INTEGER DEFAULT 0,
                    completed_today INTEGER DEFAULT 0,
                    failed_today INTEGER DEFAULT 0,

                    -- API Usage
                    claude_api_calls_today INTEGER DEFAULT 0,
                    openai_api_calls_today INTEGER DEFAULT 0,

                    -- Queue Status
                    pending_videos INTEGER DEFAULT 0,
                    in_progress_videos INTEGER DEFAULT 0
                )
            """)

            # Quality gate results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_gate_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    stage_name TEXT NOT NULL,
                    gate_type TEXT NOT NULL,  -- story_quality, character_quality, etc.

                    -- Gate Results
                    passed BOOLEAN NOT NULL,
                    score REAL DEFAULT 0.0,
                    threshold REAL DEFAULT 0.0,

                    -- Details
                    criteria_checked TEXT,  -- JSON
                    failure_reasons TEXT,   -- JSON array
                    recommendations TEXT,   -- JSON array

                    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (video_id) REFERENCES video_production (id)
                )
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_status ON topics(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_priority ON topics(priority)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_created_at ON topics(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_keywords ON topics(keywords)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_historical_period ON topics(historical_period)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_done ON topics(done)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_status ON video_production(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_stage ON video_production(current_stage)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stage_execution_video ON stage_execution(video_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stage_execution_stage ON stage_execution(stage_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_gate_video ON quality_gate_results(video_id)")

            conn.commit()
            print("✅ Database tables created successfully!")

    def import_csv_data(self, csv_file_path: str = "topics.csv"):
        """CSV verilerinizi topics tablosuna KAPSAMLI import eder (keywords ve historical_period DAHİL)"""

        csv_path = Path(csv_file_path)
        if not csv_path.exists():
            print(f"❌ CSV file not found: {csv_path}")
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                with open(csv_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)

                    imported_count = 0
                    print("\n📊 Processing CSV data...")

                    for row in reader:
                        # CSV'deki GERÇEK veriler
                        topic = row.get('topic', '').strip()
                        description = row.get('description', '').strip()
                        keywords = row.get('keywords', '').strip()                    # "Vesuvius, eruption, Roman Empire, Campania, tragedy"
                        historical_period = row.get('historical_period', '').strip() # "1st Century CE"
                        estimated_duration = int(row.get('estimated_duration', 135))

                        # Production status fields from CSV
                        done = int(row.get('done', 0))
                        audio_generated = int(row.get('audio_generated', 0))
                        cover_image_created = int(row.get('cover_image_created', 0))
                        teaser_video_created = int(row.get('teaser_video_created', 0))
                        images_generated = int(row.get('images_generated', 0))
                        background_music_added = int(row.get('background_music_added', 0))
                        editing_completed = int(row.get('editing_completed', 0))
                        published = int(row.get('published', 0))
                        thumbnail = int(row.get('thumbnail', 0))

                        # Status belirleme
                        status = 'completed' if done == 1 else 'pending'

                        # Priority belirleme
                        if done == 1:
                            priority = 3  # Düşük (tamamlanmış)
                        elif (audio_generated == 1 or cover_image_created == 1 or thumbnail == 1):
                            priority = 1  # Yüksek (başlanmış)
                        else:
                            priority = 2  # Orta (beklemede)

                        # Category belirleme
                        category = 'historical_narrative' if historical_period and 'century' in historical_period.lower() else 'sleep_story'

                        # Insert data with ALL CSV fields
                        cursor.execute("""
                            INSERT OR IGNORE INTO topics (
                                topic, description, keywords, historical_period,
                                category, priority, target_duration_minutes, status,
                                done, audio_generation_status, cover_image_creation_status,
                                teaser_video_creation_status, images_generation_status,
                                background_music_status, editing_status, publishing_status,
                                thumbnail_generation_status,
                                cover_image_created, teaser_video_created, images_generated,
                                background_music_added, editing_completed, published, thumbnail_generated,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """, (
                            topic,
                            description,
                            keywords,                                               # ✅ KEYWORDS DAHİL
                            historical_period,                                     # ✅ HISTORICAL PERIOD DAHİL
                            category,
                            priority,
                            estimated_duration,
                            status,
                            done,
                            'completed' if audio_generated == 1 else 'pending',
                            'completed' if cover_image_created == 1 else 'pending',
                            'completed' if teaser_video_created == 1 else 'pending',
                            'completed' if images_generated == 1 else 'pending',
                            'completed' if background_music_added == 1 else 'pending',
                            'completed' if editing_completed == 1 else 'pending',
                            'completed' if published == 1 else 'pending',
                            'completed' if thumbnail == 1 else 'pending',
                            cover_image_created,
                            teaser_video_created,
                            images_generated,
                            background_music_added,
                            editing_completed,
                            published,
                            thumbnail
                        ))

                        imported_count += 1
                        print(f"  ✅ Imported: {topic}")
                        if keywords:
                            print(f"     Keywords: {keywords}")
                        if historical_period:
                            print(f"     Period: {historical_period}")

                conn.commit()
                print(f"\n✅ Successfully imported {imported_count} topics from CSV!")
                print(f"   📝 Keywords and historical periods included!")
                return True

        except Exception as e:
            print(f"❌ Error importing CSV: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_database(self):
        """Test database operations"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Test topics table
                cursor.execute("SELECT COUNT(*) FROM topics")
                topics_count = cursor.fetchone()[0]
                print(f"📊 Topics in DB: {topics_count}")

                # Test keywords and historical periods
                cursor.execute("SELECT COUNT(*) FROM topics WHERE keywords IS NOT NULL AND keywords != ''")
                keywords_count = cursor.fetchone()[0]
                print(f"📊 Topics with keywords: {keywords_count}")

                cursor.execute("SELECT COUNT(*) FROM topics WHERE historical_period IS NOT NULL AND historical_period != ''")
                periods_count = cursor.fetchone()[0]
                print(f"📊 Topics with historical periods: {periods_count}")

                # Test recent topics WITH keywords and periods
                cursor.execute("""
                    SELECT topic, keywords, historical_period, status 
                    FROM topics 
                    WHERE keywords IS NOT NULL AND keywords != ''
                    ORDER BY created_at DESC 
                    LIMIT 3
                """)
                recent_topics = cursor.fetchall()

                print(f"\n📚 Recent topics with full data:")
                for topic, keywords, period, status in recent_topics:
                    print(f"  🎬 {topic}: {status}")
                    print(f"     📝 Keywords: {keywords}")
                    print(f"     📅 Period: {period}")
                    print()

                # Test production stages
                cursor.execute("""
                    SELECT 
                        SUM(CASE WHEN done = 1 THEN 1 ELSE 0 END) as completed,
                        SUM(CASE WHEN thumbnail_generated = 1 THEN 1 ELSE 0 END) as thumbnails,
                        SUM(CASE WHEN cover_image_created = 1 THEN 1 ELSE 0 END) as covers,
                        SUM(CASE WHEN editing_completed = 1 THEN 1 ELSE 0 END) as edited
                    FROM topics
                """)
                stages = cursor.fetchone()
                print(f"📊 Production stages: {stages[0]} completed, {stages[1]} thumbnails, {stages[2]} covers, {stages[3]} edited")

                print("✅ Database test successful!")
                return True

        except Exception as e:
            print(f"❌ Database test failed: {e}")
            return False

    def get_database_info(self):
        """Get database information"""

        info = {
            "database_path": str(self.db_path),
            "database_exists": self.db_path.exists(),
            "database_size_mb": 0
        }

        if self.db_path.exists():
            info["database_size_mb"] = round(self.db_path.stat().st_size / 1024 / 1024, 2)

        return info


def main():
    """Main setup function"""
    print("🚀 Professional Production Database Setup + FULL CSV Import")
    print("=" * 65)

    # Initialize database
    db = ProductionDatabase()

    # Show database info
    info = db.get_database_info()
    print(f"📁 Database location: {info['database_path']}")
    print(f"📏 Database size: {info['database_size_mb']} MB")

    # Create tables
    print("\n🏗️  Creating database tables...")
    db.create_tables()

    # Import CSV data (WITH keywords and historical_period)
    print("\n📊 Importing CSV data with keywords and historical periods...")
    csv_imported = db.import_csv_data("topics.csv")

    # Test database
    print("\n🧪 Testing database...")
    success = db.test_database()

    if success:
        print("\n🎉 Database setup completed successfully!")
        print("\n📋 What was imported:")
        print("1. Database schema created ✅")
        print("2. Topics with keywords ✅" if csv_imported else "2. CSV import failed ❌")
        print("3. Historical periods ✅" if csv_imported else "3. Historical periods ❌")
        print("4. Production stages tracking ✅")
        print("5. Complete pipeline: story → thumbnail → editing → publishing ✅")
        print("\n🔄 Ready for full production pipeline!")
    else:
        print("\n❌ Database setup failed!")
        return False

    return True


if __name__ == "__main__":
    main()