"""
Production Database Setup - Professional Status Tracking
Location: src/database/database_setup.py
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
import json


class ProductionDatabase:
    """Professional production status tracking database"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to project root/data/production.db
            project_root = Path(__file__).parent.parent.parent
            self.db_path = project_root / "data" / "production.db"
        else:
            self.db_path = Path(db_path)

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"📁 Database path: {self.db_path}")

    def create_tables(self):
        """Create all production tracking tables"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Main video production table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_production (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    description TEXT,
                    clickbait_title TEXT,
                    font_design TEXT,

                    -- Overall Status
                    status TEXT DEFAULT 'pending',  -- pending, in_progress, completed, failed, retry
                    current_stage TEXT DEFAULT 'story_generation',

                    -- Stage Status Tracking
                    story_generation_status TEXT DEFAULT 'pending',
                    character_extraction_status TEXT DEFAULT 'pending',
                    visual_generation_status TEXT DEFAULT 'pending',
                    tts_generation_status TEXT DEFAULT 'pending',
                    video_composition_status TEXT DEFAULT 'pending',
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
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Stage execution details table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stage_execution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    stage_name TEXT NOT NULL,
                    status TEXT NOT NULL,  -- pending, in_progress, completed, failed

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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_status ON video_production(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_stage ON video_production(current_stage)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stage_execution_video ON stage_execution(video_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stage_execution_stage ON stage_execution(stage_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_gate_video ON quality_gate_results(video_id)")

            conn.commit()
            print("✅ Database tables created successfully!")

    def insert_sample_data(self):
        """Insert sample data for testing"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Sample video production entry
            cursor.execute("""
                INSERT INTO video_production 
                (topic, description, status, current_stage, story_completion_rate, character_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "Ancient Roman Villa",
                "A peaceful exploration of a Roman villa at sunset",
                "completed",
                "youtube_upload",
                1.0,
                5
            ))

            video_id = cursor.lastrowid

            # Sample stage executions
            stages = [
                ("story_generation", "completed", 330, 2, 0.95),
                ("character_extraction", "completed", 120, 1, 0.90),
                ("visual_generation", "completed", 1200, 0, 0.85),
                ("tts_generation", "in_progress", 0, 0, 0.0)
            ]

            for stage_name, status, duration, api_calls, quality_score in stages:
                cursor.execute("""
                    INSERT INTO stage_execution 
                    (video_id, stage_name, status, duration_seconds, api_calls, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (video_id, stage_name, status, duration, api_calls, quality_score))

            # Sample system metrics
            cursor.execute("""
                INSERT INTO system_metrics 
                (cpu_usage_percent, memory_usage_percent, disk_usage_percent, 
                 active_productions, completed_today, claude_api_calls_today)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (45.2, 62.8, 23.1, 1, 3, 12))

            conn.commit()
            print("✅ Sample data inserted!")

    def test_database(self):
        """Test database operations"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Test video_production table
                cursor.execute("SELECT COUNT(*) FROM video_production")
                video_count = cursor.fetchone()[0]
                print(f"📊 Video productions in DB: {video_count}")

                # Test stage_execution table
                cursor.execute("SELECT COUNT(*) FROM stage_execution")
                stage_count = cursor.fetchone()[0]
                print(f"📊 Stage executions in DB: {stage_count}")

                # Test system_metrics table
                cursor.execute("SELECT COUNT(*) FROM system_metrics")
                metrics_count = cursor.fetchone()[0]
                print(f"📊 System metrics in DB: {metrics_count}")

                # Test recent activity
                cursor.execute("""
                    SELECT topic, status, current_stage 
                    FROM video_production 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """)
                recent_videos = cursor.fetchall()

                print(f"\n📺 Recent videos:")
                for topic, status, stage in recent_videos:
                    print(f"  • {topic}: {status} ({stage})")

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
    print("🚀 Professional Production Database Setup")
    print("=" * 50)

    # Initialize database
    db = ProductionDatabase()

    # Show database info
    info = db.get_database_info()
    print(f"📁 Database location: {info['database_path']}")
    print(f"📏 Database size: {info['database_size_mb']} MB")

    # Create tables
    print("\n🏗️  Creating database tables...")
    db.create_tables()

    # Insert sample data
    print("\n📊 Inserting sample data...")
    db.insert_sample_data()

    # Test database
    print("\n🧪 Testing database...")
    success = db.test_database()

    if success:
        print("\n🎉 Database setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Database schema created ✅")
        print("2. Sample data inserted ✅")
        print("3. Database tested ✅")
        print("\n🔄 Ready for Status Tracking System setup!")
    else:
        print("\n❌ Database setup failed!")
        return False

    return True


if __name__ == "__main__":
    main()