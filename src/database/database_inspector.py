"""
Database Inspector - Check Topics Table Structure
Check current topics table structure and data
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime


def inspect_topics_table():
    """Check topics table structure and sample data"""

    db_path = Path("data/production.db")

    if not db_path.exists():
        print("❌ Database not found: data/production.db")
        return

    print("🔍 TOPICS TABLE INSPECTION")
    print("=" * 60)
    print(f"📁 Database: {db_path}")
    print(f"📦 Size: {db_path.stat().st_size / 1024:.1f} KB")
    print(f"📅 Modified: {datetime.fromtimestamp(db_path.stat().st_mtime)}")
    print()

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Get table schema
            print("📋 TABLE STRUCTURE:")
            print("-" * 40)
            cursor.execute("PRAGMA table_info(topics);")
            columns = cursor.fetchall()

            print(f"{'Column':<35} {'Type':<15} {'Null':<8} {'Default':<20} {'PK'}")
            print("-" * 90)

            for col in columns:
                col_id, name, type_name, not_null, default_val, pk = col
                null_str = "NOT NULL" if not_null else "NULL OK"
                default_str = str(default_val) if default_val is not None else "None"
                pk_str = "PK" if pk else ""

                print(f"{name:<35} {type_name:<15} {null_str:<8} {default_str:<20} {pk_str}")

            print()

            # Get record counts
            print("📊 RECORD COUNTS:")
            print("-" * 20)
            cursor.execute("SELECT COUNT(*) FROM topics;")
            total_count = cursor.fetchone()[0]
            print(f"Total records: {total_count}")

            if total_count > 0:
                # Status distribution
                print("\n📈 STATUS DISTRIBUTION:")
                print("-" * 25)

                # Main status
                cursor.execute("""
                    SELECT status, COUNT(*) 
                    FROM topics 
                    GROUP BY status 
                    ORDER BY COUNT(*) DESC
                """)
                status_dist = cursor.fetchall()

                for status, count in status_dist:
                    percentage = (count / total_count * 100) if total_count > 0 else 0
                    print(f"  {status or 'NULL':<15} {count:>3} ({percentage:>5.1f}%)")

                # Check stage status columns
                stage_columns = [
                    'story_generation_status',
                    'character_generation_status',
                    'scene_generation_status',
                    'audio_generation_status',
                    'video_generation_status',
                    'thumbnail_generation_status',
                    'youtube_upload_status'
                ]

                print("\n🔄 STAGE STATUS COLUMNS:")
                print("-" * 30)

                existing_stage_columns = []

                for col_name in stage_columns:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM topics WHERE {col_name} IS NOT NULL;")
                        non_null_count = cursor.fetchone()[0]

                        cursor.execute(f"SELECT COUNT(*) FROM topics WHERE {col_name} IS NULL;")
                        null_count = cursor.fetchone()[0]

                        existing_stage_columns.append(col_name)

                        # Get unique values for this column
                        cursor.execute(f"SELECT DISTINCT {col_name} FROM topics WHERE {col_name} IS NOT NULL;")
                        unique_values = [row[0] for row in cursor.fetchall()]

                        print(f"  ✅ {col_name:<30} NULL: {null_count:>3} | Non-NULL: {non_null_count:>3}")
                        if unique_values:
                            print(f"     Values: {', '.join(unique_values)}")

                    except sqlite3.OperationalError:
                        print(f"  ❌ {col_name:<30} Column doesn't exist")

                # Check other important columns
                print("\n📋 OTHER IMPORTANT COLUMNS:")
                print("-" * 35)

                important_columns = [
                    'thumbnail_generated',
                    'api_calls_used',
                    'total_cost',
                    'scenes_generated',
                    'output_path',
                    'created_at',
                    'updated_at'
                ]

                for col_name in important_columns:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM topics WHERE {col_name} IS NOT NULL;")
                        non_null_count = cursor.fetchone()[0]

                        print(f"  📝 {col_name:<25} Non-NULL: {non_null_count:>3}")

                        # Show sample values for some columns
                        if col_name in ['api_calls_used', 'total_cost', 'scenes_generated']:
                            cursor.execute(f"SELECT AVG({col_name}) FROM topics WHERE {col_name} IS NOT NULL;")
                            avg_val = cursor.fetchone()[0]
                            if avg_val is not None:
                                print(f"     Average: {avg_val:.2f}")

                    except sqlite3.OperationalError:
                        print(f"  ❌ {col_name:<25} Column doesn't exist")

                # Sample records
                print("\n📄 SAMPLE RECORDS:")
                print("-" * 20)

                cursor.execute("""
                    SELECT id, topic, status, 
                           story_generation_status,
                           character_generation_status,
                           scene_generation_status,
                           audio_generation_status,
                           video_generation_status,
                           thumbnail_generated,
                           created_at
                    FROM topics 
                    ORDER BY id DESC 
                    LIMIT 5
                """)

                sample_records = cursor.fetchall()

                print(f"{'ID':<4} {'Topic':<25} {'Status':<12} {'Story':<10} {'Char':<8} {'Scene':<8} {'Audio':<8} {'Video':<8} {'Thumb':<6}")
                print("-" * 100)

                for record in sample_records:
                    id_val, topic, status, story_st, char_st, scene_st, audio_st, video_st, thumb, created = record

                    topic_short = (topic[:22] + "...") if topic and len(topic) > 25 else (topic or "")
                    status_short = (status[:9] + "...") if status and len(status) > 12 else (status or "")

                    print(f"{id_val:<4} {topic_short:<25} {status_short:<12} {story_st or 'NULL':<10} {char_st or 'NULL':<8} {scene_st or 'NULL':<8} {audio_st or 'NULL':<8} {video_st or 'NULL':<8} {thumb or 'NULL':<6}")

                # Pipeline readiness check
                print("\n🚀 PIPELINE READINESS CHECK:")
                print("-" * 35)

                # Topics ready for each stage
                pipeline_checks = [
                    ("Story Generation", "status = 'pending'"),
                    ("Character Generation", "story_generation_status = 'completed' AND (character_generation_status IS NULL OR character_generation_status = 'pending')"),
                    ("Scene Generation", "character_generation_status = 'completed' AND (scene_generation_status IS NULL OR scene_generation_status = 'pending')"),
                    ("Audio Generation", "scene_generation_status = 'completed' AND (audio_generation_status IS NULL OR audio_generation_status = 'pending')"),
                    ("Video Composition", "audio_generation_status = 'completed' AND (video_generation_status IS NULL OR video_generation_status = 'pending')"),
                ]

                for stage_name, condition in pipeline_checks:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM topics WHERE {condition};")
                        ready_count = cursor.fetchone()[0]

                        if ready_count > 0:
                            print(f"  🎯 {stage_name:<20} {ready_count:>3} topics ready")

                            # Show some ready topics
                            cursor.execute(f"SELECT id, topic FROM topics WHERE {condition} LIMIT 3;")
                            ready_topics = cursor.fetchall()
                            for topic_id, topic_title in ready_topics:
                                topic_short = (topic_title[:30] + "...") if topic_title and len(topic_title) > 33 else (topic_title or "")
                                print(f"     - ID {topic_id}: {topic_short}")
                        else:
                            print(f"  ⚪ {stage_name:<20} No topics ready")

                    except sqlite3.OperationalError as e:
                        print(f"  ❌ {stage_name:<20} Error: {e}")

            else:
                print("No records found in topics table")

    except Exception as e:
        print(f"❌ Error inspecting database: {e}")
        import traceback
        traceback.print_exc()


def suggest_missing_columns():
    """Suggest missing columns for autonomous pipeline"""

    print("\n🔧 AUTONOMOUS PIPELINE COLUMN SUGGESTIONS:")
    print("=" * 50)

    suggested_columns = [
        ("story_generation_status", "TEXT DEFAULT 'pending'", "Track story generation progress"),
        ("story_generation_started_at", "DATETIME", "When story generation started"),
        ("story_generation_completed_at", "DATETIME", "When story generation completed"),

        ("character_generation_status", "TEXT DEFAULT NULL", "Track character generation progress"),
        ("character_generation_started_at", "DATETIME", "When character generation started"),
        ("character_generation_completed_at", "DATETIME", "When character generation completed"),

        ("scene_generation_status", "TEXT DEFAULT NULL", "Track scene generation progress"),
        ("scene_generation_started_at", "DATETIME", "When scene generation started"),
        ("scene_generation_completed_at", "DATETIME", "When scene generation completed"),

        ("audio_generation_status", "TEXT DEFAULT NULL", "Track audio generation progress"),
        ("audio_generation_started_at", "DATETIME", "When audio generation started"),
        ("audio_generation_completed_at", "DATETIME", "When audio generation completed"),

        ("video_generation_status", "TEXT DEFAULT NULL", "Track video generation progress"),
        ("video_generation_started_at", "DATETIME", "When video generation started"),
        ("video_generation_completed_at", "DATETIME", "When video generation completed"),

        ("thumbnail_generated", "BOOLEAN DEFAULT FALSE", "Whether thumbnail is ready"),
        ("thumbnail_path", "TEXT", "Path to thumbnail file"),

        ("orchestration_status", "TEXT DEFAULT 'pending'", "Overall pipeline status"),
        ("current_pipeline_stage", "TEXT", "Current stage in pipeline"),
        ("pipeline_started_at", "DATETIME", "When pipeline started"),
        ("pipeline_completed_at", "DATETIME", "When pipeline completed"),

        ("error_count", "INTEGER DEFAULT 0", "Number of errors encountered"),
        ("last_error", "TEXT", "Last error message"),
        ("retry_count", "INTEGER DEFAULT 0", "Number of retries attempted"),
    ]

    print("📋 Suggested columns for autonomous pipeline:")
    print()

    for col_name, col_type, description in suggested_columns:
        print(f"  📝 {col_name:<35} {col_type:<25} # {description}")

    print("\n🔧 SQL to add missing columns:")
    print("-" * 30)

    for col_name, col_type, description in suggested_columns:
        print(f"ALTER TABLE topics ADD COLUMN {col_name} {col_type};")


if __name__ == "__main__":
    inspect_topics_table()
    suggest_missing_columns()

    print("\n✅ Database inspection completed!")
    print("🚀 Ready to implement autonomous pipeline!")