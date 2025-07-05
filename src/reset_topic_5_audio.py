#!/usr/bin/env python3
"""
Reset Topic 5 Audio Generation Status to Pending
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime


def find_database_path():
    """Find production.db database file"""
    possible_paths = [
        "data/production.db",
        "../data/production.db",
        "../../data/production.db",
        "/home/youtube-automation/channels/sleepy-dull-stories/data/production.db"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Try relative to current script location
    current_dir = Path(__file__).parent
    for i in range(3):  # Try up to 3 levels up
        db_path = current_dir / "data" / "production.db"
        if db_path.exists():
            return str(db_path)
        current_dir = current_dir.parent

    return None


def reset_topic_5_audio_status():
    """Reset Topic 5 audio generation status to pending"""

    print("🔍 Finding production.db...")
    db_path = find_database_path()

    if not db_path:
        print("❌ production.db not found!")
        print("💡 Make sure you're running this from the correct directory")
        return False

    print(f"✅ Database found: {db_path}")

    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check current status of topic 5
        print("\n🔍 Checking current status of Topic 5...")
        cursor.execute('''
            SELECT id, topic, status, scene_generation_status, audio_generation_status, 
                   audio_generation_started_at, audio_generation_completed_at,
                   audio_chunks_generated, audio_duration_seconds, audio_cost_usd
            FROM topics 
            WHERE id = 5
        ''')

        result = cursor.fetchone()
        if not result:
            print("❌ Topic 5 not found in database!")
            conn.close()
            return False

        # Display current status
        (topic_id, topic, status, scene_status, audio_status,
         audio_started, audio_completed, chunks, duration, cost) = result

        print(f"📊 CURRENT STATUS - Topic {topic_id}:")
        print(f"   📚 Topic: {topic}")
        print(f"   ✅ Status: {status}")
        print(f"   🎬 Scene status: {scene_status}")
        print(f"   🎵 Audio status: {audio_status}")
        print(f"   🕐 Audio started: {audio_started}")
        print(f"   🕐 Audio completed: {audio_completed}")
        print(f"   📊 Chunks generated: {chunks}")
        print(f"   ⏱️  Duration: {duration}s" if duration else "   ⏱️  Duration: None")
        print(f"   💰 Cost: ${cost}" if cost else "   💰 Cost: None")

        # Reset audio generation status
        print(f"\n🔄 Resetting audio generation status to 'pending'...")

        cursor.execute('''
            UPDATE topics 
            SET audio_generation_status = 'pending',
                audio_generation_started_at = NULL,
                audio_generation_completed_at = NULL,
                audio_chunks_generated = 0,
                audio_duration_seconds = 0.0,
                audio_cost_usd = 0.0,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 5
        ''')

        # Verify the update
        if cursor.rowcount > 0:
            conn.commit()
            print("✅ Topic 5 audio status reset successfully!")

            # Show updated status
            cursor.execute('''
                SELECT audio_generation_status, audio_generation_started_at, 
                       audio_generation_completed_at, audio_chunks_generated,
                       audio_duration_seconds, audio_cost_usd, updated_at
                FROM topics 
                WHERE id = 5
            ''')

            updated_result = cursor.fetchone()
            (new_audio_status, new_started, new_completed, new_chunks,
             new_duration, new_cost, updated_at) = updated_result

            print(f"\n📊 UPDATED STATUS - Topic 5:")
            print(f"   🎵 Audio status: {new_audio_status}")
            print(f"   🕐 Audio started: {new_started}")
            print(f"   🕐 Audio completed: {new_completed}")
            print(f"   📊 Chunks: {new_chunks}")
            print(f"   ⏱️  Duration: {new_duration}s")
            print(f"   💰 Cost: ${new_cost}")
            print(f"   🕐 Updated at: {updated_at}")

            print(f"\n🎉 SUCCESS! Topic 5 is now ready for audio generation!")
            print(f"🚀 Next run of audio generator will pick up Topic 5")

        else:
            print("❌ No rows were updated!")
            conn.rollback()

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Database error: {e}")
        return False


def show_topics_ready_for_audio():
    """Show all topics ready for audio generation"""

    db_path = find_database_path()
    if not db_path:
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("\n📋 TOPICS READY FOR AUDIO GENERATION:")
        print("=" * 60)

        cursor.execute('''
            SELECT id, topic, status, scene_generation_status, audio_generation_status,
                   scene_generation_completed_at
            FROM topics 
            WHERE status = 'completed' 
            AND scene_generation_status = 'completed'
            AND (audio_generation_status IS NULL OR audio_generation_status = 'pending')
            ORDER BY scene_generation_completed_at ASC
        ''')

        ready_topics = cursor.fetchall()

        if not ready_topics:
            print("❌ No topics ready for audio generation")
        else:
            for i, (topic_id, topic, status, scene_status, audio_status, completed_at) in enumerate(ready_topics, 1):
                print(f"   {i}. Topic {topic_id}: {topic}")
                print(f"      📊 Status: {status} | Scene: {scene_status} | Audio: {audio_status or 'pending'}")
                print(f"      🕐 Scene completed: {completed_at}")
                if i == 1:
                    print(f"      🎯 NEXT TO PROCESS ⬆️")
                print()

        conn.close()

    except Exception as e:
        print(f"❌ Error checking topics: {e}")


if __name__ == "__main__":
    print("🎵 RESET TOPIC 5 AUDIO STATUS")
    print("=" * 40)

    # Reset Topic 5 status
    success = reset_topic_5_audio_status()

    if success:
        # Show all topics ready for audio
        show_topics_ready_for_audio()
    else:
        print("💡 Troubleshooting:")
        print("   1. Check if you're in the correct directory")
        print("   2. Verify production.db exists in data/ folder")
        print("   3. Make sure database is not locked by another process")