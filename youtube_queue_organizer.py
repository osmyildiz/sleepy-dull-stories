#!/usr/bin/env python3
"""
YouTube Queue Organizer
Organize video upload queue by scanning output directories and setting proper timestamps
"""

import os
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta


def scan_and_organize_youtube_queue():
    """Scan output directories and organize YouTube upload queue"""

    print("🔍 YouTube Queue Organizer")
    print("=" * 50)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    output_base = project_root / 'output'
    db_path = project_root / 'data' / 'production.db'

    print(f"📁 Output directory: {output_base}")
    print(f"🗄️ Database: {db_path}")

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False

    # Connect to database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Find videos with final_video.mp4
    videos_found = []

    for video_id in range(4, 34):  # 4 to 33
        video_dir = output_base / str(video_id)
        final_video_path = video_dir / 'final_video.mp4'

        if final_video_path.exists():
            file_size = final_video_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ Found ID {video_id}: final_video.mp4 ({file_size:.1f} MB)")

            # Check database status
            cursor.execute('''
                SELECT id, topic, status, video_generation_status, youtube_upload_status, 
                       video_generation_completed_at
                FROM topics WHERE id = ?
            ''', (video_id,))

            result = cursor.fetchone()
            if result:
                db_id, topic, status, video_status, youtube_status, completed_at = result
                videos_found.append({
                    'id': video_id,
                    'topic': topic,
                    'status': status,
                    'video_status': video_status,
                    'youtube_status': youtube_status,
                    'completed_at': completed_at,
                    'file_size_mb': file_size
                })
                print(f"   📋 Topic: {topic[:50]}...")
                print(f"   📊 Status: {status} | Video: {video_status} | YouTube: {youtube_status}")
            else:
                print(f"   ⚠️ ID {video_id} not found in database")
        else:
            print(f"❌ ID {video_id}: No final_video.mp4 found")

    if not videos_found:
        print("❌ No videos found with final_video.mp4")
        conn.close()
        return False

    print(f"\n📊 Found {len(videos_found)} videos with final_video.mp4")

    # Sort by ID (smallest to largest)
    videos_found.sort(key=lambda x: x['id'])

    # Update database with sequential timestamps
    base_time = datetime(2025, 7, 1, 10, 0, 0)  # Start from July 1, 2025

    print(f"\n🔄 Updating database timestamps...")

    for i, video in enumerate(videos_found):
        # Calculate timestamp (1 hour apart)
        new_timestamp = base_time + timedelta(hours=i)
        timestamp_str = new_timestamp.strftime('%Y-%m-%d %H:%M:%S')

        # Update video_generation_completed_at for proper ordering
        cursor.execute('''
            UPDATE topics 
            SET video_generation_completed_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (timestamp_str, video['id']))

        # Update statuses if needed
        updates = []

        if video['status'] != 'completed':
            cursor.execute('UPDATE topics SET status = ? WHERE id = ?', ('completed', video['id']))
            updates.append('status→completed')

        if video['video_status'] != 'completed':
            cursor.execute('UPDATE topics SET video_generation_status = ? WHERE id = ?', ('completed', video['id']))
            updates.append('video_status→completed')

        if video['youtube_status'] not in ['pending', None]:
            cursor.execute('UPDATE topics SET youtube_upload_status = ? WHERE id = ?', ('pending', video['id']))
            updates.append('youtube_status→pending')

        update_text = ', '.join(updates) if updates else 'no changes'
        print(f"✅ ID {video['id']:2d}: {timestamp_str} ({update_text})")

    # Commit changes
    conn.commit()
    print(f"\n💾 Database updated successfully!")

    # Show final queue order
    print(f"\n📺 Final YouTube Upload Queue:")
    cursor.execute('''
        SELECT id, topic, video_generation_completed_at, youtube_upload_status
        FROM topics 
        WHERE status = 'completed' 
        AND video_generation_status = 'completed'
        AND (youtube_upload_status IS NULL OR youtube_upload_status = 'pending')
        ORDER BY video_generation_completed_at ASC
    ''')

    queue = cursor.fetchall()
    for i, (video_id, topic, completed_at, youtube_status) in enumerate(queue, 1):
        print(f"   {i:2d}. ID {video_id:2d}: {topic[:40]}... ({completed_at})")

    conn.close()

    print(f"\n🎉 YouTube queue organized!")
    print(f"✅ {len(videos_found)} videos ready for upload")
    print(f"📊 Queue order: ID {videos_found[0]['id']} → ID {videos_found[-1]['id']}")
    print(f"\n🚀 Run YouTube uploader to start processing:")
    print(f"python3 src/generators/7_youtube_uploader_autonomous.py")

    return True


def verify_queue_order():
    """Verify the queue order after organization"""

    print("\n🔍 Verifying queue order...")

    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / 'src' / 'data' / 'production.db'

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, topic, video_generation_completed_at
        FROM topics 
        WHERE status = 'completed' 
        AND video_generation_status = 'completed'
        AND (youtube_upload_status IS NULL OR youtube_upload_status = 'pending')
        ORDER BY video_generation_completed_at ASC
        LIMIT 5
    ''')

    next_videos = cursor.fetchall()

    print("📋 Next 5 videos in queue:")
    for i, (video_id, topic, completed_at) in enumerate(next_videos, 1):
        print(f"   {i}. ID {video_id}: {topic[:50]}... ({completed_at})")

    conn.close()


if __name__ == "__main__":
    try:
        print("🚀 YouTube Queue Organizer v1.0")
        print("🔄 Scanning output directories (ID 4-33)")
        print("📊 Organizing upload queue by ID order")
        print("🗄️ Updating database timestamps")
        print("=" * 60)

        success = scan_and_organize_youtube_queue()

        if success:
            verify_queue_order()
            print("\n✅ Queue organization completed successfully!")

            choice = input("\n▶️ Start YouTube upload process now? (y/N): ").strip().lower()
            if choice == 'y':
                print("🚀 Starting YouTube uploader...")
                os.system("python3 src/generators/7_youtube_uploader_autonomous.py")
            else:
                print("⏸️ Ready to start uploads when you're ready!")
        else:
            print("\n❌ Queue organization failed!")

    except KeyboardInterrupt:
        print("\n⏹️ Queue organization stopped by user")
    except Exception as e:
        print(f"💥 Queue organization failed: {e}")
        import traceback

        traceback.print_exc()