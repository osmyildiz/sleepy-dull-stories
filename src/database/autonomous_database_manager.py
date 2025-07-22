"""
Autonomous Database Manager - Pipeline Query System
Each generator will use this to find ready topics and update status
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List


class AutonomousDatabaseManager:
    """Database manager for autonomous pipeline operations"""

    def __init__(self, db_path: str = "data/production.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        print(f"✅ Database connected: {self.db_path}")

    def get_next_story_generation_topic(self) -> Optional[Tuple[int, str, str]]:
        """Get next topic ready for story generation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, topic, description 
                FROM topics 
                WHERE status = 'pending'
                ORDER BY priority ASC, created_at ASC 
                LIMIT 1
            """)

            result = cursor.fetchone()
            return result if result else None

    def get_next_character_generation_topic(self) -> Optional[Tuple[int, str, str]]:
        """Get next topic ready for character generation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, topic, description 
                FROM topics 
                WHERE status = 'completed' 
                AND character_generation_status = 'pending'
                ORDER BY production_completed_at ASC 
                LIMIT 1
            """)

            result = cursor.fetchone()
            return result if result else None

    def get_next_scene_generation_topic(self) -> Optional[Tuple[int, str, str]]:
        """Get next topic ready for scene generation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, topic, description 
                FROM topics 
                WHERE character_generation_status = 'completed' 
                AND scene_generation_status = 'pending'
                ORDER BY character_generation_completed_at ASC 
                LIMIT 1
            """)

            result = cursor.fetchone()
            return result if result else None

    def get_next_audio_generation_topic(self) -> Optional[Tuple[int, str, str]]:
        """Get next topic ready for audio generation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, topic, description, output_path
                FROM topics 
                WHERE scene_generation_status = 'completed' 
                AND audio_generation_status = 'pending'
                ORDER BY scene_generation_completed_at ASC 
                LIMIT 1
            """)

            result = cursor.fetchone()
            return result if result else None

    def get_next_video_generation_topic(self) -> Optional[Tuple[int, str, str, str]]:
        """Get next topic ready for video generation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, topic, description, output_path
                FROM topics 
                WHERE audio_generation_status = 'completed' 
                AND video_generation_status = 'pending'
                ORDER BY audio_generation_completed_at ASC 
                LIMIT 1
            """)

            result = cursor.fetchone()
            return result if result else None

    # Story Generation Status Management (uses main 'status' column)
    def mark_story_generation_started(self, topic_id: int):
        """Mark story generation as started"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE topics 
                SET production_started_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (topic_id,))

            conn.commit()
        print(f"🔄 Story generation started for topic {topic_id}")

    def mark_story_generation_completed(self, topic_id: int, output_path: str,
                                        scene_count: int = 0, api_calls: int = 0, cost: float = 0.0):
        """Mark story generation as completed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE topics 
                SET status = 'completed',
                    production_completed_at = CURRENT_TIMESTAMP,
                    output_path = ?,
                    scene_count = ?,
                    api_calls_used = api_calls_used + ?,
                    total_cost = total_cost + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (output_path, scene_count, api_calls, cost, topic_id))

            conn.commit()
        print(f"✅ Story generation completed for topic {topic_id}")

    def mark_story_generation_failed(self, topic_id: int, error_message: str):
        """Mark story generation as failed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE topics 
                SET status = 'failed',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (topic_id,))

            conn.commit()
        print(f"❌ Story generation failed for topic {topic_id}: {error_message}")

    # Character Generation Status Management
    def mark_character_generation_started(self, topic_id: int):
        """Mark character generation as started"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE topics 
                SET character_generation_status = 'in_progress',
                    character_generation_started_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (topic_id,))

            conn.commit()
        print(f"🔄 Character generation started for topic {topic_id}")

    def mark_character_generation_completed(self, topic_id: int, characters_generated: int = 0):
        """Mark character generation as completed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE topics 
                SET character_generation_status = 'completed',
                    character_generation_completed_at = CURRENT_TIMESTAMP,
                    characters_generated = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (characters_generated, topic_id))

            conn.commit()
        print(f"✅ Character generation completed for topic {topic_id}")

    # Scene Generation Status Management
    def mark_scene_generation_started(self, topic_id: int):
        """Mark scene generation as started"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE topics 
                SET scene_generation_status = 'in_progress',
                    scene_generation_started_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (topic_id,))

            conn.commit()
        print(f"🔄 Scene generation started for topic {topic_id}")

    def mark_scene_generation_completed(self, topic_id: int, scenes_generated: int = 0):
        """Mark scene generation as completed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE topics 
                SET scene_generation_status = 'completed',
                    scene_generation_completed_at = CURRENT_TIMESTAMP,
                    scenes_generated = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (scenes_generated, topic_id))

            conn.commit()
        print(f"✅ Scene generation completed for topic {topic_id}")

    # Audio Generation Status Management
    def mark_audio_generation_started(self, topic_id: int):
        """Mark audio generation as started"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE topics 
                SET audio_generation_status = 'in_progress',
                    audio_generation_started_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (topic_id,))

            conn.commit()
        print(f"🔄 Audio generation started for topic {topic_id}")

    def mark_audio_generation_completed(self, topic_id: int, audio_duration: float = 0.0,
                                        audio_cost: float = 0.0, chunks_generated: int = 0):
        """Mark audio generation as completed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE topics 
                SET audio_generation_status = 'completed',
                    audio_generation_completed_at = CURRENT_TIMESTAMP,
                    audio_duration_seconds = ?,
                    audio_cost_usd = ?,
                    audio_chunks_generated = ?,
                    total_cost = total_cost + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (audio_duration, audio_cost, chunks_generated, audio_cost, topic_id))

            conn.commit()
        print(f"✅ Audio generation completed for topic {topic_id}")

    # Video Generation Status Management
    def mark_video_generation_started(self, topic_id: int):
        """Mark video generation as started"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE topics 
                SET video_generation_status = 'in_progress',
                    video_generation_started_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (topic_id,))

            conn.commit()
        print(f"🔄 Video generation started for topic {topic_id}")

    def mark_video_generation_completed(self, topic_id: int, video_duration: float = 0.0,
                                        video_file_size: float = 0.0, processing_time: float = 0.0):
        """Mark video generation as completed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE topics 
                SET video_generation_status = 'completed',
                    video_generation_completed_at = CURRENT_TIMESTAMP,
                    video_duration_seconds = ?,
                    video_file_size_mb = ?,
                    video_processing_time_minutes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (video_duration, video_file_size, processing_time, topic_id))

            conn.commit()
        print(f"✅ Video generation completed for topic {topic_id}")

    def get_pipeline_status(self) -> Dict[str, int]:
        """Get comprehensive pipeline status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            status = {}

            # Story generation queue (using main status column)
            cursor.execute("SELECT COUNT(*) FROM topics WHERE status = 'pending'")
            status['story_generation_queue'] = cursor.fetchone()[0]

            # Character generation queue
            cursor.execute("""
                SELECT COUNT(*) FROM topics 
                WHERE status = 'completed' 
                AND character_generation_status = 'pending'
            """)
            status['character_generation_queue'] = cursor.fetchone()[0]

            # Scene generation queue
            cursor.execute("""
                SELECT COUNT(*) FROM topics 
                WHERE character_generation_status = 'completed' 
                AND scene_generation_status = 'pending'
            """)
            status['scene_generation_queue'] = cursor.fetchone()[0]

            # Audio generation queue
            cursor.execute("""
                SELECT COUNT(*) FROM topics 
                WHERE scene_generation_status = 'completed' 
                AND audio_generation_status = 'pending'
            """)
            status['audio_generation_queue'] = cursor.fetchone()[0]

            # Video generation queue
            cursor.execute("""
                SELECT COUNT(*) FROM topics 
                WHERE audio_generation_status = 'completed' 
                AND video_generation_status = 'pending'
            """)
            status['video_generation_queue'] = cursor.fetchone()[0]

            # Active processing counts (no story_generation_status anymore)
            status['story_generation_active'] = 0  # No separate status tracking

            cursor.execute("SELECT COUNT(*) FROM topics WHERE character_generation_status = 'in_progress'")
            status['character_generation_active'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM topics WHERE scene_generation_status = 'in_progress'")
            status['scene_generation_active'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM topics WHERE audio_generation_status = 'in_progress'")
            status['audio_generation_active'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM topics WHERE video_generation_status = 'in_progress'")
            status['video_generation_active'] = cursor.fetchone()[0]

            return status


def test_autonomous_db_manager():
    """Test the autonomous database manager"""
    print("🧪 Testing Autonomous Database Manager")
    print("=" * 50)

    try:
        # Initialize manager
        db_manager = AutonomousDatabaseManager()

        # Get pipeline status
        status = db_manager.get_pipeline_status()

        print("📊 PIPELINE STATUS:")
        print(f"   📝 Story generation queue: {status['story_generation_queue']}")
        print(f"   👥 Character generation queue: {status['character_generation_queue']}")
        print(f"   🎬 Scene generation queue: {status['scene_generation_queue']}")
        print(f"   🎵 Audio generation queue: {status['audio_generation_queue']}")
        print(f"   🎥 Video generation queue: {status['video_generation_queue']}")

        print("\n🔄 ACTIVE PROCESSING:")
        print(f"   📝 Story generation active: {status['story_generation_active']}")
        print(f"   👥 Character generation active: {status['character_generation_active']}")
        print(f"   🎬 Scene generation active: {status['scene_generation_active']}")
        print(f"   🎵 Audio generation active: {status['audio_generation_active']}")
        print(f"   🎥 Video generation active: {status['video_generation_active']}")

        # Test getting next topics
        print("\n🎯 NEXT READY TOPICS:")

        story_topic = db_manager.get_next_story_generation_topic()
        if story_topic:
            print(f"   📝 Next story generation: ID {story_topic[0]} - {story_topic[1]}")
        else:
            print(f"   📝 No topics ready for story generation")

        char_topic = db_manager.get_next_character_generation_topic()
        if char_topic:
            print(f"   👥 Next character generation: ID {char_topic[0]} - {char_topic[1]}")
        else:
            print(f"   👥 No topics ready for character generation")

        scene_topic = db_manager.get_next_scene_generation_topic()
        if scene_topic:
            print(f"   🎬 Next scene generation: ID {scene_topic[0]} - {scene_topic[1]}")
        else:
            print(f"   🎬 No topics ready for scene generation")

        audio_topic = db_manager.get_next_audio_generation_topic()
        if audio_topic:
            print(f"   🎵 Next audio generation: ID {audio_topic[0]} - {audio_topic[1]}")
        else:
            print(f"   🎵 No topics ready for audio generation")

        video_topic = db_manager.get_next_video_generation_topic()
        if video_topic:
            print(f"   🎥 Next video generation: ID {video_topic[0]} - {video_topic[1]}")
        else:
            print(f"   🎥 No topics ready for video generation")

        print("\n✅ Autonomous Database Manager test completed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_autonomous_db_manager()