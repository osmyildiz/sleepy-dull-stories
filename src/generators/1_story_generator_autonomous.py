#!/usr/bin/env python3
"""
Autonomous Story Generator - Continuously processes pending topics
Based on 1_story_generator_claude_server.py but runs autonomously
Location: src/generators/1_story_generator_autonomous.py
"""

import os
import sys
import time
import signal
from pathlib import Path
from datetime import datetime

# Add paths for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root / 'src' / 'database'))
sys.path.append(str(current_dir))

# Import the autonomous database manager
try:
    from autonomous_database_manager import AutonomousDatabaseManager

    print("✅ Database manager imported")
except ImportError as e:
    print(f"❌ Failed to import database manager: {e}")
    print(f"Expected path: {project_root / 'src' / 'database' / 'autonomous_database_manager.py'}")
    sys.exit(1)

# Check if we can import the real story generator
try:
    # Try to import the actual story generator
    sys.path.append(str(current_dir))
    exec(open(current_dir / '1_story_generator_claude_server.py').read())
    from autonomous_story_generator_core import AutomatedStoryGenerator, CONFIG

    REAL_GENERATOR_AVAILABLE = True
    print("✅ Real story generator imported")
except Exception as e:
    print(f"⚠️ Real story generator not available: {e}")
    print("🔄 Using simplified simulation mode")
    REAL_GENERATOR_AVAILABLE = False


class AutonomousStoryGenerator:
    """Autonomous story generator that continuously processes topics"""

    def __init__(self):
        self.db_manager = AutonomousDatabaseManager()
        self.running = True
        self.processed_count = 0
        self.start_time = datetime.now()

        # Initialize real generator if available
        if REAL_GENERATOR_AVAILABLE:
            self.real_generator = AutomatedStoryGenerator()
            print("✅ Real story generator initialized")
        else:
            self.real_generator = None
            print("🔄 Using simulation mode")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("🤖 Autonomous Story Generator initialized")
        print("🔄 Press Ctrl+C to stop gracefully")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n⏹️ Received shutdown signal ({signum})")
        print("🔄 Finishing current topic and shutting down...")
        self.running = False

    def process_single_topic(self, topic_id: int, topic: str, description: str) -> bool:
        """Process a single topic using the story generation logic"""
        try:
            print(f"🔄 Processing topic {topic_id}: {topic}")

            # Mark as started
            self.db_manager.mark_story_generation_started(topic_id)

            if REAL_GENERATOR_AVAILABLE and self.real_generator:
                # Use real story generator
                print(f"📝 Running REAL story generation for: {topic}")

                # Get topic details from database
                clickbait_title = ""
                font_design = ""

                # Generate using real generator
                result = self.real_generator.generate_complete_story_with_characters(
                    topic, description, clickbait_title, font_design
                )

                # Create output directory
                output_dir = Path("output") / str(topic_id)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save outputs (simplified version of save_production_outputs)
                import json

                # Save complete story
                story_file = output_dir / "complete_story.txt"
                with open(story_file, 'w', encoding='utf-8') as f:
                    f.write(result.get("complete_story", ""))

                # Save scene plan
                scene_file = output_dir / "scene_plan.json"
                with open(scene_file, 'w', encoding='utf-8') as f:
                    json.dump(result.get("scene_plan", []), f, indent=2)

                # Save visual prompts
                visual_file = output_dir / "visual_generation_prompts.json"
                with open(visual_file, 'w', encoding='utf-8') as f:
                    json.dump(result.get("visual_prompts", []), f, indent=2)

                # Mark as completed with real stats
                stats = result.get("generation_stats", {})
                self.db_manager.mark_story_generation_completed(
                    topic_id=topic_id,
                    output_path=str(output_dir),
                    scene_count=stats.get("scenes_planned", 0),
                    api_calls=stats.get("api_calls_used", 0),
                    cost=result.get("total_cost", 0.0)
                )

                print(f"✅ REAL story generation completed for topic {topic_id}")
                print(f"   📊 Scenes: {stats.get('scenes_planned', 0)}")
                print(f"   💰 Cost: ${result.get('total_cost', 0.0):.4f}")

            else:
                # Simulation mode
                print(f"📝 SIMULATING story generation for: {topic}")

                # Create output directory
                output_dir = Path("output") / str(topic_id)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Simulate story generation time
                print("⏳ Generating story... (simulated)")
                time.sleep(10)  # Simulate processing time

                # Create simulated story file
                story_file = output_dir / "complete_story.txt"
                with open(story_file, 'w') as f:
                    f.write(f"SIMULATED story for: {topic}\nDescription: {description}\nGenerated at: {datetime.now()}")

                # Mark as completed with simulated stats
                self.db_manager.mark_story_generation_completed(
                    topic_id=topic_id,
                    output_path=str(output_dir),
                    scene_count=30,  # Simulated
                    api_calls=5,  # Simulated
                    cost=0.15  # Simulated
                )

                print(f"✅ SIMULATED story generation completed for topic {topic_id}")

            return True

        except Exception as e:
            print(f"❌ Error processing topic {topic_id}: {e}")
            self.db_manager.mark_story_generation_failed(topic_id, str(e))
            import traceback
            traceback.print_exc()
            return False

    def run_autonomous_loop(self, check_interval: int = 60):
        """Main autonomous loop"""
        print("🚀 Starting autonomous story generation loop")
        print(f"⏱️ Check interval: {check_interval} seconds")
        print(f"🤖 Mode: {'REAL GENERATION' if REAL_GENERATOR_AVAILABLE else 'SIMULATION'}")
        print("=" * 60)

        while self.running:
            try:
                # Get next topic ready for story generation
                next_topic = self.db_manager.get_next_story_generation_topic()

                if next_topic:
                    topic_id, topic, description = next_topic

                    # Process the topic
                    success = self.process_single_topic(topic_id, topic, description)

                    if success:
                        self.processed_count += 1
                        print(f"📊 Progress: {self.processed_count} topics processed")

                    # Short pause between topics
                    if self.running:
                        time.sleep(5)

                else:
                    # No topics ready, wait and check again
                    print(f"😴 No topics ready for story generation. Waiting {check_interval}s...")

                    # Wait with periodic checks for shutdown signal
                    for i in range(check_interval):
                        if not self.running:
                            break
                        time.sleep(1)

                # Print status every few iterations
                if self.processed_count > 0 and self.processed_count % 5 == 0:
                    self.print_status_summary()

            except KeyboardInterrupt:
                print("\n⏹️ Keyboard interrupt received")
                break
            except Exception as e:
                print(f"❌ Unexpected error in autonomous loop: {e}")
                print("⏳ Waiting 30 seconds before retry...")
                time.sleep(30)

        self.shutdown()

    def print_status_summary(self):
        """Print current status summary"""
        runtime = datetime.now() - self.start_time
        status = self.db_manager.get_pipeline_status()

        print(f"\n📊 AUTONOMOUS STORY GENERATOR STATUS")
        print(f"⏱️ Runtime: {runtime}")
        print(f"✅ Topics processed: {self.processed_count}")
        print(f"📝 Topics in queue: {status['story_generation_queue']}")
        print(f"🔄 Active processing: {status['story_generation_active']}")
        print(f"🤖 Mode: {'REAL' if REAL_GENERATOR_AVAILABLE else 'SIMULATION'}")
        print("=" * 40)

    def shutdown(self):
        """Graceful shutdown"""
        print(f"\n🏁 AUTONOMOUS STORY GENERATOR SHUTDOWN")
        print(f"⏱️ Total runtime: {datetime.now() - self.start_time}")
        print(f"✅ Topics processed: {self.processed_count}")
        print("👋 Goodbye!")


def test_autonomous_mode():
    """Test autonomous mode with database manager"""
    print("🧪 Testing Autonomous Story Generator")
    print("=" * 50)

    try:
        # Test database manager first
        db_manager = AutonomousDatabaseManager()
        status = db_manager.get_pipeline_status()

        print("📊 Current pipeline status:")
        print(f"   📝 Story generation queue: {status['story_generation_queue']}")

        if status['story_generation_queue'] > 0:
            print("✅ Topics available for processing")

            # Get next topic
            next_topic = db_manager.get_next_story_generation_topic()
            if next_topic:
                topic_id, topic, description = next_topic
                print(f"🎯 Next topic: ID {topic_id} - {topic}")
                print(f"📝 Description: {description}")

                return True
            else:
                print("❌ No topic returned despite queue count > 0")
                return False
        else:
            print("⚠️ No topics in story generation queue")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main function"""
    print("🤖 AUTONOMOUS STORY GENERATOR")
    print("🔄 Continuously processes pending topics from database")
    print("=" * 60)

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test mode
        success = test_autonomous_mode()
        if success:
            print("✅ Test passed - ready for autonomous operation")
        else:
            print("❌ Test failed - check database and configuration")
        return

    try:
        # Initialize autonomous generator
        generator = AutonomousStoryGenerator()

        # Check if there are topics to process
        status = generator.db_manager.get_pipeline_status()

        if status['story_generation_queue'] == 0:
            print("⚠️ No topics in story generation queue")
            print("💡 Add topics to database first:")
            print(
                "   sqlite3 data/production.db \"INSERT INTO topics (topic, description, status) VALUES ('Test Topic', 'Test Description', 'pending');\"")
            return

        print(f"🎯 Found {status['story_generation_queue']} topics ready for story generation")

        # Start autonomous processing
        generator.run_autonomous_loop(check_interval=60)

    except KeyboardInterrupt:
        print("\n⏹️ Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()