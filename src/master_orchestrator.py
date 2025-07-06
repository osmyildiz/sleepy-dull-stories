"""
Sleepy Dull Stories - MASTER ORCHESTRATOR
Complete Autonomous YouTube Content Factory
Production-optimized with complete automation and database integration
"""

import os
import sys
import json
import time
import sqlite3
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dotenv import load_dotenv
import threading

# Load environment first
load_dotenv()


class PipelineStage(Enum):
    """Pipeline stages in execution order"""
    STORY_GENERATION = "story_generation"
    CHARACTER_GENERATION = "character_generation"
    SCENE_GENERATION = "scene_generation"
    AUDIO_GENERATION = "audio_generation"
    VIDEO_COMPOSITION = "video_composition"
    THUMBNAIL_CREATION = "thumbnail_creation"
    YOUTUBE_UPLOAD = "youtube_upload"


class ExecutionMode(Enum):
    """Orchestrator execution modes"""
    SINGLE_TOPIC = "single"
    BATCH_PROCESSING = "batch"
    CONTINUOUS = "continuous"
    AUTO_DETECT = "auto"


class StageStatus(Enum):
    """Stage execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# Server Configuration for Master Orchestrator
class OrchestratorConfig:
    """Master Orchestrator configuration management"""

    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.setup_orchestrator_config()
        self.ensure_directories()

    def setup_paths(self):
        """Setup server-friendly paths"""
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent.parent

        self.paths = {
            'BASE_DIR': str(self.project_root),
            'SRC_DIR': str(current_file.parent),
            'GENERATORS_DIR': str(current_file.parent / 'generators'),
            'DATA_DIR': str(self.project_root / 'data'),
            'OUTPUT_DIR': str(self.project_root / 'output'),
            'LOGS_DIR': str(self.project_root / 'logs'),
            'CONFIG_DIR': str(self.project_root / 'config')
        }

        print(f"✅ Master Orchestrator paths configured:")
        print(f"   📁 Project root: {self.paths['BASE_DIR']}")
        print(f"   🔧 Generators: {self.paths['GENERATORS_DIR']}")

    def setup_orchestrator_config(self):
        """Setup orchestrator configuration"""
        self.orchestrator_config = {
            "max_retry_attempts": 3,
            "stage_timeout_minutes": 60,
            "inter_stage_delay_seconds": 5,
            "health_check_interval_seconds": 30,
            "auto_detect_interval_minutes": 10,
            "batch_size": 5,
            "budget_controls": {
                "max_concurrent_productions": 3,
                "max_daily_productions": 20,
                "max_session_duration_hours": 8
            },
            "stage_scripts": {
                PipelineStage.STORY_GENERATION.value: "1_story_generator_server.py",
                PipelineStage.CHARACTER_GENERATION.value: "2_character_generator_server.py",
                PipelineStage.SCENE_GENERATION.value: "3_scene_generator_server.py",
                PipelineStage.AUDIO_GENERATION.value: "4_tts_generator_server.py",
                PipelineStage.VIDEO_COMPOSITION.value: "6_video_composer_final_server.py",
                PipelineStage.THUMBNAIL_CREATION.value: "5_thumbnail_create_server.py",
                PipelineStage.YOUTUBE_UPLOAD.value: "7_youtube_uploader_server.py"
            },
            "dependency_requirements": {
                PipelineStage.STORY_GENERATION.value: [],
                PipelineStage.CHARACTER_GENERATION.value: ["story_generation"],
                PipelineStage.SCENE_GENERATION.value: ["character_generation"],
                PipelineStage.AUDIO_GENERATION.value: ["scene_generation"],
                PipelineStage.VIDEO_COMPOSITION.value: ["audio_generation"],
                PipelineStage.THUMBNAIL_CREATION.value: ["scene_generation"],  # Can run parallel with audio/video
                PipelineStage.YOUTUBE_UPLOAD.value: ["video_composition", "thumbnail_creation"]
            },
            "server_mode": True,
            "production_ready": True
        }

        print("✅ Master Orchestrator configuration loaded")
        print(f"🎬 Pipeline stages: {len(self.orchestrator_config['stage_scripts'])}")

    def setup_logging(self):
        """Setup comprehensive logging"""
        logs_dir = Path(self.project_root) / 'logs' / 'orchestrator'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"master_orchestrator_{datetime.now().strftime('%Y%m%d')}.log"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("MasterOrchestrator")
        self.logger.info(f"✅ Master Orchestrator logging initialized: {log_file}")

    def ensure_directories(self):
        """Ensure all required directories exist"""
        dirs_to_create = [
            'DATA_DIR', 'OUTPUT_DIR', 'LOGS_DIR', 'CONFIG_DIR'
        ]

        for dir_key in dirs_to_create:
            dir_path = Path(self.paths[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)

        print("✅ All orchestrator directories created/verified")


# Initialize config
try:
    CONFIG = OrchestratorConfig()
    print("🚀 Master Orchestrator configuration loaded successfully")
except Exception as e:
    print(f"❌ Master Orchestrator configuration failed: {e}")
    sys.exit(1)


class DatabaseOrchestrationManager:
    """Professional orchestration management using production.db"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            self.db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
        else:
            self.db_path = Path(db_path)

    def get_topics_by_status(self, status: str = None) -> List[Dict]:
        """Get topics filtered by status"""
        if not self.db_path.exists():
            print(f"❌ Database not found: {self.db_path}")
            return []

        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if status:
                    cursor.execute("SELECT * FROM topics WHERE status = ? ORDER BY updated_at ASC", (status,))
                else:
                    cursor.execute("SELECT * FROM topics ORDER BY updated_at ASC")

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            print(f"❌ Database query error: {e}")
            return []

    def get_next_ready_topic_for_stage(self, stage: PipelineStage) -> Optional[Dict]:
        """Get next topic ready for specific stage"""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Define stage readiness conditions
                stage_conditions = {
                    PipelineStage.STORY_GENERATION: "status = 'pending'",
                    PipelineStage.CHARACTER_GENERATION: "story_generation_status = 'completed' AND (character_generation_status IS NULL OR character_generation_status = 'pending')",
                    PipelineStage.SCENE_GENERATION: "character_generation_status = 'completed' AND (scene_generation_status IS NULL OR scene_generation_status = 'pending')",
                    PipelineStage.AUDIO_GENERATION: "scene_generation_status = 'completed' AND (audio_generation_status IS NULL OR audio_generation_status = 'pending')",
                    PipelineStage.VIDEO_COMPOSITION: "audio_generation_status = 'completed' AND (video_generation_status IS NULL OR video_generation_status = 'pending')",
                    PipelineStage.THUMBNAIL_CREATION: "scene_generation_status = 'completed' AND (thumbnail_generation_status IS NULL OR thumbnail_generation_status = 'pending')",
                    PipelineStage.YOUTUBE_UPLOAD: "video_generation_status = 'completed' AND thumbnail_generation_status = 'completed' AND (youtube_upload_status IS NULL OR youtube_upload_status = 'pending')"
                }

                condition = stage_conditions.get(stage)
                if not condition:
                    return None

                query = f"SELECT * FROM topics WHERE {condition} ORDER BY updated_at ASC LIMIT 1"
                cursor.execute(query)

                result = cursor.fetchone()
                return dict(result) if result else None

        except Exception as e:
            print(f"❌ Error finding ready topic for {stage.value}: {e}")
            return None

    def mark_orchestration_started(self, topic_id: int, stage: PipelineStage):
        """Mark orchestration stage as started"""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()

                # Add orchestration columns if they don't exist
                cursor.execute('PRAGMA table_info(topics)')
                columns = [row[1] for row in cursor.fetchall()]

                if 'orchestration_status' not in columns:
                    cursor.execute('ALTER TABLE topics ADD COLUMN orchestration_status TEXT DEFAULT "pending"')
                if 'current_orchestration_stage' not in columns:
                    cursor.execute('ALTER TABLE topics ADD COLUMN current_orchestration_stage TEXT')
                if 'orchestration_started_at' not in columns:
                    cursor.execute('ALTER TABLE topics ADD COLUMN orchestration_started_at DATETIME')

                # Update orchestration status
                cursor.execute('''
                    UPDATE topics 
                    SET orchestration_status = 'in_progress',
                        current_orchestration_stage = ?,
                        orchestration_started_at = COALESCE(orchestration_started_at, CURRENT_TIMESTAMP),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (stage.value, topic_id))

                conn.commit()

        except Exception as e:
            print(f"❌ Error marking orchestration started: {e}")

    def mark_orchestration_completed(self, topic_id: int):
        """Mark orchestration as completed"""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    UPDATE topics 
                    SET orchestration_status = 'completed',
                        current_orchestration_stage = 'completed',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (topic_id,))

                conn.commit()

        except Exception as e:
            print(f"❌ Error marking orchestration completed: {e}")

    def get_orchestration_status(self) -> Dict:
        """Get comprehensive orchestration status"""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get stage counts
                stage_counts = {}
                for stage in PipelineStage:
                    if stage == PipelineStage.STORY_GENERATION:
                        cursor.execute("SELECT COUNT(*) FROM topics WHERE status = 'pending'")
                    elif stage == PipelineStage.CHARACTER_GENERATION:
                        cursor.execute(
                            "SELECT COUNT(*) FROM topics WHERE story_generation_status = 'completed' AND (character_generation_status IS NULL OR character_generation_status = 'pending')")
                    elif stage == PipelineStage.SCENE_GENERATION:
                        cursor.execute(
                            "SELECT COUNT(*) FROM topics WHERE character_generation_status = 'completed' AND (scene_generation_status IS NULL OR scene_generation_status = 'pending')")
                    elif stage == PipelineStage.AUDIO_GENERATION:
                        cursor.execute(
                            "SELECT COUNT(*) FROM topics WHERE scene_generation_status = 'completed' AND (audio_generation_status IS NULL OR audio_generation_status = 'pending')")
                    elif stage == PipelineStage.VIDEO_COMPOSITION:
                        cursor.execute(
                            "SELECT COUNT(*) FROM topics WHERE audio_generation_status = 'completed' AND (video_generation_status IS NULL OR video_generation_status = 'pending')")
                    elif stage == PipelineStage.THUMBNAIL_CREATION:
                        cursor.execute(
                            "SELECT COUNT(*) FROM topics WHERE scene_generation_status = 'completed' AND (thumbnail_generation_status IS NULL OR thumbnail_generation_status = 'pending')")
                    elif stage == PipelineStage.YOUTUBE_UPLOAD:
                        cursor.execute(
                            "SELECT COUNT(*) FROM topics WHERE video_generation_status = 'completed' AND thumbnail_generation_status = 'completed' AND (youtube_upload_status IS NULL OR youtube_upload_status = 'pending')")

                    stage_counts[stage.value] = cursor.fetchone()[0]

                # Get active orchestrations
                cursor.execute("SELECT COUNT(*) FROM topics WHERE orchestration_status = 'in_progress'")
                active_orchestrations = cursor.fetchone()[0]

                # Get completed today
                cursor.execute(
                    "SELECT COUNT(*) FROM topics WHERE DATE(updated_at) = DATE('now') AND orchestration_status = 'completed'")
                completed_today = cursor.fetchone()[0]

                return {
                    "stage_queues": stage_counts,
                    "active_orchestrations": active_orchestrations,
                    "completed_today": completed_today,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            print(f"❌ Error getting orchestration status: {e}")
            return {}


class PipelineExecutor:
    """Execute individual pipeline stages"""

    def __init__(self):
        self.generators_dir = Path(CONFIG.paths['GENERATORS_DIR'])
        self.stage_scripts = CONFIG.orchestrator_config['stage_scripts']

    def execute_stage(self, stage: PipelineStage, timeout_minutes: int = 60) -> Tuple[bool, str, Dict]:
        """Execute a single pipeline stage"""
        script_name = self.stage_scripts.get(stage.value)
        if not script_name:
            return False, f"No script defined for stage {stage.value}", {}

        script_path = self.generators_dir / script_name
        if not script_path.exists():
            return False, f"Script not found: {script_path}", {}

        start_time = time.time()

        try:
            print(f"🔄 Executing {stage.value}: {script_name}")

            # Execute script with timeout
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.generators_dir),
                capture_output=True,
                text=True,
                timeout=timeout_minutes * 60
            )

            execution_time = time.time() - start_time

            success = result.returncode == 0

            if success:
                print(f"✅ {stage.value} completed successfully ({execution_time:.1f}s)")
            else:
                print(f"❌ {stage.value} failed (exit code: {result.returncode})")
                print(f"Error output: {result.stderr[:500]}...")

            return success, result.stderr if not success else "", {
                "execution_time_seconds": execution_time,
                "exit_code": result.returncode,
                "stdout_length": len(result.stdout),
                "stderr_length": len(result.stderr)
            }

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return False, f"Stage timeout after {timeout_minutes} minutes", {
                "execution_time_seconds": execution_time,
                "timeout": True
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return False, str(e), {
                "execution_time_seconds": execution_time,
                "exception": str(e)
            }

    def check_stage_dependencies(self, stage: PipelineStage, topic_data: Dict) -> Tuple[bool, List[str]]:
        """Check if stage dependencies are satisfied"""
        dependencies = CONFIG.orchestrator_config['dependency_requirements'].get(stage.value, [])

        missing_deps = []

        for dep_stage in dependencies:
            dep_status_column = f"{dep_stage}_status"

            if dep_status_column not in topic_data:
                missing_deps.append(f"{dep_stage} (column not found)")
            elif topic_data[dep_status_column] != 'completed':
                missing_deps.append(f"{dep_stage} (status: {topic_data.get(dep_status_column, 'unknown')})")

        return len(missing_deps) == 0, missing_deps


class MasterOrchestrator:
    """Master Orchestrator - Complete Autonomous YouTube Content Factory"""

    def __init__(self):
        self.db_manager = DatabaseOrchestrationManager()
        self.executor = PipelineExecutor()
        self.session_start = datetime.now()
        self.active_productions = {}
        self.stats = {
            "total_productions_started": 0,
            "total_productions_completed": 0,
            "total_productions_failed": 0,
            "total_stages_executed": 0,
            "total_execution_time": 0.0
        }

        print("🎬 Master Orchestrator v1.0 Initialized")
        print(f"📊 Database: {self.db_manager.db_path}")

    def log_step(self, message: str, level: str = "INFO"):
        """Log orchestration steps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icon = "🔄" if level == "INFO" else "✅" if level == "SUCCESS" else "❌" if level == "ERROR" else "⚠️"
        print(f"{timestamp} {icon} {message}")
        CONFIG.logger.info(f"{message}")

    def check_budget_limits(self) -> Tuple[bool, str]:
        """Check orchestrator budget limits"""
        budget = CONFIG.orchestrator_config.get("budget_controls", {})

        # Check session duration
        session_duration_hours = (datetime.now() - self.session_start).total_seconds() / 3600
        max_session_hours = budget.get("max_session_duration_hours", 8)

        if session_duration_hours >= max_session_hours:
            return False, f"Session duration limit exceeded: {session_duration_hours:.1f}h >= {max_session_hours}h"

        # Check daily productions
        if self.stats["total_productions_completed"] >= budget.get("max_daily_productions", 20):
            return False, f"Daily production limit exceeded: {self.stats['total_productions_completed']}"

        # Check concurrent productions
        if len(self.active_productions) >= budget.get("max_concurrent_productions", 3):
            return False, f"Concurrent production limit exceeded: {len(self.active_productions)}"

        return True, "OK"

    def execute_single_topic(self, topic_id: int) -> bool:
        """Execute complete pipeline for a single topic"""

        self.log_step(f"🎯 Starting pipeline execution for topic {topic_id}")

        # Get topic data
        topics = self.db_manager.get_topics_by_status()
        topic_data = next((t for t in topics if t['id'] == topic_id), None)

        if not topic_data:
            self.log_step(f"❌ Topic {topic_id} not found", "ERROR")
            return False

        self.log_step(f"📚 Topic: {topic_data.get('topic', 'Unknown')}")

        # Track production
        self.active_productions[topic_id] = {
            "topic": topic_data.get('topic', 'Unknown'),
            "started_at": datetime.now(),
            "current_stage": None,
            "stages_completed": []
        }

        self.stats["total_productions_started"] += 1
        overall_success = True

        # Execute pipeline stages in order
        for stage in PipelineStage:

            # Check budget limits
            can_continue, budget_reason = self.check_budget_limits()
            if not can_continue:
                self.log_step(f"🚨 Budget limit: {budget_reason}", "ERROR")
                overall_success = False
                break

            # Update current stage
            self.active_productions[topic_id]["current_stage"] = stage.value
            self.db_manager.mark_orchestration_started(topic_id, stage)

            # Check if topic is ready for this stage
            current_topic = self.db_manager.get_next_ready_topic_for_stage(stage)
            if not current_topic or current_topic['id'] != topic_id:
                self.log_step(f"⏭️  Skipping {stage.value} - topic not ready or already completed")
                continue

            # Check dependencies
            deps_satisfied, missing_deps = self.executor.check_stage_dependencies(stage, current_topic)
            if not deps_satisfied:
                self.log_step(f"⚠️  {stage.value} dependencies not satisfied: {missing_deps}", "ERROR")
                overall_success = False
                break

            # Execute stage
            self.log_step(f"🚀 Executing {stage.value}")

            max_retries = CONFIG.orchestrator_config.get("max_retry_attempts", 3)
            stage_success = False

            for attempt in range(max_retries):
                if attempt > 0:
                    self.log_step(f"🔄 Retry {attempt}/{max_retries - 1} for {stage.value}")
                    time.sleep(CONFIG.orchestrator_config.get("inter_stage_delay_seconds", 5))

                success, error_msg, execution_data = self.executor.execute_stage(
                    stage, CONFIG.orchestrator_config.get("stage_timeout_minutes", 60)
                )

                self.stats["total_stages_executed"] += 1
                self.stats["total_execution_time"] += execution_data.get("execution_time_seconds", 0)

                if success:
                    stage_success = True
                    self.active_productions[topic_id]["stages_completed"].append(stage.value)
                    break
                else:
                    self.log_step(f"❌ {stage.value} attempt {attempt + 1} failed: {error_msg}", "ERROR")

            if not stage_success:
                self.log_step(f"💥 {stage.value} failed after {max_retries} attempts", "ERROR")
                overall_success = False
                break

            # Inter-stage delay
            delay = CONFIG.orchestrator_config.get("inter_stage_delay_seconds", 5)
            if delay > 0:
                time.sleep(delay)

        # Complete orchestration
        if overall_success:
            self.db_manager.mark_orchestration_completed(topic_id)
            self.stats["total_productions_completed"] += 1
            self.log_step(f"🎉 Pipeline completed successfully for topic {topic_id}", "SUCCESS")
        else:
            self.stats["total_productions_failed"] += 1
            self.log_step(f"💥 Pipeline failed for topic {topic_id}", "ERROR")

        # Remove from active productions
        if topic_id in self.active_productions:
            del self.active_productions[topic_id]

        return overall_success

    def execute_auto_mode(self):
        """Auto-detect and execute ready topics continuously"""

        self.log_step("🤖 Starting auto-detect mode")

        check_interval = CONFIG.orchestrator_config.get("auto_detect_interval_minutes", 10) * 60

        while True:
            try:
                # Check budget limits
                can_continue, budget_reason = self.check_budget_limits()
                if not can_continue:
                    self.log_step(f"🚨 Budget limit reached: {budget_reason}")
                    break

                # Find topics ready for any stage
                topics_processed = 0

                for stage in PipelineStage:
                    ready_topic = self.db_manager.get_next_ready_topic_for_stage(stage)

                    if ready_topic:
                        topic_id = ready_topic['id']
                        topic_title = ready_topic.get('topic', 'Unknown')

                        if topic_id not in self.active_productions:
                            self.log_step(f"🎯 Auto-detected topic {topic_id} ready for {stage.value}: {topic_title}")

                            success = self.execute_single_topic(topic_id)
                            topics_processed += 1

                            if topics_processed >= CONFIG.orchestrator_config.get("batch_size", 5):
                                break

                if topics_processed == 0:
                    self.log_step(f"😴 No topics ready for processing. Waiting {check_interval / 60:.1f} minutes...")
                    time.sleep(check_interval)
                else:
                    self.log_step(f"✅ Processed {topics_processed} topics in this cycle")

            except KeyboardInterrupt:
                self.log_step("⏹️  Auto-mode stopped by user")
                break
            except Exception as e:
                self.log_step(f"❌ Auto-mode error: {e}", "ERROR")
                time.sleep(60)  # Wait before retrying

    def execute_batch_mode(self, max_topics: int = 5):
        """Execute batch of ready topics"""

        self.log_step(f"📦 Starting batch mode (max {max_topics} topics)")

        processed = 0

        for stage in PipelineStage:
            if processed >= max_topics:
                break

            ready_topics = []

            # Find multiple topics ready for this stage
            for _ in range(max_topics - processed):
                topic = self.db_manager.get_next_ready_topic_for_stage(stage)
                if topic and topic['id'] not in [t['id'] for t in ready_topics]:
                    ready_topics.append(topic)
                else:
                    break

            # Process found topics
            for topic in ready_topics:
                if processed >= max_topics:
                    break

                topic_id = topic['id']
                if topic_id not in self.active_productions:
                    self.log_step(f"📦 Batch processing topic {topic_id}: {topic.get('topic', 'Unknown')}")

                    success = self.execute_single_topic(topic_id)
                    processed += 1

        self.log_step(f"📦 Batch processing completed: {processed} topics processed")

    def print_status_summary(self):
        """Print comprehensive status summary"""

        # Get orchestration status
        status = self.db_manager.get_orchestration_status()

        session_duration = (datetime.now() - self.session_start).total_seconds() / 3600

        print(f"\n🎬 MASTER ORCHESTRATOR STATUS SUMMARY")
        print(f"=" * 60)
        print(f"⏱️  Session duration: {session_duration:.1f} hours")
        print(f"🎯 Productions started: {self.stats['total_productions_started']}")
        print(f"✅ Productions completed: {self.stats['total_productions_completed']}")
        print(f"❌ Productions failed: {self.stats['total_productions_failed']}")
        print(f"🔄 Total stages executed: {self.stats['total_stages_executed']}")
        print(f"⚡ Total execution time: {self.stats['total_execution_time'] / 60:.1f} minutes")

        if status.get("stage_queues"):
            print(f"\n📊 STAGE QUEUES:")
            for stage, count in status["stage_queues"].items():
                print(f"   {stage}: {count} topics ready")

        if self.active_productions:
            print(f"\n🔄 ACTIVE PRODUCTIONS:")
            for topic_id, prod in self.active_productions.items():
                print(f"   Topic {topic_id}: {prod['topic']} ({prod['current_stage']})")

        # Calculate success rate
        total_completed = self.stats['total_productions_completed'] + self.stats['total_productions_failed']
        if total_completed > 0:
            success_rate = (self.stats['total_productions_completed'] / total_completed) * 100
            print(f"\n📈 Success rate: {success_rate:.1f}%")

    def run_orchestrator(self, mode: ExecutionMode, topic_id: int = None, max_topics: int = 5):
        """Main orchestrator execution function"""

        print("🚀" * 60)
        print("MASTER ORCHESTRATOR - AUTONOMOUS YOUTUBE CONTENT FACTORY")
        print("🎬 Complete end-to-end automation system")
        print("🔗 Database integrated with comprehensive tracking")
        print("🚀" * 60)

        self.log_step(f"🎬 Master Orchestrator starting in {mode.value} mode")

        try:
            if mode == ExecutionMode.SINGLE_TOPIC:
                if not topic_id:
                    self.log_step("❌ Topic ID required for single topic mode", "ERROR")
                    return False
                return self.execute_single_topic(topic_id)

            elif mode == ExecutionMode.BATCH_PROCESSING:
                self.execute_batch_mode(max_topics)
                return True

            elif mode == ExecutionMode.CONTINUOUS or mode == ExecutionMode.AUTO_DETECT:
                self.execute_auto_mode()
                return True

            else:
                self.log_step(f"❌ Unknown execution mode: {mode.value}", "ERROR")
                return False

        except KeyboardInterrupt:
            self.log_step("⏹️  Orchestrator stopped by user")
            return True
        except Exception as e:
            self.log_step(f"💥 Orchestrator error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.print_status_summary()


def main():
    """Main function for Master Orchestrator"""

    parser = argparse.ArgumentParser(description="Master Orchestrator - Autonomous YouTube Content Factory")

    parser.add_argument('--mode',
                        choices=['single', 'batch', 'continuous', 'auto'],
                        default='auto',
                        help='Execution mode (default: auto)')

    parser.add_argument('--topic-id',
                        type=int,
                        help='Topic ID for single topic mode')

    parser.add_argument('--max-topics',
                        type=int,
                        default=5,
                        help='Maximum topics to process in batch mode (default: 5)')

    parser.add_argument('--status',
                        action='store_true',
                        help='Show orchestration status and exit')

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = MasterOrchestrator()

    # Handle status command
    if args.status:
        orchestrator.print_status_summary()
        return

    # Map mode string to enum
    mode_map = {
        'single': ExecutionMode.SINGLE_TOPIC,
        'batch': ExecutionMode.BATCH_PROCESSING,
        'continuous': ExecutionMode.CONTINUOUS,
        'auto': ExecutionMode.AUTO_DETECT
    }

    mode = mode_map[args.mode]

    print(f"\n🎯 EXECUTION PARAMETERS:")
    print(f"   Mode: {mode.value}")
    if args.topic_id:
        print(f"   Topic ID: {args.topic_id}")
    if args.mode == 'batch':
        print(f"   Max topics: {args.max_topics}")

    print(f"\n🚀 Starting Master Orchestrator...")

    # Run orchestrator
    success = orchestrator.run_orchestrator(mode, args.topic_id, args.max_topics)

    if success:
        print(f"\n🎉 Master Orchestrator completed successfully!")
        print(f"🎬 Autonomous YouTube Content Factory operational!")
    else:
        print(f"\n❌ Master Orchestrator encountered errors")
        print(f"📊 Check logs for details")


if __name__ == "__main__":
    main()