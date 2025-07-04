"""
Professional Production Status Tracker
Location: src/database/production_status_tracker.py
"""

import sqlite3
import json
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ProductionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


class StageStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ProductionStage(Enum):
    STORY_GENERATION = "story_generation"
    CHARACTER_EXTRACTION = "character_extraction"
    VISUAL_GENERATION = "visual_generation"
    TTS_GENERATION = "tts_generation"
    VIDEO_COMPOSITION = "video_composition"
    YOUTUBE_UPLOAD = "youtube_upload"


@dataclass
class QualityGateResult:
    passed: bool
    score: float
    threshold: float
    criteria_checked: Dict
    failure_reasons: List[str]
    recommendations: List[str]


class ProductionStatusTracker:
    """Professional production status tracking system"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.db_path = project_root / "data" / "production.db"
        else:
            self.db_path = Path(db_path)

        self.current_video_id = None
        self.stages = [stage.value for stage in ProductionStage]

    def create_new_production(self, topic: str, description: str,
                              clickbait_title: str = None, font_design: str = None) -> int:
        """Create new production entry and return video_id"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO video_production 
                (topic, description, clickbait_title, font_design, status, current_stage, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                topic, description, clickbait_title, font_design,
                ProductionStatus.PENDING.value,
                ProductionStage.STORY_GENERATION.value,
                datetime.now()
            ))

            video_id = cursor.lastrowid
            self.current_video_id = video_id

            print(f"🎬 New production created: ID {video_id}")
            print(f"📚 Topic: {topic}")

            return video_id

    def start_production(self, video_id: int):
        """Start production tracking"""

        self.current_video_id = video_id

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE video_production 
                SET status = ?, started_at = ?, updated_at = ?
                WHERE id = ?
            """, (ProductionStatus.IN_PROGRESS.value, datetime.now(), datetime.now(), video_id))

            conn.commit()

        print(f"🚀 Production {video_id} started")
        self.log_system_metrics()

    def start_stage(self, stage: ProductionStage) -> int:
        """Start a production stage"""

        if not self.current_video_id:
            raise ValueError("No active video production")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Update current stage in video_production
            cursor.execute("""
                UPDATE video_production 
                SET current_stage = ?, updated_at = ?
                WHERE id = ?
            """, (stage.value, datetime.now(), self.current_video_id))

            # Update stage status
            stage_column = f"{stage.value}_status"
            cursor.execute(f"""
                UPDATE video_production 
                SET {stage_column} = ?
                WHERE id = ?
            """, (StageStatus.IN_PROGRESS.value, self.current_video_id))

            # Create stage execution entry
            cursor.execute("""
                INSERT INTO stage_execution
                (video_id, stage_name, status, started_at)
                VALUES (?, ?, ?, ?)
            """, (
                self.current_video_id, stage.value,
                StageStatus.IN_PROGRESS.value, datetime.now()
            ))

            stage_execution_id = cursor.lastrowid
            conn.commit()

        print(f"🔄 Stage started: {stage.value}")
        return stage_execution_id

    def complete_stage(self, stage: ProductionStage, stage_execution_id: int,
                       api_calls: int = 0, output_files: List[str] = None,
                       quality_score: float = 0.0, metadata: Dict = None):
        """Complete a production stage"""

        if output_files is None:
            output_files = []

        if metadata is None:
            metadata = {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get stage start time
            cursor.execute("""
                SELECT started_at FROM stage_execution 
                WHERE id = ?
            """, (stage_execution_id,))

            result = cursor.fetchone()
            if result:
                start_time = datetime.fromisoformat(result[0])
                duration = (datetime.now() - start_time).total_seconds()
            else:
                duration = 0

            # Update stage execution
            cursor.execute("""
                UPDATE stage_execution 
                SET status = ?, completed_at = ?, duration_seconds = ?, 
                    api_calls = ?, output_files = ?, quality_score = ?
                WHERE id = ?
            """, (
                StageStatus.COMPLETED.value, datetime.now(), duration,
                api_calls, json.dumps(output_files), quality_score,
                stage_execution_id
            ))

            # Update video production stage status
            stage_column = f"{stage.value}_status"
            cursor.execute(f"""
                UPDATE video_production 
                SET {stage_column} = ?, api_calls_used = api_calls_used + ?, updated_at = ?
                WHERE id = ?
            """, (StageStatus.COMPLETED.value, api_calls, datetime.now(), self.current_video_id))

            # Update metadata if provided
            if metadata:
                self.update_production_metadata(metadata)

            conn.commit()

        print(f"✅ Stage completed: {stage.value} ({duration:.1f}s, {api_calls} API calls)")

    def fail_stage(self, stage: ProductionStage, stage_execution_id: int,
                   error_message: str, error_details: Dict = None):
        """Mark stage as failed"""

        if error_details is None:
            error_details = {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Update stage execution
            cursor.execute("""
                UPDATE stage_execution 
                SET status = ?, completed_at = ?, error_message = ?, error_details = ?
                WHERE id = ?
            """, (
                StageStatus.FAILED.value, datetime.now(),
                error_message, json.dumps(error_details),
                stage_execution_id
            ))

            # Update video production stage status
            stage_column = f"{stage.value}_status"
            cursor.execute(f"""
                UPDATE video_production 
                SET {stage_column} = ?, error_count = error_count + 1, 
                    last_error = ?, updated_at = ?
                WHERE id = ?
            """, (
                StageStatus.FAILED.value, error_message,
                datetime.now(), self.current_video_id
            ))

            conn.commit()

        print(f"❌ Stage failed: {stage.value} - {error_message}")

    def complete_production(self, success: bool, output_directory: str = None,
                            files_generated: List[str] = None):
        """Complete entire production"""

        if files_generated is None:
            files_generated = []

        final_status = ProductionStatus.COMPLETED if success else ProductionStatus.FAILED

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get production start time
            cursor.execute("""
                SELECT started_at FROM video_production WHERE id = ?
            """, (self.current_video_id,))

            result = cursor.fetchone()
            if result and result[0]:
                start_time = datetime.fromisoformat(result[0])
                total_duration = (datetime.now() - start_time).total_seconds()
            else:
                total_duration = 0

            # Update production
            cursor.execute("""
                UPDATE video_production 
                SET status = ?, completed_at = ?, total_duration_seconds = ?,
                    output_directory = ?, files_generated = ?, updated_at = ?
                WHERE id = ?
            """, (
                final_status.value, datetime.now(), total_duration,
                output_directory, json.dumps(files_generated),
                datetime.now(), self.current_video_id
            ))

            conn.commit()

        status_icon = "🎉" if success else "💥"
        status_text = "completed successfully" if success else "failed"
        print(f"{status_icon} Production {self.current_video_id} {status_text} ({total_duration:.1f}s)")

        self.log_system_metrics()

    def update_production_metadata(self, metadata: Dict):
        """Update production metadata"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            update_fields = []
            values = []

            if 'story_completion_rate' in metadata:
                update_fields.append("story_completion_rate = ?")
                values.append(metadata['story_completion_rate'])

            if 'character_count' in metadata:
                update_fields.append("character_count = ?")
                values.append(metadata['character_count'])

            if 'visual_prompts_count' in metadata:
                update_fields.append("visual_prompts_count = ?")
                values.append(metadata['visual_prompts_count'])

            if 'thumbnail_generated' in metadata:
                update_fields.append("thumbnail_generated = ?")
                values.append(metadata['thumbnail_generated'])

            if 'intro_visuals_generated' in metadata:
                update_fields.append("intro_visuals_generated = ?")
                values.append(metadata['intro_visuals_generated'])

            if update_fields:
                update_fields.append("updated_at = ?")
                values.append(datetime.now())
                values.append(self.current_video_id)

                query = f"""
                    UPDATE video_production 
                    SET {', '.join(update_fields)}
                    WHERE id = ?
                """

                cursor.execute(query, values)
                conn.commit()

    def log_quality_gate(self, stage: ProductionStage, gate_type: str,
                         result: QualityGateResult):
        """Log quality gate results"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO quality_gate_results
                (video_id, stage_name, gate_type, passed, score, threshold,
                 criteria_checked, failure_reasons, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.current_video_id, stage.value, gate_type,
                result.passed, result.score, result.threshold,
                json.dumps(result.criteria_checked),
                json.dumps(result.failure_reasons),
                json.dumps(result.recommendations)
            ))

            conn.commit()

        gate_icon = "✅" if result.passed else "❌"
        print(f"{gate_icon} Quality gate: {gate_type} ({result.score:.2f}/{result.threshold:.2f})")

    def log_system_metrics(self):
        """Log current system metrics"""

        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Get production counts
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Active productions
                cursor.execute("""
                    SELECT COUNT(*) FROM video_production 
                    WHERE status = ?
                """, (ProductionStatus.IN_PROGRESS.value,))
                active_count = cursor.fetchone()[0]

                # Completed today
                today = datetime.now().date()
                cursor.execute("""
                    SELECT COUNT(*) FROM video_production 
                    WHERE DATE(completed_at) = ? AND status = ?
                """, (today, ProductionStatus.COMPLETED.value))
                completed_today = cursor.fetchone()[0]

                # Failed today
                cursor.execute("""
                    SELECT COUNT(*) FROM video_production 
                    WHERE DATE(updated_at) = ? AND status = ?
                """, (today, ProductionStatus.FAILED.value))
                failed_today = cursor.fetchone()[0]

                # API calls today
                cursor.execute("""
                    SELECT COALESCE(SUM(api_calls_used), 0) FROM video_production 
                    WHERE DATE(created_at) = ?
                """, (today,))
                api_calls_today = cursor.fetchone()[0]

                # Insert system metrics
                cursor.execute("""
                    INSERT INTO system_metrics
                    (cpu_usage_percent, memory_usage_percent, disk_usage_percent,
                     active_productions, completed_today, failed_today, claude_api_calls_today)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    cpu_percent, memory.percent, disk.percent,
                    active_count, completed_today, failed_today, api_calls_today
                ))

                conn.commit()

        except Exception as e:
            print(f"⚠️ Could not log system metrics: {e}")

    def get_production_status(self, video_id: int = None) -> Dict:
        """Get comprehensive production status"""

        if video_id is None:
            video_id = self.current_video_id

        if not video_id:
            return {}

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get main production info
            cursor.execute("""
                SELECT * FROM video_production WHERE id = ?
            """, (video_id,))

            production = cursor.fetchone()
            if not production:
                return {}

            # Get stage executions
            cursor.execute("""
                SELECT * FROM stage_execution 
                WHERE video_id = ? 
                ORDER BY created_at
            """, (video_id,))

            stages = cursor.fetchall()

            # Get quality gates
            cursor.execute("""
                SELECT * FROM quality_gate_results 
                WHERE video_id = ? 
                ORDER BY checked_at
            """, (video_id,))

            quality_gates = cursor.fetchall()

            # Build comprehensive status
            status = {
                "video_id": video_id,
                "topic": production["topic"],
                "description": production["description"],
                "status": production["status"],
                "current_stage": production["current_stage"],

                "progress": self._calculate_progress(production),
                "estimated_completion": self._estimate_completion(production, stages),

                "stages": {
                    stage.value: {
                        "status": production[f"{stage.value}_status"],
                        "details": self._get_stage_details(stage.value, stages)
                    }
                    for stage in ProductionStage
                },

                "quality_metrics": {
                    "story_completion_rate": production["story_completion_rate"],
                    "character_count": production["character_count"],
                    "visual_prompts_count": production["visual_prompts_count"],
                    "thumbnail_generated": bool(production["thumbnail_generated"]),
                    "intro_visuals_generated": bool(production["intro_visuals_generated"])
                },

                "performance": {
                    "api_calls_used": production["api_calls_used"],
                    "total_duration_seconds": production["total_duration_seconds"],
                    "error_count": production["error_count"]
                },

                "timestamps": {
                    "created_at": production["created_at"],
                    "started_at": production["started_at"],
                    "completed_at": production["completed_at"],
                    "updated_at": production["updated_at"]
                },

                "files": {
                    "output_directory": production["output_directory"],
                    "files_generated": json.loads(production["files_generated"] or "[]")
                },

                "quality_gates": [dict(gate) for gate in quality_gates]
            }

            return status

    def _calculate_progress(self, production) -> float:
        """Calculate overall production progress (0.0 - 1.0)"""

        stage_weights = {
            "story_generation": 0.25,
            "character_extraction": 0.15,
            "visual_generation": 0.25,
            "tts_generation": 0.15,
            "video_composition": 0.15,
            "youtube_upload": 0.05
        }

        total_progress = 0.0

        for stage, weight in stage_weights.items():
            status = production[f"{stage}_status"]
            if status == "completed":
                total_progress += weight
            elif status == "in_progress":
                total_progress += weight * 0.5

        return round(total_progress, 2)

    def _estimate_completion(self, production, stages) -> Optional[str]:
        """Estimate completion time"""

        if production["status"] == "completed":
            return production["completed_at"]

        if production["status"] != "in_progress":
            return None

        # Calculate average stage duration
        completed_stages = [s for s in stages if s["status"] == "completed"]
        if not completed_stages:
            return None

        avg_duration = sum(s["duration_seconds"] for s in completed_stages) / len(completed_stages)
        remaining_stages = len([s for s in ProductionStage]) - len(completed_stages)

        estimated_seconds = remaining_stages * avg_duration
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)

        return estimated_completion.isoformat()

    def _get_stage_details(self, stage_name: str, stages) -> Dict:
        """Get detailed stage information"""

        stage_executions = [s for s in stages if s["stage_name"] == stage_name]

        if not stage_executions:
            return {"executions": 0}

        latest = stage_executions[-1]

        return {
            "executions": len(stage_executions),
            "latest_execution": {
                "started_at": latest["started_at"],
                "completed_at": latest["completed_at"],
                "duration_seconds": latest["duration_seconds"],
                "api_calls": latest["api_calls"],
                "quality_score": latest["quality_score"],
                "error_message": latest["error_message"]
            }
        }


def test_status_tracker():
    """Test the status tracker"""

    print("🧪 Testing Production Status Tracker")
    print("=" * 40)

    tracker = ProductionStatusTracker()

    # Create new production
    video_id = tracker.create_new_production(
        "Test Ancient Villa",
        "A test story for status tracking"
    )

    # Start production
    tracker.start_production(video_id)

    # Test stage progression
    stage_id = tracker.start_stage(ProductionStage.STORY_GENERATION)
    time.sleep(1)  # Simulate work
    tracker.complete_stage(
        ProductionStage.STORY_GENERATION,
        stage_id,
        api_calls=2,
        output_files=["story.txt", "scenes.json"],
        quality_score=0.95,
        metadata={"story_completion_rate": 1.0, "character_count": 3}
    )

    # Test quality gate
    quality_result = QualityGateResult(
        passed=True,
        score=0.95,
        threshold=0.8,
        criteria_checked={"scene_count": 40, "word_count": 15000},
        failure_reasons=[],
        recommendations=["Consider adding more sensory details"]
    )
    tracker.log_quality_gate(ProductionStage.STORY_GENERATION, "story_quality", quality_result)

    # Complete production
    tracker.complete_production(
        success=True,
        output_directory="/output/1",
        files_generated=["story.txt", "scenes.json", "characters.json"]
    )

    # Get status
    status = tracker.get_production_status(video_id)

    print(f"\n📊 Production Status:")
    print(f"Topic: {status['topic']}")
    print(f"Status: {status['status']}")
    print(f"Progress: {status['progress'] * 100:.1f}%")
    print(f"API Calls: {status['performance']['api_calls_used']}")
    print(f"Duration: {status['performance']['total_duration_seconds']:.1f}s")

    print("\n✅ Status tracker test completed!")


if __name__ == "__main__":
    test_status_tracker()