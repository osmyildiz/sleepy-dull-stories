"""
Professional Production Dashboard - Real-time Web Interface
Location: src/dashboard/production_dashboard.py
"""

import sqlite3
import json
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from typing import Dict, List, Any


class ProductionDashboard:
    """Professional production monitoring dashboard"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.db_path = project_root / "data" / "production.db"
        else:
            self.db_path = Path(db_path)

        self.app = Flask(__name__,
                         template_folder='templates',
                         static_folder='static')

        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')

        @self.app.route('/api/status')
        def api_status():
            """API endpoint for dashboard status"""
            return jsonify(self.get_dashboard_data())

        @self.app.route('/api/video/<int:video_id>')
        def api_video_details(video_id):
            """Get detailed video information"""
            return jsonify(self.get_video_details(video_id))

        @self.app.route('/api/system')
        def api_system_metrics():
            """Get current system metrics"""
            return jsonify(self.get_system_metrics())

        @self.app.route('/api/logs/<int:video_id>')
        def api_video_logs(video_id):
            """Get video production logs"""
            return jsonify(self.get_video_logs(video_id))

    def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data"""

        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get active productions
                cursor.execute("""
                    SELECT * FROM video_production 
                    WHERE status IN ('pending', 'in_progress') 
                    ORDER BY created_at DESC
                """)
                active_productions = [dict(row) for row in cursor.fetchall()]

                # Get recent completed productions
                cursor.execute("""
                    SELECT * FROM video_production 
                    WHERE status = 'completed' 
                    ORDER BY completed_at DESC 
                    LIMIT 5
                """)
                recent_completed = [dict(row) for row in cursor.fetchall()]

                # Get today's statistics
                today = datetime.now().date()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_today,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_today,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_today,
                        SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress_today,
                        SUM(api_calls_used) as total_api_calls,
                        AVG(total_duration_seconds) as avg_duration
                    FROM video_production 
                    WHERE DATE(created_at) = ?
                """, (today,))

                stats = dict(cursor.fetchone())

                # Get system info
                system_info = self.get_current_system_info()

                return {
                    "timestamp": datetime.now().isoformat(),
                    "active_productions": active_productions,
                    "recent_completed": recent_completed,
                    "today_stats": stats,
                    "system_info": system_info,
                    "status": "healthy"
                }

        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "error"
            }

    def get_video_details(self, video_id: int) -> Dict:
        """Get detailed video production information"""

        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get video production
                cursor.execute("""
                    SELECT * FROM video_production WHERE id = ?
                """, (video_id,))

                video = cursor.fetchone()
                if not video:
                    return {"error": "Video not found"}

                # Get stage executions
                cursor.execute("""
                    SELECT * FROM stage_execution 
                    WHERE video_id = ? 
                    ORDER BY created_at
                """, (video_id,))

                stages = [dict(row) for row in cursor.fetchall()]

                # Get quality gates
                cursor.execute("""
                    SELECT * FROM quality_gate_results 
                    WHERE video_id = ? 
                    ORDER BY checked_at DESC
                """, (video_id,))

                quality_gates = [dict(row) for row in cursor.fetchall()]

                # Calculate progress
                progress = self._calculate_video_progress(dict(video))

                return {
                    "video": dict(video),
                    "stages": stages,
                    "quality_gates": quality_gates,
                    "progress": progress,
                    "files_generated": json.loads(video["files_generated"] or "[]")
                }

        except Exception as e:
            return {"error": str(e)}

    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""

        try:
            # Current system info
            current = self.get_current_system_info()

            # Historical data from database
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get last 24 hours of metrics
                since = datetime.now() - timedelta(hours=24)
                cursor.execute("""
                    SELECT * FROM system_metrics 
                    WHERE timestamp > ? 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """, (since,))

                historical = [dict(row) for row in cursor.fetchall()]

            return {
                "current": current,
                "historical": historical,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": str(e)}

    def get_video_logs(self, video_id: int) -> Dict:
        """Get video production logs"""

        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get stage executions with details
                cursor.execute("""
                    SELECT 
                        stage_name,
                        status,
                        started_at,
                        completed_at,
                        duration_seconds,
                        api_calls,
                        quality_score,
                        error_message,
                        error_details
                    FROM stage_execution 
                    WHERE video_id = ? 
                    ORDER BY created_at
                """, (video_id,))

                stage_logs = [dict(row) for row in cursor.fetchall()]

                # Get quality gate logs
                cursor.execute("""
                    SELECT 
                        stage_name,
                        gate_type,
                        passed,
                        score,
                        threshold,
                        failure_reasons,
                        recommendations,
                        checked_at
                    FROM quality_gate_results 
                    WHERE video_id = ? 
                    ORDER BY checked_at
                """, (video_id,))

                quality_logs = [dict(row) for row in cursor.fetchall()]

                return {
                    "video_id": video_id,
                    "stage_logs": stage_logs,
                    "quality_logs": quality_logs,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            return {"error": str(e)}

    def get_current_system_info(self) -> Dict:
        """Get current system information"""

        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "cpu": {
                    "usage_percent": round(cpu_percent, 1),
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "usage_percent": round(memory.percent, 1),
                    "used_gb": round(memory.used / 1024 ** 3, 1),
                    "total_gb": round(memory.total / 1024 ** 3, 1)
                },
                "disk": {
                    "usage_percent": round(disk.percent, 1),
                    "used_gb": round(disk.used / 1024 ** 3, 1),
                    "total_gb": round(disk.total / 1024 ** 3, 1)
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": str(e)}

    def _calculate_video_progress(self, video: Dict) -> Dict:
        """Calculate video production progress"""

        stages = [
            "story_generation",
            "character_extraction",
            "visual_generation",
            "tts_generation",
            "video_composition",
            "youtube_upload"
        ]

        completed = 0
        in_progress = 0

        for stage in stages:
            status = video.get(f"{stage}_status", "pending")
            if status == "completed":
                completed += 1
            elif status == "in_progress":
                in_progress += 1

        total_progress = (completed + in_progress * 0.5) / len(stages)

        return {
            "total_progress": round(total_progress, 2),
            "completed_stages": completed,
            "in_progress_stages": in_progress,
            "pending_stages": len(stages) - completed - in_progress,
            "current_stage": video.get("current_stage", "pending")
        }

    def run(self, host='0.0.0.0', port=5000, debug=True):
        """Run the dashboard server"""

        print(f"🎛️  Starting Production Dashboard")
        print(f"🌐 URL: http://{host}:{port}")
        print(f"📊 Database: {self.db_path}")

        self.app.run(host=host, port=port, debug=debug)


def create_dashboard_template():
    """Create basic HTML template for dashboard"""

    template_dir = Path("src/dashboard/templates")
    template_dir.mkdir(parents=True, exist_ok=True)

    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sleepy Dull Stories - Production Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: #1a1a1a; 
            color: #ffffff; 
            line-height: 1.6;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .header h1 { font-size: 2rem; font-weight: 300; }
        .container { padding: 2rem; max-width: 1400px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; }
        .card {
            background: #2d2d2d;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            border: 1px solid #444;
        }
        .card h3 { color: #667eea; margin-bottom: 1rem; font-size: 1.2rem; }
        .stat { display: flex; justify-content: space-between; margin: 0.5rem 0; }
        .stat-label { color: #aaa; }
        .stat-value { color: #fff; font-weight: bold; }
        .status-active { color: #4ade80; }
        .status-completed { color: #22d3ee; }
        .status-failed { color: #f87171; }
        .status-pending { color: #fbbf24; }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #444;
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
        .video-item {
            background: #333;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #667eea;
        }
        .video-title { font-weight: bold; margin-bottom: 0.5rem; }
        .video-meta { font-size: 0.9rem; color: #aaa; }
        .system-metric {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 0.75rem 0;
        }
        .metric-circle {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9rem;
        }
        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            margin-bottom: 1rem;
        }
        .refresh-btn:hover { background: #5a67d8; }
        .last-updated { 
            text-align: center; 
            color: #666; 
            margin-top: 2rem; 
            font-size: 0.9rem; 
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Sleepy Dull Stories - Production Dashboard</h1>
    </div>

    <div class="container">
        <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh Data</button>

        <div class="grid">
            <!-- Today's Statistics -->
            <div class="card">
                <h3>📊 Today's Production</h3>
                <div id="today-stats">Loading...</div>
            </div>

            <!-- System Metrics -->
            <div class="card">
                <h3>🖥️ System Status</h3>
                <div id="system-metrics">Loading...</div>
            </div>

            <!-- Active Productions -->
            <div class="card">
                <h3>🔄 Active Productions</h3>
                <div id="active-productions">Loading...</div>
            </div>

            <!-- Recent Completed -->
            <div class="card">
                <h3>✅ Recent Completed</h3>
                <div id="recent-completed">Loading...</div>
            </div>
        </div>

        <div class="last-updated" id="last-updated">Last updated: Never</div>
    </div>

    <script>
        async function fetchDashboardData() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                if (data.error) {
                    console.error('Dashboard error:', data.error);
                    return;
                }

                updateTodayStats(data.today_stats);
                updateSystemMetrics(data.system_info);
                updateActiveProductions(data.active_productions);
                updateRecentCompleted(data.recent_completed);

                document.getElementById('last-updated').textContent = 
                    `Last updated: ${new Date().toLocaleTimeString()}`;

            } catch (error) {
                console.error('Failed to fetch dashboard data:', error);
            }
        }

        function updateTodayStats(stats) {
            const html = `
                <div class="stat">
                    <span class="stat-label">Total Productions</span>
                    <span class="stat-value">${stats.total_today || 0}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Completed</span>
                    <span class="stat-value status-completed">${stats.completed_today || 0}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Failed</span>
                    <span class="stat-value status-failed">${stats.failed_today || 0}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">In Progress</span>
                    <span class="stat-value status-active">${stats.in_progress_today || 0}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">API Calls</span>
                    <span class="stat-value">${stats.total_api_calls || 0}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Avg Duration</span>
                    <span class="stat-value">${Math.round(stats.avg_duration || 0)}s</span>
                </div>
            `;
            document.getElementById('today-stats').innerHTML = html;
        }

        function updateSystemMetrics(system) {
            if (system.error) {
                document.getElementById('system-metrics').innerHTML = 
                    `<div class="stat-value status-failed">Error: ${system.error}</div>`;
                return;
            }

            const html = `
                <div class="system-metric">
                    <span>CPU Usage</span>
                    <div class="metric-circle" style="background: conic-gradient(#667eea ${system.cpu.usage_percent * 3.6}deg, #444 0deg)">
                        ${system.cpu.usage_percent}%
                    </div>
                </div>
                <div class="system-metric">
                    <span>Memory Usage</span>
                    <div class="metric-circle" style="background: conic-gradient(#764ba2 ${system.memory.usage_percent * 3.6}deg, #444 0deg)">
                        ${system.memory.usage_percent}%
                    </div>
                </div>
                <div class="system-metric">
                    <span>Disk Usage</span>
                    <div class="metric-circle" style="background: conic-gradient(#f87171 ${system.disk.usage_percent * 3.6}deg, #444 0deg)">
                        ${system.disk.usage_percent}%
                    </div>
                </div>
                <div class="stat">
                    <span class="stat-label">Memory</span>
                    <span class="stat-value">${system.memory.used_gb}GB / ${system.memory.total_gb}GB</span>
                </div>
            `;
            document.getElementById('system-metrics').innerHTML = html;
        }

        function updateActiveProductions(productions) {
            if (productions.length === 0) {
                document.getElementById('active-productions').innerHTML = 
                    '<div class="stat-value">No active productions</div>';
                return;
            }

            const html = productions.map(video => `
                <div class="video-item">
                    <div class="video-title">${video.topic}</div>
                    <div class="video-meta">
                        Status: <span class="status-${video.status}">${video.status}</span> | 
                        Stage: ${video.current_stage}
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${calculateProgress(video)}%"></div>
                    </div>
                </div>
            `).join('');

            document.getElementById('active-productions').innerHTML = html;
        }

        function updateRecentCompleted(videos) {
            if (videos.length === 0) {
                document.getElementById('recent-completed').innerHTML = 
                    '<div class="stat-value">No completed videos today</div>';
                return;
            }

            const html = videos.map(video => `
                <div class="video-item">
                    <div class="video-title">${video.topic}</div>
                    <div class="video-meta">
                        Completed: ${new Date(video.completed_at).toLocaleTimeString()} | 
                        Duration: ${Math.round(video.total_duration_seconds || 0)}s
                    </div>
                </div>
            `).join('');

            document.getElementById('recent-completed').innerHTML = html;
        }

        function calculateProgress(video) {
            const stages = ['story_generation', 'character_extraction', 'visual_generation', 
                          'tts_generation', 'video_composition', 'youtube_upload'];
            let completed = 0;

            stages.forEach(stage => {
                if (video[stage + '_status'] === 'completed') completed++;
            });

            return Math.round((completed / stages.length) * 100);
        }

        function refreshDashboard() {
            fetchDashboardData();
        }

        // Auto-refresh every 30 seconds
        setInterval(fetchDashboardData, 30000);

        // Initial load
        fetchDashboardData();
    </script>
</body>
</html>'''

    template_path = template_dir / "dashboard.html"
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Dashboard template created: {template_path}")


def main():
    """Main dashboard function"""

    print("🎛️  Professional Production Dashboard Setup")
    print("=" * 50)

    # Create template
    create_dashboard_template()

    # Initialize dashboard
    dashboard = ProductionDashboard()

    print("\n🔧 Dashboard Features:")
    print("• Real-time production monitoring")
    print("• System resource tracking")
    print("• Video progress visualization")
    print("• API endpoints for data")
    print("• Auto-refresh every 30 seconds")

    print("\n🌐 Starting dashboard server...")
    print("📊 Access dashboard at: http://localhost:5000")
    print("🔄 Press Ctrl+C to stop")

    try:
        dashboard.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")


if __name__ == "__main__":
    main()