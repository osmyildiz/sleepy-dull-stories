"""
Professional Production Dashboard - Server Optimized Version
Location: src/dashboard/production_dashboard_server.py
"""

import sqlite3
import json
import psutil
import os
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
from typing import Dict, List, Any


class ProductionDashboardServer:
    """Server-optimized production monitoring dashboard"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.db_path = project_root / "data" / "production.db"
        else:
            self.db_path = Path(db_path)

        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'sleepy-dull-stories-dashboard-2025'

        # Server optimizations
        self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300  # 5 min cache

        # Dashboard HTML template as string (no external files needed)
        self.html_template = self._get_html_template()

        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template_string(self.html_template)

        @self.app.route('/production/status')
        def production_status():
            """Production status page (alternative URL)"""
            return render_template_string(self.html_template)

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

        @self.app.route('/api/health')
        def api_health():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "database": str(self.db_path),
                "database_exists": self.db_path.exists()
            })

    def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data with error handling"""

        try:
            # Ensure database exists
            if not self.db_path.exists():
                return {
                    "timestamp": datetime.now().isoformat(),
                    "error": "Database not found",
                    "status": "error"
                }

            with sqlite3.connect(self.db_path, timeout=30) as conn:
                conn.row_factory = sqlite3.Row

                # Enable WAL mode for better concurrency
                conn.execute('PRAGMA journal_mode=WAL;')
                conn.execute('PRAGMA synchronous=NORMAL;')

                cursor = conn.cursor()

                # Get active productions
                cursor.execute("""
                    SELECT * FROM video_production 
                    WHERE status IN ('pending', 'in_progress') 
                    ORDER BY created_at DESC
                    LIMIT 10
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
                        COALESCE(SUM(api_calls_used), 0) as total_api_calls,
                        COALESCE(AVG(total_duration_seconds), 0) as avg_duration
                    FROM video_production 
                    WHERE DATE(created_at) = ?
                """, (today,))

                stats_row = cursor.fetchone()
                stats = dict(stats_row) if stats_row else {}

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
            with sqlite3.connect(self.db_path, timeout=30) as conn:
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

                return {
                    "video": dict(video),
                    "stages": stages,
                    "files_generated": json.loads(video["files_generated"] or "[]")
                }

        except Exception as e:
            return {"error": str(e)}

    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""

        try:
            return self.get_current_system_info()
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

    def _get_html_template(self) -> str:
        """Get dashboard HTML template as string"""

        return '''<!DOCTYPE html>
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
        .error-message {
            background: #dc2626;
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Sleepy Dull Stories - Production Dashboard</h1>
    </div>

    <div class="container">
        <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh Data</button>

        <div id="error-container"></div>

        <div class="grid">
            <div class="card">
                <h3>📊 Today's Production</h3>
                <div id="today-stats">Loading...</div>
            </div>

            <div class="card">
                <h3>🖥️ System Status</h3>
                <div id="system-metrics">Loading...</div>
            </div>

            <div class="card">
                <h3>🔄 Active Productions</h3>
                <div id="active-productions">Loading...</div>
            </div>

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

                document.getElementById('error-container').innerHTML = '';

                if (data.error) {
                    showError(data.error);
                    return;
                }

                updateTodayStats(data.today_stats || {});
                updateSystemMetrics(data.system_info || {});
                updateActiveProductions(data.active_productions || []);
                updateRecentCompleted(data.recent_completed || []);

                document.getElementById('last-updated').textContent = 
                    `Last updated: ${new Date().toLocaleTimeString()}`;

            } catch (error) {
                showError(`Failed to fetch data: ${error.message}`);
            }
        }

        function showError(message) {
            document.getElementById('error-container').innerHTML = 
                `<div class="error-message">❌ ${message}</div>`;
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
            `;
            document.getElementById('today-stats').innerHTML = html;
        }

        function updateSystemMetrics(system) {
            if (system.error) {
                document.getElementById('system-metrics').innerHTML = 
                    `<div class="stat-value status-failed">Error: ${system.error}</div>`;
                return;
            }

            const cpu = system.cpu || {};
            const memory = system.memory || {};
            const disk = system.disk || {};

            const html = `
                <div class="system-metric">
                    <span>CPU Usage</span>
                    <div class="metric-circle" style="background: conic-gradient(#667eea ${(cpu.usage_percent || 0) * 3.6}deg, #444 0deg)">
                        ${cpu.usage_percent || 0}%
                    </div>
                </div>
                <div class="system-metric">
                    <span>Memory Usage</span>
                    <div class="metric-circle" style="background: conic-gradient(#764ba2 ${(memory.usage_percent || 0) * 3.6}deg, #444 0deg)">
                        ${memory.usage_percent || 0}%
                    </div>
                </div>
                <div class="stat">
                    <span class="stat-label">Memory</span>
                    <span class="stat-value">${memory.used_gb || 0}GB / ${memory.total_gb || 0}GB</span>
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
                    <div class="video-title">${video.topic || 'Unknown Topic'}</div>
                    <div class="video-meta">
                        Status: <span class="status-${video.status}">${video.status}</span> | 
                        Stage: ${video.current_stage || 'pending'}
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
                    <div class="video-title">${video.topic || 'Unknown Topic'}</div>
                    <div class="video-meta">
                        Completed: ${video.completed_at ? new Date(video.completed_at).toLocaleTimeString() : 'Unknown'} | 
                        Duration: ${Math.round(video.total_duration_seconds || 0)}s
                    </div>
                </div>
            `).join('');

            document.getElementById('recent-completed').innerHTML = html;
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

    def run_server(self, host='0.0.0.0', port=5000, debug=False):
        """Run the dashboard server (production mode)"""

        print(f"🎛️  Starting Production Dashboard Server")
        print(f"🌐 URL: http://{host}:{port}")
        print(f"📊 Database: {self.db_path}")
        print(f"🔧 Mode: {'Debug' if debug else 'Production'}")

        self.app.run(host=host, port=port, debug=debug, threaded=True)


def main():
    """Main function for server dashboard"""

    print("🎛️  Sleepy Dull Stories - Production Dashboard Server")
    print("=" * 60)

    dashboard = ProductionDashboardServer()

    print("\n🔧 Dashboard Features:")
    print("• Real-time production monitoring")
    print("• System resource tracking")
    print("• Error handling and recovery")
    print("• Server-optimized performance")
    print("• No external template dependencies")

    print("\n🌐 Starting server...")
    print("📊 Access: http://server-ip:5000")
    print("🏥 Health: http://server-ip:5000/api/health")
    print("🔄 Press Ctrl+C to stop")

    try:
        dashboard.run_server(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Dashboard server stopped")
    except Exception as e:
        print(f"\n❌ Server error: {e}")


if __name__ == "__main__":
    main()