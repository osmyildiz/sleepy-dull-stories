"""
Find Database - Locate production.db file
"""

import os
from pathlib import Path

def find_production_db():
    """Find production.db file in project structure"""

    print("🔍 SEARCHING FOR PRODUCTION.DB")
    print("=" * 40)

    # Current directory
    current_dir = Path.cwd()
    print(f"📍 Current directory: {current_dir}")

    # Search patterns
    search_patterns = [
        "production.db",
        "data/production.db",
        "../data/production.db",
        "../../data/production.db",
        "../../../data/production.db"
    ]

    print("\n🔎 Searching in common locations:")

    found_databases = []

    for pattern in search_patterns:
        db_path = Path(pattern)
        if db_path.exists():
            abs_path = db_path.resolve()
            size_kb = abs_path.stat().st_size / 1024
            print(f"  ✅ Found: {pattern} -> {abs_path} ({size_kb:.1f} KB)")
            found_databases.append(abs_path)
        else:
            print(f"  ❌ Not found: {pattern}")

    # Search recursively in parent directories
    print(f"\n🔍 Recursive search from current directory...")

    search_root = current_dir
    max_levels = 5

    for level in range(max_levels):
        for db_file in search_root.rglob("production.db"):
            if db_file not in found_databases:
                size_kb = db_file.stat().st_size / 1024
                print(f"  ✅ Found: {db_file} ({size_kb:.1f} KB)")
                found_databases.append(db_file)

        # Go up one level
        search_root = search_root.parent
        if search_root == search_root.parent:  # Reached filesystem root
            break

    # Look for any .db files
    print(f"\n🔍 Looking for any .db files...")

    search_root = current_dir
    for level in range(3):
        for db_file in search_root.rglob("*.db"):
            size_kb = db_file.stat().st_size / 1024
            print(f"  📄 Found DB: {db_file} ({size_kb:.1f} KB)")

        search_root = search_root.parent
        if search_root == search_root.parent:
            break

    print(f"\n📊 SUMMARY:")
    if found_databases:
        print(f"✅ Found {len(found_databases)} production.db file(s):")
        for i, db_path in enumerate(found_databases, 1):
            print(f"  {i}. {db_path}")

        # Use the first found database
        chosen_db = found_databases[0]
        print(f"\n🎯 Using: {chosen_db}")
        return chosen_db
    else:
        print("❌ No production.db found!")

        # Check if we need to create one
        project_root = find_project_root()
        if project_root:
            data_dir = project_root / "data"
            suggested_path = data_dir / "production.db"
            print(f"\n💡 Suggested location: {suggested_path}")

            if not data_dir.exists():
                print(f"📁 Creating data directory: {data_dir}")
                data_dir.mkdir(parents=True, exist_ok=True)

            return suggested_path

        return None

def find_project_root():
    """Find project root directory"""

    current = Path.cwd()

    # Look for indicators of project root
    indicators = [
        'src',
        'data',
        'output',
        'config',
        '.env',
        'requirements.txt',
        '.git'
    ]

    max_levels = 10

    for level in range(max_levels):
        for indicator in indicators:
            if (current / indicator).exists():
                print(f"📍 Project root detected: {current} (found {indicator})")
                return current

        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    # Fallback - assume current directory's parent or grandparent
    current_dir = Path.cwd()

    if current_dir.name in ['src', 'database', 'generators']:
        return current_dir.parent
    elif current_dir.parent.name == 'src':
        return current_dir.parent.parent

    return current_dir

if __name__ == "__main__":
    db_path = find_production_db()

    if db_path and db_path.exists():
        print(f"\n🚀 Ready to inspect database: {db_path}")

        # Quick check if it's a SQLite database
        try:
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                print(f"\n📋 Tables found: {[table[0] for table in tables]}")

                if ('topics',) in tables:
                    cursor.execute("SELECT COUNT(*) FROM topics;")
                    count = cursor.fetchone()[0]
                    print(f"📊 Topics count: {count}")

        except Exception as e:
            print(f"❌ Error reading database: {e}")

    else:
        print(f"\n💭 Database needs to be created or path corrected")