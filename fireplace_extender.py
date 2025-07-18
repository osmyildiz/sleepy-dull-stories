#!/usr/bin/env python3
"""
Sleepy Dull Stories - Fireplace Extender
One-time script to create extended fireplace video for video composition
Creates fireplace_200min.mp4 from original fireplace.mp4
"""

import os
import sys
import time
from pathlib import Path


def create_extended_fireplace():
    """Create 200-minute extended fireplace video once"""

    print("🔥" * 60)
    print("FIREPLACE EXTENDER - ONE-TIME SETUP")
    print("Creating 200-minute extended fireplace video")
    print("🔥" * 60)

    # Check MoviePy installation first
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        print("✅ MoviePy found and imported successfully")
    except ImportError as e:
        print("❌ MoviePy not installed or not found")
        print(f"   Error: {e}")
        print()
        print("🔧 To install MoviePy:")
        print("   pip install moviepy")
        print("   # or")
        print("   pip3 install moviepy")
        print()
        print("🔄 After installation, run this script again")
        return False

    # Auto-detect project structure
    script_dir = Path(__file__).resolve().parent

    # Try to find project root
    if 'generators' in str(script_dir):
        project_root = script_dir.parent.parent.parent
    elif 'src' in str(script_dir):
        project_root = script_dir.parent.parent
    else:
        project_root = script_dir

    # Fireplace paths
    overlay_dir = project_root / "src" / "data" / "overlay_videos"
    original_fireplace = overlay_dir / "fireplace.mp4"
    extended_fireplace = overlay_dir / "fireplace_200min.mp4"

    print(f"📁 Project root: {project_root}")
    print(f"📁 Overlay directory: {overlay_dir}")
    print(f"🔥 Original fireplace: {original_fireplace}")
    print(f"🎬 Extended fireplace: {extended_fireplace}")
    print()

    # Check if directory exists
    if not overlay_dir.exists():
        print(f"❌ Overlay directory not found: {overlay_dir}")
        print("   Please ensure you're running this from the correct location")
        return False

    # Check if original exists
    if not original_fireplace.exists():
        print(f"❌ Original fireplace not found: {original_fireplace}")
        print("   Please ensure fireplace.mp4 exists in src/data/overlay_videos/")
        print()
        print("📋 Available files in overlay directory:")
        try:
            for file in overlay_dir.iterdir():
                if file.is_file():
                    print(f"   📄 {file.name}")
        except:
            print("   ⚠️  Cannot list directory contents")
        return False

    # Check if extended already exists
    if extended_fireplace.exists():
        print(f"✅ Extended fireplace already exists: {extended_fireplace}")

        # Get file info
        try:
            clip = VideoFileClip(str(extended_fireplace))
            duration_minutes = clip.duration / 60
            file_size_mb = extended_fireplace.stat().st_size / (1024 * 1024)
            clip.close()

            print(f"   📏 Duration: {duration_minutes:.1f} minutes")
            print(f"   📦 File size: {file_size_mb:.1f} MB")

            if duration_minutes >= 190:  # At least 190 minutes
                print("   ✅ Extended fireplace is ready for use!")
                return True
            else:
                print("   ⚠️  Duration too short, recreating...")
        except Exception as e:
            print(f"   ⚠️  File seems corrupted: {e}")
            print("   🔄 Recreating extended fireplace...")

    print("🚀 CREATING EXTENDED FIREPLACE:")
    print("   This is a ONE-TIME process that will take 5-10 minutes")
    print("   But will save HOURS in future video rendering!")
    print()

    try:
        # Load original fireplace
        print("🔥 Loading original fireplace video...")
        start_time = time.time()

        original_clip = VideoFileClip(str(original_fireplace))
        original_duration = original_clip.duration

        load_time = time.time() - start_time
        print(f"   ✅ Loaded: {original_duration:.1f}s duration in {load_time:.1f}s")

        # Calculate loops needed for 200 minutes (12000 seconds)
        target_duration = 200 * 60  # 200 minutes in seconds
        loops_needed = int(target_duration / original_duration) + 1

        print(f"📊 EXTENSION CALCULATION:")
        print(f"   🎯 Target duration: {target_duration}s ({target_duration / 60:.1f} minutes)")
        print(f"   🔄 Loops needed: {loops_needed}")
        print(f"   📏 Final duration: {loops_needed * original_duration:.1f}s")
        print()

        print("🔄 Creating extended video clips...")
        extended_start = time.time()

        # Create list of clips
        print("   📝 Preparing clip list...")
        clips = []
        for i in range(loops_needed):
            if i % 10 == 0:  # Progress every 10 clips
                print(f"      🔄 Preparing clip {i + 1}/{loops_needed}")
            clips.append(original_clip.copy())

        clip_prep_time = time.time() - extended_start
        print(f"   ✅ {len(clips)} clips prepared in {clip_prep_time:.1f}s")

        # Concatenate all clips
        print("🔗 Concatenating clips (this will take several minutes)...")
        concat_start = time.time()

        extended_clip = concatenate_videoclips(clips, method="compose")

        concat_time = time.time() - concat_start
        print(f"   ✅ Concatenation completed in {concat_time / 60:.1f} minutes")

        # Trim to exact 200 minutes
        print("✂️  Trimming to exact 200 minutes...")
        final_clip = extended_clip.subclip(0, target_duration)

        # Export extended fireplace
        print("💾 Exporting extended fireplace video...")
        print("   📁 Output file: fireplace_200min.mp4")
        print("   ⏱️  Expected export time: 3-7 minutes")
        print("   🔄 Progress will be shown below:")
        print()

        export_start = time.time()

        # Fixed: Removed unsupported 'crf' parameter and simplified codec options
        final_clip.write_videofile(
            str(extended_fireplace),
            fps=30,
            codec="libx264",
            preset="fast",
            bitrate="2000k",  # Use bitrate instead of crf for quality control
            verbose=False,
            logger='bar'
        )

        export_time = time.time() - export_start
        total_time = time.time() - start_time

        # Cleanup
        print("\n🧹 Cleaning up temporary clips...")
        for clip in clips:
            try:
                clip.close()
            except:
                pass

        try:
            original_clip.close()
            extended_clip.close()
            final_clip.close()
        except:
            pass

        # Verify result
        if extended_fireplace.exists():
            file_size_mb = extended_fireplace.stat().st_size / (1024 * 1024)

            print("\n✅ EXTENDED FIREPLACE CREATED SUCCESSFULLY!")
            print("=" * 50)
            print(f"📁 File: {extended_fireplace.name}")
            print(f"📏 Duration: 200 minutes")
            print(f"📦 File size: {file_size_mb:.1f} MB")
            print(f"⏱️  Export time: {export_time / 60:.1f} minutes")
            print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
            print("=" * 50)
            print()
            print("🎬 Your video composer will now use this extended fireplace")
            print("   for MUCH faster rendering!")
            print("🔥 This file will be automatically used by video composer")
            print()
            return True
        else:
            print("❌ Extended fireplace file not created")
            return False

    except Exception as e:
        print(f"❌ Error creating extended fireplace: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("🚀 Starting fireplace extension process...")

    success = create_extended_fireplace()

    if success:
        print("🎉 FIREPLACE EXTENSION COMPLETED!")
        print("   Your video composer is now optimized for fast rendering!")
    else:
        print("❌ FIREPLACE EXTENSION FAILED!")
        print("   Video composer will fall back to original method")

    return success


if __name__ == "__main__":
    main()