import os
import ffmpeg
from pathlib import Path


class FireplaceDebugTest:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        if 'generators' in script_dir:
            self.base_dir = Path(script_dir).parent.parent
        elif 'src' in script_dir:
            self.base_dir = Path(script_dir).parent
        else:
            self.base_dir = Path(script_dir)

        self.project_dir = self.base_dir / "src" / "output" / "1"
        self.scenes_dir = self.project_dir / "scenes"
        self.overlay_path = self.base_dir / "src" / "data" / "overlay_videos"

        print("🔥 FIREPLACE DEBUG TEST")
        print(f"📁 Project Dir: {self.project_dir}")
        print(f"🔥 Overlay Path: {self.overlay_path}")

    def find_test_image(self):
        """Find test image"""
        for scene_id in [1, 2, 3, 4, 5]:
            for format_str in [f"scene_{scene_id:02d}", f"scene_{scene_id}"]:
                test_file = self.scenes_dir / f"{format_str}.png"
                if test_file.exists():
                    print(f"✅ Found test image: {test_file}")
                    return test_file
        return None

    def debug_fireplace_file(self):
        """Debug fireplace file in detail"""
        fireplace_video = self.overlay_path / "fireplace.mp4"

        print(f"\n🔍 DEBUGGING FIREPLACE FILE:")
        print(f"📁 Path: {fireplace_video}")
        print(f"📁 Exists: {fireplace_video.exists()}")

        if not fireplace_video.exists():
            print("❌ Fireplace file not found!")
            return None

        try:
            # Get file size
            file_size = fireplace_video.stat().st_size
            print(f"📊 File size: {file_size / (1024 * 1024):.1f} MB")

            # Detailed probe
            probe = ffmpeg.probe(str(fireplace_video))

            print(f"\n📊 FORMAT INFO:")
            format_info = probe['format']
            print(f"   Format: {format_info.get('format_name', 'unknown')}")
            print(f"   Duration: {format_info.get('duration', 'unknown')}s")
            print(f"   Bitrate: {format_info.get('bit_rate', 'unknown')}")

            print(f"\n📊 STREAMS INFO:")
            for i, stream in enumerate(probe['streams']):
                print(f"   Stream {i}:")
                print(f"      Type: {stream.get('codec_type', 'unknown')}")
                print(f"      Codec: {stream.get('codec_name', 'unknown')}")
                if stream['codec_type'] == 'video':
                    print(f"      Size: {stream.get('width', '?')}x{stream.get('height', '?')}")
                    print(f"      FPS: {stream.get('r_frame_rate', 'unknown')}")
                    print(f"      Pixel format: {stream.get('pix_fmt', 'unknown')}")

            return fireplace_video

        except Exception as e:
            print(f"❌ Could not probe fireplace file: {e}")
            return None

    def test_1_simple_conversion(self, fireplace_path):
        """Test 1: Just convert fireplace to verify it works"""
        output = self.project_dir / "debug_1_fireplace_convert.mp4"

        try:
            print("\n🧪 DEBUG 1: Simple fireplace conversion")

            (
                ffmpeg
                .input(str(fireplace_path))
                .output(str(output), t=10, vcodec='libx264', pix_fmt='yuv420p')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            if output.exists():
                print(f"✅ Fireplace conversion works: {output}")
                return True
            else:
                print("❌ Fireplace conversion failed")
                return False

        except ffmpeg.Error as e:
            print(f"❌ Fireplace conversion error:")
            print(f"   STDERR: {e.stderr.decode()}")
            return False

    def test_2_simple_loop(self, fireplace_path):
        """Test 2: Test fireplace looping"""
        output = self.project_dir / "debug_2_fireplace_loop.mp4"

        try:
            print("\n🧪 DEBUG 2: Fireplace looping test")

            (
                ffmpeg
                .input(str(fireplace_path))
                .filter('loop', loop=2, size=32767)  # Alternative loop method
                .output(str(output), t=15, vcodec='libx264', pix_fmt='yuv420p')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            if output.exists():
                print(f"✅ Fireplace looping works: {output}")
                return True
            else:
                print("❌ Fireplace looping failed")
                return False

        except ffmpeg.Error as e:
            print(f"❌ Fireplace looping error:")
            print(f"   STDERR: {e.stderr.decode()}")
            return False

    def test_3_image_to_video(self, image_path):
        """Test 3: Convert image to video (baseline)"""
        output = self.project_dir / "debug_3_image_video.mp4"

        try:
            print("\n🧪 DEBUG 3: Image to video conversion")

            (
                ffmpeg
                .input(str(image_path), loop=1, t=10, r=30)
                .filter('scale', 1920, 1080, force_original_aspect_ratio='decrease')
                .filter('pad', 1920, 1080, '(ow-iw)/2', '(oh-ih)/2')
                .output(str(output), vcodec='libx264', pix_fmt='yuv420p')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            if output.exists():
                print(f"✅ Image to video works: {output}")
                return True
            else:
                print("❌ Image to video failed")
                return False

        except ffmpeg.Error as e:
            print(f"❌ Image to video error:")
            print(f"   STDERR: {e.stderr.decode()}")
            return False

    def test_4_simple_overlay_alpha(self, image_path, fireplace_path):
        """Test 4: Simple alpha overlay (no blend modes)"""
        output = self.project_dir / "debug_4_alpha_overlay.mp4"

        try:
            print("\n🧪 DEBUG 4: Simple alpha overlay")

            duration = 10

            # Base image video
            base = (
                ffmpeg
                .input(str(image_path), loop=1, t=duration, r=30)
                .filter('scale', 1920, 1080, force_original_aspect_ratio='decrease')
                .filter('pad', 1920, 1080, '(ow-iw)/2', '(oh-ih)/2')
            )

            # Fireplace with alpha
            fireplace = (
                ffmpeg
                .input(str(fireplace_path))
                .filter('loop', loop=-1, size=32767)
                .filter('scale', 1920, 1080)
                .filter('format', 'yuva420p')  # Add alpha channel
                .filter('colorchannelmixer', aa=0.3)  # 30% transparency
            )

            # Simple overlay
            result = ffmpeg.filter([base, fireplace], 'overlay')

            (
                result
                .output(str(output), vcodec='libx264', pix_fmt='yuv420p', t=duration)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            if output.exists():
                print(f"✅ Alpha overlay works: {output}")
                return True
            else:
                print("❌ Alpha overlay failed")
                return False

        except ffmpeg.Error as e:
            print(f"❌ Alpha overlay error:")
            print(f"   STDERR: {e.stderr.decode()}")
            return False

    def test_5_ultra_simple_overlay(self, image_path, fireplace_path):
        """Test 5: Ultra simple overlay without any fancy stuff"""
        output = self.project_dir / "debug_5_ultra_simple.mp4"

        try:
            print("\n🧪 DEBUG 5: Ultra simple overlay")

            duration = 8

            # Convert fireplace to image sequence first
            temp_fireplace = self.project_dir / "temp_fireplace_simple.mp4"

            # Step 1: Create simple fireplace video
            (
                ffmpeg
                .input(str(fireplace_path))
                .output(str(temp_fireplace), t=duration, vcodec='libx264', pix_fmt='yuv420p', r=30)
                .overwrite_output()
                .run(quiet=True)
            )

            # Step 2: Create base video
            temp_base = self.project_dir / "temp_base_simple.mp4"
            (
                ffmpeg
                .input(str(image_path), loop=1, t=duration, r=30)
                .filter('scale', 1920, 1080, force_original_aspect_ratio='decrease')
                .filter('pad', 1920, 1080, '(ow-iw)/2', '(oh-ih)/2')
                .output(str(temp_base), vcodec='libx264', pix_fmt='yuv420p')
                .overwrite_output()
                .run(quiet=True)
            )

            # Step 3: Simple overlay
            (
                ffmpeg
                .filter([
                    ffmpeg.input(str(temp_base)),
                    ffmpeg.input(str(temp_fireplace))
                ], 'overlay', eval='init', opacity=0.3)
                .output(str(output), vcodec='libx264', pix_fmt='yuv420p')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            # Cleanup
            if temp_fireplace.exists():
                temp_fireplace.unlink()
            if temp_base.exists():
                temp_base.unlink()

            if output.exists():
                print(f"✅ Ultra simple overlay works: {output}")
                return True
            else:
                print("❌ Ultra simple overlay failed")
                return False

        except ffmpeg.Error as e:
            print(f"❌ Ultra simple overlay error:")
            print(f"   STDERR: {e.stderr.decode()}")
            return False

    def run_debug_tests(self):
        """Run all debug tests to find the problem"""
        print("🔥" * 50)
        print("FIREPLACE DEBUG TESTS - FIND THE PROBLEM")
        print("🔥" * 50)

        # Check files
        image_path = self.find_test_image()
        fireplace_path = self.debug_fireplace_file()

        if not image_path:
            print("❌ No test image found")
            return

        if not fireplace_path:
            print("❌ Fireplace file problem")
            return

        # Run debug tests in order
        tests = [

            ("Alpha Overlay", lambda: self.test_4_simple_overlay_alpha(image_path, fireplace_path)),

        ]

        results = {}

        for test_name, test_func in tests:
            print(f"\n{'=' * 60}")
            try:
                result = test_func()
                results[test_name] = result
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                results[test_name] = False

        # Analysis
        print(f"\n{'🔍' * 30}")
        print("DEBUG ANALYSIS")
        print(f"{'🔍' * 30}")

        for test_name, result in results.items():
            status = "✅ WORKS" if result else "❌ FAILS"
            print(f"{status} | {test_name}")

        # Diagnosis
        print(f"\n💡 DIAGNOSIS:")

        if not results.get("Fireplace Conversion", False):
            print("🔥 Problem: Fireplace video file is corrupted or unreadable")
        elif not results.get("Fireplace Looping", False):
            print("🔄 Problem: Video looping doesn't work")
        elif not results.get("Image to Video", False):
            print("🖼️ Problem: Image to video conversion fails")
        elif not results.get("Alpha Overlay", False):
            print("🎭 Problem: Overlay filters don't work")
        elif not results.get("Ultra Simple", False):
            print("🔧 Problem: Even simplest overlay fails")
        else:
            print("✅ All basic functions work - original blend issue")

        working_tests = [name for name, result in results.items() if result]
        if working_tests:
            print(f"\n🎯 WORKING METHODS: {', '.join(working_tests)}")
            print("💡 We can use the working method for fireplace overlay!")


if __name__ == "__main__":
    try:
        debugger = FireplaceDebugTest()
        debugger.run_debug_tests()
    except Exception as e:
        print(f"\n💥 Debug failed: {e}")
        import traceback

        traceback.print_exc()