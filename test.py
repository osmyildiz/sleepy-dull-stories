import os
import ffmpeg
import subprocess
import random
from pathlib import Path


class VideoTestSimple:
    def __init__(self):
        # Hard-coded paths for test
        script_dir = os.path.dirname(os.path.abspath(__file__))

        if 'generators' in script_dir:
            self.base_dir = Path(script_dir).parent.parent
        else:
            self.base_dir = Path(script_dir)

        # Test project 1
        self.test_project = 1
        self.project_dir = self.base_dir / "src" / "output" / str(self.test_project)
        self.audio_dir = self.project_dir / "audio_parts"
        self.scenes_dir = self.project_dir / "scenes"
        self.overlay_path = self.base_dir / "src" / "data" / "overlay_videos"

        print("🧪 VIDEO TEST - Hook+Subscribe+2 Scenes")
        print(f"📁 Project Directory: {self.project_dir}")
        print(f"🎵 Audio Directory: {self.audio_dir}")
        print(f"🎨 Scenes Directory: {self.scenes_dir}")

        self.check_files()

    def check_files(self):
        """Test için gerekli dosyaları kontrol et"""
        print("\n🔍 Checking required files...")

        required_files = [
            self.audio_dir / "hook_audio.mp3",
            self.audio_dir / "subscribe_audio.mp3",
            self.audio_dir / "scene_1_audio.mp3",
            self.audio_dir / "scene_2_audio.mp3"
        ]

        required_images = [
            self.scenes_dir / "scene_01.png",
            self.scenes_dir / "scene_02.png"
        ]

        all_good = True

        for file_path in required_files:
            if file_path.exists():
                print(f"✅ {file_path.name}")
            else:
                print(f"❌ {file_path.name} - NOT FOUND")
                all_good = False

        for file_path in required_images:
            if file_path.exists():
                print(f"✅ {file_path.name}")
            else:
                print(f"❌ {file_path.name} - NOT FOUND")
                all_good = False

        if not all_good:
            print("❌ Missing required files! Please check paths.")
            exit(1)
        else:
            print("✅ All required files found!")

    def get_audio_duration(self, audio_file_path):
        """Get audio duration"""
        try:
            probe = ffmpeg.probe(str(audio_file_path))
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            print(f"⚠️ Could not get duration for {audio_file_path}: {e}")
            return 5.0

    def create_test_sequence(self):
        """Create test video sequence"""
        print("\n📝 Creating test sequence...")

        sequence = []
        total_duration = 0

        # 1. HOOK - 2 random scenes
        hook_file = self.audio_dir / "hook_audio.mp3"
        hook_duration = self.get_audio_duration(hook_file)
        hook_scenes = 2
        hook_scene_duration = hook_duration / hook_scenes

        print(f"🎬 Hook: {hook_duration:.1f}s / {hook_scenes} scenes = {hook_scene_duration:.1f}s each")

        # Use scene 1 and 2 for hook
        for i in [1, 2]:
            scene_file = self.scenes_dir / f"scene_{i:02d}.png"
            sequence.append({
                "type": "hook",
                "image": str(scene_file),
                "duration": hook_scene_duration
            })
        total_duration += hook_duration

        # 2. SUBSCRIBE - 1 scene
        subscribe_file = self.audio_dir / "subscribe_audio.mp3"
        subscribe_duration = self.get_audio_duration(subscribe_file)

        print(f"🔔 Subscribe: {subscribe_duration:.1f}s / 1 scene")

        scene_file = self.scenes_dir / "scene_01.png"
        sequence.append({
            "type": "subscribe",
            "image": str(scene_file),
            "duration": subscribe_duration
        })
        total_duration += subscribe_duration

        # 3. SCENE 1
        scene1_audio = self.audio_dir / "scene_1_audio.mp3"
        scene1_duration = self.get_audio_duration(scene1_audio)
        scene1_image = self.scenes_dir / "scene_01.png"

        print(f"📺 Scene 1: {scene1_duration:.1f}s")

        sequence.append({
            "type": "scene",
            "scene_id": 1,
            "image": str(scene1_image),
            "duration": scene1_duration
        })
        total_duration += scene1_duration

        # 4. SCENE 2
        scene2_audio = self.audio_dir / "scene_2_audio.mp3"
        scene2_duration = self.get_audio_duration(scene2_audio)
        scene2_image = self.scenes_dir / "scene_02.png"

        print(f"📺 Scene 2: {scene2_duration:.1f}s")

        sequence.append({
            "type": "scene",
            "scene_id": 2,
            "image": str(scene2_image),
            "duration": scene2_duration
        })
        total_duration += scene2_duration

        print(f"✅ Total sequence: {len(sequence)} segments, {total_duration:.1f}s ({total_duration / 60:.1f} minutes)")
        return sequence, total_duration

    def create_image_list(self, sequence):
        """Create FFmpeg image list file"""
        list_file = self.project_dir / "test_image_list.txt"

        print(f"\n📝 Creating image list: {list_file}")

        with open(list_file, 'w') as f:
            for i, segment in enumerate(sequence):
                f.write(f"file '{segment['image']}'\n")
                f.write(f"duration {segment['duration']:.2f}\n")
                print(f"  {i + 1}. {Path(segment['image']).name} - {segment['duration']:.1f}s ({segment['type']})")

        print(f"✅ Image list created: {list_file}")
        return list_file

    def combine_test_audio(self):
        """Combine audio files for test"""
        print(f"\n🎵 Combining audio files...")

        audio_files = [
            self.audio_dir / "hook_audio.mp3",
            self.audio_dir / "subscribe_audio.mp3",
            self.audio_dir / "scene_1_audio.mp3",
            self.audio_dir / "scene_2_audio.mp3"
        ]

        audio_list_file = self.project_dir / "test_audio_list.txt"
        combined_audio = self.project_dir / "test_combined_audio.wav"

        # Create audio list
        with open(audio_list_file, 'w') as f:
            for audio_file in audio_files:
                f.write(f"file '{audio_file}'\n")
                duration = self.get_audio_duration(audio_file)
                print(f"  ✅ {audio_file.name} - {duration:.1f}s")

        # Combine with FFmpeg
        try:
            (
                ffmpeg
                .input(str(audio_list_file), format='concat', safe=0)
                .output(str(combined_audio), acodec='pcm_s16le')
                .overwrite_output()
                .run(quiet=True)
            )
            print(f"✅ Combined audio: {combined_audio}")
            return combined_audio
        except Exception as e:
            print(f"❌ Audio combination failed: {e}")
            return None

    def create_test_video(self, image_list_file):
        """Create test video from images"""
        print(f"\n🎬 Creating test video...")

        test_video = self.project_dir / "test_video.mp4"

        try:
            (
                ffmpeg
                .input(str(image_list_file), format='concat', safe=0)
                .filter('scale', 1920, 1080)
                .filter('setsar', 1)
                .output(
                    str(test_video),
                    vcodec='libx264',
                    pix_fmt='yuv420p',
                    framerate=30
                )
                .overwrite_output()
                .run(quiet=True)
            )

            print(f"✅ Test video created: {test_video}")
            return test_video
        except Exception as e:
            print(f"❌ Video creation failed: {e}")
            return None

    def combine_video_audio(self, video_file, audio_file):
        """Combine video and audio"""
        print(f"\n🔗 Combining video and audio...")

        final_video = self.project_dir / "test_final_video.mp4"

        try:
            (
                ffmpeg
                .output(
                    ffmpeg.input(str(video_file)),
                    ffmpeg.input(str(audio_file)),
                    str(final_video),
                    vcodec='libx264',
                    acodec='aac',
                    preset='medium',
                    crf=23
                )
                .overwrite_output()
                .run(quiet=True)
            )

            print(f"✅ Final test video: {final_video}")
            return final_video
        except Exception as e:
            print(f"❌ Video-audio combination failed: {e}")
            return None

    def verify_final_video(self, video_file):
        """Verify final video properties"""
        print(f"\n🔍 Verifying final video...")

        try:
            probe = ffmpeg.probe(str(video_file))

            # Video stream info
            video_stream = None
            audio_stream = None

            for stream in probe['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                elif stream['codec_type'] == 'audio':
                    audio_stream = stream

            duration = float(probe['format']['duration'])

            print(f"✅ Duration: {duration:.1f}s ({duration / 60:.1f} minutes)")

            if video_stream:
                print(f"✅ Video: {video_stream['width']}x{video_stream['height']}, {video_stream['codec_name']}")

            if audio_stream:
                print(f"✅ Audio: {audio_stream['codec_name']}, {audio_stream.get('sample_rate', 'unknown')} Hz")

            file_size = os.path.getsize(video_file) / (1024 * 1024)  # MB
            print(f"✅ File size: {file_size:.1f} MB")

            return True

        except Exception as e:
            print(f"❌ Video verification failed: {e}")
            return False

    def run_test(self):
        """Run complete test"""
        print("\n" + "🧪" * 50)
        print("STARTING VIDEO TEST")
        print("🧪" * 50)

        try:
            # 1. Create sequence
            sequence, total_duration = self.create_test_sequence()

            # 2. Create image list
            image_list = self.create_image_list(sequence)

            # 3. Combine audio
            combined_audio = self.combine_test_audio()
            if not combined_audio:
                return False

            # 4. Create video
            test_video = self.create_test_video(image_list)
            if not test_video:
                return False

            # 5. Combine video + audio
            final_video = self.combine_video_audio(test_video, combined_audio)
            if not final_video:
                return False

            # 6. Verify
            if self.verify_final_video(final_video):
                print("\n" + "🎉" * 50)
                print("TEST SUCCESSFUL!")
                print(f"🎬 Final video: {final_video}")
                print("🎉" * 50)
                return True
            else:
                print("\n❌ TEST FAILED - Video verification failed")
                return False

        except Exception as e:
            print(f"\n💥 TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    try:
        tester = VideoTestSimple()
        success = tester.run_test()

        if success:
            print("\n✅ Test completed successfully!")
        else:
            print("\n❌ Test failed!")

    except KeyboardInterrupt:
        print("\n⏹️ Test stopped by user")
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        import traceback

        traceback.print_exc()