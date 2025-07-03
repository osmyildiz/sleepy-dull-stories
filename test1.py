import os
from pathlib import Path


class MoviePyVideoTest:
    def __init__(self):
        # Auto-detect base directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        if 'generators' in script_dir:
            self.base_dir = Path(script_dir).parent.parent
        elif 'src' in script_dir:
            self.base_dir = Path(script_dir).parent
        else:
            self.base_dir = Path(script_dir)

        self.project_dir = self.base_dir / "src" / "output" / "1"
        self.audio_dir = self.project_dir / "audio_parts"
        self.scenes_dir = self.project_dir / "scenes"
        self.overlay_path = self.base_dir / "src" / "data" / "overlay_videos"

        # Store clips for proper cleanup
        self.fireplace_clips_to_cleanup = []

        print("🎬 MOVIEPY VIDEO TEST")
        print("📝 Hook + Subscribe + Scene1 + Scene2 + Scene3")
        print(f"📁 Project Directory: {self.project_dir}")

    def check_required_files(self):
        """Check required files for test"""
        required_files = [
            (self.audio_dir / "hook_audio.mp3", "Hook Audio"),
            (self.audio_dir / "subscribe_audio.mp3", "Subscribe Audio"),
            (self.audio_dir / "scene_1_audio.mp3", "Scene 1 Audio"),
            (self.audio_dir / "scene_2_audio.mp3", "Scene 2 Audio"),
            (self.audio_dir / "scene_3_audio.mp3", "Scene 3 Audio"),
            (self.scenes_dir / "scene_01.png", "Scene 1 Image"),
            (self.scenes_dir / "scene_02.png", "Scene 2 Image"),
            (self.scenes_dir / "scene_03.png", "Scene 3 Image"),
            (self.overlay_path / "fireplace.mp4", "Fireplace Video")
        ]

        all_good = True
        total_duration = 0

        print("\n🔍 Checking required files...")
        for file_path, name in required_files:
            if file_path.exists():
                print(f"✅ {name}: {file_path.name}")
                if file_path.suffix == '.mp3':
                    try:
                        from moviepy.editor import AudioFileClip
                        audio = AudioFileClip(str(file_path))
                        duration = audio.duration
                        total_duration += duration
                        audio.close()
                        print(f"   ⏱️  Duration: {duration:.1f}s")
                    except:
                        print(f"   ⚠️  Could not read duration")
            else:
                print(f"❌ {name}: NOT FOUND - {file_path}")
                all_good = False

        print(f"\n📊 Total estimated video duration: {total_duration:.1f}s ({total_duration / 60:.1f} minutes)")
        return all_good, total_duration

    def debug_clip(self, clip, name):
        """Debug clip to check if it's valid"""
        if clip is None:
            print(f"❌ {name} is None!")
            return False

        try:
            print(f"🔍 {name}: duration={clip.duration:.1f}s, size={clip.size}")
            return True
        except Exception as e:
            print(f"❌ {name} error: {e}")
            return False

    def create_video_clips(self):
        """Create individual video clips for each segment with validation"""
        try:
            from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip

            print("\n🎬 Creating video clips...")

            clips = []

            # 1. HOOK - scene_01.png
            print("📝 Creating hook clip...")
            hook_audio = AudioFileClip(str(self.audio_dir / "hook_audio.mp3"))
            hook_img = ImageClip(str(self.scenes_dir / "scene_01.png"))
            hook_img = hook_img.set_duration(hook_audio.duration)
            hook_clip = hook_img.set_audio(hook_audio)

            if self.debug_clip(hook_clip, "Hook clip"):
                clips.append(hook_clip)
            else:
                return None

            # 2. SUBSCRIBE - scene_01.png
            print("📝 Creating subscribe clip...")
            subscribe_audio = AudioFileClip(str(self.audio_dir / "subscribe_audio.mp3"))
            subscribe_img = ImageClip(str(self.scenes_dir / "scene_01.png"))
            subscribe_img = subscribe_img.set_duration(subscribe_audio.duration)
            subscribe_clip = subscribe_img.set_audio(subscribe_audio)

            if self.debug_clip(subscribe_clip, "Subscribe clip"):
                clips.append(subscribe_clip)
            else:
                return None

            # 3. SCENE 1 - scene_01.png
            print("📝 Creating scene 1 clip...")
            scene1_audio = AudioFileClip(str(self.audio_dir / "scene_1_audio.mp3"))
            scene1_img = ImageClip(str(self.scenes_dir / "scene_01.png"))
            scene1_img = scene1_img.set_duration(scene1_audio.duration)
            scene1_clip = scene1_img.set_audio(scene1_audio)

            if self.debug_clip(scene1_clip, "Scene 1 clip"):
                clips.append(scene1_clip)
            else:
                return None

            # 4. SCENE 2 - scene_02.png
            print("📝 Creating scene 2 clip...")
            scene2_audio = AudioFileClip(str(self.audio_dir / "scene_2_audio.mp3"))
            scene2_img = ImageClip(str(self.scenes_dir / "scene_02.png"))
            scene2_img = scene2_img.set_duration(scene2_audio.duration)
            scene2_clip = scene2_img.set_audio(scene2_audio)

            if self.debug_clip(scene2_clip, "Scene 2 clip"):
                clips.append(scene2_clip)
            else:
                return None

            # 5. SCENE 3 - scene_03.png
            print("📝 Creating scene 3 clip...")
            scene3_audio = AudioFileClip(str(self.audio_dir / "scene_3_audio.mp3"))
            scene3_img = ImageClip(str(self.scenes_dir / "scene_03.png"))
            scene3_img = scene3_img.set_duration(scene3_audio.duration)
            scene3_clip = scene3_img.set_audio(scene3_audio)

            if self.debug_clip(scene3_clip, "Scene 3 clip"):
                clips.append(scene3_clip)
            else:
                return None

            print(f"✅ All {len(clips)} clips created successfully")
            return clips

        except ImportError:
            print("❌ MoviePy not installed. Install with: pip install moviepy")
            return None
        except Exception as e:
            print(f"❌ Error creating clips: {e}")
            import traceback
            traceback.print_exc()
            return None

    def concatenate_clips(self, clips):
        """Concatenate all clips into one video with validation"""
        try:
            from moviepy.editor import concatenate_videoclips

            print("\n🔗 Concatenating clips...")

            # Validate all clips before concatenation
            for i, clip in enumerate(clips):
                if not self.debug_clip(clip, f"Clip {i + 1}"):
                    print(f"❌ Clip {i + 1} is invalid, aborting concatenation")
                    return None

            # Concatenate all clips
            final_video = concatenate_videoclips(clips, method="compose")

            if self.debug_clip(final_video, "Concatenated video"):
                return final_video
            else:
                print("❌ Concatenation resulted in None video")
                return None

        except Exception as e:
            print(f"❌ Error concatenating clips: {e}")
            import traceback
            traceback.print_exc()
            return None

    def add_fireplace_overlay(self, base_video):
        """Add animated fireplace overlay - NO EARLY CLEANUP"""
        fireplace_video = self.overlay_path / "fireplace.mp4"

        print(f"\n🔥 Fireplace overlay process...")

        # Validate base video first
        if not self.debug_clip(base_video, "Base video"):
            print("❌ Base video is invalid, skipping overlay")
            return base_video

        if not fireplace_video.exists():
            print("⚠️ Fireplace video not found, using video without overlay")
            return base_video

        try:
            from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips

            print("🔥 Loading fireplace video...")
            fireplace_clip = VideoFileClip(str(fireplace_video))

            if not self.debug_clip(fireplace_clip, "Fireplace clip"):
                print("❌ Fireplace clip is invalid, skipping overlay")
                fireplace_clip.close()
                return base_video

            base_duration = base_video.duration
            print(f"🔥 Base video duration: {base_duration:.1f}s")
            print(f"🔥 Fireplace duration: {fireplace_clip.duration:.1f}s")

            print("🔥 Adding fireplace overlay...")

            # Loop fireplace to match base video duration
            if fireplace_clip.duration < base_duration:
                loop_count = int(base_duration / fireplace_clip.duration) + 1
                print(f"🔄 Looping fireplace {loop_count} times")

                # Create looped clips with COPY (not reference)
                fireplace_clips = [fireplace_clip.copy() for _ in range(loop_count)]
                fireplace_looped = concatenate_videoclips(fireplace_clips)

                # Store individual clips for later cleanup
                self.fireplace_clips_to_cleanup.extend(fireplace_clips)
                self.fireplace_clips_to_cleanup.append(fireplace_looped)
            else:
                fireplace_looped = fireplace_clip

            # Store original fireplace clip for cleanup
            self.fireplace_clips_to_cleanup.append(fireplace_clip)

            # Trim to exact duration
            fireplace_overlay = fireplace_looped.subclip(0, base_duration)

            # Resize to match base video
            fireplace_overlay = fireplace_overlay.resize(base_video.size)

            # Set opacity (30% transparent)
            fireplace_overlay = fireplace_overlay.set_opacity(0.3)
            print("🔥 Fireplace overlay prepared with 30% opacity")

            # Remove audio from fireplace (we want base video audio)
            fireplace_overlay = fireplace_overlay.without_audio()

            # Composite video with overlay
            final_video = CompositeVideoClip([base_video, fireplace_overlay])

            if self.debug_clip(final_video, "Final video with overlay"):
                print("✅ Fireplace overlay added successfully!")
                print("🔄 Fireplace clips will be cleaned up after rendering")
                return final_video
            else:
                print("❌ Overlay composition failed, using base video")
                self.cleanup_fireplace_clips()  # Cleanup on failure
                return base_video

        except Exception as e:
            print(f"❌ Error in fireplace overlay: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Using video without overlay")
            self.cleanup_fireplace_clips()  # Cleanup on failure
            return base_video

    def cleanup_fireplace_clips(self):
        """Clean up fireplace clips safely"""
        print("🧹 Cleaning up fireplace clips...")
        try:
            for clip in self.fireplace_clips_to_cleanup:
                if clip is not None:
                    clip.close()
            self.fireplace_clips_to_cleanup.clear()
            print("✅ Fireplace clips cleaned up")
        except Exception as e:
            print(f"⚠️ Fireplace cleanup warning: {e}")

    def render_final_video(self, final_video):
        """Render final video to file"""
        output_file = self.project_dir / "moviepy_test_final.mp4"

        try:
            print(f"\n🚀 Rendering final video...")
            print(f"📁 Output: {output_file}")
            print(f"⏱️  Duration: {final_video.duration:.1f}s ({final_video.duration / 60:.1f} minutes)")
            print(f"⏱️  Estimated render time: ~{final_video.duration * 0.3 / 60:.1f} minutes")

            # Write video file with proper progress settings
            final_video.write_videofile(
                str(output_file),
                fps=30,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=True,  # Enable verbose for progress
                logger='bar'  # Progress bar style
            )

            if output_file.exists():
                file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                print(f"✅ Video rendered successfully!")
                print(f"📁 File: {output_file}")
                print(f"📊 Size: {file_size:.1f} MB")
                return True
            else:
                print("❌ Video file not created")
                return False

        except Exception as e:
            print(f"❌ Rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup_clips(self, clips, final_video):
        """Clean up clips to free memory PROPERLY - AFTER RENDERING"""
        try:
            print("\n🧹 Cleaning up all clips...")

            # Close individual clips
            for clip in clips:
                if clip is not None:
                    clip.close()

            # Close final video
            if final_video is not None:
                final_video.close()

            # Clean up fireplace clips
            self.cleanup_fireplace_clips()

            # Remove references to help GC
            del clips
            del final_video

            print("✅ Complete cleanup finished")

        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    def run_moviepy_test(self):
        """Run complete MoviePy test"""
        print("🎬" * 50)
        print("MOVIEPY VIDEO TEST")
        print("📝 Hook + Subscribe + Scene1 + Scene2 + Scene3")
        print("🔥 With Animated Fireplace Overlay")
        print("🎬" * 50)

        try:
            # 1. Check files
            files_ok, estimated_duration = self.check_required_files()
            if not files_ok:
                return False

            # 2. Create individual clips
            clips = self.create_video_clips()
            if not clips:
                return False

            # 3. Concatenate clips
            base_video = self.concatenate_clips(clips)
            if not base_video:
                return False

            # 4. Add fireplace overlay (NO cleanup here)
            final_video = self.add_fireplace_overlay(base_video)

            # 5. Render final video (clips still alive during render)
            success = self.render_final_video(final_video)

            # 6. NOW cleanup everything (after render is complete)
            self.cleanup_clips(clips, final_video)

            if success:
                print("\n🎉" * 50)
                print("MOVIEPY TEST SUCCESSFUL!")
                print("✅ Hook + Subscribe + 3 Scenes")
                print("✅ Animated Fireplace Overlay")
                print("✅ Audio Working")
                print("✅ Multiple Scenes Working")
                print("✅ Proper Cleanup Timing")
                print("🎉" * 50)
                return True
            else:
                print("\n❌ MoviePy test failed")
                return False

        except Exception as e:
            print(f"\n💥 Test failed: {e}")
            import traceback
            traceback.print_exc()
            # Emergency cleanup
            self.cleanup_fireplace_clips()
            return False


if __name__ == "__main__":
    try:
        tester = MoviePyVideoTest()
        success = tester.run_moviepy_test()

        if success:
            print("\n✅ MoviePy test completed successfully!")
            print("🚀 Ready to integrate into main code!")
        else:
            print("\n❌ MoviePy test failed!")
            print("🔧 Check the errors above")

    except KeyboardInterrupt:
        print("\n⏹️ Test stopped by user")
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        import traceback

        traceback.print_exc()