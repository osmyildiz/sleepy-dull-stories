"""
Hızlı Hook & Subscribe Video Test
Sadece hook ve subscribe videolarını render eder - 2-3 dakika
"""

import json
import os
import random
import time
from pathlib import Path
import subprocess

def quick_hook_subscribe_test():
    """Hızlı hook ve subscribe video testi"""

    # Paths - src/generators'dan çalışıyor
    project_root = Path(__file__).parent.parent.parent  # src/generators -> sleepy-dull-stories
    output_dir = project_root / "output"

    # İlk mevcut proje klasörünü bul
    project_folders = [d for d in output_dir.glob("*") if d.is_dir()]
    if not project_folders:
        print("❌ output/ dizininde hiç proje bulunamadı!")
        return

    # İlk projeyi seç
    selected_project = sorted(project_folders)[0]
    print(f"📁 Seçilen proje: {selected_project.name}")

    audio_dir = selected_project / "audio_parts"
    scenes_dir = selected_project / "scenes"
    fireplace_path = project_root / "overlays" / "fireplace.mp4"

    print("🚀 HIZLI HOOK & SUBSCRIBE TEST")
    print("=" * 50)

    # Check directories
    if not selected_project.exists():
        print(f"❌ {selected_project} dizini bulunamadı!")
        return

    if not audio_dir.exists():
        print("❌ audio_parts dizini bulunamadı!")
        return

    if not scenes_dir.exists():
        print("❌ scenes dizini bulunamadı!")
        return

    # Timeline dosyasını yükle
    timeline_file = selected_project / "story_audio_youtube_timeline.json"
    if not timeline_file.exists():
        print("❌ Timeline dosyası bulunamadı!")
        return

    with open(timeline_file, 'r', encoding='utf-8') as f:
        timeline_data = json.load(f)

    # Hook ve Subscribe verilerini bul
    hook_scene = None
    subscribe_scene = None
    story_scenes = []

    for scene in timeline_data.get('scenes', []):
        if scene['type'] == 'youtube_hook':
            hook_scene = scene
        elif scene['type'] == 'youtube_subscribe':
            subscribe_scene = scene
        elif scene['type'] == 'story_scene':
            story_scenes.append(scene)

    print(f"📊 Story scenes bulundu: {len(story_scenes)}")
    print(f"🎣 Hook scene: {'✅' if hook_scene else '❌'}")
    print(f"📺 Subscribe scene: {'✅' if subscribe_scene else '❌'}")
    print(f"🔥 Fireplace: {'✅' if fireplace_path.exists() else '❌'}")

    if not hook_scene and not subscribe_scene:
        print("❌ Hook veya Subscribe scene bulunamadı!")
        return

    # Test dizini oluştur
    test_dir = selected_project / "quick_test"
    test_dir.mkdir(exist_ok=True)

    def get_audio_duration(audio_file):
        """Audio dosya süresini al"""
        try:
            import ffmpeg
            probe = ffmpeg.probe(str(audio_file))
            duration = float(probe['format']['duration'])
            return duration
        except:
            return 10.0  # Fallback

    def find_scene_image(scene_id):
        """Scene image dosyasını bul"""
        for format_str in [f"scene_{scene_id:02d}", f"scene_{scene_id}"]:
            # 4K versiyonu önce
            for suffix in ["_4k.png", ".png"]:
                image_file = scenes_dir / f"{format_str}{suffix}"
                if image_file.exists():
                    return image_file
        return None

    def create_quick_video(scene_data, scene_type, output_filename):
        """Hızlı video oluştur"""
        print(f"\n🎬 {scene_type.upper()} video oluşturuluyor...")

        # Audio dosyası
        audio_file = audio_dir / scene_data['audio_file']
        if not audio_file.exists():
            print(f"   ❌ Audio dosyası bulunamadı: {scene_data['audio_file']}")
            return False

        scene_duration = get_audio_duration(audio_file)
        print(f"   ⏱️ Süre: {scene_duration:.1f}s")

        # Random scene seçimi (hook/subscribe için)
        if scene_type == "hook":
            selected_scenes = random.sample(story_scenes[:10], min(3, len(story_scenes)))
        else:  # subscribe
            selected_scenes = random.sample(story_scenes[:8], min(2, len(story_scenes)))

        print(f"   🎲 Seçilen sahneler: {[s['scene_number'] for s in selected_scenes]}")

        # Her image için süre
        image_duration = scene_duration / len(selected_scenes)

        # FFmpeg input listesi oluştur
        inputs = []
        filters = []

        for i, scene_info in enumerate(selected_scenes):
            scene_id = scene_info['scene_number']
            image_file = find_scene_image(scene_id)

            if not image_file:
                print(f"   ⚠️ Scene {scene_id} image bulunamadı, atlanıyor...")
                continue

            print(f"   📸 Scene {scene_id}: {image_file.name}")

            # FFmpeg input
            inputs.extend(['-loop', '1', '-t', str(image_duration), '-i', str(image_file)])

            # Scale to 4K ve overlay hazırlığı
            filters.append(f"[{i}:v]scale=3840:2160,setpts=PTS-STARTPTS[v{i}]")

        if not filters:
            print("   ❌ Hiç image bulunamadı!")
            return False

        # Video concatenation
        video_inputs = ''.join(f"[v{i}]" for i in range(len(filters)))
        filters.append(f"{video_inputs}concat=n={len(filters)}:v=1:a=0[video_main]")

        # Fireplace overlay ekle
        fireplace_filter = ""
        if fireplace_path.exists():
            print(f"   🔥 Fireplace overlay ekleniyor...")
            inputs.extend(['-stream_loop', '-1', '-i', str(fireplace_path)])
            fireplace_idx = len(selected_scenes)

            # Fireplace'i scale et ve opacity ver
            filters.append(f"[{fireplace_idx}:v]scale=3840:2160,format=yuva420p,colorchannelmixer=aa=0.3[fireplace]")
            filters.append(f"[video_main][fireplace]overlay=0:0[video_final]")
            video_output = "[video_final]"
        else:
            print(f"   ⚠️ Fireplace bulunamadı, sadece ana video")
            video_output = "[video_main]"

        # Audio input
        inputs.extend(['-i', str(audio_file)])
        audio_idx = len(selected_scenes) + (1 if fireplace_path.exists() else 0)

        # Output file
        output_file = test_dir / output_filename

        # FFmpeg command
        cmd = [
            'ffmpeg', '-y'
        ] + inputs + [
            '-filter_complex', '; '.join(filters),
            '-map', video_output,
            '-map', f'{audio_idx}:a',
            '-c:v', 'libx264',
            '-preset', 'fast',  # Hızlı encode
            '-crf', '23',       # Orta kalite
            '-c:a', 'aac',
            '-t', str(scene_duration),  # Süre sınırı
            str(output_file)
        ]

        print(f"   🚀 FFmpeg render başlıyor...")
        start_time = time.time()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout

            if result.returncode == 0:
                render_time = time.time() - start_time
                file_size = os.path.getsize(output_file) / (1024 * 1024)
                print(f"   ✅ Render tamamlandı!")
                print(f"   ⏱️ Süre: {render_time:.1f}s")
                print(f"   📦 Boyut: {file_size:.1f}MB")
                print(f"   📁 Dosya: {output_file}")
                return True
            else:
                print(f"   ❌ FFmpeg hatası:")
                print(f"   {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"   ❌ Timeout (5 dakika aşıldı)")
            return False
        except Exception as e:
            print(f"   ❌ Hata: {e}")
            return False

    # Hook video oluştur
    hook_success = False
    if hook_scene:
        hook_success = create_quick_video(hook_scene, "hook", "hook_test.mp4")

    # Subscribe video oluştur
    subscribe_success = False
    if subscribe_scene:
        subscribe_success = create_quick_video(subscribe_scene, "subscribe", "subscribe_test.mp4")

    # Sonuçlar
    print(f"\n📊 HIZLI TEST SONUÇLARI:")
    print(f"   🎣 Hook: {'✅ Başarılı' if hook_success else '❌ Başarısız'}")
    print(f"   📺 Subscribe: {'✅ Başarılı' if subscribe_success else '❌ Başarısız'}")

    if hook_success or subscribe_success:
        print(f"\n🎉 Test videolarınız hazır!")
        print(f"📁 Konum: {test_dir}")
        print(f"🔍 İçindekiler:")
        for file in test_dir.glob("*.mp4"):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"   📹 {file.name} ({size_mb:.1f}MB)")

if __name__ == "__main__":
    quick_hook_subscribe_test()