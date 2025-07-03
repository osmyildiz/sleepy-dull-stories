from moviepy.editor import ImageClip, VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips

scene_videos = []

for i in range(1, 41):
    # Dosya yolları
    img_file = f"src/output/1/scenes/scene_{i:02d}.png"
    audio_file = f"src/output/1/audio_parts/scene_{i}_audio.mp3"
    overlay_file = "src/data/overlay_videos/fireplace.mp4"

    # Resim klibi
    img_clip = ImageClip(img_file)

    # Ses klibi
    audio_clip = AudioFileClip(audio_file)
    duration = audio_clip.duration

    # Resmi ses süresine kadar uzat
    img_clip = img_clip.set_duration(duration)

    # Overlay klibi
    overlay_clip = VideoFileClip(overlay_file).subclip(0, duration)
    overlay_clip = overlay_clip.resize(img_clip.size)

    # Katmanları birleştir
    final_video = CompositeVideoClip([img_clip, overlay_clip])
    final_video = final_video.set_audio(audio_clip)

    # Listeye ekle
    scene_videos.append(final_video)

# Hepsini birleştir
full_video = concatenate_videoclips(scene_videos, method="compose")

# Dosyaya yaz
full_video.write_videofile("final_output.mp4", fps=30, codec="libx264", audio_codec="aac")
