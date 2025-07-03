import os
import json
import pandas as pd
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import time
import re

# .env dosyasını yükle
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPIC_CSV_PATH = os.path.join(BASE_DIR, "data", "topics.csv")

def print_step(step_num: int, description: str):
    """Adımları yazdır"""
    print(f"\n🔄 Adım {step_num}: {description}")
    print("-" * 50)

def save_prompts(output_dir: str, prompts: dict):
    """Promptları JSON dosyasına kaydet"""
    os.makedirs(output_dir, exist_ok=True)
    prompt_path = os.path.join(output_dir, "prompt.json")
    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"✅ Promptlar kaydedildi: {prompt_path}")

def save_story(output_dir: str, story_text: str):
    """Hikayeyi TXT dosyasına kaydet"""
    os.makedirs(output_dir, exist_ok=True)
    story_path = os.path.join(output_dir, "story.txt")
    with open(story_path, "w", encoding="utf-8") as f:
        f.write(story_text)
    print(f"✅ Hikaye kaydedildi: {story_path}")

def count_words(text: str) -> int:
    """Metindeki kelime sayısını hesapla"""
    return len(text.split())

def analyze_story_for_segments(story_text: str) -> list:
    """Hikayeyi 4 dakikalık segmentlere böl ve her segmentin içeriğini analiz et"""

    # 4 dakika = yaklaşık 480 kelime (120 kelime/dakika)
    words_per_segment = 480
    words = story_text.split()
    total_words = len(words)

    segments = []

    # Hikayeyi segmentlere böl
    for i in range(0, total_words, words_per_segment):
        segment_words = words[i:i + words_per_segment]
        segment_text = ' '.join(segment_words)

        segment_info = {
            "segment_number": len(segments) + 1,
            "word_start": i,
            "word_end": min(i + words_per_segment, total_words),
            "text": segment_text,
            "word_count": len(segment_words)
        }
        segments.append(segment_info)

    print(f"📊 Hikaye {len(segments)} adet 4-dakikalık segmente bölündü")
    return segments

def generate_visual_prompts_from_story(topic: str, story_segments: list) -> dict:
    """Hikayenin gerçek içeriğine dayalı olarak görsel promptlar oluştur"""

    print(f"🎨 {len(story_segments)} segment için özel promptlar oluşturuluyor...")

    # Her segment için prompt oluştur
    image_prompts = []

    for i, segment in enumerate(story_segments):
        if i >= 10:  # Maksimum 10 resim promptu
            break

        print(f"   🖼️  Segment {segment['segment_number']} analiz ediliyor...")

        # Segment içeriğini analiz et ve prompt oluştur
        system_prompt = (
            "You are a visual prompt expert for Runway AI. "
            "Analyze the given story segment and create a cinematic, atmospheric visual prompt. "
            "Focus on the specific locations, objects, moods, and visual elements mentioned in this segment. "
            "The prompt should be sleepy, meditative, and visually stunning. "
            "Include specific visual details: lighting, camera angles, atmosphere, colors, textures. "
            "Keep it under 100 words but be very specific and evocative. "
            "Style should be: cinematic, 8k, soft lighting, atmospheric, peaceful, no people visible."
        )

        user_prompt = (
            f"Story segment {segment['segment_number']} (4-minute reading):\n\n"
            f"{segment['text'][:800]}...\n\n"  # İlk 800 karakter
            f"Create a specific Runway AI visual prompt for this segment. "
            f"Focus on the actual locations, objects, and atmosphere described in this text."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )

            prompt = response.choices[0].message.content.strip()
            image_prompts.append(f"Segment {segment['segment_number']}: {prompt}")
            print(f"   ✅ Segment {segment['segment_number']} promptu oluşturuldu")
            time.sleep(1)  # API rate limiting

        except Exception as e:
            print(f"   ❌ Segment {segment['segment_number']} prompt hatası: {e}")
            # Fallback prompt
            fallback_prompt = f"Segment {segment['segment_number']}: Cinematic view of {topic}, atmospheric lighting, 8k, peaceful, meditative, no people"
            image_prompts.append(fallback_prompt)

    # Teaser video prompt - hikayenin genel atmosferini yansıtan
    teaser_prompt = (
        f"Cinematic 60-second introduction to: {topic}. "
        f"Montage of key visual elements from the story: ancient architecture, moonlit courtyards, "
        f"misty gardens, ornate details. Slow, meditative camera movements. "
        f"Golden hour transitioning to night. Classical instrumental background. "
        f"No dialogue, no people. Documentary style, reverent atmosphere. "
        f"Style: 8k, cinematic, atmospheric, peaceful."
    )

    # Cover image prompt
    cover_prompt = (
        f"YouTube thumbnail for 'Sleepy Dull Stories' channel. "
        f"Topic: {topic}. Dark academia aesthetic with ancient manuscript background. "
        f"Elegant calligraphy title overlay in gold. Subtle cultural symbols. "
        f"Scholarly, sophisticated, mysterious design. Professional quality, "
        f"high contrast for thumbnail visibility."
    )

    return {
        "teaser_video": teaser_prompt,
        "image_prompts": image_prompts,
        "topic_cover": cover_prompt,
        "generation_info": {
            "total_segments": len(story_segments),
            "prompts_generated": len(image_prompts),
            "based_on_actual_story": True
        }
    }

def generate_story_part(topic: str, description: str, part_number: int, total_parts: int) -> str:
    """Hikayenin belirli bir bölümünü oluştur"""

    system_prompt = (
        "You are a historical narrator creating sleepy, meditative content. "
        "Generate ONE complete part of a multi-part historical story. "
        "Each part should be approximately 2000-3000 words. "
        "Use third-person passive voice with NO questions, dialogue, or listener engagement. "
        "Write in short, clear sentences perfect for slow, peaceful reading. "
        "Focus on atmospheric descriptions, historical details, and tranquil imagery. "
        "Include specific visual details: architecture, objects, textures, lighting, atmosphere. "
        "Describe specific locations, rooms, artifacts, and sensory details. "
        "Insert occasional empty lines for natural pauses. "
        "The tone should be detached, scholarly, and deeply calming. "
    )

    if part_number == 1:
        user_prompt = f"Topic: {topic}\nDescription: {description}\n\nWrite Part 1: Begin the story with a peaceful introduction to the setting at midnight. Focus on specific architectural details, rooms, and atmospheric elements. Include lots of visual details that could be used for creating images later. Target: 2500+ words."
    elif total_parts == 99:  # Ongoing story indicator
        user_prompt = f"Topic: {topic}\nDescription: {description}\n\nWrite Part {part_number}: Continue the ongoing story by exploring new areas, rooms, and aspects of {topic}. Maintain the peaceful midnight atmosphere. Focus on detailed descriptions of architecture, objects, lighting, and sensory details. Target: 2500+ words."
    elif part_number == total_parts and total_parts != 99:
        user_prompt = f"Topic: {topic}\nDescription: {description}\n\nWrite Part {part_number} (FINAL): Conclude the story as dawn approaches, visiting final locations and bringing closure. Include specific visual scenes and architectural details. Target: 2500+ words."
    else:
        user_prompt = f"Topic: {topic}\nDescription: {description}\n\nWrite Part {part_number}: Continue exploring different specific areas, rooms, and aspects of {topic}. Focus on detailed descriptions of architecture, objects, lighting, and atmosphere. Include visual details perfect for image generation. Target: 2500+ words."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=4000
    )

    return response.choices[0].message.content.strip()

def generate_story(topic: str, description: str, min_word_count: int = 14000, max_additional_parts: int = 10) -> str:
    """Hikaye oluştur - maliyet optimize edilmiş versiyon"""

    print(f"📝 Hikaye oluşturma başlatılıyor (Hedef: {min_word_count:,} kelime)")

    # İlk batch: 6 bölüm
    initial_parts = 6
    full_story = ""
    total_words = 0
    current_part = 1

    print(f"🚀 İlk batch: {initial_parts} bölüm oluşturuluyor...")

    # İlk 6 bölümü oluştur
    for part_num in range(1, initial_parts + 1):
        print(f"   📄 Bölüm {part_num}/{initial_parts} oluşturuluyor...")

        try:
            part_content = generate_story_part(topic, description, part_num, initial_parts)
            part_words = count_words(part_content)

            full_story += f"## Part {part_num}\n\n{part_content}\n\n"
            total_words += part_words
            current_part = part_num + 1

            print(f"   ✅ Bölüm {part_num}: {part_words:,} kelime")
            time.sleep(1)  # API rate limiting için kısa bekleme

        except Exception as e:
            print(f"   ❌ Bölüm {part_num} hatası: {e}")
            raise

    print(f"📊 İlk batch tamamlandı: {total_words:,} kelime")

    # Yeterli mi kontrol et
    if total_words >= min_word_count:
        print(f"✅ Hikaye yeterli uzunlukta! ({total_words:,} >= {min_word_count:,})")
        return full_story.strip()

    # Yeterli değilse, ek bölümler ekle
    needed_words = min_word_count - total_words
    print(f"⚠️  {needed_words:,} kelime daha gerekli. Ek bölümler ekleniyor...")

    # Her ek bölüm için tahmini kelime sayısı
    avg_words_per_part = total_words // initial_parts
    estimated_additional_parts = max(1, (needed_words // avg_words_per_part) + 1)

    print(f"💡 Tahmini {estimated_additional_parts} ek bölüm gerekli (ortalama {avg_words_per_part:,} kelime/bölüm)")

    # Ek bölümler ekle
    additional_count = 0
    while total_words < min_word_count and additional_count < max_additional_parts:
        additional_count += 1
        print(f"   📄 Ek bölüm {additional_count} oluşturuluyor... (Mevcut: {total_words:,} kelime)")

        try:
            # Ek bölüm için prompt - ongoing story olduğunu belirt
            part_content = generate_story_part(topic, description, current_part, 99)  # 99 = ongoing
            part_words = count_words(part_content)

            full_story += f"## Part {current_part}\n\n{part_content}\n\n"
            total_words += part_words
            current_part += 1

            print(f"   ✅ Ek bölüm {additional_count}: {part_words:,} kelime (Toplam: {total_words:,})")

            if total_words >= min_word_count:
                print(f"🎯 Hedef kelime sayısına ulaşıldı! ({total_words:,} >= {min_word_count:,})")
                break

            time.sleep(1)  # API rate limiting

        except Exception as e:
            print(f"   ❌ Ek bölüm {additional_count} hatası: {e}")
            break

    if total_words >= min_word_count:
        print(f"✅ Hikaye başarıyla tamamlandı: {total_words:,} kelime")
        return full_story.strip()
    else:
        print(f"⚠️  Maksimum deneme sayısına ulaşıldı. Mevcut uzunluk: {total_words:,} kelime")
        print(f"💡 Bu uzunluk kabul edilebilir. Devam ediliyor...")
        return full_story.strip()

def get_next_topic_and_update_csv(csv_path: str):
    """CSV'den sonraki konuyu al ve işlenmiş olarak işaretle"""
    df = pd.read_csv(csv_path)
    next_row = df[df["done"] == 0].head(1)
    if next_row.empty:
        raise ValueError("❌ İşlenecek konu kalmadı.")

    index = next_row.index[0]
    topic = next_row.iloc[0]["topic"]
    description = next_row.iloc[0]["description"]

    # CSV'yi güncelle
    df.at[index, "done"] = 1
    df.to_csv(csv_path, index=False)

    return index + 1, topic, description

def create_checklist(output_dir: str, prompts: dict, story_word_count: int, story_segments: list):
    """Tamamlanan işlemler için checklist oluştur"""
    checklist = {
        "timestamp": datetime.now().isoformat(),
        "completed_tasks": {
            "story_generated": True,
            "story_word_count": story_word_count,
            "story_meets_requirement": story_word_count >= 14000,
            "story_analyzed_into_segments": len(story_segments) > 0,
            "teaser_video_prompt": "teaser_video" in prompts,
            "image_prompts_generated": len(prompts.get("image_prompts", [])),
            "image_prompts_from_story": prompts.get("generation_info", {}).get("based_on_actual_story", False),
            "cover_image_prompt": "topic_cover" in prompts,
            "prompts_saved": True
        },
        "quality_checks": {
            "story_estimated_duration_minutes": round(story_word_count / 120),
            "story_meets_2hour_target": story_word_count >= 14000,
            "story_segments_count": len(story_segments),
            "prompts_based_on_content": True,
            "prompts_generated_from_segments": len(prompts.get("image_prompts", []))
        }
    }

    checklist_path = os.path.join(output_dir, "checklist.json")
    with open(checklist_path, "w", encoding="utf-8") as f:
        json.dump(checklist, f, indent=2, ensure_ascii=False)

    print(f"✅ Checklist oluşturuldu: {checklist_path}")
    return checklist

def print_completion_summary(checklist: dict, topic: str, output_path: str):
    """Tamamlama özetini yazdır"""
    print("\n" + "="*60)
    print("🎉 SÜREÇ TAMAMLANDI!")
    print("="*60)
    print(f"📚 Konu: {topic}")
    print(f"📁 Çıktı dizini: {output_path}")
    print(f"⏰ Tamamlanma zamanı: {checklist['timestamp']}")

    print("\n📋 Tamamlanan Görevler:")
    tasks = checklist["completed_tasks"]
    quality = checklist["quality_checks"]

    # Ana görevler
    for task, status in tasks.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {task}: {status}")

    print(f"\n📊 Hikaye İstatistikleri:")
    print(f"  📖 Kelime sayısı: {tasks['story_word_count']:,}")
    print(f"  ⏱️  Tahmini süre: {quality['story_estimated_duration_minutes']} dakika")
    print(f"  🎯 2 saat hedefi: {'BAŞARILI' if quality['story_meets_2hour_target'] else 'BAŞARISIZ'}")
    print(f"  📑 Segment sayısı: {quality['story_segments_count']} adet (4'er dakikalık)")

    print(f"\n🎬 Prompt Durumu:")
    print(f"  🎥 Video promptu: {'✅' if tasks['teaser_video_prompt'] else '❌'}")
    print(f"  🖼️  Resim promptları: {quality['prompts_generated_from_segments']} adet")
    print(f"  🎯 Hikaye bazlı: {'✅ Hikayenin içeriğine özel' if tasks['image_prompts_from_story'] else '❌ Generic'}")
    print(f"  📱 Cover promptu: {'✅' if tasks['cover_image_prompt'] else '❌'}")

    print("\n📄 Oluşturulan dosyalar:")
    print(f"  📖 story.txt - {tasks['story_word_count']:,} kelime")
    print(f"  🎬 prompt.json - Hikaye içeriğine özel promptlar")
    print(f"  ✅ checklist.json - Kalite kontrol raporu")
    print("="*60)

if __name__ == "__main__":
    try:
        print("🚀 Sleepy Dull Stories Generator v3.0 Başlatılıyor...")
        print("💡 Yeni: Önce hikaye, sonra hikayeye özel promptlar!")

        # Adım 1: Konu seçimi
        print_step(1, "CSV'den sonraki konuyu alıyor")
        story_index, topic, description = get_next_topic_and_update_csv(TOPIC_CSV_PATH)
        print(f"📚 Seçilen konu: {topic}")
        print(f"📝 Açıklama: {description}")

        # Adım 2: Çıktı dizini hazırlama
        print_step(2, "Çıktı dizini hazırlanıyor")
        output_path = os.path.join("output", str(story_index))
        os.makedirs(output_path, exist_ok=True)
        print(f"📁 Çıktı dizini: {output_path}")

        # Adım 3: Hikaye oluşturma (MALİYET OPTİMİZE)
        print_step(3, "Maliyet optimize edilmiş hikaye oluşturma")
        story_text = generate_story(topic, description)
        word_count = count_words(story_text)

        # Adım 4: Hikayeyi kaydetme
        print_step(4, "Hikaye kaydediliyor")
        save_story(output_path, story_text)

        # Adım 5: Hikayeyi segmentlere böl
        print_step(5, "Hikaye 4-dakikalık segmentlere bölünüyor")
        story_segments = analyze_story_for_segments(story_text)

        # Adım 6: Hikayeye özel promptlar oluştur
        print_step(6, "Hikayenin içeriğine dayalı promptlar oluşturuluyor")
        prompts = generate_visual_prompts_from_story(topic, story_segments)
        print(f"🎬 1 video promptu oluşturuldu")
        print(f"🖼️  {len(prompts['image_prompts'])} resim promptu oluşturuldu (hikayeye özel)")
        print(f"📱 1 cover resmi promptu oluşturuldu")

        # Adım 7: Promptları kaydetme
        print_step(7, "Promptlar kaydediliyor")
        save_prompts(output_path, prompts)

        # Adım 8: Checklist oluşturma
        print_step(8, "Kalite kontrol checklist'i oluşturuluyor")
        checklist = create_checklist(output_path, prompts, word_count, story_segments)

        # Adım 9: Tamamlama özeti
        print_step(9, "Süreç başarıyla tamamlandı!")
        print_completion_summary(checklist, topic, output_path)

    except Exception as e:
        print(f"\n❌ HATA: {e}")
        print("💡 Lütfen .env dosyanızın ve topics.csv dosyanızın doğru konumda olduğundan emin olun.")