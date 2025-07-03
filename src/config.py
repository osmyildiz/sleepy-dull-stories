import os
import json
import pandas as pd
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import time

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


def generate_runway_prompts(topic: str, description: str) -> dict:
    """Runway için çeşitli promptlar oluştur - HER BİRİ FARKLI"""

    # 10 FARKLI sahne promptu
    scene_variations = [
        f"Wide establishing shot: {topic}. {description} [Cinematic, golden hour lighting, aerial view, 8k, atmospheric]",
        f"Close-up architectural details: Ancient elements of {topic}. {description} [Macro lens, weathered textures, soft shadows, vintage film aesthetic]",
        f"Misty aerial drone view: {topic} shrouded in morning fog. {description} [Bird's eye perspective, ethereal atmosphere, desaturated palette]",
        f"Ground level walkthrough: Moving through {topic} corridors. {description} [Tracking shot, dramatic lighting, stone textures, mysterious ambiance]",
        f"Moonlit courtyard scene: {topic} under starlight. {description} [Night photography, silver moonbeams, deep shadows, romantic lighting]",
        f"Interior chamber view: Sacred spaces within {topic}. {description} [Intimate lighting, golden lanterns, ornate details, reverent atmosphere]",
        f"Garden panorama: Natural elements of {topic}. {description} [Ultra-wide lens, seasonal beauty, flowing water, peaceful serenity]",
        f"Architectural symmetry: Geometric patterns of {topic}. {description} [Perfect symmetry, mathematical beauty, cultural artistry, meditative framing]",
        f"Historical artifacts: Ancient objects within {topic}. {description} [Museum quality lighting, precious materials, cultural treasures, respectful presentation]",
        f"Transitional spaces: Doorways and passages of {topic}. {description} [Threshold moments, symbolic passages, layered depth, contemplative mood]"
    ]

    return {
        "teaser_video": f"Cinematic, moody 60-second introduction to: {topic}. Start with dawn breaking over ancient walls. Slow, meditative camera movements. No dialogue, no people. Desaturated color palette shifting to warm golden tones. Classical Chinese instrumental background. Duration: 60 seconds. Style: Documentary, reverent, sleepy.",
        "image_prompts": scene_variations,
        "topic_cover": f"YouTube thumbnail for 'Sleepy Dull Stories'. Topic: {topic}. Dark academia aesthetic with aged parchment background. Elegant calligraphy title overlay. Subtle imperial symbols. Scholarly, sophisticated design. Text: '{topic}' in gold lettering. Minimalist, high-quality, professional."
    }


def count_words(text: str) -> int:
    """Metindeki kelime sayısını hesapla"""
    return len(text.split())


def generate_story_part(topic: str, description: str, part_number: int, total_parts: int) -> str:
    """Hikayenin belirli bir bölümünü oluştur"""

    system_prompt = (
        "You are a historical narrator creating sleepy, meditative content. "
        "Generate ONE complete part of a multi-part historical story. "
        "Each part should be approximately 2000-3000 words. "
        "Use third-person passive voice with NO questions, dialogue, or listener engagement. "
        "Write in short, clear sentences perfect for slow, peaceful reading. "
        "Focus on atmospheric descriptions, historical details, and tranquil imagery. "
        "Insert occasional empty lines for natural pauses. "
        "The tone should be detached, scholarly, and deeply calming. "
        f"This is Part {part_number} of {total_parts} parts."
    )

    if part_number == 1:
        user_prompt = f"Topic: {topic}\nDescription: {description}\n\nWrite Part 1: Begin the story with a peaceful introduction to the setting at midnight. Focus on the atmosphere and initial descriptions. Target: 2500+ words."
    elif part_number == total_parts:
        user_prompt = f"Topic: {topic}\nDescription: {description}\n\nWrite Part {part_number} (FINAL): Conclude the story as dawn approaches, bringing closure and tranquility. Target: 2500+ words."
    else:
        user_prompt = f"Topic: {topic}\nDescription: {description}\n\nWrite Part {part_number}: Continue exploring different areas and aspects of the topic. Maintain the peaceful, meditative atmosphere. Target: 2500+ words."

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


def generate_story(topic: str, description: str, min_word_count: int = 14000, max_attempts: int = 3) -> str:
    """Hikaye oluştur - bölüm bölüm, uzunluk kontrolü ile"""

    target_parts = 6  # 6 bölüm x ~2500 kelime = ~15000 kelime

    for attempt in range(max_attempts):
        print(f"📝 Hikaye oluşturma denemesi {attempt + 1}/{max_attempts}")

        full_story = ""
        total_words = 0

        for part_num in range(1, target_parts + 1):
            print(f"   📄 Bölüm {part_num}/{target_parts} oluşturuluyor...")

            try:
                part_content = generate_story_part(topic, description, part_num, target_parts)
                part_words = count_words(part_content)

                full_story += f"## Part {part_num}\n\n{part_content}\n\n"
                total_words += part_words

                print(f"   ✅ Bölüm {part_num}: {part_words} kelime")
                time.sleep(1)  # API rate limiting için kısa bekleme

            except Exception as e:
                print(f"   ❌ Bölüm {part_num} hatası: {e}")
                raise

        print(f"📊 Toplam hikaye: {total_words} kelime")

        if total_words >= min_word_count:
            print(f"✅ Hikaye yeterli uzunlukta ({total_words} >= {min_word_count})")
            return full_story.strip()
        else:
            print(f"⚠️  Hikaye hala kısa ({total_words} < {min_word_count})")
            if attempt < max_attempts - 1:
                print("🔄 Daha fazla bölümle tekrar deneniyor...")
                target_parts += 2  # Bir sonraki denemede daha fazla bölüm
                time.sleep(3)

    raise Exception(f"❌ {max_attempts} denemeden sonra yeterli uzunlukta hikaye oluşturulamadı")


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


def create_checklist(output_dir: str, prompts: dict, story_word_count: int):
    """Tamamlanan işlemler için checklist oluştur"""
    checklist = {
        "timestamp": datetime.now().isoformat(),
        "completed_tasks": {
            "story_generated": True,
            "story_word_count": story_word_count,
            "story_meets_requirement": story_word_count >= 14000,
            "teaser_video_prompt": "teaser_video" in prompts,
            "image_prompts_generated": len(prompts.get("image_prompts", [])) == 10,
            "image_prompts_unique": len(set(prompts.get("image_prompts", []))) == 10,
            # Hepsinin farklı olduğunu kontrol et
            "cover_image_prompt": "topic_cover" in prompts,
            "prompts_saved": True
        },
        "quality_checks": {
            "story_estimated_duration_minutes": round(story_word_count / 120),
            "story_meets_2hour_target": story_word_count >= 14000,
            "all_prompts_different": len(set(prompts.get("image_prompts", []))) == 10
        }
    }

    checklist_path = os.path.join(output_dir, "checklist.json")
    with open(checklist_path, "w", encoding="utf-8") as f:
        json.dump(checklist, f, indent=2, ensure_ascii=False)

    print(f"✅ Checklist oluşturuldu: {checklist_path}")
    return checklist


def print_completion_summary(checklist: dict, topic: str, output_path: str):
    """Tamamlama özetini yazdır"""
    print("\n" + "=" * 60)
    print("🎉 SÜREÇ TAMAMLANDI!")
    print("=" * 60)
    print(f"📚 Konu: {topic}")
    print(f"📁 Çıktı dizini: {output_path}")
    print(f"⏰ Tamamlanma zamanı: {checklist['timestamp']}")

    print("\n📋 Kalite Kontrolleri:")
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

    print(f"\n🎬 Prompt Durumu:")
    print(f"  🎥 Video promptu: {'✅' if tasks['teaser_video_prompt'] else '❌'}")
    print(f"  🖼️  Resim promptları: {len(prompts.get('image_prompts', []))} adet")
    print(f"  🔄 Benzersizlik: {'✅ Hepsi farklı' if quality['all_prompts_different'] else '❌ Tekrar var'}")
    print(f"  📱 Cover promptu: {'✅' if tasks['cover_image_prompt'] else '❌'}")

    print("\n📄 Oluşturulan dosyalar:")
    print(f"  📖 story.txt - {tasks['story_word_count']:,} kelime")
    print(f"  🎬 prompt.json - 1 video + 10 benzersiz resim + 1 cover")
    print(f"  ✅ checklist.json - Kalite kontrol raporu")
    print("=" * 60)


if __name__ == "__main__":
    try:
        print("🚀 Sleepy Dull Stories Generator v2.0 Başlatılıyor...")

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

        # Adım 3: Runway promptları oluşturma (FİKSLENDİ)
        print_step(3, "Runway video ve resim promptları oluşturuluyor")
        prompts = generate_runway_prompts(topic, description)
        print(f"🎬 1 video promptu oluşturuldu")
        print(f"🖼️  10 FARKLI resim promptu oluşturuldu")
        print(f"📱 1 cover resmi promptu oluşturuldu")

        # Adım 4: Promptları kaydetme
        print_step(4, "Promptlar kaydediliyor")
        save_prompts(output_path, prompts)

        # Adım 5: Hikaye oluşturma (İYİLEŞTİRİLDİ)
        print_step(5, "ChatGPT'den uzun hikaye oluşturuluyor (6 bölüm)")
        story_text = generate_story(topic, description)
        word_count = count_words(story_text)

        # Adım 6: Hikayeyi kaydetme
        print_step(6, "Hikaye kaydediliyor")
        save_story(output_path, story_text)

        # Adım 7: Checklist oluşturma (GELİŞTİRİLDİ)
        print_step(7, "Kalite kontrol checklist'i oluşturuluyor")
        checklist = create_checklist(output_path, prompts, word_count)

        # Adım 8: Tamamlama özeti
        print_step(8, "Süreç başarıyla tamamlandı!")
        print_completion_summary(checklist, topic, output_path)

    except Exception as e:
        print(f"\n❌ HATA: {e}")
        print("💡 Lütfen .env dosyanızın ve topics.csv dosyanızın doğru konumda olduğundan emin olun.")