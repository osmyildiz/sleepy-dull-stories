import os
import json
import time
import requests
import re
from pydub import AudioSegment
from datetime import datetime

# ElevenLabs Configuration
ELEVENLABS_API_KEY = "sk_d48c501bddb3e09f0ba95f7143202925e3c7482003ccb2a6"  # .env veya direkt buraya
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Sarah (default) - Rachel için farklı ID gerekir
BASE_SPEED = 0.85  # Daha hızlı


def load_voice_guide():
    """story_1_voice_guide.json dosyasını yükle"""
    guide_file = "story_1_voice_guide.json"

    if not os.path.exists(guide_file):
        print(f"❌ {guide_file} bulunamadı!")
        return None

    try:
        with open(guide_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ JSON okuma hatası: {e}")
        return None


def get_elevenlabs_api_key():
    """ElevenLabs API key'ini al"""
    api_key = os.getenv('ELEVENLABS_API_KEY') or ELEVENLABS_API_KEY

    if not api_key or api_key == "your_api_key_here":
        print("❌ ElevenLabs API key gerekli!")
        print("💡 https://elevenlabs.io/speech-synthesis adresinden key al")
        print("💡 Sonra buraya ekle veya ELEVENLABS_API_KEY env variable set et")
        return None

    return api_key


def pace_to_stability_similarity(pace, pitch=0, emotion_type="peaceful"):
    """Pace + pitch + emotion'u ElevenLabs parametrelerine çevir - ENHANCED"""

    # Emotional base settings
    emotion_settings = {
        "cinematic_opening": {"base_stability": 0.4, "base_similarity": 0.6, "style_boost": 0.7},
        "profound_gravitas": {"base_stability": 0.6, "base_similarity": 0.8, "style_boost": 0.5},
        "sensory_immersion": {"base_stability": 0.3, "base_similarity": 0.5, "style_boost": 0.8},
        "catalog_of_wonders": {"base_stability": 0.2, "base_similarity": 0.4, "style_boost": 0.9},
        "reflective_nostalgia": {"base_stability": 0.5, "base_similarity": 0.7, "style_boost": 0.6},
        "tactile_reverence": {"base_stability": 0.4, "base_similarity": 0.6, "style_boost": 0.7},
        "expansive_vista": {"base_stability": 0.3, "base_similarity": 0.5, "style_boost": 0.8},
        "poetic_imagery": {"base_stability": 0.4, "base_similarity": 0.6, "style_boost": 0.7},
        "philosophical_depth": {"base_stability": 0.6, "base_similarity": 0.8, "style_boost": 0.4},
        "melancholic_beauty": {"base_stability": 0.7, "base_similarity": 0.9, "style_boost": 0.3},
        "serene_acceptance": {"base_stability": 0.8, "base_similarity": 0.9, "style_boost": 0.2},
        "transcendent_finale": {"base_stability": 0.9, "base_similarity": 0.95, "style_boost": 0.1}
    }

    # Get emotion base or default
    emotion_base = emotion_settings.get(emotion_type,
                                        {"base_stability": 0.5, "base_similarity": 0.7, "style_boost": 0.5})

    # Pace influence (0.45-0.7 range)
    pace_factor = (0.7 - pace) / 0.25  # 0.0-1.0 range, higher = slower

    # Pitch influence (-5 to +5)
    pitch_factor = pitch / 10.0  # -0.5 to +0.5

    # Calculate final values
    stability = emotion_base["base_stability"] + (pace_factor * 0.3) + (pitch_factor * 0.1)
    similarity = emotion_base["base_similarity"] + (pace_factor * 0.2) - abs(pitch_factor * 0.1)
    style = emotion_base["style_boost"] - (pace_factor * 0.3) + abs(pitch_factor * 0.2)

    # Clamp values
    stability = max(0.0, min(1.0, stability))
    similarity = max(0.0, min(1.0, similarity))
    style = max(0.0, min(1.0, style))

    # Speed based on pace
    speed = min(1.0, pace * 1.4)  # 0.45*1.4=0.63, 0.7*1.4=0.98

    return stability, similarity, style, speed


def preprocess_text_for_elevenlabs(raw_text: str, emotion="neutral") -> str:
    """
    ElevenLabs için doğal, insani, nefesli bir okuma elde etmek için metni işler.
    SSML yerine dramatik ve mikro duraklamalar için '...' kullanır.
    """
    # --------------------------
    # 1. Kısa doğal duraklamalar (virgül, noktalı virgül, bağlaçlar)
    # --------------------------
    short_pause_words = [' and ', ' but ', ' while ', ' though ', ' as ', ' because ']
    for word in short_pause_words:
        raw_text = raw_text.replace(word, f"{word.strip()}... ")

    # --------------------------
    # 2. Orta duraklamalar (ikinci cümleye geçiş gibi yerler)
    # --------------------------
    mid_pause_words = [' therefore ', ' however ', ' although ', ' instead ', ' nonetheless ']
    for word in mid_pause_words:
        raw_text = raw_text.replace(word, f"... {word.strip()}... ")

    # --------------------------
    # 3. Virgül ve noktalama sonrası doğal duraksama
    # --------------------------
    raw_text = re.sub(r",\s*", "... ", raw_text)
    raw_text = re.sub(r";\s*", "... ", raw_text)

    # --------------------------
    # 4. Uzun cümlelerde anlam bölümü sonrası duraksama
    # --------------------------
    words = raw_text.split()
    if len(words) > 16:  # cümle uzunsa, ortalara bir '...' yerleştir
        midpoint = len(words) // 2
        for i in range(midpoint, min(len(words) - 1, midpoint + 4)):
            if words[i].endswith('.') or words[i].endswith(',') or words[i] in ['and', 'but']:
                words[i] = words[i] + "..."
                break
        raw_text = ' '.join(words)

    # --------------------------
    # 5. Özel kelime/duygu bazlı manipülasyonlar - CLEOPATRA ENHANCED
    # --------------------------
    if emotion in ["whisper", "intimate", "tactile_reverence"]:
        raw_text = raw_text.replace("whispers", "whispers...")
        raw_text = raw_text.replace("gently", "gently...")
        raw_text = raw_text.replace("softly", "softly...")

    elif emotion in ["dramatic", "cinematic_opening", "profound_gravitas"]:
        raw_text = raw_text.replace("suddenly", "suddenly... everything changed")
        raw_text = raw_text.replace("magnificent", "magnificent...")
        raw_text = raw_text.replace("pharaoh", "pharaoh...")
        raw_text = raw_text.replace("Alexandria", "Alexandria...")

    elif emotion in ["contemplative", "philosophical_depth", "transcendent_finale"]:
        raw_text = raw_text.replace("eternal", "eternal...")
        raw_text = raw_text.replace("forever", "forever...")
        raw_text = raw_text.replace("infinite", "infinite...")
        raw_text = raw_text.replace("millennia", "millennia...")

    elif emotion in ["poetic_imagery", "expansive_vista"]:
        raw_text = raw_text.replace("golden", "golden...")
        raw_text = raw_text.replace("like", "like...")
        raw_text = raw_text.replace("through", "through...")
        raw_text = raw_text.replace("Mediterranean", "Mediterranean...")

    # --------------------------
    # 6. [PAUSE] varsa onu da '...' ile değiştir
    # --------------------------
    raw_text = raw_text.replace("[PAUSE]", "...")

    # --------------------------
    # 7. Çift '...' temizle
    # --------------------------
    raw_text = re.sub(r'\.\.\.+', '...', raw_text)

    return raw_text.strip()


def create_elevenlabs_audio(text, voice_settings, api_key, sentence_num):
    """ElevenLabs ile tek cümle seslendirme - ENHANCED"""

    # Voice settings'i ElevenLabs formatına çevir - ENHANCED
    pace = voice_settings['pace']
    pitch = voice_settings.get('pitch', 0)
    volume = voice_settings.get('volume', 80)
    tone = voice_settings.get('tone', 'peaceful')

    stability, similarity, style, speed = pace_to_stability_similarity(pace, pitch, tone)

    # Text'i doğal nefes alması için işle - SENIN GELIŞMIŞ FONKSİYON
    processed_text = preprocess_text_for_elevenlabs(text, tone)

    print(
        f"   🎭 ElevenLabs: stability={stability:.2f}, similarity={similarity:.2f}, style={style:.2f}, speed={speed:.2f}")
    print(f"   🎨 Emotion: {tone} | Advanced breathing algorithm applied")
    print(f"   📝 Text preview: {processed_text[:60]}...")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }

    # ElevenLabs optimized settings - ENHANCED
    data = {
        "text": processed_text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity,
            "style": style,  # Enhanced emotional variation
            "use_speaker_boost": True
        }
    }

    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()

        # Save temporary file
        temp_file = f"temp_eleven_{sentence_num}.mp3"
        with open(temp_file, 'wb') as f:
            f.write(response.content)

        return temp_file, None

    except Exception as e:
        return None, str(e)


def create_audio_from_detailed_guide_elevenlabs(voice_guide, output_file="test_story1_elevenlabs1.mp3"):
    """ElevenLabs ile detaylı voice guide'dan audio oluştur"""

    if not voice_guide:
        return False

    print("🎙️ ELEVENLABS DETAYLI VOICE GUIDE TEST")
    print(f"📚 Story: {voice_guide['title']}")
    print(f"⏱️ Target: {voice_guide['total_duration_target']}")
    print(f"🎭 Base Speed: {BASE_SPEED} (daha hızlı)")
    print(f"🔊 Voice ID: {VOICE_ID}")
    print("=" * 50)

    # API key kontrolü
    api_key = get_elevenlabs_api_key()
    if not api_key:
        return False

    try:
        combined_audio = AudioSegment.empty()
        temp_files = []  # Cleanup için

        # Her sentence için ayrı ayrı işle
        for i, direction in enumerate(voice_guide['detailed_voice_directions']):
            sentence_num = direction['sentence']
            text = direction['text']
            voice_settings = direction['voice_settings']

            print(f"🔊 Cümle {sentence_num}: {text[:50]}...")

            # [PAUSE] işaretlerini handle et - DAHA KISA
            if text.startswith("[PAUSE]"):
                pause_before = direction.get('pause_before', 2.0) * 0.5  # %50 azalt
                text = text.replace("[PAUSE]", "").strip()

                # Önce pause ekle
                pause_audio = AudioSegment.silent(duration=int(pause_before * 1000))
                combined_audio += pause_audio
                print(f"   ⏸️ Pause before: {pause_before:.1f}s")

            # ElevenLabs TTS request
            temp_file, error = create_elevenlabs_audio(text, voice_settings, api_key, sentence_num)

            if error:
                print(f"   ❌ ElevenLabs error: {error}")
                continue

            if temp_file and os.path.exists(temp_file):
                # AudioSegment'e ekle
                sentence_audio = AudioSegment.from_mp3(temp_file)
                combined_audio += sentence_audio
                temp_files.append(temp_file)

                print(f"   ✅ Generated: {len(sentence_audio) / 1000:.1f}s")

            # Cümle sonrası pause - DAHA KISA VE DOGAL
            pause_after = direction.get('pause_after', 1.0)
            if 'final_pause' in direction:
                pause_after = min(direction['final_pause'], 4.0)  # Max 4 saniye

            # Pause'ları daha kısa yap
            pause_after = pause_after * 0.6  # %40 azalt

            pause_audio = AudioSegment.silent(duration=int(pause_after * 1000))
            combined_audio += pause_audio

            print(f"   ⏸️ Pause after: {pause_after}s")

            # Rate limiting - ElevenLabs için
            time.sleep(1.5)

        # Cleanup temp files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

        # Final audio export
        combined_audio.export(output_file, format="mp3", bitrate="192k")

        duration_min = len(combined_audio) / 60000
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

        print(f"\n🎉 ELEVENLABS BAŞARILI!")
        print(f"📁 Dosya: {output_file}")
        print(f"⏱️ Süre: {duration_min:.1f} dakika")
        print(f"📦 Boyut: {file_size_mb:.1f} MB")
        print(f"🎭 Cümle sayısı: {len(voice_guide['detailed_voice_directions'])}")
        print(f"🚀 Daha hızlı tempo: {BASE_SPEED}")
        print(f"🎨 Duygusal variation: Her cümle farklı stability/similarity")

        # Orijinal hedef ile karşılaştır
        target_duration = float(voice_guide['total_duration_target'].replace(' minutes', ''))
        accuracy = (duration_min / target_duration) * 100
        print(f"🎯 Hedef accuracy: {accuracy:.1f}% ({duration_min:.1f}/{target_duration})")

        return True

    except Exception as e:
        print(f"❌ ElevenLabs audio oluşturma hatası: {e}")
        return False


def test_elevenlabs():
    """ElevenLabs test fonksiyonu"""
    print("🧪 ELEVENLABS DETAYLI VOICE GUIDE TEST")
    print("🎨 Duygusal kontrol + Daha hızlı tempo")
    print("=" * 60)

    # Voice guide'ı yükle
    voice_guide = load_voice_guide()
    if not voice_guide:
        print("💡 story_1_voice_guide.json dosyasını aynı klasöre kaydet")
        return

    # ElevenLabs test çalıştır
    success = create_audio_from_detailed_guide_elevenlabs(voice_guide)

    if success:
        print("\n✅ ElevenLabs Test başarılı!")
        print("🎧 test_story1_elevenlabs1.mp3 dosyasını dinleyebilirsin")
        print("🎭 Google TTS'e göre çok daha duygusal olmalı")
        print("🚀 0.85 hızında daha akıcı")
        print("🎨 Her cümle farklı stability/similarity ile")
        print("📊 Pace değişiklikleri emotional variation'a çevrildi")
    else:
        print("\n❌ ElevenLabs Test başarısız!")


if __name__ == "__main__":
    test_elevenlabs()