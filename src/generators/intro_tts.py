import os
from google.cloud import texttospeech
from dotenv import load_dotenv
import time


# Project root'u bul ve .env dosyasını yükle
def setup_environment():
    """Environment'ı düzgün ayarla"""
    # Current script path
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Project root'u bul (src klasörünün parent'ı)
    if 'src' in current_dir:
        project_root = current_dir.split('src')[0]
    else:
        project_root = os.path.dirname(os.path.dirname(current_dir))

    # .env dosyasını project root'dan yükle
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)

    # Google credentials path'ini düzelt
    credentials_rel_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if credentials_rel_path and not os.path.isabs(credentials_rel_path):
        credentials_abs_path = os.path.join(project_root, credentials_rel_path)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_abs_path
        print(f"🔑 Credentials yolu: {credentials_abs_path}")

    return project_root


# Environment'ı ayarla
project_root = setup_environment()

# İngilizce intro metinleri
INTRO_SCRIPTS = {
    "classic_friendly": {
        "name": "Classic Friendly",
        "text": """Hello dear listeners, and welcome to Sleepy Dull Stories. If you're enjoying these relaxing tales, please don't forget to subscribe to our channel. This way, you can access more peaceful stories. Also, hitting the like button is a small gesture but it means great support for us. Now, let's dive into today's beautifully boring story."""
    },

    "casual_chat": {
        "name": "Casual Chat",
        "text": """Hey there friends! I hope you're having a wonderful and peaceful day. If you enjoy these kinds of calm, sleep-inducing stories, you might consider joining our channel. Clicking that subscribe button takes just a second, but it boosts our motivation for months. Now get comfortable and let's meet today's delightfully dull story."""
    },

    "gentle_request": {
        "name": "Gentle Request",
        "text": """Good evening, beloved listeners. I hope the time you spend on this channel brings you peace. If these calming contents appeal to you, we'd be delighted to have you stay with us. Subscribing is entirely your choice, but it's truly valuable to us. If you're ready, we're now diving into today's sweetly boring story."""
    },

    "gratitude_focused": {
        "name": "Gratitude Focused",
        "text": """Hello wonderful souls! First, thank you for being here. Knowing these stories bring you peace truly makes me happy. If you enjoy this experience, you can become part of this journey by subscribing to our channel. Every new listener is very special to us. Now let's explore today's magnificently boring story together."""
    },

    "relaxed_approach": {
        "name": "Relaxed Approach",
        "text": """Hello everyone! I hope you're feeling comfortable. If you like this channel and want to listen to more peaceful stories, you can take a look at the subscribe button. No rush at all, whenever you feel like it. We'll continue creating tranquil moments for you here. Now let's take a peaceful journey with today's wonderfully boring story."""
    },

    "community_feeling": {
        "name": "Community Feeling",
        "text": """Hello Sleepy Dull Stories family! Yes, you're part of this family too. If you'd like to become an official member of our peaceful community, you can join us by clicking the subscribe button. Being part of this beautiful growing community is wonderful. Now let's all listen to today's exquisite boring story together."""
    },

    "personal_connection": {
        "name": "Personal Connection",
        "text": """Hello there, my dear friend! Yes, I'm talking directly to you. This channel exists for you, and I hope you feel at home here. If these stories are good for you, you might consider subscribing to stay with me on this journey. Your presence is very precious to me. Now together, we're starting today's perfect boring story."""
    },

    "whispered_intimate": {
        "name": "Whispered Intimate",
        "text": """Psst... Hello there. I know you came here for some peace and relaxation. And you're absolutely right. If you'd like to experience these moments more often, how about joining us? Subscribing is just one click, but it's worth the world to us. Now let's quietly listen to today's enchanting boring story."""
    },

    "sharing_spirit": {
        "name": "Sharing Spirit",
        "text": """Hello sharing spirits! I'm grateful that you're experiencing this beautiful moment with me. If you love these peaceful minutes and want others to discover them too, you can help spread this beauty by subscribing to our channel. Together is better. Now let's get lost in today's amazing boring story."""
    },

    "meditation_style": {
        "name": "Meditation Style",
        "text": """Take a deep breath... You are here, in this moment. Every minute you spend on this channel is for your peace. If you want to continue this journey, you can stay with us by subscribing. No obligation, just an invitation. Whenever you're ready. Now let's peacefully focus on today's tranquil boring story."""
    }
}


def create_intro_audio(text, filename, voice_name="en-US-Chirp3-HD-Enceladus"):
    """İngilizce intro metinlerini seslendirme"""

    client = texttospeech.TextToSpeechClient()

    # Metni ayarla
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Ses ayarları
    voice = texttospeech.VoiceSelectionParams(
        name=voice_name,
        language_code="en-US"
    )

    # Audio config - intro için özel ayarlar
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=0.9,  # Biraz daha hızlı (intro için)
        volume_gain_db=1.0,  # Biraz daha yüksek ses
        sample_rate_hertz=24000
    )

    try:
        # Ses sentezi
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Dosyayı kaydet
        with open(filename, "wb") as out:
            out.write(response.audio_content)

        return True, None

    except Exception as e:
        return False, str(e)


def setup_intro_directory():
    """İntro ses dosyaları için klasör yapısı oluştur"""

    intro_dir = os.path.join(project_root, "data", "intro_audios")

    # Klasörü oluştur
    os.makedirs(intro_dir, exist_ok=True)

    return intro_dir


def generate_all_intro_audios():
    """Tüm intro metinlerini seslendirme"""

    print("🎙️ SLEEPY DULL STORIES - İngilizce İntro Seslendirme")
    print("=" * 60)

    # Klasör hazırlama
    intro_dir = setup_intro_directory()
    print(f"📁 Klasör hazırlandı: {intro_dir}")

    success_count = 0
    error_count = 0
    total_count = len(INTRO_SCRIPTS)

    for idx, (script_id, script_data) in enumerate(INTRO_SCRIPTS.items(), 1):

        print(f"\n🎤 İşleniyor {idx}/{total_count}: {script_data['name']}")

        # Dosya adını oluştur
        filename = f"intro_{script_id}.mp3"
        filepath = os.path.join(intro_dir, filename)

        # Metin bilgisi
        word_count = len(script_data['text'].split())
        print(f"📝 Kelime sayısı: {word_count}")
        print(f"💾 Dosya: {filename}")

        # Seslendirme
        start_time = time.time()
        success, error = create_intro_audio(script_data['text'], filepath)
        end_time = time.time()

        processing_time = round(end_time - start_time, 1)

        if success:
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"✅ Başarılı! ({processing_time}s, {file_size:.1f} KB)")
            success_count += 1
        else:
            print(f"❌ Hata: {error}")
            error_count += 1

        # API rate limiting
        if idx < total_count:
            time.sleep(1)

    # Sonuç raporu
    print(f"\n📊 SONUÇ RAPORU:")
    print("=" * 30)
    print(f"✅ Başarılı: {success_count}")
    print(f"❌ Hatalı: {error_count}")
    print(f"📁 Toplam: {total_count}")
    print(f"📂 Konum: {intro_dir}")

    if success_count > 0:
        print(f"\n🎉 {success_count} intro başarıyla oluşturuldu!")
        print("🎵 Dosyalar:")
        for script_id, script_data in INTRO_SCRIPTS.items():
            filename = f"intro_{script_id}.mp3"
            filepath = os.path.join(intro_dir, filename)
            if os.path.exists(filepath):
                print(f"  📀 {script_data['name']}: {filename}")


def create_intro_list_file():
    """İntro dosyaları listesi oluştur"""

    intro_dir = setup_intro_directory()
    list_file = os.path.join(intro_dir, "intro_list.txt")

    with open(list_file, 'w', encoding='utf-8') as f:
        f.write("SLEEPY DULL STORIES - İngilizce İntro Ses Dosyaları\n")
        f.write("=" * 50 + "\n\n")

        for idx, (script_id, script_data) in enumerate(INTRO_SCRIPTS.items(), 1):
            f.write(f"{idx}. {script_data['name']}\n")
            f.write(f"   Dosya: intro_{script_id}.mp3\n")
            f.write(f"   Metin: {script_data['text'][:100]}...\n")
            f.write(f"   Kelime: {len(script_data['text'].split())}\n\n")

    print(f"📄 Liste dosyası oluşturuldu: intro_list.txt")


def test_single_intro(script_id="classic_friendly"):
    """Tek bir intro'yu test etme"""

    if script_id not in INTRO_SCRIPTS:
        print(f"❌ Geçersiz script ID: {script_id}")
        return

    print(f"🧪 Test: {INTRO_SCRIPTS[script_id]['name']}")

    intro_dir = setup_intro_directory()
    filename = f"test_intro_{script_id}.mp3"
    filepath = os.path.join(intro_dir, filename)

    success, error = create_intro_audio(INTRO_SCRIPTS[script_id]['text'], filepath)

    if success:
        print(f"✅ Test başarılı: {filename}")
    else:
        print(f"❌ Test hatası: {error}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test modu
        test_script = sys.argv[2] if len(sys.argv) > 2 else "classic_friendly"
        test_single_intro(test_script)
    else:
        # Ana işlem
        try:
            generate_all_intro_audios()
            create_intro_list_file()
            print("\n🎯 Tüm işlemler tamamlandı!")

        except KeyboardInterrupt:
            print("\n⚠️ İşlem kullanıcı tarafından durduruldu")
        except Exception as e:
            print(f"\n❌ Beklenmeyen hata: {e}")