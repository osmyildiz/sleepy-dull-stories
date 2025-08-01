import json
import os
from PIL import Image, ImageDraw, ImageFont
import re


def load_thumbnail_config():
    """thumbnail_features.json'ı yükle - src/generators/ dizininden src/data/ dizinine"""
    try:
        # src/generators/ dizininden src/data/ dizinine çık
        config_path = "../data/thumbnail_features.json"
        print(f"🔍 Config path: {config_path}")

        if not os.path.exists(config_path):
            print(f"❌ Config dosyası bulunamadı: {config_path}")
            return None

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✅ Thumbnail config yüklendi")
        return config
    except Exception as e:
        print(f"❌ Config yükleme hatası: {e}")
        return None


def load_fonts():
    """Fontları yükle - src/generators/ dizininden fonts/ dizinine"""
    # src/generators/ dizininden proje root'una çık
    font_paths = {
        'shocking': '../../fonts/Poppins-Bold.ttf',
        'main': '../../fonts/CrimsonText-Bold.ttf',
        'bottom': '../../fonts/CrimsonText-Italic.ttf',
        'channel': '../../fonts/Lora-VariableFont_wght.ttf'
    }

    loaded_fonts = {}
    for font_type, path in font_paths.items():
        if os.path.exists(path):
            loaded_fonts[font_type] = path
            print(f"✅ {font_type}: {os.path.basename(path)}")
        else:
            loaded_fonts[font_type] = None
            print(f"❌ {font_type}: {path} bulunamadı, varsayılan font kullanılacak")

    return loaded_fonts


def hex_to_rgb(hex_color):
    """Hex rengi RGB'ye çevir"""
    try:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    except:
        return (255, 255, 255)


def remove_emojis(text):
    """Emoji karakterlerini kaldır"""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    clean_text = emoji_pattern.sub(r'', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text


def find_topic_by_id(topic_id, config):
    """ID ile config'de topic bul"""
    for topic in config['topics']:
        if topic['id'] == topic_id:
            print(f"✅ Topic bulundu: ID={topic_id} -> {topic['topic']}")
            return topic

    print(f"❌ Topic bulunamadı: ID={topic_id}")
    return None


def create_thumbnail(background_image_path, topic_id, output_path):
    """Basit thumbnail oluştur"""

    print(f"\n🎨 ID {topic_id} için thumbnail oluşturuluyor...")
    print(f"📸 Background: {background_image_path}")
    print(f"💾 Output: {output_path}")

    # Config'i yükle
    config = load_thumbnail_config()
    if not config:
        return False

    # Topic'i bul
    topic_config = find_topic_by_id(topic_id, config)
    if not topic_config:
        return False

    # Template ayarları
    template = config['thumbnail_template']
    width = template['layout']['width']
    height = template['layout']['height']

    try:
        # Background resmi yükle
        if not os.path.exists(background_image_path):
            print(f"❌ Background resim bulunamadı: {background_image_path}")
            return False

        img = Image.open(background_image_path)
        print(f"📸 Orijinal resim: {img.size}, mode: {img.mode}")

        # RGB'ye çevir ve boyutlandır
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((width, height), Image.Resampling.LANCZOS)
        print(f"✅ Resim ayarlandı: {width}x{height}")

        draw = ImageDraw.Draw(img)

        # Fontları yükle
        fonts = load_fonts()

        # Font boyutları
        shocking_size = template['sections']['shocking_word']['font_size']
        main_size = template['sections']['main_title']['font_size']
        bottom_size = template['sections']['bottom_text']['font_size']
        channel_size = template['sections']['channel']['font_size']

        try:
            shocking_font = ImageFont.truetype(fonts['shocking'], shocking_size) if fonts[
                'shocking'] else ImageFont.load_default()
            main_font = ImageFont.truetype(fonts['main'], main_size) if fonts['main'] else ImageFont.load_default()
            bottom_font = ImageFont.truetype(fonts['bottom'], bottom_size) if fonts[
                'bottom'] else ImageFont.load_default()
            channel_font = ImageFont.truetype(fonts['channel'], channel_size) if fonts[
                'channel'] else ImageFont.load_default()
        except:
            print("⚠️ Font yükleme sorunu, varsayılan fontlar kullanılıyor")
            shocking_font = main_font = bottom_font = channel_font = ImageFont.load_default()

        print("\n📝 Yazılar ekleniyor...")

        # 1. SHOCKING WORD
        try:
            shocking_section = topic_config['sections']['shocking_word']
            shocking_pos = (
                template['sections']['shocking_word']['x_offset'],
                template['sections']['shocking_word']['y_offset']
            )

            shocking_text = remove_emojis(shocking_section['text'])
            shocking_color = hex_to_rgb(shocking_section['color'])

            # Gölge
            draw.text((shocking_pos[0] + 3, shocking_pos[1] + 3),
                      shocking_text, font=shocking_font, fill=(0, 0, 0))
            # Ana yazı
            draw.text(shocking_pos, shocking_text, font=shocking_font, fill=shocking_color)
            print(f"✅ Shocking word: {shocking_text}")
        except Exception as e:
            print(f"❌ Shocking word hatası: {e}")

        # 2. MAIN TITLE
        try:
            main_section = topic_config['sections']['main_title']
            x_offset = template['sections']['main_title']['x_offset']
            y_start = template['sections']['main_title']['y_start']
            line_height = template['sections']['main_title']['line_height']
            main_color = hex_to_rgb(main_section['color'])

            for i, line in enumerate(main_section['lines']):
                y_pos = y_start + (i * line_height)
                clean_line = remove_emojis(line)

                # Gölge
                draw.text((x_offset + 3, y_pos + 3),
                          clean_line, font=main_font, fill=(0, 0, 0))
                # Ana yazı
                draw.text((x_offset, y_pos), clean_line, font=main_font, fill=main_color)

            print(f"✅ Main title: {len(main_section['lines'])} satır")
        except Exception as e:
            print(f"❌ Main title hatası: {e}")

        # 3. BOTTOM TEXT
        try:
            bottom_section = topic_config['sections']['bottom_text']
            x_offset = template['sections']['bottom_text']['x_offset']
            y_start = template['sections']['bottom_text']['y_start']
            line_height = template['sections']['bottom_text']['line_height']
            bottom_color = hex_to_rgb(bottom_section['color'])

            for i, line in enumerate(bottom_section['lines']):
                y_pos = y_start + (i * line_height)
                clean_line = remove_emojis(line)

                # Gölge
                draw.text((x_offset + 2, y_pos + 2),
                          clean_line, font=bottom_font, fill=(139, 69, 19))  # Kahverengi gölge
                # Ana yazı
                draw.text((x_offset, y_pos), clean_line, font=bottom_font, fill=bottom_color)

            print(f"✅ Bottom text: {len(bottom_section['lines'])} satır")
        except Exception as e:
            print(f"❌ Bottom text hatası: {e}")

        # 4. CHANNEL
        try:
            channel_section = topic_config['sections']['channel']
            channel_pos = (
                template['sections']['channel']['x_offset'],
                height - template['sections']['channel']['y_offset']
            )
            channel_color = hex_to_rgb(template['sections']['channel']['color'])
            channel_text = remove_emojis(channel_section['text'])

            # Gölge
            draw.text((channel_pos[0] + 2, channel_pos[1] + 2),
                      channel_text, font=channel_font, fill=(0, 0, 0))
            # Ana yazı
            draw.text(channel_pos, channel_text, font=channel_font, fill=channel_color)
            print(f"✅ Channel: {channel_text}")
        except Exception as e:
            print(f"❌ Channel hatası: {e}")

        # Kaydet
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, 'JPEG', quality=95, optimize=True)

        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"\n✅ THUMBNAIL BAŞARIYLA OLUŞTURULDU!")
            print(f"📁 Dosya: {output_path}")
            print(f"📦 Boyut: {file_size:,} bytes")
            return True
        else:
            print(f"❌ Dosya kaydedilemedi")
            return False

    except Exception as e:
        print(f"❌ Genel hata: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ana fonksiyon - static input"""
    print("🎨 BASIT THUMBNAIL OLUŞTURUCU")
    print("=" * 50)

    # STATİK DEĞERLER - INPUT YOK!
    background_path = "../../improved_thumbnails/topic_0_version_1.png"
    topic_id = 1
    output_path = "../../output/final_thumbnail.jpg"

    print(f"📸 Background: {background_path}")
    print(f"🆔 Topic ID: {topic_id}")
    print(f"💾 Output: {output_path}")

    # Thumbnail oluştur
    success = create_thumbnail(background_path, topic_id, output_path)

    if success:
        print("\n🎉 İşlem tamamlandı!")
    else:
        print("\n❌ İşlem başarısız!")


if __name__ == "__main__":
    main()