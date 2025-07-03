import pandas as pd
import json
import os
import re
from PIL import Image, ImageDraw, ImageFont

# Base directory'yi düzelt - script src/generators/ dizininde çalışıyor
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/ dizini
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Ana proje dizini

TOPIC_CSV_PATH = os.path.join(BASE_DIR, "data", "topics.csv")
THUMBNAIL_JSON_PATH = os.path.join(BASE_DIR, "data", "thumbnail_features.json")


def load_thumbnail_config():
    """src/data/thumbnail_features.json'ı yükle"""
    try:
        with open(THUMBNAIL_JSON_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✅ Thumbnail config yüklendi")
        return config
    except Exception as e:
        print(f"❌ Config yükleme hatası: {e}")
        return None


def load_topics_csv():
    """src/data/topics.csv'yi yükle"""
    try:
        df = pd.read_csv(TOPIC_CSV_PATH)
        print(f"✅ Topics CSV yüklendi: {len(df)} topic")

        # thumbnail kolonu yoksa ekle
        if 'thumbnail' not in df.columns:
            df['thumbnail'] = 0
            df.to_csv(TOPIC_CSV_PATH, index=False)
            print("📝 thumbnail kolonu eklendi")

        # cover_image_created=1 olanları filtrele
        ready_topics = df[df['cover_image_created'] == 1]
        print(f"📸 Cover image hazır olanlar: {len(ready_topics)}")

        if len(ready_topics) == 0:
            print("⚠️ Hiç hazır topic yok. cover_image_created=1 olan topic'ler gerekli.")
            return df, None

        return df, ready_topics
    except Exception as e:
        print(f"❌ CSV yükleme hatası: {e}")
        return None, None


def find_topic_in_config(row_number, config):
    """CSV'deki satır numarasına göre config'de topic bul"""
    # ID ile eşleştir (satır numarası = id)
    for topic in config['topics']:
        if topic['id'] == row_number:
            print(f"✅ Config'de bulundu: ID={row_number} -> {topic['topic']}")
            return topic

    # Bulunamazsa None döndür
    print(f"⚠️ Config'de bulunamadı: ID={row_number}")
    return None


def load_fonts():
    """Fontları yükle - absolute paths kullan"""
    # Font dizinini proje root'undan başlat
    font_dir = os.path.join(PROJECT_ROOT, "fonts")

    font_paths = {
        'shocking': os.path.join(font_dir, 'Poppins-Bold.ttf'),
        'main': os.path.join(font_dir, 'CrimsonText-Bold.ttf'),
        'bottom': os.path.join(font_dir, 'CrimsonText-Italic.ttf'),
        'channel': os.path.join(font_dir, 'Lora-VariableFont_wght.ttf')
    }

    loaded_fonts = {}

    for font_type, path in font_paths.items():
        if os.path.exists(path):
            loaded_fonts[font_type] = path
            print(f"✅ {font_type}: {os.path.basename(path)}")
        else:
            loaded_fonts[font_type] = None
            print(f"❌ {font_type}: {path} bulunamadı")

    return loaded_fonts


def hex_to_rgb(hex_color):
    """Hex rengi RGB'ye çevir"""
    try:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    except:
        return (255, 255, 255)  # Default beyaz


def remove_emojis(text):
    """Emoji karakterlerini kaldır"""
    import re
    # Unicode emoji pattern'i
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    # Unicode escape sequence'leri de kaldır (\ud83d\ude31 gibi)
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)

    # Normal emoji'leri kaldır
    clean_text = emoji_pattern.sub(r'', text)

    # Fazla boşlukları temizle
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text


def create_thumbnail_from_config(row_number, topic_row, topic_config, template_config):
    """Simple thumbnail creation - back to basics with minimal shadows"""

    # Story output klasörü - src/output/{row_number}/
    story_output_dir = os.path.join(BASE_DIR, "output", str(row_number))

    # Final thumbnail için scenes klasörü oluştur
    final_output_dir = os.path.join(story_output_dir, "scenes")
    os.makedirs(final_output_dir, exist_ok=True)

    try:
        print(f"\n🎨 {topic_row['topic']} için simple thumbnail oluşturuluyor...")
        print(f"📁 Story Dir: {story_output_dir}")
        print(f"📁 Final Output: {final_output_dir}")

        # Template ayarları
        template = template_config['thumbnail_template']
        width = template['layout']['width']
        height = template['layout']['height']
        print(f"🎛️ Template boyut: {width}x{height}")

        # Ham thumbnail.png dosyasını DOĞRU konumdan yükle
        thumbnail_path = os.path.join(story_output_dir, "thumbnail.png")

        print(f"🔍 Background aranıyor: {thumbnail_path}")

        if os.path.exists(thumbnail_path):
            # Ham resmi yükle ve RGB'ye çevir
            img = Image.open(thumbnail_path)
            original_size = img.size
            original_mode = img.mode

            print(f"🔍 Orijinal resim: {original_size}, mode: {original_mode}")

            # RGB'ye çevir
            if img.mode != 'RGB':
                print(f"🔄 Converting {original_mode} → RGB")
                img = img.convert('RGB')

            # Boyutu template boyutuna ayarla
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            print(f"✅ Background processed: {original_size} ({original_mode}) → {width}x{height} (RGB)")
        else:
            # Ham resim yoksa RGB arkaplan oluştur
            img = Image.new('RGB', (width, height), (30, 30, 30))
            print(f"⚠️ Ham thumbnail bulunamadı: {thumbnail_path}")
            print(f"📝 RGB arkaplan oluşturuldu ({width}x{height})")

        draw = ImageDraw.Draw(img)
        print("✅ Simple draw objesi oluşturuldu")

        # Fontları yükle - ORIGINAL SIZES
        fonts = load_fonts()
        print(f"📝 Font durumu: {[k for k, v in fonts.items() if v is not None]}")

        # Font boyutları template'den al - NO OPTIMIZATION
        try:
            shocking_size = template['sections']['shocking_word']['font_size']
            main_size = template['sections']['main_title']['font_size']
            bottom_size = template['sections']['bottom_text']['font_size']
            channel_size = template['sections']['channel']['font_size']

            print(
                f"📏 Font boyutları: SHOCKING={shocking_size}, Main={main_size}, Bottom={bottom_size}, Channel={channel_size}")

            shocking_font = ImageFont.truetype(fonts['shocking'], shocking_size) if fonts[
                'shocking'] else ImageFont.load_default()
            main_font = ImageFont.truetype(fonts['main'], main_size) if fonts['main'] else ImageFont.load_default()
            bottom_font = ImageFont.truetype(fonts['bottom'], bottom_size) if fonts[
                'bottom'] else ImageFont.load_default()
            channel_font = ImageFont.truetype(fonts['channel'], channel_size) if fonts[
                'channel'] else ImageFont.load_default()

            print("✅ Original fonts loaded")

        except Exception as font_error:
            print(f"❌ Font yükleme hatası: {font_error}")
            shocking_font = main_font = bottom_font = channel_font = ImageFont.load_default()
            print("⚠️ Tüm fontlar varsayılan olarak ayarlandı")

        print("\n📝 SIMPLE TEXT RENDERING...")

        # 1. SHOCKING WORD - With Shadow Only
        print("📍 1. SHOCKING WORD (with shadow)")
        try:
            shocking_section = topic_config['sections']['shocking_word']
            shocking_pos = (
                template['sections']['shocking_word']['x_offset'],
                template['sections']['shocking_word']['y_offset']
            )

            shocking_text = remove_emojis(shocking_section['text'])
            shocking_color = hex_to_rgb(shocking_section['color'])

            print(f"   📝 Text: '{shocking_text}'")
            print(f"   📍 Position: {shocking_pos}")
            print(f"   🎨 Color: {shocking_color}")

            # Simple shadow for SHOCKING
            shadow_offset = [3, 3]
            draw.text((shocking_pos[0] + shadow_offset[0], shocking_pos[1] + shadow_offset[1]),
                      shocking_text, font=shocking_font, fill=(0, 0, 0))

            # Main text
            draw.text(shocking_pos, shocking_text, font=shocking_font, fill=shocking_color)
            print("   ✅ Shocking word with simple shadow")
        except Exception as e:
            print(f"   ❌ Shocking word hatası: {e}")

        # 2. MAIN TITLE - Clean, No Effects
        print("📍 2. MAIN TITLE (clean)")
        try:
            main_section = topic_config['sections']['main_title']
            x_offset = template['sections']['main_title']['x_offset']
            y_start = template['sections']['main_title']['y_start']
            line_height = template['sections']['main_title']['line_height']

            main_color = hex_to_rgb(main_section['color'])

            print(f"   📝 Lines: {len(main_section['lines'])}")
            print(f"   📍 Position: x={x_offset}, y_start={y_start}, line_height={line_height}")
            print(f"   🎨 Color: {main_color}")

            for i, line in enumerate(main_section['lines']):
                y_pos = y_start + (i * line_height)
                clean_line = remove_emojis(line)

                print(f"     Line {i + 1}: '{clean_line}' @ y={y_pos}")

                # NO EFFECTS - Just clean text
                draw.text((x_offset, y_pos), clean_line, font=main_font, fill=main_color)

            print("   ✅ Main title - clean")
        except Exception as e:
            print(f"   ❌ Main title hatası: {e}")

        # 3. BOTTOM TEXT - Clean, No Effects
        print("📍 3. BOTTOM TEXT (clean)")
        try:
            bottom_section = topic_config['sections']['bottom_text']
            x_offset = template['sections']['bottom_text']['x_offset']
            y_start = template['sections']['bottom_text']['y_start']
            line_height = template['sections']['bottom_text']['line_height']

            bottom_color = hex_to_rgb(bottom_section['color'])

            print(f"   📝 Lines: {len(bottom_section['lines'])}")
            print(f"   📍 Position: x={x_offset}, y_start={y_start}")
            print(f"   🎨 Color: {bottom_color}")

            for i, line in enumerate(bottom_section['lines']):
                y_pos = y_start + (i * line_height)
                clean_line = remove_emojis(line)

                print(f"     Line {i + 1}: '{clean_line}' @ y={y_pos}")

                # NO EFFECTS - Just clean text
                draw.text((x_offset, y_pos), clean_line, font=bottom_font, fill=bottom_color)

            print("   ✅ Bottom text - clean")
        except Exception as e:
            print(f"   ❌ Bottom text hatası: {e}")

        # 4. CHANNEL - With Shadow Only
        print("📍 4. CHANNEL (with shadow)")
        try:
            channel_section = topic_config['sections']['channel']
            channel_pos = (
                template['sections']['channel']['x_offset'],
                height - template['sections']['channel']['y_offset']
            )
            channel_color = hex_to_rgb(template['sections']['channel']['color'])

            channel_text = remove_emojis(channel_section['text'])

            print(f"   📝 Text: '{channel_text}'")
            print(f"   📍 Position: {channel_pos}")
            print(f"   🎨 Color: {channel_color}")

            # Simple shadow for CHANNEL
            shadow_offset = [2, 2]
            draw.text((channel_pos[0] + shadow_offset[0], channel_pos[1] + shadow_offset[1]),
                      channel_text, font=channel_font, fill=(0, 0, 0))

            # Main text
            draw.text(channel_pos, channel_text, font=channel_font, fill=channel_color)
            print("   ✅ Channel with simple shadow")
        except Exception as e:
            print(f"   ❌ Channel hatası: {e}")

        print("\n💾 SIMPLE SAVE...")

        # Klasör var mı kontrol et
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir, exist_ok=True)
            print(f"📁 Klasör oluşturuldu: {final_output_dir}")

        # Final mode check before saving
        print(f"🔍 Final image mode before save: {img.mode}")
        if img.mode != 'RGB':
            print(f"🔄 Final conversion: {img.mode} → RGB")
            img = img.convert('RGB')

        # Simple JPEG save
        output_path = os.path.join(final_output_dir, "final_thumbnail.jpg")
        print(f"📁 Hedef dosya: {output_path}")

        # Simple YouTube JPEG
        img.save(output_path, 'JPEG', quality=92, optimize=True)

        # Dosya kontrol
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ SIMPLE THUMBNAIL BAŞARIYLA KAYDEDİLDİ!")
            print(f"   📁 Path: {output_path}")
            print(f"   📦 Boyut: {file_size:,} bytes")
            print(f"   🎯 Clean design - No overlapping text")
            print(f"   🌑 Shadows: SHOCKING + Channel only")
            return output_path
        else:
            print(f"❌ Dosya kaydedilemedi: {output_path}")
            return None

    except Exception as e:
        print(f"❌ GENEL THUMBNAIL HATASI: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_csv_thumbnail_status(df, csv_index, row_number):
    """CSV'de thumbnail kolonu 1 yap ve kaydet"""
    try:
        df.loc[csv_index, 'thumbnail'] = 1
        df.to_csv(TOPIC_CSV_PATH, index=False)  # Absolute path kullan
        print(f"✅ CSV güncellendi: Satır {row_number} thumbnail=1")
        return True
    except Exception as e:
        print(f"❌ CSV güncelleme hatası: {e}")
        return False


def create_all_thumbnails():
    """Simple thumbnail creation system"""

    print("🎨 SIMPLE THUMBNAIL CREATE SYSTEM")
    print(f"📁 Base Dir: {BASE_DIR}")
    print(f"📁 Project Root: {PROJECT_ROOT}")
    print("📁 Paths: src/data/ -> src/output/{id}/scenes/")
    print("🎯 Clean Design - No Text Overlap")
    print("🌑 Minimal Shadows: SHOCKING + Channel only")
    print("=" * 60)

    # Config ve CSV'yi yükle
    config = load_thumbnail_config()
    full_df, ready_topics = load_topics_csv()

    if config is None or full_df is None or ready_topics is None:
        print("❌ Sistem başlatılamadı")
        return

    print(f"\n🎯 {len(ready_topics)} topic simple processing")

    created_thumbnails = []
    updated_rows = []

    # Her ready topic için simple thumbnail oluştur
    for csv_index, topic_row in ready_topics.iterrows():
        try:
            # Satır numarası 1'den başlar (csv_index + 1)
            row_number = csv_index + 1

            # Config'de bu satır numarasına karşılık gelen topic'i bul
            topic_config = find_topic_in_config(row_number, config)

            if topic_config is None:
                print(f"⚠️ Satır {row_number} ({topic_row['topic']}) config'de bulunamadı, atlanıyor...")
                continue

            # Simple thumbnail oluştur
            thumbnail_path = create_thumbnail_from_config(
                row_number, topic_row, topic_config, config
            )

            if thumbnail_path:
                created_thumbnails.append(thumbnail_path)

                # CSV'yi güncelle
                if update_csv_thumbnail_status(full_df, csv_index, row_number):
                    updated_rows.append(row_number)

        except Exception as e:
            print(f"❌ Satır {row_number} ({topic_row['topic']}) için hata: {e}")
            continue

    print(f"\n🎉 SIMPLE CREATION COMPLETE!")
    print(f"✅ {len(created_thumbnails)} clean thumbnails oluşturuldu")
    print(f"💾 {len(updated_rows)} satır CSV'de güncellendi")
    print("📁 Dosyalar: src/output/{satır_no}/scenes/final_thumbnail.jpg")
    print("🎯 Clean Design - No Overlapping Text")
    print("🌑 Simple Shadows: SHOCKING + Channel name only")
    print("📱 Original Font Sizes")

    # Özet
    if created_thumbnails:
        print(f"\n📋 Oluşturulan simple thumbnail'lar:")
        for i, path in enumerate(created_thumbnails, 1):
            print(f"   {i}. {path}")

    return created_thumbnails


if __name__ == "__main__":
    simple_files = create_all_thumbnails()