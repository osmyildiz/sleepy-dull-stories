import json
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import re


def load_thumbnail_config():
    """thumbnail_features.json'ı yükle"""
    try:
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


def get_brand_design_config():
    """Sleepy Dull Stories brand tasarımı - 4K optimize"""
    return {
        "name": "SLEEPY DULL STORIES BRAND",
        "description": "Kırmızı-Beyaz-Sarı brand identity - 4K kalite",
        "resolution": {
            "width": 3840,  # 4K width (1920x2 = 3840)
            "height": 2160  # 4K height (1080x2 = 2160)
        },
        "fonts": {
            "shocking": "Anton-Regular.ttf",
            "main": "Poppins-Black.ttf",
            "bottom": "Merriweather_36pt-ExtraBold.ttf",
            "channel": "SourceSans3-Bold.ttf"
        },
        "sizes": {
            "shocking": 340,  # Daha dengeli boyut
            "main": 270,  # Ana yazı küçültüldü
            "bottom": 196,  # Alt yazı küçültüldü
            "channel": 156  # Channel küçültüldü
        },
        "effects": {
            "shocking_stroke": 30,  # Daha ince kontur
            "main_stroke": 15,  # Daha ince kontur
            "shadow_offset": 22,  # Daha hafif gölge
            "contrast_boost": 1.9,
            "double_shadow": True,
            "glow_effect": True
        },
        "colors": {
            "shocking": "#FF0000",  # Brand kırmızı
            "main": "#FFFFFF",  # Brand beyaz
            "bottom": "#FFFF00",  # Brand sarı
            "stroke": "#000000",  # Brand siyah kontur
            "glow": "#FF4444"  # Kırmızı parıltı
        },
        "positioning": {
            "shocking_y_offset": -80,  # Shocking daha yukarı
            "main_y_offset": 80,  # Main title daha aşağı
            "bottom_y_offset": 100  # Bottom text daha yakın
        }
    }


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


def load_font_safe(font_path, size):
    """Font'u güvenli şekilde yükle - 4K optimize"""
    try:
        full_path = f"../../fonts/{font_path}"
        if os.path.exists(full_path):
            return ImageFont.truetype(full_path, size)
        else:
            print(f"⚠️ Font bulunamadı: {font_path}, varsayılan kullanılıyor")
            # 4K için büyük varsayılan font
            return ImageFont.load_default()
    except Exception as e:
        print(f"⚠️ Font yükleme hatası: {e}")
        return ImageFont.load_default()


def add_premium_text_effects(draw, text, position, font, color, brand_config, text_type):
    """Premium metin efektleri - 4K kalite"""
    effects = brand_config['effects']
    colors = brand_config['colors']

    x, y = position
    stroke_width = effects.get(f'{text_type}_stroke', 8)
    shadow_offset = effects.get('shadow_offset', 8)

    # Çift gölge efekti (brand identity için)
    if effects.get('double_shadow', False):
        # Büyük gölge
        draw.text((x + shadow_offset + 4, y + shadow_offset + 4),
                  text, font=font, fill=(0, 0, 0, 120))
        # Normal gölge
        draw.text((x + shadow_offset, y + shadow_offset),
                  text, font=font, fill=(0, 0, 0, 200))
    else:
        # Tek gölge
        draw.text((x + shadow_offset, y + shadow_offset),
                  text, font=font, fill=(0, 0, 0))

    # Glow efekti (shocking kelime için)
    if effects.get('glow_effect', False) and text_type == 'shocking' and 'glow' in colors:
        glow_color = hex_to_rgb(colors['glow'])
        for offset in range(2, 8, 2):  # 4K için daha büyük glow
            draw.text((x - offset, y), text, font=font, fill=glow_color + (80,),
                      stroke_width=2, stroke_fill=glow_color)
            draw.text((x + offset, y), text, font=font, fill=glow_color + (80,),
                      stroke_width=2, stroke_fill=glow_color)

    # Ana metin - kalın kontur
    stroke_color = hex_to_rgb(colors['stroke'])
    draw.text(position, text, font=font, fill=color,
              stroke_width=stroke_width, stroke_fill=stroke_color)


def enhance_image_4k(img, brand_config):
    """4K resim kalitesi optimize edilmiş iyileştirme"""
    effects = brand_config['effects']

    # Kontrast artırma
    if 'contrast_boost' in effects:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(effects['contrast_boost'])

    # Renk doygunluğu artırma (brand renkleri için)
    color_enhancer = ImageEnhance.Color(img)
    img = color_enhancer.enhance(1.25)

    # Netlik artırma (4K için kritik)
    sharpness_enhancer = ImageEnhance.Sharpness(img)
    img = sharpness_enhancer.enhance(1.15)

    return img


def create_4k_thumbnail(background_image_path, topic_id, output_path):
    """4K kalitesinde brand thumbnail oluştur"""

    print(f"\n🎨 4K SLEEPY DULL STORIES THUMBNAIL OLUŞTURULUYOR...")
    print(f"📐 Çözünürlük: 3840x2160 (4K)")
    print(f"🎯 Brand: Kırmızı-Beyaz-Sarı")

    # Config'leri yükle
    config = load_thumbnail_config()
    if not config:
        return False

    brand_config = get_brand_design_config()

    # Topic'i bul
    topic_config = find_topic_by_id(topic_id, config)
    if not topic_config:
        return False

    # 4K boyutları
    width = brand_config['resolution']['width']
    height = brand_config['resolution']['height']

    try:
        # Background resmi yükle ve 4K'ya çevir
        if not os.path.exists(background_image_path):
            print(f"❌ Background resim bulunamadı: {background_image_path}")
            return False

        print("📸 Background resmi 4K'ya dönüştürülüyor...")
        img = Image.open(background_image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 4K boyutuna yeniden boyutlandır (yüksek kalite)
        img = img.resize((width, height), Image.Resampling.LANCZOS)

        # 4K kalite iyileştirmesi
        img = enhance_image_4k(img, brand_config)

        draw = ImageDraw.Draw(img)

        print("🔤 4K fontlar yükleniyor...")
        # 4K fontları yükle
        fonts = {
            'shocking': load_font_safe(brand_config['fonts']['shocking'],
                                       brand_config['sizes']['shocking']),
            'main': load_font_safe(brand_config['fonts']['main'],
                                   brand_config['sizes']['main']),
            'bottom': load_font_safe(brand_config['fonts']['bottom'],
                                     brand_config['sizes']['bottom']),
            'channel': load_font_safe(brand_config['fonts']['channel'],
                                      brand_config['sizes']['channel'])
        }

        # Template pozisyonlarını 4K'ya çevir (2x büyüt)
        template = config['thumbnail_template']

        # Pozisyon ayarları
        positioning = brand_config['positioning']

        print("✍️ Yazılar 4K kalitede ekleniyor...")

        # 1. SHOCKING WORD - Brand kırmızısı
        try:
            shocking_section = topic_config['sections']['shocking_word']
            shocking_y_base = template['sections']['shocking_word']['y_offset'] * 2  # 4K için 2x
            shocking_y_adjusted = shocking_y_base + positioning['shocking_y_offset']

            shocking_pos = (
                template['sections']['shocking_word']['x_offset'] * 2,  # 4K için 2x
                shocking_y_adjusted
            )

            shocking_text = remove_emojis(shocking_section['text'])
            shocking_color = hex_to_rgb(brand_config['colors']['shocking'])

            add_premium_text_effects(draw, shocking_text, shocking_pos,
                                     fonts['shocking'], shocking_color, brand_config, 'shocking')

            print(f"✅ Shocking word (4K): {shocking_text}")
        except Exception as e:
            print(f"❌ Shocking word hatası: {e}")

        # 2. MAIN TITLE - Brand beyazı
        try:
            main_section = topic_config['sections']['main_title']
            x_offset = template['sections']['main_title']['x_offset'] * 2  # 4K için 2x
            y_base = template['sections']['main_title']['y_start'] * 2  # 4K için 2x
            y_adjusted = y_base + positioning['main_y_offset']
            line_height = brand_config['sizes']['main'] + 12  # Daha sıkı satır aralığı
            main_color = hex_to_rgb(brand_config['colors']['main'])

            for i, line in enumerate(main_section['lines']):
                y_pos = y_adjusted + (i * line_height)
                clean_line = remove_emojis(line)

                add_premium_text_effects(draw, clean_line, (x_offset, y_pos),
                                         fonts['main'], main_color, brand_config, 'main')

            print(f"✅ Main title (4K): {len(main_section['lines'])} satır")
        except Exception as e:
            print(f"❌ Main title hatası: {e}")

        # 3. BOTTOM TEXT - Brand sarısı
        try:
            bottom_section = topic_config['sections']['bottom_text']
            x_offset = template['sections']['bottom_text']['x_offset'] * 2  # 4K için 2x
            y_base = template['sections']['bottom_text']['y_start'] * 2  # 4K için 2x
            y_adjusted = y_base + positioning['bottom_y_offset']
            line_height = brand_config['sizes']['bottom'] + 10  # Daha sıkı alt yazı
            bottom_color = hex_to_rgb(brand_config['colors']['bottom'])

            for i, line in enumerate(bottom_section['lines']):
                y_pos = y_adjusted + (i * line_height)
                clean_line = remove_emojis(line)

                add_premium_text_effects(draw, clean_line, (x_offset, y_pos),
                                         fonts['bottom'], bottom_color, brand_config, 'bottom')

            print(f"✅ Bottom text (4K): {len(bottom_section['lines'])} satır")
        except Exception as e:
            print(f"❌ Bottom text hatası: {e}")

        # 4. CHANNEL - Brand beyazı
        try:
            channel_section = topic_config['sections']['channel']
            channel_pos = (
                template['sections']['channel']['x_offset'] * 2,  # 4K için 2x
                height - (template['sections']['channel']['y_offset'] * 2) - 60  # 4K için 2x
            )
            channel_color = hex_to_rgb(brand_config['colors']['main'])  # Beyaz
            channel_text = remove_emojis(channel_section['text'])

            # Channel için özel gölge (4K optimize)
            draw.text((channel_pos[0] + 6, channel_pos[1] + 6),
                      channel_text, font=fonts['channel'], fill=(0, 0, 0, 200))
            draw.text(channel_pos, channel_text, font=fonts['channel'],
                      fill=channel_color, stroke_width=6, stroke_fill=(0, 0, 0))

            print(f"✅ Channel (4K): {channel_text}")
        except Exception as e:
            print(f"❌ Channel hatası: {e}")

        # 4K kalitede kaydet
        print("💾 4K thumbnail kaydediliyor...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 4K için en yüksek kalite ayarları
        img.save(output_path, 'JPEG', quality=100, optimize=True, dpi=(300, 300))

        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"\n🎉 4K THUMBNAIL BAŞARIYLA OLUŞTURULDU!")
            print(f"📁 Dosya: {output_path}")
            print(f"📦 Boyut: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
            print(f"📐 Çözünürlük: {width}x{height} (4K)")
            print(f"🎨 Brand: Sleepy Dull Stories (Kırmızı-Beyaz-Sarı)")
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
    """Ana fonksiyon - 4K Brand Thumbnail"""

    print("🚀 SLEEPY DULL STORIES 4K THUMBNAIL GENERATOR")
    print("=" * 70)
    print("📐 4K Çözünürlük (3840x2160)")
    print("🎨 Brand Identity: Kırmızı-Beyaz-Sarı")
    print("🏷️ Dosya formatı: thumbnail_[ID].jpg")
    print("=" * 70)

    # STATİK DEĞERLER
    background_path = "../../thumbnails/thumbnail_31_variant_2.png"
    topic_id = 31
    output_dir = "../../output/thumbnails"
    output_path = f"{output_dir}/thumbnail_{topic_id}.jpg"

    print(f"📸 Background: {background_path}")
    print(f"🆔 Topic ID: {topic_id}")
    print(f"📁 Output: {output_path}")

    # 4K thumbnail oluştur
    success = create_4k_thumbnail(background_path, topic_id, output_path)

    if success:
        print("\n🎊 4K BRAND THUMBNAIL HAZIR!")
        print("🎯 YouTube'da maksimum kalite!")
        print("📱 Mobil ve desktop'ta mükemmel görünüm!")
        print("🏆 Sleepy Dull Stories brand tutarlılığı!")
    else:
        print("\n❌ Thumbnail oluşturulamadı")


if __name__ == "__main__":
    main()