from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path


class YouTubeTextOverlay:
    def __init__(self):
        # YouTube banner boyutları
        self.BANNER_WIDTH = 2560
        self.BANNER_HEIGHT = 1440

        # Safe area boyutları (merkezde)
        self.SAFE_AREA_WIDTH = 1546
        self.SAFE_AREA_HEIGHT = 423

        # Safe area pozisyonu (merkezde)
        self.safe_x = (self.BANNER_WIDTH - self.SAFE_AREA_WIDTH) // 2
        self.safe_y = (self.BANNER_HEIGHT - self.SAFE_AREA_HEIGHT) // 2

    def get_available_fonts(self):
        """Mevcut fontları listele"""
        font_preferences = [
            "fonts/CrimsonText-Bold.ttf",  # En iyi seçim - Bold ve elegant
            "fonts/CrimsonText-SemiBold.ttf",  # İkinci seçim
            "fonts/Lora-VariableFont_wght.ttf",  # Üçüncü seçim - Variable font
            "fonts/CrimsonText-Regular.ttf",  # Dördüncü seçim
            "fonts/Poppins-Bold.ttf",  # Modern alternatif
            "fonts/Poppins-Black.ttf",  # Daha bold alternatif
            "fonts/JustAnotherHand-Regular.ttf",  # Handwritten style
        ]

        available_fonts = []
        for font_path in font_preferences:
            if os.path.exists(font_path):
                available_fonts.append(font_path)
                print(f"✅ Found font: {font_path}")
            else:
                print(f"❌ Not found: {font_path}")

        return available_fonts

    def find_best_font_size(self, draw, text, max_width, max_height, font_path=None):
        """Text için en uygun font boyutunu bul"""

        # Önce mevcut fontları kontrol et
        if not font_path:
            available_fonts = self.get_available_fonts()
            if available_fonts:
                font_path = available_fonts[0]  # En iyi seçimi kullan
                print(f"🎨 Using font: {font_path}")

        font_size = 140  # Daha büyük başla

        while font_size > 30:
            try:
                if font_path and os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    # Fallback system fonts
                    font_options = [
                        "times.ttf", "georgia.ttf", "arial.ttf",
                        "/System/Library/Fonts/Times.ttc",
                        "C:/Windows/Fonts/times.ttf",
                    ]

                    font = None
                    for font_file in font_options:
                        try:
                            font = ImageFont.truetype(font_file, font_size)
                            break
                        except:
                            continue

                    if font is None:
                        font = ImageFont.load_default()

                # Text boyutunu ölç
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                if text_width <= max_width and text_height <= max_height:
                    print(f"📏 Perfect font size: {font_size}px ({text_width}x{text_height})")
                    return font, text_width, text_height

            except Exception as e:
                print(f"⚠️ Font size {font_size} failed: {e}")

            font_size -= 8  # Biraz daha hızlı azalt

        # Fallback
        return ImageFont.load_default(), max_width // 2, max_height // 2

    def add_elegant_text(self, image, text="SLEEPY DULL STORIES"):
        """Görselin üzerine elegant text ekle"""

        # Kopya oluştur
        result = image.copy()
        draw = ImageDraw.Draw(result)

        # Safe area içinde text alanı belirle
        text_margin = 120  # Biraz daha fazla padding
        text_area_width = self.SAFE_AREA_WIDTH - (text_margin * 2)
        text_area_height = self.SAFE_AREA_HEIGHT - (text_margin * 2)

        print(f"📝 Text area: {text_area_width}x{text_area_height}")

        # Font boyutunu otomatik hesapla (mevcut fontları kullanarak)
        font, text_width, text_height = self.find_best_font_size(
            draw, text, text_area_width, text_area_height
        )

        # Text'i safe area'nın merkezine hizala
        text_x = self.safe_x + (self.SAFE_AREA_WIDTH - text_width) // 2
        text_y = self.safe_y + (self.SAFE_AREA_HEIGHT - text_height) // 2

        print(f"📍 Text position: ({text_x}, {text_y})")

        # ENHANCED SHADOW EFFECTS - Daha sinematik
        shadow_layers = [
            {"offset": (8, 8), "color": (0, 0, 0, 200)},  # Ana gölge
            {"offset": (6, 6), "color": (0, 0, 0, 150)},  # Orta gölge
            {"offset": (4, 4), "color": (0, 0, 0, 100)},  # Hafif gölge
            {"offset": (2, 2), "color": (20, 20, 40, 80)},  # Mavi-gri ton
        ]

        for shadow in shadow_layers:
            offset_x, offset_y = shadow["offset"]
            draw.text(
                (text_x + offset_x, text_y + offset_y),
                text,
                font=font,
                fill=shadow["color"]
            )

        # MAIN TEXT - Gradient effect simulation
        # Base layer (koyu altın)
        draw.text(
            (text_x + 2, text_y + 2),
            text,
            font=font,
            fill=(180, 120, 0, 255)  # Koyu altın base
        )

        # Middle layer (orta altın)
        draw.text(
            (text_x + 1, text_y + 1),
            text,
            font=font,
            fill=(220, 170, 20, 255)  # Orta altın
        )

        # Top layer (parlak altın)
        draw.text(
            (text_x, text_y),
            text,
            font=font,
            fill=(255, 220, 50, 255)  # Parlak altın
        )

        # HIGHLIGHT EFFECT (parlaklık)
        draw.text(
            (text_x - 1, text_y - 1),
            text,
            font=font,
            fill=(255, 255, 200, 120)  # Çok açık sarı highlight
        )

        # GLOW EFFECT (dış parlaklık) - Çok hafif
        glow_offsets = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
        for glow_x, glow_y in glow_offsets:
            draw.text(
                (text_x + glow_x, text_y + glow_y),
                text,
                font=font,
                fill=(255, 215, 0, 30)  # Çok hafif altın glow
            )

        print(f"✅ Enhanced text added with multi-layer effects!")
        return result

    def create_debug_overlay(self, image):
        """Safe area sınırlarını göster (debug için)"""

        result = image.copy()
        draw = ImageDraw.Draw(result)

        # Safe area çerçevesi
        draw.rectangle([
            self.safe_x,
            self.safe_y,
            self.safe_x + self.SAFE_AREA_WIDTH,
            self.safe_y + self.SAFE_AREA_HEIGHT
        ], outline=(255, 0, 0, 255), width=3)

        # Safe area label
        draw.text(
            (self.safe_x, self.safe_y - 30),
            "SAFE AREA",
            fill=(255, 0, 0, 255)
        )

        return result

    def process_banner(self, input_path, output_path="final_sleepy_banner.png", debug=True):
        """Ana işlem fonksiyonu"""

        print("🎨 Processing YouTube banner...")

        # Görseli yükle
        image = Image.open(input_path)
        image = image.convert('RGBA')

        # Banner boyutuna resize et
        image = image.resize((self.BANNER_WIDTH, self.BANNER_HEIGHT), Image.Resampling.LANCZOS)
        print(f"✅ Image resized to: {image.size}")

        # Text ekle
        final_image = self.add_elegant_text(image)

        # Kaydet
        final_image = final_image.convert('RGB')
        final_image.save(output_path, 'PNG', quality=95)
        print(f"🎉 Final banner saved: {output_path}")

        # Debug versiyonu
        if debug:
            debug_image = self.create_debug_overlay(final_image)
            debug_path = "debug_" + output_path
            debug_image.save(debug_path, 'PNG')
            print(f"🐛 Debug version saved: {debug_path}")

        return final_image


def main():
    # Mevcut banner dosyası (background dosyanız)
    INPUT_FILE = "banners/topic_101_16x9_v1.png"  # Güzel banner dosyanız
    OUTPUT_FILE = "sleepy_dull_stories_final.png"

    # Dosyanın varlığını kontrol et
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        print("Mevcut dosyalar:")

        # banners klasörünü kontrol et
        if os.path.exists("banners"):
            for f in Path("banners").glob("*.png"):
                print(f"  📁 {f}")
        else:
            print("banners klasörü bulunamadı!")

        # Mevcut klasördeki png dosyalarını da göster
        for f in Path(".").glob("*.png"):
            print(f"  📁 {f}")
        return

    # Text overlay işlemi
    processor = YouTubeTextOverlay()

    try:
        final_banner = processor.process_banner(
            input_path=INPUT_FILE,
            output_path=OUTPUT_FILE,
            debug=True
        )

        print("\n🎉 Banner processing completed!")
        print(f"📁 Final file: {OUTPUT_FILE}")
        print(f"📐 Size: {final_banner.size}")
        print(f"💡 Safe area: Centered, perfect for YouTube!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()