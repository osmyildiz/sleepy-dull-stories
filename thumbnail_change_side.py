from PIL import Image, ImageEnhance
import os
from pathlib import Path


class ImageCompositionProcessor:
    def __init__(self):
        self.input_dir = "thumbnails"
        self.output_dir = "thumbnails_flipped"

    def flip_horizontal(self, image_path, output_path):
        """Resmi yatay olarak çevir (Photoshop gibi - sol sağa, sağ sola)"""
        try:
            with Image.open(image_path) as img:
                # DOĞRU yatay çevirme - Photoshop'taki gibi
                flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                flipped.save(output_path, quality=95, optimize=True)
                print(f"✅ Horizontally flipped: {output_path}")
                return True
        except Exception as e:
            print(f"❌ Flip error: {e}")
            return False

    def crop_and_recompose(self, image_path, output_path, character_side='left'):
        """Resmi kırp ve yeniden kompoze et"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size

                if character_side == 'left':
                    # Karakter sol tarafta ise, onu sağa taşı
                    # Sol yarıyı (karakter) al
                    character_crop = img.crop((0, 0, width // 2, height))
                    # Sağ yarıyı (background) al
                    background_crop = img.crop((width // 2, 0, width, height))

                    # Yeni kompozisyon: background sol, karakter sağ
                    new_img = Image.new('RGB', (width, height))
                    new_img.paste(background_crop, (0, 0))
                    new_img.paste(character_crop, (width // 2, 0))

                else:  # character_side == 'right'
                    # Karakter zaten sağda, sadece flip yap
                    new_img = img.transpose(Image.FLIP_LEFT_RIGHT)

                new_img.save(output_path, quality=95, optimize=True)
                print(f"✅ Recomposed: {output_path}")
                return True

        except Exception as e:
            print(f"❌ Recompose error: {e}")
            return False

    def smart_composition_fix(self, image_path, output_path):
        """Akıllı kompozisyon düzeltme - karakter sağa geçsin"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size

                # Resmi 3 parçaya böl: sol, orta, sağ
                left_section = img.crop((0, 0, width // 3, height))
                middle_section = img.crop((width // 3, 0, 2 * width // 3, height))
                right_section = img.crop((2 * width // 3, 0, width, height))

                # Yeni kompozisyon: background solda, karakter sağda
                new_img = Image.new('RGB', (width, height))

                # Sol tarafı background ile doldur (middle + right kısmı)
                new_img.paste(middle_section, (0, 0))
                new_img.paste(right_section, (width // 3, 0))

                # Sağ tarafta karakter (left section'ı sağa taşı)
                new_img.paste(left_section, (2 * width // 3, 0))

                new_img.save(output_path, quality=95, optimize=True)
                print(f"✅ Smart composition: {output_path}")
                return True

        except Exception as e:
            print(f"❌ Smart composition error: {e}")
            return False

    def process_all_thumbnails(self, method='flip'):
        """Tüm thumbnail'ları işle"""

        Path(self.output_dir).mkdir(exist_ok=True)

        if not os.path.exists(self.input_dir):
            print(f"❌ Input directory not found: {self.input_dir}")
            return False

        # Thumbnail dosyalarını bul
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(Path(self.input_dir).glob(ext))

        if not image_files:
            print(f"❌ No images found in {self.input_dir}")
            return False

        print(f"🎨 PROCESSING {len(image_files)} THUMBNAILS")
        print(f"📊 Method: {method}")
        print("=" * 50)

        success_count = 0

        for img_file in image_files:
            # Output path oluştur
            output_name = f"{img_file.stem}{img_file.suffix}"
            output_path = Path(self.output_dir) / output_name

            print(f"🔄 Processing: {img_file.name}")

            # Metoda göre işle
            if method == 'flip':
                success = self.flip_horizontal(str(img_file), str(output_path))
            elif method == 'recompose':
                success = self.crop_and_recompose(str(img_file), str(output_path), 'left')
            elif method == 'smart':
                success = self.smart_composition_fix(str(img_file), str(output_path))
            else:
                print(f"❌ Unknown method: {method}")
                continue

            if success:
                success_count += 1

        print("\n" + "=" * 50)
        print(f"🎉 COMPLETED: {success_count}/{len(image_files)} processed")
        print(f"📁 Output directory: {self.output_dir}")

        return success_count > 0

    def process_single_image(self, image_path, method='flip'):
        """Tek resmi işle"""

        Path(self.output_dir).mkdir(exist_ok=True)

        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False

        # Output path
        img_file = Path(image_path)
        output_name = f"{img_file.stem}{img_file.suffix}"
        output_path = Path(self.output_dir) / output_name

        print(f"🎨 PROCESSING SINGLE IMAGE")
        print(f"📊 Method: {method}")
        print(f"📁 Input: {image_path}")
        print(f"📁 Output: {output_path}")
        print("=" * 50)

        # Metoda göre işle
        if method == 'flip':
            success = self.flip_horizontal(image_path, str(output_path))
        elif method == 'recompose':
            success = self.crop_and_recompose(image_path, str(output_path), 'left')
        elif method == 'smart':
            success = self.smart_composition_fix(image_path, str(output_path))
        else:
            print(f"❌ Unknown method: {method}")
            return False

        if success:
            print(f"\n🎉 SUCCESS! Fixed image saved to:")
            print(f"📁 {output_path}")

        return success


def main():
    """Ana fonksiyon - Basit horizontal flip"""

    processor = ImageCompositionProcessor()

    print("🎨 IMAGE HORIZONTAL FLIP PROCESSOR")
    print("=" * 60)
    print("🔄 Photoshop style horizontal flip:")
    print("   ↔️ Sol taraf → Sağ tarafa geçer")
    print("   ↔️ Sağ taraf → Sol tarafa geçer")
    print("=" * 60)

    # TEK RESİM İÇİN - DOSYA ADI BURAYA
    single_image = "thumbnails/thumbnail_31_variant_2.png"  # <-- DEĞİŞTİR

    if os.path.exists(single_image):
        print(f"📁 Processing: {single_image}")
        success = processor.process_single_image(single_image, method='flip')

        if success:
            print(f"\n🎉 SUCCESS! Karakter artık sağ tarafta!")
            print(f"📱 Thumbnail yazılar için hazır!")
        else:
            print(f"\n❌ FAILED!")
    else:
        print(f"❌ File not found: {single_image}")
        print(f"💡 Available files in thumbnails/:")
        if os.path.exists("thumbnails"):
            for file in os.listdir("thumbnails"):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    print(f"   📄 {file}")


if __name__ == "__main__":
    main()