"""
Sleepy Dull Stories - Simple Watermark Test
ACTUALLY WORKING test script that processes real images
"""

import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

class SleepyDullStoriesWatermark:
    """Working watermark tool for testing"""

    def __init__(self):
        self.channel_name = "Sleepy Dull Stories"
        self.watermark_text = "© Sleepy Dull Stories"
        self.setup_watermark_style()

    def setup_watermark_style(self):
        """Setup watermark appearance"""
        self.watermark_config = {
            "font_size_ratio": 0.04,  # 4% of image width for testing
            "opacity": 140,  # Semi-transparent
            "color": (255, 255, 255),  # White text
            "outline_color": (0, 0, 0),  # Black outline
            "outline_width": 2,
            "position": "bottom_right",
            "margin_ratio": 0.02,  # 2% margin from edges
        }

    def get_font(self, image_width: int):
        """Get font - project fonts or fallback"""
        font_size = int(image_width * self.watermark_config["font_size_ratio"])

        # Project fonts (adjust path as needed)
        project_fonts = [
            "fonts/Poppins-Bold.ttf",
            "fonts/OpenSans-VariableFont_wdth,wght.ttf",
            "fonts/Roboto-VariableFont_wdth,wght.ttf",
            "fonts/CrimsonText-Bold.ttf",
            "fonts/oswald.ttf",
            "./fonts/Poppins-Bold.ttf",  # Try with ./
            "../fonts/Poppins-Bold.ttf",  # Try parent directory
        ]

        # Try project fonts
        for font_path in project_fonts:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    print(f"✅ Using font: {font_path}")
                    return font
            except Exception as e:
                continue

        # System fonts fallback
        system_fonts = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

        for font_path in system_fonts:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    print(f"⚠️ Using system font: {font_path}")
                    return font
            except:
                continue

        print("⚠️ Using default system font")
        return ImageFont.load_default()

    def add_watermark_to_image(self, image_path: str, output_path: str = None) -> bool:
        """Add watermark to image"""
        try:
            print(f"🔍 Processing: {image_path}")

            # Check if input exists
            if not os.path.exists(image_path):
                print(f"❌ File not found: {image_path}")
                return False

            with Image.open(image_path) as img:
                print(f"📏 Image size: {img.size} ({img.mode})")

                # Convert to RGBA for transparency
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')

                # Create transparent overlay
                overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)

                # Get font and text dimensions
                font = self.get_font(img.width)
                text_bbox = draw.textbbox((0, 0), self.watermark_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                print(f"💧 Watermark size: {text_width}x{text_height}")

                # Calculate position (bottom right)
                margin_x = int(img.width * self.watermark_config["margin_ratio"])
                margin_y = int(img.height * self.watermark_config["margin_ratio"])
                x = img.width - text_width - margin_x
                y = img.height - text_height - margin_y

                print(f"📍 Watermark position: ({x}, {y})")

                # Draw outline for visibility
                outline_width = self.watermark_config["outline_width"]
                outline_color = self.watermark_config["outline_color"] + (self.watermark_config["opacity"],)

                for adj_x in range(-outline_width, outline_width + 1):
                    for adj_y in range(-outline_width, outline_width + 1):
                        if adj_x != 0 or adj_y != 0:
                            draw.text((x + adj_x, y + adj_y), self.watermark_text,
                                    font=font, fill=outline_color)

                # Draw main text
                text_color = self.watermark_config["color"] + (self.watermark_config["opacity"],)
                draw.text((x, y), self.watermark_text, font=font, fill=text_color)

                # Composite images
                watermarked = Image.alpha_composite(img, overlay)

                # Convert back to RGB if needed
                if watermarked.mode == 'RGBA':
                    background = Image.new('RGB', watermarked.size, (255, 255, 255))
                    background.paste(watermarked, mask=watermarked.split()[-1])
                    watermarked = background

                # Save result
                if output_path is None:
                    # Create output filename
                    path_obj = Path(image_path)
                    output_path = path_obj.parent / f"{path_obj.stem}_watermarked{path_obj.suffix}"

                watermarked.save(output_path, 'PNG', quality=95)

                print(f"✅ Watermarked image saved: {output_path}")
                print(f"📁 Output size: {os.path.getsize(output_path)} bytes")
                return True

        except Exception as e:
            print(f"❌ Watermark failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def find_test_images():
    """Find images to test with"""
    print("🔍 Looking for test images...")

    # Common image locations
    search_paths = [
        ".",  # Current directory
        "./images",
        "./test_images",
        "./scenes",
        "./output",
        "./data",
        "../images",
        "../scenes"
    ]

    # Image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

    found_images = []

    for search_path in search_paths:
        if os.path.exists(search_path):
            for file in os.listdir(search_path):
                if Path(file).suffix.lower() in image_extensions:
                    full_path = os.path.join(search_path, file)
                    found_images.append(full_path)

    return found_images


def test_watermark():
    """Test watermark functionality"""
    print("🎨 SLEEPY DULL STORIES WATERMARK TEST")
    print("=" * 50)

    # Initialize watermark tool
    watermark_tool = SleepyDullStoriesWatermark()

    # Find test images
    test_images = find_test_images()

    if not test_images:
        print("❌ No test images found!")
        print("\n📁 To test the watermark:")
        print("1. Put a test image (PNG, JPG) in the same folder")
        print("2. Name it something like: test.png, scene.jpg, etc.")
        print("3. Run this script again")
        print("\n🔍 Searched in these locations:")
        print("- Current directory (.)")
        print("- ./images/")
        print("- ./test_images/")
        print("- ./scenes/")
        print("- ./output/")
        return False

    print(f"🖼️ Found {len(test_images)} test images:")
    for i, img_path in enumerate(test_images[:5]):  # Show max 5
        print(f"   {i+1}. {img_path}")

    if len(test_images) > 5:
        print(f"   ... and {len(test_images) - 5} more")

    # Test with first image
    test_image = test_images[0]
    print(f"\n🧪 Testing with: {test_image}")

    success = watermark_tool.add_watermark_to_image(test_image)

    if success:
        print("\n🎉 WATERMARK TEST SUCCESSFUL!")
        print("✅ Check the '_watermarked' version of your image")
        print("✅ You should see '© Sleepy Dull Stories' in bottom-right")
    else:
        print("\n❌ WATERMARK TEST FAILED!")
        print("Check the error messages above")

    return success


def manual_test(image_path: str):
    """Test with specific image path"""
    print(f"🧪 Manual test with: {image_path}")

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return False

    watermark_tool = SleepyDullStoriesWatermark()
    return watermark_tool.add_watermark_to_image(image_path)


if __name__ == "__main__":
    print("🚀 Starting watermark test...")

    # Auto test
    success = test_watermark()

    if not success:
        print("\n" + "="*50)
        print("🔧 MANUAL TEST OPTION:")
        print("If you have a specific image, try:")
        print('manual_test("path/to/your/image.png")')
        print("\nExample:")
        print('manual_test("test.png")')
        print('manual_test("scene_01.png")')
        print('manual_test("C:/Users/YourName/Desktop/image.jpg")')