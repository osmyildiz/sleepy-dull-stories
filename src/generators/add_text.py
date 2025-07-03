from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import os

# === Ayarlar ===
STORY_ID = 1
BASE_DIR = Path("../../output") / str(STORY_ID)
PROMPT_PATH = BASE_DIR / "prompt.json"
INPUT_IMAGE_PATH = BASE_DIR / "runway_visuals" / "cover_test_right.jpg"
OUTPUT_IMAGE_PATH = BASE_DIR / "runway_visuals" / "final_thumbnail.jpg"

# Font ve stil ayarları
CHANNEL_NAME = "SLEEPY DULL STORIES"
CHANNEL_FONT_SIZE = 50
TITLE_FONT_SIZE = 100
TEXT_COLOR = "#FFD700"  # Altın sarısı
STROKE_COLOR = "#000000"  # Siyah kontur
STROKE_WIDTH = 3
TEXT_MARGIN_X = 60
TEXT_MARGIN_TOP = 120
TEXT_LINE_SPACING = 10
MAX_TEXT_AREA_WIDTH = 600  # Yazının genişlik sınırı

def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_paths = [
        "/System/Library/Fonts/Times.ttc",
        "/System/Library/Fonts/TimesNewRomanPSMT.ttc",
        "/Library/Fonts/Times New Roman.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
    ]
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    return ImageFont.load_default()

def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = font.getbbox(test_line)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def draw_text_with_stroke(draw: ImageDraw.Draw, position: tuple, text: str,
                          font: ImageFont.FreeTypeFont, fill_color: str,
                          stroke_color: str, stroke_width: int):
    x, y = position
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)
    draw.text((x, y), text, font=font, fill=fill_color)

def add_text_overlay(image_path: Path, title: str, output_path: Path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Kanal adı üstte büyükçe yazılsın
    channel_font = get_font(CHANNEL_FONT_SIZE, bold=True)
    draw_text_with_stroke(draw, (TEXT_MARGIN_X, TEXT_MARGIN_TOP - 70), CHANNEL_NAME,
                          channel_font, TEXT_COLOR, STROKE_COLOR, 2)

    # Başlık satırlara bölünsün
    title_font = get_font(TITLE_FONT_SIZE, bold=True)
    wrapped_lines = wrap_text(title, title_font, MAX_TEXT_AREA_WIDTH)

    y = TEXT_MARGIN_TOP
    for line in wrapped_lines:
        draw_text_with_stroke(draw, (TEXT_MARGIN_X, y), line, title_font,
                              TEXT_COLOR, STROKE_COLOR, STROKE_WIDTH)
        y += title_font.getbbox("Ay")[3] - title_font.getbbox("Ay")[1] + TEXT_LINE_SPACING

    img.save(output_path, "JPEG", quality=95)
    print(f"✅ Modern wrap tabanlı thumbnail oluşturuldu: {output_path}")

def main():
    if not PROMPT_PATH.exists():
        print(f"❌ prompt.json bulunamadı: {PROMPT_PATH}")
        return
    if not INPUT_IMAGE_PATH.exists():
        print(f"❌ Giriş görseli bulunamadı: {INPUT_IMAGE_PATH}")
        return
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    clickbait_title = prompt_data.get("clickbait_title", "Untitled Story")
    print(f"📝 Başlık: {clickbait_title}")
    print(f"🖼️  Giriş görseli: {INPUT_IMAGE_PATH}")
    os.makedirs(OUTPUT_IMAGE_PATH.parent, exist_ok=True)
    add_text_overlay(INPUT_IMAGE_PATH, clickbait_title, OUTPUT_IMAGE_PATH)

main()
