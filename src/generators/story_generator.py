# src/generators/chatgpt_prompt_optimizer.py
import os
import json
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# .env dosyasını yükle
load_dotenv()

# OpenAI for prompt optimization
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
    print("✅ OpenAI available for prompt optimization")
except ImportError:
    print("❌ OpenAI not found!")
    print("💡 Install with: pip install openai")
    OPENAI_AVAILABLE = False

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# ChatGPT optimization settings
CHATGPT_SETTINGS = {
    'model': 'gpt-4',  # GPT-4 for best prompt optimization
    'max_tokens': 500,  # Shorter responses for prompts
    'temperature': 0.7,  # Some creativity for visual descriptions
    'max_prompt_words': 600,  # Limit input to avoid 650+ error
    'target_output_words': 50,  # Target optimized prompt length
}

# Sleepy Dull Stories style guide
SLEEPY_STYLE_GUIDE = """
SLEEPY DULL STORIES - Image Style Guide:

VISUAL STYLE:
- Soft watercolor painting aesthetic
- Warm, muted color palette: amber, ochre, soft gold, brown tones
- Gentle, diffused lighting (no harsh contrasts)
- Peaceful, meditative atmosphere
- No people visible in scenes
- Ancient historical settings

MOOD KEYWORDS TO USE:
- peaceful, serene, quiet, gentle, soft, calm, tranquil
- warm lighting, golden hour, soft shadows
- muted tones, earthy colors, weathered textures

KEYWORDS TO AVOID:
- dramatic, intense, bright, vivid, sharp, bold
- cinematic, action, movement, chaos
- modern elements, technology

FORMAT:
Keep prompts concise: 40-60 words maximum
Focus on: Setting + Mood + Visual Style + Lighting
"""


def find_correct_paths():
    """Doğru dosya yollarını bul"""
    print("🔍 Path detection starting...")

    possible_csv_paths = [
        "src/data/topics.csv",
        "data/topics.csv",
        "../data/topics.csv",
        "../../src/data/topics.csv",
        "../../data/topics.csv"
    ]

    possible_output_paths = [
        "../../output",
        "../output",
        "output",
        "src/output",
        "../src/output"
    ]

    csv_path = None
    output_path = None

    # CSV path bul
    for path in possible_csv_paths:
        if os.path.exists(path):
            csv_path = path
            print(f"   ✅ CSV found: {path}")
            break

    # Output path bul
    for path in possible_output_paths:
        if os.path.exists(path):
            test_prompt_path = os.path.join(path, "1", "prompt.json")
            if os.path.exists(test_prompt_path):
                output_path = path
                print(f"   ✅ Output dir found: {path}")
                break

    return csv_path, output_path


# Doğru yolları bul
TOPIC_CSV_PATH, OUTPUT_BASE_PATH = find_correct_paths()


def print_step(step_num: str, description: str):
    """Adımları yazdır"""
    print(f"\n🔄 Adım {step_num}: {description}")
    print("-" * 60)


def check_csv_for_prompt_tasks():
    """CSV'den prompt optimization'a hazır hikayeleri bul"""
    try:
        df = pd.read_csv(TOPIC_CSV_PATH)

        # Gerekli kolonları ekle
        required_columns = ['prompts_optimized', 'chatgpt_prompts_created']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0

        df.to_csv(TOPIC_CSV_PATH, index=False)

        # Prompt optimization'a hazır olan hikayeler
        ready_for_prompts = df[
            (df['done'] == 1) &
            (df['audio_generated'] == 1) &
            (df['chatgpt_prompts_created'] == 0)
            ]

        return df, ready_for_prompts

    except Exception as e:
        print(f"❌ CSV okuma hatası: {e}")
        return None, None


def update_csv_prompt_status(csv_path: str, row_index: int, **kwargs):
    """CSV'de prompt optimization durumunu güncelle"""
    try:
        df = pd.read_csv(csv_path)
        for column, value in kwargs.items():
            if column in df.columns:
                df.at[row_index, column] = value
                print(f"   ✅ {column} = {value} güncellendi")
        df.to_csv(csv_path, index=False)
        return True, None
    except Exception as e:
        return False, f"CSV güncelleme hatası: {e}"


def load_prompts_from_json(prompt_file_path: str):
    """prompt.json dosyasını yükle"""
    try:
        if not os.path.exists(prompt_file_path):
            return None, f"Prompt dosyası bulunamadı: {prompt_file_path}"

        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        return prompts, None
    except Exception as e:
        return None, f"Dosya okuma hatası: {e}"


def truncate_long_prompt(prompt: str, max_words: int = 600) -> str:
    """Çok uzun prompt'ları kısalt ChatGPT limiti için"""
    words = prompt.split()
    if len(words) <= max_words:
        return prompt

    # İlk max_words kelimeyi al
    truncated = ' '.join(words[:max_words])
    print(f"   ⚠️ Prompt truncated: {len(words)} -> {max_words} words")
    return truncated


class SleepyChatGPTOptimizer:
    """ChatGPT ile Sleepy Dull Stories prompt optimization"""

    def __init__(self):
        if not OPENAI_AVAILABLE:
            raise Exception("OpenAI not available. Install with: pip install openai")

        if not OPENAI_API_KEY:
            raise Exception("OPENAI_API_KEY not configured in .env")

        # OpenAI client
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        # Cost tracking
        self.total_cost = 0.0
        self.prompts_optimized = 0

        print("🤖 Sleepy ChatGPT Optimizer initialized")
        print(f"🔗 Model: {CHATGPT_SETTINGS['model']}")
        print(
            f"📊 Word limits: {get_flexible_word_limit()[1]} (±{CHATGPT_SETTINGS['tolerance_percentage']}% = {get_flexible_word_limit()[0]}-{get_flexible_word_limit()[2]})")
        print(f"🎭 Style: Sleepy Dull Stories optimized")

    def optimize_single_prompt(self, original_prompt: str, segment_number: int) -> Tuple[bool, str, Dict]:
        """Tek bir prompt'u optimize et"""
        try:
            # Check prompt length with tolerance
            safe_prompt, was_truncated = truncate_long_prompt(original_prompt)

            print(f"🤖 Optimizing segment {segment_number}...")

            # ChatGPT optimization request
            system_prompt = f"""You are an expert at creating image generation prompts for "Sleepy Dull Stories" - a YouTube channel focused on peaceful historical content for sleep and relaxation.

{SLEEPY_STYLE_GUIDE}

Your task: Transform the given historical scene description into a concise, optimized image prompt suitable for AI image generation.

REQUIREMENTS:
1. Keep it 40-60 words maximum
2. Focus on peaceful, sleepy atmosphere  
3. Use warm, muted colors
4. No people in scenes
5. Emphasize tranquil, meditative mood
6. Ancient historical setting

Example transformation:
INPUT: "A grand Roman basilica with intricate columns, busy marketplace, people trading, dramatic lighting, cinematic atmosphere, 8k quality..."
OUTPUT: "Peaceful Roman basilica at sunset, empty stone columns casting soft shadows, warm golden light, muted earth tones, tranquil ancient architecture, no people, serene atmosphere"
"""

            user_prompt = f"""Transform this historical scene into a Sleepy Dull Stories image prompt:

ORIGINAL SCENE:
{safe_prompt}

OPTIMIZED PROMPT (40-60 words, peaceful & sleepy):"""

            # ChatGPT API call
            response = self.client.chat.completions.create(
                model=CHATGPT_SETTINGS['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=CHATGPT_SETTINGS['max_tokens'],
                temperature=CHATGPT_SETTINGS['temperature']
            )

            optimized_prompt = response.choices[0].message.content.strip()

            # Remove any quotes or extra formatting
            optimized_prompt = optimized_prompt.replace('"', '').replace("'", "").strip()

            # Count words in result
            word_count = len(optimized_prompt.split())

            print(f"   ✅ Optimized: {word_count} words")
            print(f"   📝 Result: {optimized_prompt[:80]}...")

            self.prompts_optimized += 1
            # GPT-4 cost: ~$0.03 per 1K tokens (rough estimate)
            self.total_cost += 0.01

            return True, optimized_prompt, {
                'model': CHATGPT_SETTINGS['model'],
                'original_words': len(original_prompt.split()),
                'optimized_words': word_count,
                'was_truncated': was_truncated,
                'cost_estimate': 0.01
            }

        except Exception as e:
            print(f"   ❌ Optimization failed: {e}")
            return False, str(e), {}

    def optimize_thumbnail_prompt(self, clickbait_title: str, topic: str) -> Tuple[bool, str]:
        """Thumbnail için optimize edilmiş prompt oluştur"""
        try:
            print(f"🖼️ Optimizing thumbnail prompt...")

            system_prompt = f"""You are creating a YouTube thumbnail description for "Sleepy Dull Stories" - peaceful historical content for sleep.

{SLEEPY_STYLE_GUIDE}

Create a thumbnail image prompt that is:
1. Calming and peaceful (not clickbait-y)
2. Shows historical setting without people
3. Warm, inviting colors for bedtime viewing
4. Professional but soothing aesthetic
5. 40-50 words maximum

The thumbnail should attract viewers seeking peaceful, educational content for sleep."""

            user_prompt = f"""Create a sleepy thumbnail prompt for:

TITLE: {clickbait_title}
TOPIC: {topic}

SLEEPY THUMBNAIL PROMPT (40-50 words):"""

            response = self.client.chat.completions.create(
                model=CHATGPT_SETTINGS['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )

            thumbnail_prompt = response.choices[0].message.content.strip()
            thumbnail_prompt = thumbnail_prompt.replace('"', '').replace("'", "").strip()

            print(f"   ✅ Thumbnail optimized: {len(thumbnail_prompt.split())} words")

            return True, thumbnail_prompt

        except Exception as e:
            print(f"   ❌ Thumbnail optimization failed: {e}")
            return False, str(e)

    def process_story_prompts(self, story_id: int, topic: str, prompts: dict):
        """Bir hikaye için tüm prompt'ları optimize et"""
        output_dir = os.path.join(OUTPUT_BASE_PATH, str(story_id))

        results = {
            'thumbnail_prompt': {'success': False, 'prompt': None, 'error': None},
            'optimized_prompts': {'success': False, 'prompts': [], 'error': None, 'total_count': 0},
            'total_cost': 0.0,
            'prompts_processed': 0
        }

        clickbait_title = prompts.get('clickbait_title', topic)
        print(f"🎯 Optimizing Prompts: {clickbait_title}")

        try:
            # 1. Optimize Thumbnail Prompt
            print(f"🖼️ Optimizing thumbnail...")
            thumbnail_success, thumbnail_result = self.optimize_thumbnail_prompt(clickbait_title, topic)

            if thumbnail_success:
                results['thumbnail_prompt'] = {
                    'success': True,
                    'prompt': thumbnail_result,
                    'error': None
                }
            else:
                results['thumbnail_prompt'] = {
                    'success': False,
                    'prompt': None,
                    'error': thumbnail_result
                }

            # Rate limiting
            time.sleep(2)

            # 2. Optimize Segment Prompts
            original_image_prompts = prompts.get('image_prompts', [])
            print(f"🎨 Optimizing {len(original_image_prompts)} segment prompts...")

            optimized_prompts = []

            for idx, original_prompt in enumerate(original_image_prompts, 1):
                success, result, info = self.optimize_single_prompt(original_prompt, idx)

                if success:
                    optimized_prompts.append({
                        'segment_id': idx,
                        'original_prompt': original_prompt,
                        'optimized_prompt': result,
                        'original_words': info.get('original_words', 0),
                        'optimized_words': info.get('optimized_words', 0),
                        'was_truncated': info.get('was_truncated', False)
                    })
                    print(f"      ✅ Segment {idx} optimized")
                else:
                    print(f"      ❌ Segment {idx} failed: {result}")

                # Rate limiting between prompts
                if idx < len(original_image_prompts):
                    time.sleep(1)

            # Store optimized prompts results
            if optimized_prompts:
                results['optimized_prompts'] = {
                    'success': True,
                    'prompts': optimized_prompts,
                    'error': None,
                    'total_count': len(optimized_prompts)
                }

            # Update totals
            results['total_cost'] = self.total_cost
            results['prompts_processed'] = self.prompts_optimized

        except Exception as e:
            print(f"❌ Processing error: {e}")
            results['optimized_prompts']['error'] = str(e)

        return results


def create_sleepy_prompts_summary(story_id: int, topic: str, results: dict):
    """Optimize edilmiş prompt'ları kaydet"""
    output_dir = os.path.join(OUTPUT_BASE_PATH, str(story_id))
    summary_path = os.path.join(output_dir, "sleepy_optimized_prompts.json")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "story_id": story_id,
        "topic": topic,
        "optimization_method": "ChatGPT GPT-4",
        "style": "Sleepy Dull Stories Optimized",
        "chatgpt_settings": CHATGPT_SETTINGS,
        "word_limits": {
            "base_limit": get_flexible_word_limit()[1],
            "tolerance_percentage": CHATGPT_SETTINGS['tolerance_percentage'],
            "min_limit": get_flexible_word_limit()[0],
            "max_limit": get_flexible_word_limit()[2]
        },
        "results": results,
        "stats": {
            "thumbnail_success": results['thumbnail_prompt']['success'],
            "segment_prompts_count": results['optimized_prompts']['total_count'],
            "total_cost_usd": results['total_cost'],
            "prompts_processed": results['prompts_processed'],
            "overall_success": (
                    results['thumbnail_prompt']['success'] and
                    results['optimized_prompts']['total_count'] >= 5
            )
        }
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def process_sleepy_prompt_optimization():
    """Ana Sleepy Dull Stories prompt optimization işlemi"""
    print("🤖 SLEEPY DULL STORIES - ChatGPT Prompt Optimizer v1.0")
    print("😴 Optimizing Historical Prompts for Peaceful Content")
    print("🔧 GPT-4 Powered Optimization with ±15% Word Tolerance")
    print("=" * 70)

    # Library availability check
    if not OPENAI_AVAILABLE:
        print("❌ OpenAI not available!")
        print("💡 Install with: pip install openai")
        return

    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not configured!")
        print("💡 Add OPENAI_API_KEY=your-key-here to .env")
        return

    # Path check
    if not TOPIC_CSV_PATH or not OUTPUT_BASE_PATH:
        print("❌ Required paths not found!")
        return

    # Show settings
    min_limit, base_limit, max_limit = get_flexible_word_limit()
    print(f"📁 CSV Path: {TOPIC_CSV_PATH}")
    print(f"📁 Output Path: {OUTPUT_BASE_PATH}")
    print(f"🤖 Model: {CHATGPT_SETTINGS['model']}")
    print(f"📊 Word Limits: {base_limit} (±{CHATGPT_SETTINGS['tolerance_percentage']}% = {min_limit}-{max_limit})")
    print(f"🎭 Style: Sleepy Dull Stories optimized")

    print_step("1", "Checking stories ready for prompt optimization")

    df, ready_for_prompts = check_csv_for_prompt_tasks()
    if df is None:
        return

    if ready_for_prompts.empty:
        print("✅ No stories ready for prompt optimization!")
        return

    print(f"🎯 Stories ready for optimization: {len(ready_for_prompts)}")

    # Initialize optimizer
    try:
        optimizer = SleepyChatGPTOptimizer()
    except Exception as e:
        print(f"❌ Optimizer creation failed: {e}")
        return

    # Process each story
    total_stories = len(ready_for_prompts)
    success_count = 0
    error_count = 0
    total_cost = 0.0

    for idx, (csv_index, row) in enumerate(ready_for_prompts.iterrows(), 1):
        story_id = csv_index + 1
        topic = row['topic']

        print_step(f"2.{idx}", f"Optimizing Story {story_id}/{total_stories}")
        print(f"📚 Topic: {topic}")

        # Load prompts
        prompt_file_path = os.path.join(OUTPUT_BASE_PATH, str(story_id), "prompt.json")
        prompts, load_error = load_prompts_from_json(prompt_file_path)

        if load_error:
            print(f"❌ {load_error}")
            error_count += 1
            continue

        # Process prompt optimization
        start_time = time.time()

        try:
            results = optimizer.process_story_prompts(story_id, topic, prompts)

            # Save results
            summary = create_sleepy_prompts_summary(story_id, topic, results)

            # Update CSV
            update_data = {}
            if results['thumbnail_prompt']['success'] and results['optimized_prompts']['total_count'] >= 5:
                update_data['prompts_optimized'] = 1
                update_data['chatgpt_prompts_created'] = 1

            if update_data:
                update_csv_prompt_status(TOPIC_CSV_PATH, csv_index, **update_data)

            # Evaluate success
            end_time = time.time()
            processing_time = int(end_time - start_time)

            total_success = summary['stats']['overall_success']

            print(f"\n📊 Optimization Results:")
            print(f"   🖼️ Thumbnail: {'✅' if results['thumbnail_prompt']['success'] else '❌'}")
            print(f"   🎨 Segments: {results['optimized_prompts']['total_count']} optimized")
            print(f"   💰 Cost: ${results['total_cost']:.2f}")
            print(f"   ⚡ Processing time: {processing_time // 60}m {processing_time % 60}s")

            if total_success:
                print(f"✅ Optimization successful!")
                success_count += 1
            else:
                print(f"❌ Optimization failed!")
                error_count += 1

            total_cost += results['total_cost']

        except Exception as e:
            print(f"❌ Processing error: {e}")
            error_count += 1

    # Final report
    print_step("3", "Prompt optimization completed")

    print(f"📊 FINAL OPTIMIZATION REPORT:")
    print(f"  ✅ Successful: {success_count} stories")
    print(f"  ❌ Failed: {error_count} stories")
    print(f"  📁 Total processed: {total_stories} stories")
    print(f"  💰 Total cost: ${total_cost:.2f}")
    print(f"  🤖 Method: ChatGPT GPT-4")
    print(f"  🎭 Style: Sleepy Dull Stories optimized")
    print(f"  📊 Word tolerance: ±{CHATGPT_SETTINGS['tolerance_percentage']}%")

    if success_count > 0:
        print(f"\n🎉 Successfully optimized prompts for {success_count} stories!")
        print(f"📁 Results location: output/*/sleepy_optimized_prompts.json")
        print(f"\n😴 Ready for peaceful image generation!")
        print(f"   • Prompts optimized for Sleepy Dull Stories style")
        print(f"   • 40-60 words each, perfect for AI generation")
        print(f"   • Peaceful, historical, no-people scenes")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        process_sleepy_prompt_optimization()
    except KeyboardInterrupt:
        print("\n⚠️ Prompt optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()