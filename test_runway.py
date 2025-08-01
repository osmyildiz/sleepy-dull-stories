"""
Ancient Roman Prompt Enhancer
Transform modern-looking scenes to authentic ancient Roman scenes
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

# Runway SDK import
try:
    from runwayml import RunwayML
    print("✅ RunwayML SDK available")
    SDK_AVAILABLE = True
except ImportError:
    print("⚠️ RunwayML SDK not found - prompt analysis only")
    SDK_AVAILABLE = False

def enhance_scene_4_prompt():
    """Scene 4 - Ancient Roman fisherman prompt enhancement"""

    # Original prompt
    original = """Wide establishing shot of small Roman harbor at dawn, mist rising from Sarno River waters, fishing boats silhouetted against pink sky. FOREGROUND: Gaius Nautius, weathered fisherman with sun-darkened skin and gray-streaked beard, methodically checking nets with hands roughened by salt water, his deep-set eyes reading weather patterns in cloud formations. Dramatic side lighting emphasizes the texture of worn fishing equipment and weathered wooden boat. BACKGROUND: Seagulls wheeling overhead, distant Mount Vesuvius reflected in calm water, other fishing boats preparing for departure."""

    print("🏛️ ORIGINAL PROMPT:")
    print(f"📝 {original}")
    print(f"📏 Length: {len(original)} characters")

    # Enhanced with specific ancient Roman details
    enhanced = """79 CE Pompeii harbor: Ancient Roman fisherman Gaius Nautius in white tunic and leather subligaculum, Roman sandals, period-accurate fishing nets made of hemp rope, wooden Roman fishing boat with square sail and carved prow. Ancient Mediterranean fishing techniques, clay amphora for water, woven reed baskets. Mount Vesuvius background, pre-eruption atmosphere. CRITICAL: Ancient Roman period clothing, tools, boats - NO modern elements."""

    print("\n🏺 ENHANCED ANCIENT ROMAN PROMPT:")
    print(f"📝 {enhanced}")
    print(f"📏 Length: {len(enhanced)} characters")

    # Compact version for API
    compact = "79 CE Pompeii: Roman fisherman Gaius in white tunic, leather belt, Roman sandals. Ancient hemp nets, wooden Roman boat with square sail. Clay amphora, reed baskets. Vesuvius background. Ancient Mediterranean fishing, no modern elements."

    print("\n⚡ COMPACT API VERSION:")
    print(f"📝 {compact}")
    print(f"📏 Length: {len(compact)} characters")

    return {
        "original": original,
        "enhanced": enhanced,
        "compact": compact
    }

def ancient_roman_enhancement_rules():
    """Rules for making any scene authentically ancient Roman"""

    rules = {
        "clothing": [
            "tunic (white, brown, or natural linen)",
            "toga for wealthy citizens",
            "stola for women",
            "subligaculum (leather undergarment)",
            "Roman sandals (caligae for soldiers)",
            "pallium (cloak) for cold weather",
            "NO modern buttons, zippers, or synthetic fabrics"
        ],

        "tools_and_objects": [
            "clay amphora for water/wine storage",
            "woven reed baskets",
            "hemp rope (not synthetic)",
            "wooden handles and tools",
            "bronze or iron implements",
            "leather goods with period stitching",
            "NO plastic, modern metal alloys, or machine-made items"
        ],

        "architecture": [
            "Roman concrete and brick construction",
            "red clay roof tiles",
            "marble columns and decorations",
            "frescoes on interior walls",
            "mosaics on floors",
            "aqueduct systems",
            "NO modern construction materials or methods"
        ],

        "transportation": [
            "Roman galleys with square sails",
            "wooden fishing boats with carved prows",
            "horse-drawn carts with iron-rimmed wheels",
            "Roman roads with stone paving",
            "NO modern vehicles, engines, or synthetic materials"
        ],

        "people": [
            "Mediterranean skin tones",
            "Period-accurate hairstyles (no modern cuts)",
            "Roman grooming standards",
            "Natural aging and weathering",
            "Muscular build from manual labor",
            "NO modern grooming, haircuts, or fitness aesthetics"
        ]
    }

    print("\n📋 ANCIENT ROMAN ENHANCEMENT RULES:")
    for category, items in rules.items():
        print(f"\n🏛️ {category.upper()}:")
        for item in items:
            print(f"   • {item}")

    return rules

def create_enhanced_prompts():
    """Create enhanced prompts with ancient Roman specificity"""

    # Common ancient Roman elements to add
    roman_elements = [
        "79 CE Pompeii setting",
        "Mount Vesuvius in background (pre-eruption)",
        "Roman architecture with red clay tiles",
        "Characters in period-accurate Roman clothing",
        "Ancient Mediterranean tools and materials",
        "No modern elements whatsoever",
        "Historically accurate 1st century CE details"
    ]

    # Template for any scene
    template = "79 CE Pompeii: {character_description} in {roman_clothing}. {ancient_tools_objects}. {roman_architecture_background}. {vesuvius_reference}. Ancient Roman period accurate, no modern elements."

    print("\n🏺 ANCIENT ROMAN ENHANCEMENT TEMPLATE:")
    print(f"📝 {template}")

    # Scene 4 specific enhancement
    scene_4_enhanced = template.format(
        character_description="Roman fisherman Gaius Nautius",
        roman_clothing="white linen tunic, leather subligaculum, Roman sandals",
        ancient_tools_objects="Hemp fishing nets, wooden Roman boat with square sail, clay amphora, reed baskets",
        roman_architecture_background="Roman harbor with stone quays, red tile roofs",
        vesuvius_reference="Mount Vesuvius visible in background"
    )

    print("\n🎣 SCENE 4 ENHANCED:")
    print(f"📝 {scene_4_enhanced}")
    print(f"📏 Length: {len(scene_4_enhanced)} characters")

    return scene_4_enhanced

def regenerate_scene_4():
    """Regenerate Scene 4 with enhanced ancient Roman prompt"""

    if not SDK_AVAILABLE:
        print("❌ RunwayML SDK not available for regeneration")
        return False

    api_key = os.getenv("RUNWAYML_API_SECRET")
    if not api_key:
        print("❌ RUNWAYML_API_SECRET not found in environment")
        return False

    print("\n🚀 REGENERATING SCENE 4 WITH ANCIENT ROMAN ENHANCEMENT")
    print("=" * 60)

    # Enhanced prompt for Scene 4
    enhanced_prompt = "79 CE Pompeii: Roman fisherman Gaius in white tunic, leather belt, Roman sandals. Ancient hemp nets, wooden Roman boat with square sail. Clay amphora, reed baskets. Vesuvius background. Ancient Mediterranean fishing, no modern elements."

    print(f"📝 New Prompt: {enhanced_prompt}")
    print(f"📏 Length: {len(enhanced_prompt)} characters")

    try:
        # Initialize client
        client = RunwayML(api_key=api_key)
        print("✅ Runway client initialized")

        # Create generation task
        response = client.text_to_image.create(
            model="gen4_image",
            ratio="1920:1080",
            prompt_text=enhanced_prompt
        )

        if hasattr(response, 'id'):
            task_id = response.id
            print(f"📋 Task ID: {task_id}")

            # Poll for completion
            print("⏳ Waiting for generation...")
            for attempt in range(30):  # 5 minutes max
                time.sleep(10)

                try:
                    task = client.tasks.retrieve(task_id)

                    if hasattr(task, 'status'):
                        status = task.status
                        print(f"📊 Status: {status}")

                        if status == "SUCCEEDED":
                            if hasattr(task, 'output') and task.output:
                                # Get image URL
                                if isinstance(task.output, list) and len(task.output) > 0:
                                    image_url = task.output[0]
                                elif isinstance(task.output, dict):
                                    image_url = task.output.get('url') or task.output.get('image_url')
                                else:
                                    image_url = str(task.output)

                                print(f"✅ Generation complete!")
                                print(f"🖼️ Image URL: {image_url}")

                                # Save enhanced image
                                saved_path = save_enhanced_scene_4(image_url, enhanced_prompt)
                                if saved_path:
                                    print(f"💾 Saved to: {saved_path}")
                                    return True
                                else:
                                    print("❌ Failed to save image")
                                    return False
                            else:
                                print("❌ No output in successful task")
                                return False

                        elif status == "FAILED":
                            error_detail = getattr(task, 'error', 'Unknown error')
                            print(f"❌ Generation failed: {error_detail}")
                            return False

                        elif status in ["PENDING", "RUNNING", "THROTTLED"]:
                            continue
                        else:
                            print(f"⚠️ Unknown status: {status}")
                            continue

                except Exception as poll_error:
                    print(f"⚠️ Polling error: {poll_error}")
                    time.sleep(5)
                    continue

            print("❌ Timeout waiting for generation")
            return False
        else:
            print("❌ No task ID received")
            return False

    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def save_enhanced_scene_4(image_url: str, prompt: str) -> str:
    """Save the enhanced Scene 4 image"""

    import requests

    output_dir = "../output/1/scenes"
    os.makedirs(output_dir, exist_ok=True)

    # Save as enhanced version
    filename = "scene_04_enhanced.png"
    file_path = os.path.join(output_dir, filename)

    try:
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            f.write(response.content)

        file_size = os.path.getsize(file_path)
        print(f"💾 Enhanced Scene 4 saved: {filename} ({file_size:,} bytes)")

        # Save prompt info
        prompt_file = os.path.join(output_dir, "scene_04_enhanced_prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"Enhanced Scene 4 Prompt:\n{prompt}\n\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        return file_path

    except Exception as e:
        print(f"❌ Save error: {e}")
        return None

def quick_scene_test():
    """Quick test for any scene with enhanced Roman prompt"""

    if not SDK_AVAILABLE:
        print("❌ SDK not available for testing")
        return False

    print("\n🧪 QUICK ROMAN ENHANCEMENT TEST")
    print("=" * 40)

    # Test prompt
    test_prompt = "79 CE Pompeii: Roman citizen in white tunic, Roman sandals. Ancient Roman architecture, clay vessels. Mount Vesuvius background. No modern elements."

    print(f"📝 Test Prompt: {test_prompt}")
    print(f"📏 Length: {len(test_prompt)} characters")

    api_key = os.getenv("RUNWAYML_API_SECRET")
    if not api_key:
        print("❌ RUNWAYML_API_SECRET not found")
        return False

    try:
        client = RunwayML(api_key=api_key)

        response = client.text_to_image.create(
            model="gen4_image",
            ratio="1920:1080",
            prompt_text=test_prompt
        )

        print(f"✅ Test generation started: {response.id}")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main execution"""
    print("🏛️ ANCIENT ROMAN PROMPT ENHANCER & REGENERATOR")
    print("=" * 60)

    # Analyze Scene 4
    print("\n📊 SCENE 4 ANALYSIS:")
    prompts = enhance_scene_4_prompt()

    # Show enhancement rules
    ancient_roman_enhancement_rules()

    # Create template
    enhanced_prompt = create_enhanced_prompts()

    print("\n🎯 ENHANCEMENT COMPLETE!")
    print("Now you can:")
    print("1. 🚀 Regenerate Scene 4 with enhanced prompt")
    print("2. 🧪 Quick test with Roman enhancement")
    print("3. 📝 Use enhanced prompt in main generator")

    # Interactive options
    while True:
        print("\n" + "=" * 40)
        print("CHOOSE ACTION:")
        print("1. 🚀 Regenerate Scene 4 (Full process)")
        print("2. 🧪 Quick test (Just start generation)")
        print("3. 📝 Show final prompt only")
        print("4. ❌ Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            print("\n🚀 Starting Scene 4 regeneration...")
            success = regenerate_scene_4()
            if success:
                print("\n🎉 Scene 4 regeneration complete!")
                print("📁 Check ../output/1/scenes/scene_04_enhanced.png")
            else:
                print("\n❌ Regeneration failed")

        elif choice == "2":
            print("\n🧪 Starting quick test...")
            quick_scene_test()

        elif choice == "3":
            print("\n📝 FINAL ENHANCED PROMPT:")
            print(f"'{enhanced_prompt}'")

        elif choice == "4":
            print("\n👋 Exiting...")
            break

        else:
            print("❌ Invalid choice, try again")

if __name__ == "__main__":
    main()