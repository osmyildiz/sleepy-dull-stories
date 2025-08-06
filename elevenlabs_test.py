"""
ElevenLabs Cleopatra Demo - Fixed Version
Voice ID checker + fallback to default voices
"""

import requests
import json
import os
from datetime import datetime

# ElevenLabs Configuration
ELEVENLABS_API_KEY = "sk_d48c501bddb3e09f0ba95f7143202925e3c7482003ccb2a6"

# Popular voice IDs (these change, so we'll check them)
VOICE_OPTIONS = [
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Sarah"},
    {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
    {"id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi"},
    {"id": "CYw3kZ02Hs0563khs1Fj", "name": "Dave"},
    {"id": "D38z5RcWu1voky8WS1ja", "name": "Fin"},
    {"id": "JBFqnCBsd6RMkjVDRZzb", "name": "George"},
    {"id": "N2lVS1w4EtoT3dr4eOWO", "name": "Callum"},
    {"id": "TX3LPaxmHKxFdv7VOQHJ", "name": "Liam"}
]


def get_available_voices():
    """Get all available voices from ElevenLabs"""

    url = "https://api.elevenlabs.io/v1/voices"
    headers = {
        "Accept": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }

    try:
        print("🔍 Checking available voices...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        voices_data = response.json()
        voices = voices_data.get('voices', [])

        print(f"✅ Found {len(voices)} available voices:")
        for voice in voices[:10]:  # Show first 10
            print(f"   🎙️ {voice['name']}: {voice['voice_id']}")

        return voices

    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting voices: {e}")
        return []


def find_best_voice(voices):
    """Find the best voice for Cleopatra (female, elegant)"""

    # Look for female voices first
    female_voices = []
    for voice in voices:
        labels = voice.get('labels', {})
        gender = labels.get('gender', '').lower()
        if gender == 'female':
            female_voices.append(voice)

    if female_voices:
        # Prefer voices with "elegant" or similar qualities
        for voice in female_voices:
            name = voice['name'].lower()
            if any(keyword in name for keyword in ['sarah', 'rachel', 'bella', 'elli']):
                print(f"🎭 Selected voice: {voice['name']} ({voice['voice_id']})")
                return voice['voice_id']

        # Fallback to first female voice
        selected = female_voices[0]
        print(f"🎭 Selected voice: {selected['name']} ({selected['voice_id']})")
        return selected['voice_id']

    # Ultimate fallback to first available voice
    if voices:
        selected = voices[0]
        print(f"🎭 Selected voice (fallback): {selected['name']} ({selected['voice_id']})")
        return selected['voice_id']

    return None


def create_elevenlabs_demo(voice_id):
    """Create 2-minute Cleopatra demo with FULL sound effects and emotion tags"""

    # Demo script with MAXIMUM ElevenLabs features restored!
    demo_script = """
[soft dawn ambiance] [gentle breath] The golden rays of Alexandria's final dawn filtered through silk curtains, [soft fabric rustling] casting their last dance across the marble floors of the royal palace.

[footsteps on marble, slow and deliberate] [deeply contemplative] Cleopatra, the last pharaoh of Egypt, moved through her chambers with the grace that had once commanded the hearts of Caesar and Antony. [melancholic sigh] 

[whispers] "Thirty-nine years upon this earth," [voice trembling with emotion] she murmured to herself, [gentle wind through windows] her fingers tracing the ancient hieroglyphs carved into her sacred alabaster box.

[distant temple bells] [growing stronger, more resolved] But this was no ordinary morning. [dramatic pause] The Roman legions waited beyond her walls, [sound of distant armies] their bronze shields glinting like hungry eyes in the morning sun.

[soft, almost breaking] "My beloved Egypt," [voice cracking with love and loss] she whispered, [tears falling softly] placing her palm against the cool marble of her window. [soft Mediterranean waves] The sea stretched endlessly before her, [bittersweet] carrying memories of ships that once brought tribute from across the known world.

[sound of a door opening quietly] [startled, then recognizing] Her most trusted servant entered, [footsteps approaching] carrying the small wicker basket that would seal her fate. [mysterious, knowing] Inside, coiled like destiny itself, lay the asp—death's gentle messenger.

[calm acceptance] [deep, peaceful breath] "Tell me," [softly curious] she asked, [slight smile in voice] "does it truly sleep as peacefully as they say?" 

[tender, almost motherly] The servant nodded, [barely audible] tears streaming down weathered cheeks that had served three generations of pharaohs.

[sound of silk robes rustling] [final, regal determination] Cleopatra moved to her golden throne one last time, [metallic resonance] the chair that had witnessed the rise and fall of dynasties. [profound acceptance] 

[gentle, with infinite sadness but no fear] "I go to meet my ancestors," [voice growing ethereal] she declared, [soft echo effect] her words seeming to carry the weight of centuries. [divine, transcendent] "And perhaps... perhaps to find Antony waiting beyond the veil of Isis."

[very soft wind chimes] [breathing slowing] [peaceful finale] [gentle silence expanding] The last queen of Egypt closed her eyes, [final exhale] and with her, an empire that had ruled for three thousand years [fade to eternal silence] passed into legend.
"""

    print("🎭 Creating ElevenLabs Cleopatra Demo")
    print("📜 Script length:", len(demo_script.split()), "words")
    print("⏱️ Estimated duration: ~2 minutes")
    print("=" * 60)

    # ElevenLabs API call
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }

    # ElevenLabs settings for MAXIMUM feature testing
    data = {
        "text": demo_script,
        "model_id": "eleven_turbo_v2_5",  # Latest model with ALL sound effects
        "voice_settings": {
            "stability": 0.71,
            "similarity_boost": 0.8,
            "style": 0.65,  # High style for dramatic effect
            "use_speaker_boost": True
        }
    }

    try:
        print("🚀 Calling ElevenLabs API...")
        response = requests.post(url, json=data, headers=headers, timeout=120)
        response.raise_for_status()

        # Save the demo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleopatra_demo_{timestamp}.mp3"

        with open(filename, 'wb') as f:
            f.write(response.content)

        file_size = len(response.content) / 1024 / 1024  # MB

        print(f"✅ Demo created successfully!")
        print(f"📁 File: {filename}")
        print(f"📦 Size: {file_size:.1f} MB")

        # Calculate cost
        char_count = len(demo_script)
        estimated_cost = char_count * 0.00002
        print(f"💰 Estimated cost: ${estimated_cost:.4f}")

        return filename

    except requests.exceptions.RequestException as e:
        print(f"❌ ElevenLabs API Error: {e}")
        if "401" in str(e):
            print("🔑 API key might be invalid or expired")
        elif "404" in str(e):
            print("🎭 Voice ID not found - this is the issue!")
        elif "429" in str(e):
            print("⏰ Rate limit exceeded - wait a bit")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def analyze_demo_features():
    """Analyze the restored demo's FULL ElevenLabs features"""

    features_used = {
        "emotion_tags": [
            "[soft dawn ambiance]", "[gentle breath]", "[deeply contemplative]",
            "[melancholic sigh]", "[whispers]", "[voice trembling with emotion]",
            "[growing stronger, more resolved]", "[dramatic pause]", "[soft, almost breaking]",
            "[voice cracking with love and loss]", "[bittersweet]", "[startled, then recognizing]",
            "[calm acceptance]", "[tender, almost motherly]", "[final, regal determination]",
            "[gentle, with infinite sadness but no fear]", "[divine, transcendent]"
        ],
        "sound_effects": [
            "[soft fabric rustling]", "[footsteps on marble, slow and deliberate]",
            "[gentle wind through windows]", "[distant temple bells]", "[sound of distant armies]",
            "[tears falling softly]", "[soft Mediterranean waves]", "[sound of a door opening quietly]",
            "[footsteps approaching]", "[sound of silk robes rustling]", "[metallic resonance]",
            "[soft echo effect]", "[very soft wind chimes]", "[gentle silence expanding]"
        ],
        "voice_modulation": [
            "[voice trembling with emotion]", "[voice cracking with love and loss]",
            "[barely audible]", "[soft echo effect]", "[voice growing ethereal]",
            "[breathing slowing]", "[final exhale]", "[fade to eternal silence]"
        ]
    }

    print(f"\n📊 RESTORED ElevenLabs Features Analysis:")
    print(f"🎭 Emotion tags: {len(features_used['emotion_tags'])} variations")
    print(f"🎵 Sound effects: {len(features_used['sound_effects'])} different sounds")
    print(f"🎙️ Voice modulations: {len(features_used['voice_modulation'])} techniques")
    print(f"📝 Total ElevenLabs tags: {sum(len(v) for v in features_used.values())} features")
    print("🚀 This is MAXIMUM ElevenLabs feature testing!")

    return features_used


def main():
    """Main demo creation function"""

    print("👑 ELEVENLABS CLEOPATRA DEMO GENERATOR - FIXED VERSION")
    print("🔧 With automatic voice detection and fallbacks")
    print("=" * 60)

    # Step 1: Get available voices
    voices = get_available_voices()

    if not voices:
        print("❌ Could not get voice list. Check your API key!")
        return

    # Step 2: Find best voice
    voice_id = find_best_voice(voices)

    if not voice_id:
        print("❌ No suitable voice found!")
        return

    # Step 3: Analyze the full feature set
    features = analyze_demo_features()

    # Step 4: Create the FULL POWER demo
    demo_file = create_elevenlabs_demo(voice_id)

    if demo_file:
        print(f"\n🎉 SUCCESS! Demo saved as: {demo_file}")
        print("🎧 Listen to experience ElevenLabs FULL capabilities:")
        print("   • 🎭 17 emotion tags: [whispers], [voice trembling], [ethereal]")
        print("   • 🎵 14 sound effects: [footsteps], [wind chimes], [temple bells]")
        print("   • 🎙️ 8 voice modulations: [voice cracking], [breathing slowing]")
        print("   • 🌅 Atmospheric sounds: [dawn ambiance], [silk rustling]")
        print("   • 📚 Historical storytelling with dramatic progression")
        print("   • 🏛️ Alexandria setting with authentic details")

        if features:
            print("   • 🔥 ALL 39 ElevenLabs features restored!")

        print("\n💥 Maximum ElevenLabs stress test - listen to the magic!")
    else:
        print("\n❌ Demo creation failed. Check the error messages above.")


if __name__ == "__main__":
    main()