"""
ElevenLabs Text-to-Speech Generator
Simple script for Sleepy Dull Stories voice generation
"""

import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import json
from datetime import datetime

# Load environment variables
load_dotenv()

class ElevenLabsTTS:
    def __init__(self):
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("❌ ELEVENLABS_API_KEY not found in .env file!")

        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }

        # Sleepy Dull Stories custom voice
        self.sleepy_voice_id = "8rISLz2BhGPiyBbP6ulQ"  # ssselifsss

        print("✅ ElevenLabs TTS initialized")
        print(f"🔑 API Key: {self.api_key[:8]}...")
        print(f"🎙️ Sleepy Voice ID: {self.sleepy_voice_id}")

    def get_available_voices(self):
        """Get list of available voices"""
        try:
            response = requests.get(f"{self.base_url}/voices", headers=self.headers)

            if response.status_code == 200:
                voices = response.json()
                print("🎙️ Available voices:")
                for voice in voices['voices']:
                    name = voice['name']
                    voice_id = voice['voice_id']
                    category = voice.get('category', 'Unknown')
                    print(f"   {name} ({category}) - ID: {voice_id}")
                return voices['voices']
            else:
                print(f"❌ Failed to get voices: {response.status_code}")
                print(response.text)
                return []

        except Exception as e:
            print(f"❌ Error getting voices: {e}")
            return []

    def generate_speech(self, text: str, voice_id: str, output_filename: str = None):
        """Generate speech from text"""

        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"sleepy_story_{timestamp}.mp3"

        # Ensure output directory exists
        output_path = Path("output")
        output_path.mkdir(exist_ok=True)
        full_output_path = output_path / output_filename

        # API payload - Start with v2 multilingual
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",  # More stable for basic tags
            "voice_settings": {
                "stability": 0.5,        # Medium for some expressiveness
                "similarity_boost": 0.8,  # High for voice consistency
                "style": 0.6,            # Higher for emotional delivery
                "use_speaker_boost": True
            }
        }

        print(f"🎬 Generating speech...")
        print(f"   📝 Text length: {len(text)} characters")
        print(f"   🎙️ Voice ID: {voice_id}")
        print(f"   📁 Output: {full_output_path}")

        try:
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            response = requests.post(url, json=payload, headers=self.headers)

            if response.status_code == 200:
                # Save audio file
                with open(full_output_path, 'wb') as f:
                    f.write(response.content)

                file_size = os.path.getsize(full_output_path)
                print(f"✅ Audio generated successfully with v2!")
                print(f"   📁 Saved: {full_output_path}")
                print(f"   📊 File size: {file_size:,} bytes")
                print(f"   🎭 Basic audio tags tested - check if they work!")

                # Save metadata
                metadata = {
                    "text": text,
                    "voice_id": voice_id,
                    "model_used": "eleven_multilingual_v2",
                    "basic_tags_tested": True,
                    "tags_in_text": ["[pause]", "[curiously]", "[whispers]", "[sighs]"],
                    "output_file": str(full_output_path),
                    "generated_at": datetime.now().isoformat(),
                    "text_length": len(text),
                    "file_size": file_size,
                    "voice_settings": payload["voice_settings"]
                }

                json_path = full_output_path.with_suffix('.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                return str(full_output_path)

            else:
                print(f"❌ Generation failed: {response.status_code}")
                print(response.text)
                return None

        except Exception as e:
            print(f"❌ Error generating speech: {e}")
            return None

    def quick_generate(self, text: str, output_filename: str = None):
        """Quick generation with ssselifsss voice"""
        return self.generate_speech(text, self.sleepy_voice_id, output_filename)

    def interactive_generation(self):
        """Interactive mode for text input"""
        print("\n🎙️ ElevenLabs Interactive Text-to-Speech")
        print("🌙 Using ssselifsss voice for Sleepy Dull Stories")
        print("=" * 50)

        # Option to use custom voice or select different one
        use_custom = input("\nUse ssselifsss voice? (Y/n): ").strip().lower()

        if use_custom in ['', 'y', 'yes']:
            voice_id = self.sleepy_voice_id
            voice_name = "ssselifsss (Your Custom Voice)"
            print(f"✅ Using: {voice_name}")
        else:
            # Get available voices for selection
            voices = self.get_available_voices()
            if not voices:
                print("❌ No voices available!")
                return

            print(f"\n🎯 Select a voice (0-{len(voices)-1}):")
            for i, voice in enumerate(voices):
                print(f"   {i}: {voice['name']} ({voice.get('category', 'Unknown')})")

            try:
                voice_index = int(input("\nVoice selection: "))
                selected_voice = voices[voice_index]
                voice_id = selected_voice['voice_id']
                voice_name = selected_voice['name']
                print(f"✅ Selected: {voice_name}")
            except (ValueError, IndexError):
                print("❌ Invalid selection!")
                return

        # Get text input
        print(f"\n📝 Enter your text for Sleepy Dull Stories:")
        print("(Press Enter twice to finish)")

        lines = []
        while True:
            try:
                line = input()
                if line == "" and lines:  # Empty line and we have content
                    break
                lines.append(line)
            except KeyboardInterrupt:
                print("\n❌ Cancelled by user")
                return

        text = "\n".join(lines)

        if not text.strip():
            print("❌ No text provided!")
            return

        # Ask for custom filename
        default_name = f"sleepy_story_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        custom_name = input(f"\n📁 Output filename (default: {default_name}): ").strip()

        if not custom_name:
            custom_name = default_name

        if not custom_name.endswith('.mp3'):
            custom_name += '.mp3'

        # Generate speech
        print(f"\n🎬 Generating with voice: {voice_name}")
        result = self.generate_speech(text, voice_id, custom_name)

        if result:
            print(f"\n🎉 Success! Audio saved to: {result}")
        else:
            print(f"\n❌ Generation failed!")

def main():
    """Main function"""
    print("🌙 Sleepy Dull Stories - ElevenLabs TTS Generator")
    print("🎙️ Using ssselifsss custom voice")
    print("=" * 50)

    # Sample text with basic audio tags for v2 compatibility
    sample_text = """Without planning or arrangement, they found themselves drawn to the forum as if pulled by invisible threads—Marcus from his bakery street, Livia from the well, Quintus from the empty road, and Claudia from her restless wandering. [pause] [curiously] The moonlit marble columns created a natural amphitheater for their unintended gathering, each carrying private discoveries that suddenly demanded sharing. [whispers] In this moment, under ancient stars, four strangers became something more— [sighs] a constellation of human connection in the vast Roman night."""

    try:
        tts = ElevenLabsTTS()

        print(f"\n📝 Sample text to generate:")
        print(f"'{sample_text[:100]}...'")
        print(f"\n📊 Text length: {len(sample_text)} characters")

        # Generate with default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sleepy_sample_{timestamp}.mp3"

        print(f"\n🎬 Generating with ssselifsss voice...")
        print(f"📁 Output filename: {filename}")

        result = tts.quick_generate(sample_text, filename)

        if result:
            print(f"\n🎉 Success! Audio saved to: {result}")
            print(f"🎧 Ready to listen to your Sleepy Dull Stories sample!")
        else:
            print(f"\n❌ Generation failed!")

    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    main()