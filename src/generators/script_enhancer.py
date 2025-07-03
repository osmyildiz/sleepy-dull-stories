"""
Sleepy Dull Stories - Script Enhancer
ChatGPT API kullanarak Claude'dan gelen hikayeleri iyileştirir ve optimize eder
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import openai
from openai import OpenAI

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src import config


class ScriptEnhancer:
    """ChatGPT API ile script iyileştirme"""

    def __init__(self):
        """Initialize OpenAI client"""
        try:
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            print("✅ OpenAI API client başlatıldı")
        except Exception as e:
            print(f"❌ OpenAI API client hatası: {e}")
            raise

    def _create_enhancement_prompt(self, raw_story: str, topic: str) -> str:
        """ChatGPT için iyileştirme promptu oluştur"""

        prompt = f"""
You are an expert script editor specializing in sleep stories and ASMR content. Your task is to enhance and optimize a 2-hour historical sleep story for maximum relaxation and sleep induction.

ORIGINAL TOPIC: {topic}

ENHANCEMENT OBJECTIVES:
1. Optimize for sleep induction and relaxation
2. Improve flow and pacing for 2-hour duration
3. Add more sensory details and atmospheric descriptions
4. Ensure consistent, calming tone throughout
5. Remove any potentially exciting or stimulating content
6. Add natural breathing points and pauses
7. Enhance immersive "you are there" perspective

SPECIFIC IMPROVEMENTS NEEDED:
- **Pacing**: Slow down the narrative, add more descriptive passages
- **Sensory Details**: Enhance descriptions of sounds, textures, smells, temperatures
- **Repetitive Patterns**: Add soothing, repetitive language structures
- **Breathing Space**: Include natural pauses and reflection moments
- **Sleep Optimization**: Use language patterns that encourage relaxation
- **Consistency**: Maintain present tense and second-person perspective
- **Atmosphere**: Deepen the immersive historical atmosphere

TECHNICAL REQUIREMENTS:
- Maintain approximately 18,000-20,000 words for 2-hour duration
- Use present tense throughout: "You walk..." not "You walked..."
- Keep second-person perspective: "You find yourself..."
- Add strategic pauses marked with [PAUSE] for TTS processing
- Include gentle transitions between scenes and topics
- Remove any jarring or sudden changes in scene or mood

STYLE GUIDELINES:
- Use calming, flowing sentence structures
- Include repetitive, soothing phrases
- Add detailed descriptions of peaceful activities
- Focus on mundane, routine activities that are naturally boring
- Include gentle sounds: footsteps, rustling, distant conversations
- Describe comfortable temperatures, soft lighting, pleasant textures
- Use words associated with peace, calm, and relaxation

EXAMPLE IMPROVEMENTS:
Before: "The Roman senator entered the forum."
After: "You watch as the Roman senator slowly approaches the forum, his toga rustling softly in the gentle morning breeze. [PAUSE] His sandaled feet make quiet, rhythmic sounds against the worn marble stones..."

Please enhance the following story while maintaining its historical accuracy and educational value:

---ORIGINAL STORY---
{raw_story}

---ENHANCED STORY OUTPUT---
[Provide the complete enhanced story here, maintaining all historical content while optimizing for sleep and relaxation]
"""

        return prompt

    async def enhance_script(self, raw_story_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ana script iyileştirme fonksiyonu

        Args:
            raw_story_data: Claude'dan gelen ham hikaye verisi

        Returns:
            Dict containing enhanced script and metadata
        """
        print("🔄 Script iyileştirme başlatılıyor...")

        try:
            # Raw story'yi al
            raw_story = raw_story_data.get('story', '')
            topic = raw_story_data.get('topic', 'Unknown Topic')

            if not raw_story:
                raise ValueError("Raw story data bulunamadı")

            # Enhancement prompt oluştur
            prompt = self._create_enhancement_prompt(raw_story, topic)

            # ChatGPT API çağrısı
            print("🤖 ChatGPT API çağrısı yapılıyor...")
            enhanced_story = await self._call_openai_api(prompt)

            # Enhanced story'yi temizle
            cleaned_story = self._clean_enhanced_story(enhanced_story)

            # Metadata oluştur
            enhanced_data = {
                'enhanced_script': cleaned_story,
                'original_story': raw_story,
                'topic': topic,
                'enhanced_at': datetime.now().isoformat(),
                'original_word_count': len(raw_story.split()),
                'enhanced_word_count': len(cleaned_story.split()),
                'enhancement_ratio': len(cleaned_story.split()) / len(raw_story.split()) if raw_story else 0,
                'estimated_duration_minutes': len(cleaned_story.split()) / 150,  # ~150 kelime/dakika
                'api_model': getattr(config, 'OPENAI_MODEL', 'gpt-4'),
                # Original data'yı koru
                'image_prompts': raw_story_data.get('image_prompts', []),
                'teaser_video_prompt': raw_story_data.get('teaser_video_prompt', ''),
            }

            print(f"✅ Script iyileştirme tamamlandı:")
            print(f"📝 Orijinal: {enhanced_data['original_word_count']} kelime")
            print(f"✨ İyileştirilmiş: {enhanced_data['enhanced_word_count']} kelime")
            print(f"📊 İyileştirme oranı: {enhanced_data['enhancement_ratio']:.2f}x")

            return enhanced_data

        except Exception as e:
            print(f"❌ Script iyileştirme hatası: {e}")
            raise

    async def _call_openai_api(self, prompt: str) -> str:
        """OpenAI API çağrısı yap"""

        try:
            response = self.client.chat.completions.create(
                model=getattr(config, 'OPENAI_MODEL', 'gpt-4'),
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert script editor specializing in sleep stories and relaxation content. You optimize narratives for maximum calm and sleep induction while maintaining historical accuracy."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=getattr(config, 'OPENAI_MAX_TOKENS', 4000),
                temperature=getattr(config, 'OPENAI_TEMPERATURE', 0.8),
                presence_penalty=0.1,  # Tekrardan kaçınma
                frequency_penalty=0.1  # Kelime çeşitliliği
            )

            return response.choices[0].message.content

        except openai.APIError as e:
            print(f"❌ OpenAI API hatası: {e}")
            raise
        except Exception as e:
            print(f"❌ Beklenmeyen API hatası: {e}")
            raise

    def _clean_enhanced_story(self, enhanced_story: str) -> str:
        """Enhanced story'yi temizle ve optimize et"""

        print("🧹 Enhanced story temizleniyor...")

        # Output marker'ları kaldır
        cleaned = re.sub(r'---ENHANCED STORY OUTPUT---.*?(?=\n)', '', enhanced_story, flags=re.IGNORECASE)
        cleaned = re.sub(r'---.*?---', '', cleaned, flags=re.DOTALL)

        # Fazla boşlukları temizle
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = re.sub(r' +', ' ', cleaned)

        # [PAUSE] marker'larını standardize et
        cleaned = re.sub(r'\[PAUSE\]', '[PAUSE]', cleaned, flags=re.IGNORECASE)

        # TTS için optimizasyon
        cleaned = self._optimize_for_tts(cleaned)

        return cleaned.strip()

    def _optimize_for_tts(self, text: str) -> str:
        """TTS için text'i optimize et"""

        # Noktalama sonrası duraklama marker'ları ekle
        text = re.sub(r'\.(?!\s*\[PAUSE\])', '. [PAUSE]', text)
        text = re.sub(r'\?(?!\s*\[PAUSE\])', '? [PAUSE]', text)
        text = re.sub(r'!(?!\s*\[PAUSE\])', '! [PAUSE]', text)

        # Çift pause'ları temizle
        text = re.sub(r'\[PAUSE\]\s*\[PAUSE\]', '[PAUSE]', text)

        # Uzun paragrafları böl
        text = self._add_paragraph_breaks(text)

        return text

    def _add_paragraph_breaks(self, text: str) -> str:
        """Uzun paragrafları böl"""

        paragraphs = text.split('\n\n')
        processed_paragraphs = []

        for para in paragraphs:
            if len(para.split()) > 200:  # 200 kelimeden uzun paragraflar
                # Cümlelere böl
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_para = ""

                for sentence in sentences:
                    if len((current_para + sentence).split()) > 150:
                        if current_para:
                            processed_paragraphs.append(current_para.strip())
                            current_para = sentence
                        else:
                            current_para = sentence
                    else:
                        current_para += " " + sentence if current_para else sentence

                if current_para:
                    processed_paragraphs.append(current_para.strip())
            else:
                processed_paragraphs.append(para)

        return '\n\n'.join(processed_paragraphs)

    def save_enhanced_script(self, enhanced_data: Dict[str, Any], output_dir: Path = None) -> Path:
        """Enhanced script'i kaydet"""

        if not output_dir:
            processed_dir = getattr(config, 'PROCESSED_DATA_DIR', Path('data/processed'))
            output_dir = Path(processed_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Dosya adı oluştur
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        topic_clean = re.sub(r'[^\w\s-]', '', enhanced_data['topic']).strip()
        topic_clean = re.sub(r'[-\s]+', '_', topic_clean)
        filename = f"enhanced_{topic_clean}_{timestamp}.json"

        file_path = output_dir / filename

        try:
            # JSON olarak kaydet
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Enhanced script kaydedildi: {file_path}")
            return file_path

        except Exception as e:
            print(f"❌ Enhanced script kaydetme hatası: {e}")
            raise

    def analyze_enhancement_quality(self, enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhancement kalitesini analiz et"""

        enhanced_text = enhanced_data['enhanced_script']
        original_text = enhanced_data['original_story']

        analysis = {
            'sleep_keywords_count': self._count_sleep_keywords(enhanced_text),
            'pause_markers_count': enhanced_text.count('[PAUSE]'),
            'average_sentence_length': self._calculate_avg_sentence_length(enhanced_text),
            'sensory_words_count': self._count_sensory_words(enhanced_text),
            'repetitive_patterns': self._analyze_repetitive_patterns(enhanced_text),
            'readability_score': self._estimate_readability(enhanced_text),
            'enhancement_score': 0  # Hesaplanacak
        }

        # Enhancement score hesapla (1-10 arası)
        score = 5.0  # Base score

        if analysis['sleep_keywords_count'] > 50:
            score += 1
        if analysis['pause_markers_count'] > 100:
            score += 1
        if 15 <= analysis['average_sentence_length'] <= 25:
            score += 1
        if analysis['sensory_words_count'] > 100:
            score += 1

        analysis['enhancement_score'] = min(10.0, score)

        print(f"📊 Enhancement kalite analizi:")
        print(f"💤 Uyku kelimeleri: {analysis['sleep_keywords_count']}")
        print(f"⏸️ Duraklama marker'ları: {analysis['pause_markers_count']}")
        print(f"📏 Ortalama cümle uzunluğu: {analysis['average_sentence_length']:.1f}")
        print(f"🎭 Duyusal kelimeler: {analysis['sensory_words_count']}")
        print(f"⭐ Enhancement skoru: {analysis['enhancement_score']:.1f}/10")

        return analysis

    def _count_sleep_keywords(self, text: str) -> int:
        """Uyku ile ilgili kelimeleri say"""

        sleep_keywords = [
            'peaceful', 'calm', 'quiet', 'gentle', 'soft', 'slow', 'relaxing',
            'soothing', 'tranquil', 'serene', 'restful', 'comfortable', 'warm',
            'drift', 'float', 'settle', 'breathe', 'rest', 'pause'
        ]

        text_lower = text.lower()
        count = sum(text_lower.count(keyword) for keyword in sleep_keywords)
        return count

    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Ortalama cümle uzunluğunu hesapla"""

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0

        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)

    def _count_sensory_words(self, text: str) -> int:
        """Duyusal kelimeleri say"""

        sensory_words = [
            # Görsel
            'see', 'watch', 'observe', 'light', 'shadow', 'color', 'glow', 'shimmer',
            # İşitsel
            'hear', 'sound', 'whisper', 'rustle', 'echo', 'murmur', 'silence',
            # Dokunsal
            'feel', 'touch', 'texture', 'smooth', 'rough', 'warm', 'cool', 'soft',
            # Koku
            'smell', 'scent', 'aroma', 'fragrance', 'perfume',
            # Tat
            'taste', 'flavor', 'sweet', 'bitter'
        ]

        text_lower = text.lower()
        count = sum(text_lower.count(word) for word in sensory_words)
        return count

    def _analyze_repetitive_patterns(self, text: str) -> int:
        """Tekrarlayan kalıpları analiz et"""

        # "You" ile başlayan cümleleri say
        you_sentences = len(re.findall(r'\bYou\s+\w+', text))

        # Benzer yapıdaki cümleleri say
        as_patterns = len(re.findall(r'\bAs\s+you\s+', text, re.IGNORECASE))

        return you_sentences + as_patterns

    def _estimate_readability(self, text: str) -> float:
        """Basit okunabilirlik skoru (1-10, düşük daha iyi uyku için)"""

        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences or not words:
            return 5.0

        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = 1.5  # Approximation

        # Basitleştirilmiş Flesch Reading Ease benzeri
        score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)

        # 1-10 aralığına normalize et (10 en kolay)
        normalized = max(1, min(10, score / 10))

        return normalized

    async def test_api_connection(self) -> bool:
        """OpenAI API bağlantısını test et"""

        try:
            print("🧪 OpenAI API bağlantısı test ediliyor...")

            response = self.client.chat.completions.create(
                model=getattr(config, 'OPENAI_MODEL', 'gpt-4'),
                messages=[
                    {"role": "user", "content": "Please respond with exactly: 'API connection successful'"}
                ],
                max_tokens=10,
                temperature=0
            )

            if "API connection successful" in response.choices[0].message.content:
                print("✅ OpenAI API bağlantısı başarılı")
                return True
            else:
                print(f"⚠️ OpenAI API test sonucu beklenmedik: {response.choices[0].message.content}")
                return False

        except Exception as e:
            print(f"❌ OpenAI API test hatası: {e}")
            return False


# Test fonksiyonu
async def test_script_enhancer():
    """Script Enhancer test fonksiyonu"""

    try:
        print("🔄 Script Enhancer Test Başlatılıyor...")
        print("=" * 50)

        # Script enhancer oluştur
        enhancer = ScriptEnhancer()

        # API test et
        api_ok = await enhancer.test_api_connection()
        if not api_ok:
            print("❌ API bağlantısı başarısız")
            return False

        # Test story data
        test_story_data = {
            'topic': 'Ancient Roman Bath House',
            'story': """
            Welcome to Sleepy Dull Stories. Tonight we explore the Roman baths.

            You enter the caldarium, the hot room of the bath house. Steam rises from the heated pools.
            Romans spend hours here, socializing and relaxing. The marble floors are warm under your feet.
            Attendants move quietly, offering oils and strigils for cleaning.

            The atmosphere is peaceful and conducive to rest. Water drips steadily from the ceiling.
            Other bathers speak in hushed tones. The heat makes you drowsy and relaxed.
            """,
            'image_prompts': ['Roman bath interior', 'Steam rising from pools'],
            'teaser_video_prompt': 'Ancient Roman bath house atmosphere'
        }

        # Enhancement test et
        print("🔄 Script enhancement test ediliyor...")
        enhanced_data = await enhancer.enhance_script(test_story_data)

        # Kalite analizi
        quality_analysis = enhancer.analyze_enhancement_quality(enhanced_data)

        print("=" * 50)
        print(f"✅ Enhancement Test Başarılı!")
        print(f"📝 Orijinal kelime sayısı: {enhanced_data['original_word_count']}")
        print(f"✨ Enhanced kelime sayısı: {enhanced_data['enhanced_word_count']}")
        print(f"⭐ Kalite skoru: {quality_analysis['enhancement_score']:.1f}/10")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"❌ Enhancement test hatası: {e}")
        return False


if __name__ == "__main__":
    # Direkt test çalıştırma
    asyncio.run(test_script_enhancer())