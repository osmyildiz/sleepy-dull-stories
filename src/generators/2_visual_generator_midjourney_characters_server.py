# Enhanced Character Prompt System for V7 Full Body Cinematic Quality
# Add this to your character generator file

class EnhancedCharacterPromptSystem:
    """Ultra-high quality character prompt system for Midjourney v7"""

    def __init__(self):
        self.setup_v7_templates()

    def setup_v7_templates(self):
        """Setup V7 optimized character templates"""

        # V7 Core Quality Template
        self.v7_core_template = {
            "photography_base": "professional studio photography, ultra-high resolution, 8k quality, photorealistic",
            "lighting_cinematic": "cinematic lighting, dramatic shadows, warm golden hour light, three-point lighting setup",
            "camera_specs": "shot with Canon R5, 85mm lens, shallow depth of field, perfect focus",
            "quality_modifiers": "hyperrealistic, ultra-detailed textures, intricate details, masterpiece quality",
            "composition": "full body portrait, perfect human anatomy, natural pose, confident stance",
            "post_processing": "color graded, film grain, subtle vignette, enhanced contrast"
        }

        # Historical Period Templates
        self.historical_templates = {
            "ancient_roman": {
                "setting": "ancient Roman villa courtyard, marble columns, Mediterranean architecture",
                "clothing": "authentic Roman toga, tunic, or stola with period-accurate fabrics and colors",
                "accessories": "Roman jewelry, sandals, belt, period-appropriate hairstyle",
                "background": "Mount Vesuvius visible in distance, Roman garden, classical statues"
            },
            "medieval": {
                "setting": "medieval castle courtyard, stone architecture, gothic elements",
                "clothing": "period-accurate medieval garments, woolen fabrics, natural dyes",
                "accessories": "medieval jewelry, leather boots, belt pouch, period hairstyle",
                "background": "castle walls, medieval landscape, period architecture"
            },
            "victorian": {
                "setting": "Victorian mansion, ornate interior, period furniture",
                "clothing": "Victorian dress, tailcoat, or period-appropriate attire",
                "accessories": "pocket watch, jewelry, Victorian shoes, period hairstyle",
                "background": "Victorian garden, gaslight, period decoration"
            }
        }

        # Character Role Templates
        self.role_templates = {
            "protagonist": {
                "presence": "commanding presence, confident expression, natural leadership aura",
                "pose": "heroic stance, direct eye contact, open body language",
                "expression": "determined, wise, approachable yet strong"
            },
            "supporting": {
                "presence": "warm, approachable demeanor, supportive energy",
                "pose": "relaxed but attentive stance, gentle expression",
                "expression": "kind, intelligent, trustworthy"
            },
            "wise_elder": {
                "presence": "dignified bearing, years of wisdom in eyes",
                "pose": "stately posture, hands clasped or holding something meaningful",
                "expression": "serene, knowledgeable, patient"
            },
            "artisan": {
                "presence": "skilled hands, focused concentration, craftsman's pride",
                "pose": "working pose or displaying their craft",
                "expression": "dedicated, skilled, passionate about their work"
            }
        }

        # V7 Technical Parameters
        self.v7_parameters = {
            "version": "--v 7.0",
            "aspect_ratio": "--ar 3:4",  # Perfect for full body portraits
            "style": "--style raw",  # For maximum photorealism
            "quality": "--q 2",  # Highest quality
            "chaos": "--chaos 0",  # Consistent results
            "stylize": "--s 100"  # Balanced stylization
        }

    def build_enhanced_character_prompt(self, character_data: dict, historical_period: str = "ancient_roman") -> str:
        """Build ultra-high quality V7 character prompt"""

        # Extract character info
        name = character_data.get("name", "Character")
        physical_description = character_data.get("physical_description", "")
        role = character_data.get("role", "protagonist")

        print(f"🎭 Building enhanced V7 prompt for: {name}")

        # Build prompt components
        components = []

        # 1. Full body specification
        components.append("Full body portrait of")

        # 2. Enhanced physical description
        enhanced_physical = self.enhance_physical_description(physical_description, role)
        components.append(enhanced_physical)

        # 3. Historical period setting
        if historical_period in self.historical_templates:
            period_template = self.historical_templates[historical_period]
            components.append(f"wearing {period_template['clothing']}")
            components.append(f"with {period_template['accessories']}")
            components.append(f"standing in {period_template['setting']}")
            components.append(period_template['background'])

        # 4. Role-based presence
        if role in self.role_templates:
            role_template = self.role_templates[role]
            components.append(role_template['presence'])
            components.append(role_template['pose'])
            components.append(role_template['expression'])

        # 5. V7 Core quality
        components.append(self.v7_core_template['photography_base'])
        components.append(self.v7_core_template['lighting_cinematic'])
        components.append(self.v7_core_template['camera_specs'])
        components.append(self.v7_core_template['quality_modifiers'])
        components.append(self.v7_core_template['composition'])
        components.append(self.v7_core_template['post_processing'])

        # 6. V7 Technical parameters
        tech_params = " ".join([
            self.v7_parameters['aspect_ratio'],
            self.v7_parameters['version'],
            self.v7_parameters['style'],
            self.v7_parameters['quality'],
            self.v7_parameters['chaos'],
            self.v7_parameters['stylize']
        ])
        components.append(tech_params)

        # Join all components
        final_prompt = ", ".join(components)

        # Quality check and optimization
        final_prompt = self.optimize_prompt_for_v7(final_prompt)

        print(f"📊 Enhanced prompt length: {len(final_prompt)} characters")
        print(f"🎯 V7 optimizations applied: Full body, cinematic lighting, ultra-high quality")

        return final_prompt

    def enhance_physical_description(self, original_description: str, role: str) -> str:
        """Enhance physical description with cinematic details"""

        enhanced_parts = []

        # Add the original description
        enhanced_parts.append(original_description)

        # Add cinematic enhancements based on role
        if "distinguished" in original_description.lower() or role == "wise_elder":
            enhanced_parts.append("with weathered wisdom lines and noble bearing")

        if "young" in original_description.lower() or role == "protagonist":
            enhanced_parts.append("with vibrant energy and determined expression")

        if "graceful" in original_description.lower():
            enhanced_parts.append("moving with elegant poise and natural grace")

        # Add universal cinematic qualities
        enhanced_parts.append("with perfect human anatomy and natural proportions")
        enhanced_parts.append("captured in dramatic cinematic lighting")

        return ", ".join(enhanced_parts)

    def optimize_prompt_for_v7(self, prompt: str) -> str:
        """Final optimization for V7 compatibility"""

        # Remove redundant words
        optimizations = {
            ", ,": ",",
            "  ": " ",
            " , ": ", ",
        }

        optimized = prompt
        for old, new in optimizations.items():
            optimized = optimized.replace(old, new)

        # Ensure V7 compatibility keywords
        v7_keywords = [
            "photorealistic", "cinematic", "ultra-detailed",
            "professional photography", "8k quality"
        ]

        missing_keywords = []
        for keyword in v7_keywords:
            if keyword not in optimized.lower():
                missing_keywords.append(keyword)

        if missing_keywords:
            optimized += ", " + ", ".join(missing_keywords)

        return optimized

    def get_character_prompt_preview(self, character_data: dict, historical_period: str = "ancient_roman") -> dict:
        """Get detailed prompt preview for debugging"""

        prompt = self.build_enhanced_character_prompt(character_data, historical_period)

        return {
            "character_name": character_data.get("name", "Unknown"),
            "historical_period": historical_period,
            "character_role": character_data.get("role", "protagonist"),
            "final_prompt": prompt,
            "prompt_length": len(prompt),
            "v7_features": [
                "Full body portrait (3:4 aspect)",
                "V7.0 with --style raw",
                "Professional studio photography",
                "Cinematic lighting setup",
                "Ultra-high resolution (8k)",
                "Hyperrealistic quality",
                "Perfect human anatomy",
                "Period-accurate historical details"
            ],
            "technical_parameters": {
                "version": "7.0",
                "aspect_ratio": "3:4",
                "style": "raw",
                "quality": "2 (highest)",
                "chaos": "0 (consistent)",
                "stylize": "100 (balanced)"
            }
        }

# Example usage for your character generator
def integrate_enhanced_prompts_example():
    """Example of how to integrate this into your character generator"""

    # Initialize the enhanced system
    prompt_enhancer = EnhancedCharacterPromptSystem()

    # Example character data (from Claude-generated JSON)
    character_data = {
        "name": "Marcus Aurelius Valerius",
        "role": "protagonist",
        "physical_description": "Distinguished Roman patrician in his 40s with salt-and-pepper beard, wearing cream-colored toga with purple stripe, intelligent brown eyes, and weathered hands that speak of both scholarship and leadership",
        "importance_score": 9
    }

    # Generate enhanced V7 prompt
    enhanced_prompt = prompt_enhancer.build_enhanced_character_prompt(
        character_data,
        historical_period="ancient_roman"
    )

    print("\n" + "="*80)
    print("ENHANCED V7 CHARACTER PROMPT")
    print("="*80)
    print(enhanced_prompt)
    print("="*80)

    # Get detailed preview
    preview = prompt_enhancer.get_character_prompt_preview(
        character_data,
        historical_period="ancient_roman"
    )

    print("\n📊 PROMPT ANALYSIS:")
    print(f"Character: {preview['character_name']}")
    print(f"Role: {preview['character_role']}")
    print(f"Length: {preview['prompt_length']} characters")
    print(f"Period: {preview['historical_period']}")

    print("\n🎯 V7 FEATURES:")
    for feature in preview['v7_features']:
        print(f"  ✅ {feature}")

    print("\n⚙️ TECHNICAL PARAMETERS:")
    for param, value in preview['technical_parameters'].items():
        print(f"  🔧 {param}: {value}")

    return enhanced_prompt

# Integration point for your existing character generator
def enhance_existing_character_generation(character_data_list: list, historical_period: str = "ancient_roman") -> list:
    """Enhance existing character data with V7 prompts"""

    prompt_enhancer = EnhancedCharacterPromptSystem()
    enhanced_characters = []

    for character_data in character_data_list:
        # Generate enhanced prompt
        enhanced_prompt = prompt_enhancer.build_enhanced_character_prompt(
            character_data,
            historical_period
        )

        # Add enhanced prompt to character data
        enhanced_character = character_data.copy()
        enhanced_character['enhanced_v7_prompt'] = enhanced_prompt
        enhanced_character['v7_optimized'] = True
        enhanced_character['cinematic_quality'] = True
        enhanced_character['full_body_portrait'] = True

        enhanced_characters.append(enhanced_character)

        print(f"✅ Enhanced prompt for: {character_data.get('name', 'Unknown')}")

    return enhanced_characters

if __name__ == "__main__":
    # Test the enhanced prompt system
    integrate_enhanced_prompts_example()