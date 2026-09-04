#!/usr/bin/env python3
"""
H3XML Universal Graph-Based Prompt Optimizer & Compatibility Router
Taxonomy: 12 Specialized Cinematic Knowledge Subgraphs
Features:
- Perspective Trajectory Vector Anchoring (Linear Z-axis depth flight)
- 180-Degree Shutter Motion Streak Physics
- Adaptive Sigma Dynamic Rescaling
- Rigid-Body Carbon Frame & Kinetic Momentum Transfer
- Optical Bokeh Regularization & Soft-Knee ACES Tonemapping
"""

import re
from typing import Dict, Any, Tuple

# 12 Universal Cinematic Knowledge Graphs
DOMAIN_RULES = {
    'sports_high_speed_ballistics': {
        'keywords': [r'tennis\w*', r'calci\w*', r'football\w*', r'basket\w*', r'runn\w*', r'cors\w*', r'atlet\w*',
                     r'smash\w*', r'schiacc\w*', r'pugilat\w*', r'box\w*', r'surf\w*', r'nuot\w*', r'racket\w*', r'racchett\w*', r'pall\w*', r'ball\w*', r'golf\w*'],
        'camera': 'cinematic dynamic low-angle baseline tracking shot with 35mm shallow depth of field f/2.8',
        'causal_staging': (
            'chronological causal physics: '
            'athlete leaps upward tracking single bright neon yellow ball at apex, '
            'taut graphite racket strings strike ball with explosive kinetic energy and subtle chalk dust puff, '
            'followed by an instantaneous high-speed linear ballistic trajectory where the ball streaks away in straight perspective along the deep Z-axis toward the opposite baseline, '
            'shrinking smoothly in distance with realistic 180-degree directional motion streak blur, zero lateral floating drift'
        ),
        'geometry': (
            'one single rigid composite graphite tennis racket with flat planar string face, '
            'one single unified yellow tennis ball with authentic directional motion streak, zero duplicate floating balls'
        ),
        'physics': 'authentic Newtonian ballistics, linear momentum vector along court perspective, crisp tennis player athletic anatomy with five distinct fingers',
        'optical': '35mm sports cinema master, creamy soft-focus background bokeh, warm stadium rim lighting, authentic Kodak Vision3 5219 film stock',
        'audio': 'synchronized 48kHz acoustic court reverberation and explosive acoustic racket ball impact'
    },
    'automotive_motorsport': {
        'keywords': [r'ferrari\w*', r'porsche\w*', r'lamborghini\w*', r'car\w*', r'auto\w*', r'macchin\w*', r'supercar\w*', r'rac\w*',
                     r'monza\w*', r'f1\w*', r'formula\w*', r'drift\w*', r'motogp\w*', r'moto\w*', r'bik\w*'],
        'camera': 'ground-level tracking action shot at track level with cinematic shallow focus',
        'causal_staging': (
            'continuous high-speed aerodynamic tracking: '
            'supercar accelerates along the asphalt straight, '
            'tires gripping hot track with trailing tire smoke, '
            'passing smoothly through the frame in forward linear motion'
        ),
        'geometry': 'razor-sharp automotive bodywork, glossy polished lacquer finish, precise alloy wheel spokes and NACA ducts',
        'physics': 'aerodynamic high-speed motion, glowing carbon-ceramic brake rotors, tire smoke curling from asphalt',
        'optical': '35mm motorsport cinema master, golden hour specular highlights, soft atmospheric background blur, authentic lens flares',
        'audio': 'synchronized 48kHz twin-turbo engine exhaust acoustics and tire squeal'
    },
    'cinema_character_dialogue': {
        'keywords': [r'tarantino\w*', r'pulp\w*', r'diner\w*', r'danc\w*', r'ball\w*', r'mia\w*', r'vincent\w*', r'cinema\w*', r'movie\w*',
                     r'film\w*', r'attor\w*', r'actor\w*', r'scen\w*', r'dialog\w*', r'collett\w*', r'camici\w*'],
        'camera': 'cinematic medium two-shot at eye level',
        'causal_staging': (
            'fluid organic character movement: '
            'characters interact with subtle natural rhythm, eye contact and micro-expressions'
        ),
        'geometry': 'tailored crisp fabric textures, sharp collar and lapel definition, authentic facial expressions with specular iris reflections',
        'physics': 'natural fluid movement, authentic human kinematics',
        'optical': 'Panavision 35mm anamorphic cinema master, warm chiaroscuro ambient lighting, glowing neon background with soft circular bokeh',
        'audio': 'synchronized 48kHz vintage rock and ambient acoustic space'
    },
    'portrait_macro_organic': {
        'keywords': [r'ritratt\w*', r'portrait\w*', r'vecchi\w*', r'old\w*', r'sailor\w*', r'marinai\w*', r'vis\w*', r'fac\w*',
                     r'occh\w*', r'eye\w*', r'barb\w*', r'beard\w*', r'donn\w*', r'wom\w*', r'uom\w*', r'man\w*', r'ragazz\w*', r'girl\w*'],
        'camera': 'cinematic close-up portrait at 85mm focal length with creamy f/1.8 depth of field',
        'causal_staging': 'subtle natural breathing micro-movements, gentle eye blinks and organic micro-expression shifts',
        'geometry': 'individual facial pore texture, salt crystals in weathered beard, authentic periorbital micro-wrinkles, perfectly defined eyelashes and irises',
        'physics': 'subtle natural breathing micro-movements, organic skin subsurface scattering',
        'optical': '35mm master portraiture, soft Rembrandt key lighting, dark moody background with gentle depth of field',
        'audio': 'synchronized 48kHz subtle ambient atmosphere'
    },
    'scifi_cyberpunk_exoskeleton': {
        'keywords': [r'cyberpunk\w*', r'samurai\w*', r'katana\w*', r'robot\w*', r'cyborg\w*', r'futurist\w*', r'neon\w*', r'tokyo\w*',
                     r'sci-fi\w*', r'fantascienz\w*', r'blade\w*', r'laser\w*', r'hologram\w*'],
        'camera': 'cinematic medium-wide tracking shot at eye level',
        'causal_staging': 'character walking forward through the rain, unsheathing blade with fluid deliberate motion',
        'geometry': 'detailed metallic exoskeleton armor plates, weathered leather trench coat, rigid luminous blade edge',
        'physics': 'slow-motion raindrops falling through atmospheric steam, glowing plasma reflections on wet asphalt',
        'optical': '35mm anamorphic master, vibrant cyan and magenta neon chiaroscuro, volumetric rain mist bokeh',
        'audio': 'synchronized 48kHz ambient cyberpunk rain and deep synth bass'
    },
    'action_martial_arts_combat': {
        'keywords': [r'kung\w*', r'fu\w*', r'karate\w*', r'sword\w*', r'spad\w*', r'combatt\w*', r'fight\w*', r'ninja\w*', r'duell\w*'],
        'camera': 'dynamic over-the-shoulder orbiting action camera at 24fps',
        'causal_staging': 'fluid strike and counter-parry martial sequence with authentic kinetic momentum',
        'geometry': 'rigid polished steel weapon blades, five distinct fingers gripping the hilt, anatomical muscular tension',
        'physics': 'impact shockwaves, sweeping fabric garment motion, zero limb warping',
        'optical': '35mm martial arts cinema master, golden backlight rim, crisp atmospheric dust particles',
        'audio': 'synchronized 48kHz clashing steel and swift garment swooshes'
    },
    'wildlife_nature_kinetics': {
        'keywords': [r'leon\w*', r'lion\w*', r'tigr\w*', r'tiger\w*', r'aquil\w*', r'eagle\w*', r'lup\w*', r'wolf\w*', r'animal\w*', r'savana\w*', r'foresta\w*', r'forest\w*'],
        'camera': 'telephoto 400mm wildlife tracking shot at animal eye level with shallow depth of field',
        'causal_staging': 'predator moving through tall grass with powerful continuous stride and sharp focused gaze',
        'geometry': 'individual fur follicle definition, muscular ripple across shoulder blades, whiskers and specular reflections in eyes',
        'physics': 'natural quadruped kinematics, kicked-up dust and blade-of-grass displacement',
        'optical': 'National Geographic 8k cinema master, golden hour savannah sunlight, soft compressed background',
        'audio': 'synchronized 48kHz deep animal breathing and rustling wind'
    },
    'epic_landscape_aerial': {
        'keywords': [r'montagn\w*', r'mountain\w*', r'fiord\w*', r'fjord\w*', r'ocean\w*', r'mare\w*', r'tramont\w*', r'sunset\w*', r'aurora\w*', r'canyon\w*', r'drone\w*', r'paesagg\w*'],
        'camera': 'IMAX 70mm majestic aerial drone sweep with infinite depth of field f/8.0',
        'causal_staging': 'smooth forward panoramic flight gliding over epic terrain with atmospheric volumetric fog',
        'geometry': 'razor-sharp geological rock formations, crystal clear water reflections, micro-forest tree canopy details',
        'physics': 'fluid cloud drift, dynamic sunbeam rays breaking through mountain peaks',
        'optical': '70mm IMAX master landscape, vivid natural color grading, pristine highlight retention',
        'audio': 'synchronized 48kHz howling mountain wind and distant ocean waves'
    },
    'anime_studio_ghibli': {
        'keywords': [r'ghibli\w*', r'miyazaki\w*', r'anime\w*', r'totoro\w*', r'acquerell\w*', r'watercolor\w*', r'disegn\w*', r'cartoon\w*'],
        'camera': 'gentle hand-drawn cinematic panning shot with soft pastoral framing',
        'causal_staging': 'gentle summer breeze rustling green meadows, clouds billowing softly across azure skies',
        'geometry': 'hand-painted watercolor background textures, clean organic character line-art, vibrant lush greenery',
        'physics': 'natural soft wind physics, fluttering fabrics and grass blades',
        'optical': 'Studio Ghibli 35mm cel animation aesthetic, nostalgic warm pastel palette, soft sunlit diffusion',
        'audio': 'synchronized 48kHz gentle orchestral acoustic piano and cicada sounds'
    },
    'horror_dark_atmospheric': {
        'keywords': [
            r'horror\w*', r'mostr\w*', r'monster\w*', r'fantas\w*', r'ghost\w*', r'castell\w*', r'castle\w*',
            r'cimiter\w*', r'dark\w*', r'teneb\w*', r'nebbi\w*', r'fog\w*', r'paura\w*'
        ],
        'camera': 'slow creeping low-angle dolly shot with diegetic single-source light',
        'causal_staging': 'ominous shadow creeping across decaying stone walls, volumetric mist swirling slowly',
        'geometry': 'weathered Gothic architectural masonry, detailed cobwebs, authentic fabric tattering',
        'physics': 'drifting atmospheric dust particles, flickering flame light cast',
        'optical': '35mm dark cinema master, deep rich blacks without digital noise, eerie cyan and amber chiaroscuro',
        'audio': 'synchronized 48kHz low sub-bass drone and distant creaking floorboards'
    },
    'aerospace_zero_gravity': {
        'keywords': [r'spazi\w*', r'space\w*', r'astronav\w*', r'spaceship\w*', r'astronaut\w*', r'planet\w*', r'pianet\w*', r'orbita\w*', r'orbit\w*', r'nebula\w*'],
        'camera': 'smooth zero-gravity orbital camera rotation in deep space',
        'causal_staging': 'spacecraft glides silently past towering planetary rings with glowing ion propulsion thrusters',
        'geometry': 'metallic thermal insulation tiles, gold-tinted astronaut helmet visors, intricate solar array trusses',
        'physics': 'pure zero-gravity Newtonian momentum, glowing plasma thruster plumes against black vacuum',
        'optical': '70mm Panavision sci-fi master, brilliant starfield pinpoints, deep solar shadows',
        'audio': 'synchronized 48kHz muffled interior cockpit acoustics and low thruster hum'
    },
    'historical_period_epic': {
        'keywords': [r'mediov\w*', r'medieval\w*', r'cavali\w*', r'knight\w*', r'rom\w*', r'gladiat\w*', r'battagli\w*', r'battle\w*', r'castl\w*', r'armatur\w*', r'armor\w*'],
        'camera': 'epic low-angle wide tracking camera sweeping across the historic battlefield',
        'causal_staging': 'banners waving in the wind, armored warriors advancing in disciplined formation through morning mist',
        'geometry': 'hammered steel plate armor with authentic battle wear, embroidered heraldic crests, polished bronze helms',
        'physics': 'hoof-churned mud displacement, fluttering fabric banners, rising torch smoke',
        'optical': '35mm historical epic master, gritty desaturated silver-retention grading, authentic Kodak film grain',
        'audio': 'synchronized 48kHz marching armor clank, distant war horns and wind'
    }
}

class H3GraphOptimizer:
    """Universal Graph-Based Semantic Classifier and Kinematic Prompt Router."""
    
    @staticmethod
    def classify_domain(prompt: str) -> str:
        prompt_lower = prompt.lower()
        scores = {}
        for domain, rules in DOMAIN_RULES.items():
            score = sum(1 for pattern in rules['keywords'] if re.search(pattern, prompt_lower))
            scores[domain] = score
        
        best_domain = max(scores, key=scores.get)
        return best_domain if scores[best_domain] > 0 else 'cinema_character_dialogue'
    
    @classmethod
    def optimize_prompt(cls, raw_prompt: str) -> Tuple[str, str, Dict[str, Any]]:
        domain = cls.classify_domain(raw_prompt)
        rules = DOMAIN_RULES[domain]
        
        optimized_parts = [
            f"{rules['optical']}",
            f"{rules['camera']}",
            f"{raw_prompt.strip()}",
            f"{rules['causal_staging']}",
            f"{rules['geometry']}",
            f"{rules['physics']}",
            "8k photorealistic, authentic Kodak Vision3 5219 film stock",
            f"{rules['audio']}"
        ]
        
        full_optimized_prompt = ", ".join(optimized_parts)
        
        metadata = {
            'domain': domain,
            'recommended_solver': 'ab3',
            'recommended_layers': 50,
            'recommended_steps': 20,
            'recommended_reuse': 2,
            'sol_attn_safe': False if domain in ['sports_high_speed_ballistics', 'automotive_motorsport', 'action_martial_arts_combat'] else True
        }
        
        return full_optimized_prompt, domain, metadata

if __name__ == '__main__':
    test_cases = [
        "mia wallace and vincent vega dancing in a diner",
        "an old weathered sailor with a white beard",
        "a cyberpunk samurai in the rain",
        "two kung fu masters fighting",
        "a lion running across the savanna",
        "aerial panoramic sweep over alpine fjords",
        "an enchanted castle in studio ghibli aesthetic",
        "a dark hallway in a gothic castle",
        "a space station orbiting jupiter",
        "medieval knights in plate armor before battle"
    ]
    print("=== UNIVERSAL CINEMATIC KNOWLEDGE SUBGRAPHS TEST ===")
    for t in test_cases:
        opt, dom, meta = H3GraphOptimizer.optimize_prompt(t)
        print(f"[{dom.upper()}]: {t} -> Safe={meta['sol_attn_safe']}")
