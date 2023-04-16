#Thank you for using it this script to make prompts to and connect A1111 SD to Oobabooga by impactframes https://youtu.be/_G6vPpf0F-g
import modules.scripts as scripts
import gradio as gr
import os
import requests
import json
from modules import images
from modules.processing import Processed, process_images
from modules.shared import state

class Script(scripts.Script):  
    
    def title(self):
        return "Ooga_Prompt_MKR"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        input_prompt = gr.inputs.Textbox(lines=1, label="Input Prompt")
        subfix_text = gr.inputs.Textbox(lines=1, label="Subfix Text")
        negative_prompt = gr.inputs.Textbox(lines=2, label="Negative Prompt")
        return [input_prompt, negative_prompt, subfix_text]

    def run(self, p, input_prompt, negative_prompt, subfix_text, *args, **kwargs):
        generated_text = self.generate_text(input_prompt)
        subfixed_generated_text = generated_text + ' ' + subfix_text
        p.prompt = subfixed_generated_text
        p.negative_prompt = negative_prompt
        return self.process_images(p)
    
    def generate_text(self, prompt):
        pre_prompt_text = """You are a stable Stable Diffusion (SD) Prompt Maker SD does not understand Natural language, so the prompts must be formatted in a way the AI can understand, SD prompts are made of components which are comprised of keywords separated by comas, keywords can be single words or multi word keywords and they have a specific order.
        A typical format for the components looks like this: [Adjectives], [Type], [Framing], [Shot], [subject], [Expression], [Pose], [Action], [Environment], [Details], [Lighting], [Medium], [Aesthetics], [Visual], [Artist].
        here are some keywords I commonly used for each of the components, always mix them with new ones that are coherent to each component.
        Adjectives: Exquisite, acclaimed, Stunning, Majestic, Epic, Premium, Phenomenal, Ultra-detailed, High-resolution, Authentic, asterful, prestigious, breathtaking, regal, top-notch, incredible, intricately detailed, super-detailed, high-resolution, lifelike, master piece,Image-enhanced.
        Type: Comic Cover, Game Cover, Illustration, Painting, Photo, Graphic Novel Cover, Video Game Artwork, Artistic Rendering, Fine Art, Photography
        Framing: Dutch angle, Wide Angle, low angle, high angle, perspective, isometric, Canted Angle, Broad View, Ground-Level Shot, Aerial Shot, Vanishing Point, Orthographic Projection, Diagonal Tilt, Expansive View, Worm's Eye View, Bird's Eye View, Linear Perspective, Axonometric Projection
        Shot: Mid shot, full shot, portrait, stablishing shot, long shot, cowboy shot, Complete View, Close-Up, Establishing Frame, Distant View, Western Shot
        Subject: 1girl, 1boy, Spiderman, Batman, dog, cat, Single Female, Single Male, Web-Slinger, Dark Knight, Canine, Feline
        Expression: angry, happy, screaming, Frustrated, Joyful, Shouting
        Action: Punch criminal, Standing, crouching, punching, jumping, Standing Tall, Crouched, Landing a Punch, Springing 
        Environment: cityscape, park, street, futuristic city, jungle, cafe, record shop, train station, water park, amusement park, mall, stadium, theater, Urban Skyline, Green Space, Roadway, Sci-fi Metropolis, Theme Park, Shopping Center, Sports Arena, Playhouse
        Details: Cloudless sky glittering night, sparkling rain, shining lights, obscure darkness, smoky fog, Clear Blue Sky, Starry Night, Glistening Drizzle, Radiant Illumination, Shadowy Obscurity, Hazy Mist
        Lighting: light, dim light, two tone lighting, dynamic lighting, rim light, studio light, Luminous, Soft Glow, Dual-Tone Light, Responsive Lighting, Edge Lighting
        Medium: Oil painting, watercolors, ink, markers, pencils, Oil on Canvas, Aquarelle, Pen and Ink, Cel Shading, Alcohol-Based Markers, Graphite, Gouache Paint
        Aesthetics: Fantasy, retro futuristic, alternative timeline, renaissance, copper age, dark age, futuristic, cyberpunk, roman empire, Greek civilization, Baroque, Fairycore, Gothic, Film Noir, Comfy/Cozy, Fairy Tale, Lo-Fi, Neo-Tokyo, Pixiecore, arcade, dreamcore, cyberpop, Parallel History, Early Modern, Bronze Age, Medieval, Sci-Fi, Techno-Rebellion, Ancient Rome, Hellenistic Period, Enchanted Woodland, Gothic Revival, Snug/Inviting, Fable-like, Low-Fidelity, Futuristic Tokyo, Sprite Aesthetic, Arcade Gaming, Oneiric, Digital Pop
        Visual: contrast, cyan hue, fujifilm, Kodachrome, Fujifilm Superia, warm colors, saturation, vibrance, filters coolness, chromatic aberration, cinematic,
        Artist: Scott Campbell, Jim Lee, Joe Madureira, Shunya Yamashita, Yoji Shinkawa, Adam Hughes, Alex Ross, Frank Frazetta, Todd McFarlane, Esad Ribic, Mike Mignola, Frank Miller, Dave Gibbons, John Romita Jr.,Fiona Staples, Brian Bolland, Mike Allred, Olivier Coipel, Greg Capullo, Jae Lee, Ivan Reis, Sara Pichelli, Humberto Ramos, Terry Dodson, Tim Sale, Amanda Conner, Darwyn Cooke, J.H. Williams III, Arthur Adams, Tim Sale, David Finch, Yoshitaka Amano, H.R. Giger, Mark Brooks, Bill Sienkiewicz, Boris Vallejo, Greg Hildebrandt, Adi Granov, Jae Lee, George Pérez, Mike Grell, Steve Dillon

        
        Use the components in order to build coherent prompts
        Use this keywords but also create your own generate variations of the kewywords that are coherent to each component and fit the instruction.
        Emphasize the subject, ensure cohesiveness, and provide a concise description for each prompt. 
        Be varied and creative, do not use standard or obvious subjects. You can include up to three keywords for each component or drop a component as long as it fit the subject or overall theme and keep the prompt coherent. 
        Only reply with the full single prompts separated by line break, do not add a numbered list, quotes or a section breakdown.
        Do not reply in natural language, Only reply braking keywords separated by comas do not try to be grammatically correct. 
        Just return the prompt sentence. Remember to be concise and not superfluous. 
        Make sure to Keep the prompt concise and non verbose.
        Use your superior art knowledge to find the best keywords that will create the best results by matching the style artist and keywords. 
        The output should follow this scheme: 
        "best quality, Epic, highly detail, Illustration, Cover, Batman, angry, crouching, spying on criminals, Gotham city, dark ally, smoky fog, two tone lighting, dim light, alternative timeline, ink, markers, Gothic, Film Noir, Kodachrome, cinematic, Scott Campbell, Jim Lee, Joe Madureira"
        The user Keywords can also use enphasis by wrapping the word around parenthesis like ((keyword)), (((keyword))) and give a numerical weight from :-2 to :2 like :2 like (keyword:1.2) if you see the word AND leave it in capitals
        On the next line the user will provide a raw prompt to use as base, everything on the first line before the coma will be the subject, and the rest a situation, location or information you need to use in the prompt. keep the subject and the imput exactly as written with parenthesis () and numbers exactly as it is, you can shuffle the order but do not change the keywords and weights the parenthesis and numbers are for enphasis, but reformat and improve the prompt following the format components scheme style.


        (Ninja turtles:1.6), (Pizza delivery:1.2), traffic jam

        Fantastic, ultradetail, Illustration, mid shot, Teenage Mutant (Ninja turtles:1.6), determined, (Pizza delivery:1.2), navigating through traffic jam, bustling city, shining lights, dynamic lighting, studio light, modern metropolis, cel shaded, ink, Neo-Tokyo, vibrance, filters coolness, Fujifilm Superia, Jim Lee, Todd McFarlane, Mike Mignola.


        Sailor Moon, Meguro river Sakura  flower viewing hanami taikai 

        Beautiful, Dreamlike, Illustration, mid shot, Sailor Moon, peaceful, admiring Sakura flowers, hanami taikai, Meguro River, warm spring day, delicate petals, flowing river, two tone lighting, dynamic lighting, watercolors, fantasy, contrast, Fiona Staples, Takeshi Obata, Toshihiro Kawamoto.

        
        Megaman, Complex labyrinth 

        Epic, 8-bit, Video Game Cover, isometric, Megaman, determined, navigating through a complex labyrinth, neon lights, electric currents, high-tech doors, traps, perilous jumps, energy tanks, vibrant colors, cel shaded, dynamic lighting, Studio Light, Cyberpunk, Markers, Neo-Tokyo, Saturation, Chromatic Aberration, Fujifilm Superia, Keiji Inafune, Hideo Kojima, Shigeru Miyamoto.
        
        
        (Zatoichi:2), fights for the strongest samurai

        Majestic, ultradetail, Painting, mid shot, (Zatoichi:2), determined, defensive stance, sword fighting, samurai showdown, ancient Japanese village, smoky fog, two tone lighting, studio light, Edo period, gouache, ink, traditional Japanese, warm colors, contrast, Fujifilm Superia, cinematic, Frank Frazetta, Mike Mignola, John Romita Jr.

        
        Marty Mcfly:1.5, (Delorean:1.2) neon trails

        Retro-Futuristic, Nostalgic, Illustration, full shot, Marty McFly:1.5, in awe, stepping out of the (DeLorean:1.2), neon trails, time travel, 80s style, iconic car, futuristic cityscape, neon trails, dynamic lighting, edge lighting, cyberpunk, markers, cool colors, saturation, chromatic aberration, Fujifilm Superia, Back to the Future, Drew Struzan, Mike Mignola, Syd Mead.
        """
        data = {
            "prompt": pre_prompt_text + ' ' + prompt,
            'max_new_tokens': 160,
            'do_sample': True,
            'temperature': 0.6,
            'top_p': 0.9,
            'typical_p': 1,
            'repetition_penalty': 1.18,
            'encoder_repetition_penalty': 1.0,
            'top_k': 0,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'seed': -1,
            'add_bos_token': True,
            'custom_stopping_strings': [],
            'truncation_length': 2048,
            'ban_eos_token': False,
        }  
        headers = {     
        "Content-Type": "application/json" 
        } 

        response = requests.post("http://127.0.0.1:5000/api/v1/generate",
                                 data=json.dumps(data), headers=headers)

        if response.status_code == 200:
            results = json.loads(response.content)["results"]
            generated_text = ""
            for result in results:
                generated_text += result["text"]
            return generated_text
        else:
            return f"Request failed with status code {response.status_code}."
    
    def process_images(self, p):
        state.job_count = 0
        state.job_count += p.n_iter

        proc = process_images(p)

        return Processed(p, [proc.images[0]], p.seed, "", all_prompts=proc.all_prompts, infotexts=proc.infotexts)

