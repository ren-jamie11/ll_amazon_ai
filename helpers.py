
import base64
import re
import pandas as pd
from collections import Counter

# OpenAI Image Processing Functions

image_describe_system_instructions = """You are a writer for an Amazon seller of photo frames.
                                     You have decades of experience writing concise product descriptions for online stores.
                                     """


image_describe_processing_instructions =  """
                            Using the provide image, identify the specific visual details of the frame
                            and list them in 5 bullet point phrases. Focus on the overall look (1st bullet only), accents, molding,
                            color, shape.

                            Example output:
                            - refined, gallery-inspired look
                            - sculpted floral accents
                            - soft antique-gold finish

                            Advice to follow: 
                            - Use concise, positive, retail language (like ones you would see on Amazon or high-end ecommerce store)
                            - Avoid pendantic/flowery words like 'heirloom', 'applique', 'opulent'. 'old-world'. Stick to words
                            that the average American consumer can understand.
                            - Do not use pronouns like 'your'

                            YOUR FOCUS IS FIRST AND FOREMOST ON ACCURATELY DESCRIBING THE VISUAL DETAILS OF THE PRODUCT. 
                            THEREFORE, IT MUST BE TRUE TO THE IMAGE. TAKE THE TIME TO INSPECT THE IMAGE CAREFULLY, and then
                            find the best keywords/phrases to highlight what you see.
                            """


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
def generate_product_description_from_image(
    client, 
    image_file: str,
    system_instructions: str,
    processing_instructions: str,
    model: str = "gpt-5.2",
    temperature = 0.2
):
    image_base64 = encode_image_to_base64(image_file)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": processing_instructions},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ],
            },
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content

# Text generating 

generate_title_prompt_prefix = """
You are a writing assistant for Amazon listings for picture frames. Using the SEO keywords I have gathered, help me 
write an Amazon product title. 

You MUST ensure each keyword is used in the correct context.. If the keyword does not belong in the context, I'd
rather you not include it. For instance: if keywords include 'glass', 'gold', 'metal', glass is more likely to 
refer to glass front/protective layer rather than frame's material. 'metal'

Guiding instructions: 
- Prioritize maximizing # of keywords and flow/readabilitycontextual sense equally. 
  E.g. 'frame for pictures and photo' is repetitive and doesn't read well. I'd rather you just use 'picture frame' and find a way to use 'photo' later
- Aim for around 150-175 characters with spaces. You MUST keep under 200 characters with spaces
- Titles must not use the following special characters: !, $, ?, _, {, }, ^, ¬, ¦.
- Titles must not contain the same word more than twice
- Title must NOT include brand names. Ignore all provided keywords that you think are brand names or proper nouns.

Examples of real product titles:

11x14 Ornate Finish Photo Frame, with White Mat for 8x10 Picture & Real Glass, Color: Bronze
Luxury Sleep Eye Mask for Side Sleeper Men Women, Zero Eye Pressure 3D Sleeping Mask, Light Blocking Patented Design Night Blindfold, Soft Eye Shade Cover for Travel, Black
Artificial Bonsai Tree Juniper Faux Plants Indoor Small Fake Plants Decor with Ceramic Pots for Home Table Office Desk Bathroom Shelf Bedroom Living Room Farmhouse Decorations

Input:
"""

generate_suggestion_prompt_prefix = """
I'm writing an Amazon listing for picture frames. Help me come up with 3 candidate
phrases that describe either the aesthetic or functional value of the detail provided.

Step 1: Identify whether the detail is aesthetic or functional
Step 2: Come up with candidate phrases

Speak in a concise, warm, retail-ready manner. 
Use moderate, quieter words instead of loud, exagerrated marketing buzzwords. Sound like a refined gentleman instead of an annoying attention-seeking salesperson.
Acceptable words include: elegant, clean, simple, effortless, sophisticated, excellent
Unacceptable words include: hassle, incredible, game-changing, unmatched, cutting-edge, 
Avoid pedantic or flowery words like 'heirloom', 'applique', 'opulent', 'old-world'.

Example input: Sculpted floral accents and soft antique gold finish
Step 1: classify phrase as 'aesthetic'
Example output: create a rich decorative presence, adds charm and sophistication

Aesthetic keywords you could use (but don't have to):
look, charm, sophistication, elegance, classic, radiance, character, unique, special, artistic, exquisite

Example input: Soft velvet backing with swivel twist tabs 
Step 1: classify phrase as 'functional'
Example output: makes switching photos very convenient, safeguards displayed item

New input:
"""

generate_suggestions_prompt_suffix = """
Output:

Do NOT output the intermediary step. Only the final 3 candidate bullets as 3 separate bullets
- phrase 1
- phrase 2
- phrase 3

Note: Do NOT use the same words as those already included in the input phrase.
"""

generate_product_components_prompt_prefix = """
You are a writing assistant for Amazon listings. Help me come up with the key features/components of a product. 
Provide your answer in 3-10 bullet points (depending on how simple/complex the product is) 
with the component name and a very brief description. 

Example input: Picture frame 
Output: 

Frame / molding – the outer structure and decorative border 
Protective cover – glass or acrylic that protects the photo 
Mat (optional) – insert that frames the image and adds visual depth 
Backboard / backing – supports and holds the photo in place 
Closure tabs – swivel tabs, turn buttons, or clips for securing contents 
Easel stand – for tabletop display 
Hanging hardware – hooks or brackets for wall mounting 
Back panel covering – paper or velvet finish on the back

New input:
"""

generate_synonyms_prompt_prefix = """
You are a writing assistant for Amazon listings. I will give you a phrase, and you are to help me
come up with SIMPLE AND POSITIVE alternative ways of expressing the same idea.

Guiding advice:
- Your language and tone should be simple, positive, and natural-sounding. Avoid using words that do not appear
  often in human conversation (e.g. long-wearing, luxe, hassle, fuss, )
- Outputs should roughly match length of input
- Do add additional info that is not included input (e.g. setting/use case)
- Only give 3 inputs. The 3 input should vary meaningfully in word choice.
- For the last input, you may explore related but slightly different meaning 
  E.g. quick and easy --> effortless (not identical but semantically related)

Example input: makes changing photos easy
Output: 
- allows for quick and easy photo change
- makes swapping photos simple and convenient
- lets you change photos effortlessly

Example input: antique frame is sturdy and durable
Output: 
- Antique frame is sturdy and built to last
- Vintage frame features a strong, durable construction
- Traditional frame has a sturdy and robust build

Example input: Brightness and vitality
Output: 
- Energy and vigor
- Richness and vibrance
- Freshness and radiance

Example input: adds a touch of sophistication
Example GOOD inputs: Adds a refined, modern touch, adds a stylish, upscale feel
Example BAD output: gives your space a classy, polished look
Why: I never mentioned it's for my 'space'. 

New input:
"""

generate_rephrase_prompt_prefix = """
You are a writing assistant for Amazon listings for picture frames. I will give you a phrase or sentence, 
and you are to help me come up with an improved of expressing the same idea. While staying true to the 
intended meaning, look for opportunties to improve grammar, readability, concision, warmth of tone, and enthusiasm. 

Example input: Cleaning the glass surface soft cloth provided to remove finger mark if any
Output: Gently clean glass surface with included soft cloth to remove fingerpints if any

Example input: Swivel tabs allow quick, easy changes of artwork or photos
Output: Swivel tabs makes changing photos quick and effortless
Explanation: Word choices 'quick and effortless' reads more bright and cheerful

Guiding advice: 
- Your tone should be subtly breezy, bright, cheerful, positive, and retail-ready while remaining professional and concise.
- Do NOT use flowery, overly romantic, dramatic phrases. 
  E.g. I would strongly prefer "combining sentimental warmth" rather than "seamlessly uniting heartfelt significance" 
- While you are encouraged to add 1-2 new words or change wording to improve the phrase,
  your output should strongly (though not necessarily exactly) match the length of input phrase.
- You must preserve objective facts (e.g. if flower is artificial, you must keep the word artificial)
- Avoid using the word 'the' unless absolutely needed
- Only give your top 1 output. 
- If the input is in another language, you must translate to English and ensure the same quality, style, and tone described above.

Input:
"""

amazonify_prompt_prefix = """
You are a writing assistant for Amazon listings for picture frames. I will give you a phrase or sentence, 
and you are to help me 'Amazonify' the input. That is, you are to transform my input into with a positive, concise,
Amazon-ready output (1-2 sentences).

Guiding advice:
- Try your best to make the tone sound more warm, reassuring, upbeat, and enthusiastic. If needed,
  feel free to add a few impactful phrases to brighten/improve the tone (if not already)
- Do NOT use flowery, overly romantic, dramatic phrases. 
  E.g. I would strongly prefer "combining sentimental warmth" rather than "seamlessly uniting heartfelt significance" 
- Your language and tone should be simple and natural-sounding. Avoid using cliche words that do not appear
  often in human conversation (e.g. long-wearing, luxe, hassle-free)
- You must preserve objective facts (e.g. if flower is artificial, you must keep the word artificial)
- Only give your top 1 output
- Avoid run-on lengthy sentences. Do NOT have more than 2 commas ',' in 1 sentence
- Use prepositions and transition words to make sentence sound more fluid.
- Your output can be similar but not identical to input.
- Avoid self-referring pronouns (I, our, we) and 'this'
- Use 'is' and 'your' when appropriate to sound warm and personal
- If the input is in another language, you must translate to English and ensure the same quality, style, and tone described above. 

Example input: Antique photo frame is sturdy, durable, and resists scratches/chipping
Output: This antique photo frame is built to last, with a sturdy, durable design that resists scratches and chipping

Example input: Cleaning the glass surface soft cloth provided to remove finger mark if any
Output: Use provided soft cloth to gently clean surface and remove any fingerprints, if any

Example input: Wide, cushioned strap is fully adjustable between 20.5 and 26 inches. suitable for any sleeping position, 
won’t snag hair, fits snugly and stays comfortably in place while you sleep without twisting or slipping off. 
Output: The wide, cushioned strap adjusts from 20.5 to 26 inches to fit any sleeping position.
        Eye mask is sure to stays snug and comfy all night, without slipping off or getting caught in your hair
Explanation: Because input could not fit in 1 sentence, I used 2 sentences.

Example input: white photo mat, clean gallery-like display, 8x10 without mat, 5x7 with mat:
Bad output: The white photo mat creates a clean, gallery-style display, giving your frame a polished look. Display an 8x10 photo without the mat or a 5x7 photo with the mat for a perfect fit.
Great output: White photo mat creates a clean, gallery-style display, fitting 8x10 photos without mat and 5x7 photos with mat
Why: Second output is more concise! Without too much extra fluff. 
"""



keyword_enrich_header_instructions = """
You are an assistant for Amazon listing writing for photo frames You will be provided with my listing bullet points 
as well as a list of targeted keywords I have collected. Your task is to add slightly more keywords into my 
listing where possible while preserving the core meaning, content, tone, and syntax. 

Your goal is to maximize the # of unique keywords. The same keyword, no matter how many times it appears, only counts once.
Therefore, there is no need to go out of your way to jam the same keywords multiple times.

YOU MUST PRESERVE FACTUAL ACCURACY. Examples: 
- if flower is artificial, you may use synonymous keywords such as faux, life-like, but you CANNOT say it is real). 
- if color is 'gold', do NOT say it's bronze, even if bronze is a keyword. 
- If frame is 5x7, do NOT say 6x8 or 4x6 etc.   

You may NOT remove any factual details from the listing, no matter how small.
You may NOT replace existing keyword that are already in the listing with new ones.

Output format: Only output the 5 bullets and nothing else (no need to write 'here is your listing'). Bold ONLY the subheadings for each bullet.
               Do NOT bold the keywords or any other content.
               Italicize the changes you made from the input listing.

Keywords:
"""

keyword_editor_header_instructions = """
You are an assistant writer and editor for Amazon listings. Specifically, your only tasks are to:
1. Check for repetitiveness/redundancy
2. Make sure output is grammatically correct and reads naturally

You must NOT change the content, facts, or tone of the input
Make sure language reads fluently and naturally (high quality full sentence written by native English speaker)

Examples: 
1. Redundancy: 'table or tabletop' or 'present or gift' simply choose one of the two
2. Grammar: 'Photo frame comes in unique style decorate your art or home': add 'to' or connecting word
3. Flow: 'photo frame has gold beaded inlays, vintage retro style' 
         can be improved to 'photo frame has vintage style gold beaded inlays

Input:
"""

# MAIN LISTING WRITER
listing_writer_system_instructions = "You are an assistant writer for Amazon product listings."

listing_writer_instructions = """
You are an assistant for Amazon listing writing for photo frames. I will proivde you with a set of facts you must include in the listing,
and keywords you want to use. Your goal is to write a factually accurate SEO-optimized Amazon listing. Do NOT include brand names.

Guidelines:
- Produce 5 bullets. Each bullet MUST be within 200-250 characters long (with spaces).
- Be concise. Use 'this' and 'the' slightly less frequently than you normally would
- Only output the 5 bullets and nothing else (no need to write 'here is your listing'). Bold the subheadings for each bullet.

Facts:
"""

listing_writer_example = """
Example listing (to help you get sense of tone/language) 

HIGHEST QUALITY MATERIAL: 4×6 vintage gold frame is made of high quality resin, includes solid black easel back. 
Individually hand crafted, every detail of the Victorian photo frame is a concrete manifestation of the artisan spirit. 
It is easy to clean and it will be keep your picture, portrait or art prints on good view position for long time. 

ELEGANT VINTAGE DESIGN: The small gorgeous lovely antique picture frame matching with retro exquisite flower carvings. 
Ready to hang the frame on the wall or display on tabletop, counter top, desk or shelf horizontally or vertically. 
Sturdy high definition glass front gives a clear view of your picture and preserves the life of your photo. 
Extremely carefully packaged to ensure it arrives safely! 

ATTRACTIVE LOOK: Special classy and simple appearance, perfect details. Outline border of this luxury gold wall hanging picture 
frame is 4.6x0.6x 6.5 inch .The beautiful cute frame fits a 4 by 6 inches picture, comes with easy opening tabs at the back for 
easy access for installing photos . 

PERFECT FOR HOME: Timeless and old-world decor for living room, bedroom, dining room, bathroom and anywhere of home. 
This little old-fashioned family picture frames fits with shabby chic or antique-feeling design schemes, 
it will lighting your room and make your home look more like a work of art. 

BEST GIFTS: It is a wonderful pretty little picture frame molding and idea gift for relatives, friend and someone you love. 
Makes a great gift idea for multi occasion: father’s day, mother’s day, anniversary, party, Christmas, wedding, festival, birthday, new baby,
new year,thanksgiving, housewarming, gathering, Valentine's Day and at family events.
"""

def complete_phrase(client, 

                    prompt: str, 
                    model = "gpt-5.1",
                    temperature = 0.2) -> str:
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant writer for Amazon product listings."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,    
    )

    return response.choices[0].message.content.strip()


# KEYWORD COUNTER
def _plural_pattern(word: str) -> str:
    """
    Create a regex that matches a word and its common plural forms.
    """

    # words ending with 'y' → babies
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        root = word[:-1]
        return rf"{re.escape(root)}(?:y|ies)"

    # words ending with s, x, z, ch, sh → boxes, dishes
    elif word.endswith(("s", "x", "z", "ch", "sh")):
        return rf"{re.escape(word)}(?:es)?"

    # default → frames, weddings
    else:
        return rf"{re.escape(word)}s?"
    

def keyword_count_df(keywords: list[str], text: str) -> pd.DataFrame:
    """
    Counts occurrences of each keyword in text and returns a dataframe.

    - Case insensitive
    - Matches whole words/phrases
    - Supports plural forms (wedding == weddings)
    """

    text_lower = text.lower()
    results = {}

    for kw in keywords:
        kw_clean = kw.strip().lower()
        if not kw_clean:
            continue

        # build plural-aware phrase pattern
        words = kw_clean.split()
        word_patterns = [_plural_pattern(w) for w in words]
        phrase_pattern = r"\b" + r"\s+".join(word_patterns) + r"\b"

        matches = re.findall(phrase_pattern, text_lower)
        results[kw_clean] = len(matches)

    df = pd.DataFrame(
        list(results.items()),
        columns=["keyword", "count"]
    ).sort_values(by="count", ascending=False).reset_index(drop=True)

    return df[df["count"] > 0]

def summarize_listing_keyword_stats(keywords: list[str], text: str):
    df = keyword_count_df(keywords, text)
    
    # all_keywords = len(keywords)
    # total_keywords = df['count'].sum()
    total_keywords_unique = len(df.keyword.unique())

    return total_keywords_unique, df



# with st.expander("Keyword stats"):
#             stats_col1, _, stats_col2, _ = st.columns([7.2, 2.8, 5,.01])
#             with stats_col1:

#                 st.write(f"**Keywords provided**: {all_keywords}") 
#                 st.write(f"**Unique keyword count**: {total_keywords_unique1}")
#                 keyword_markdown("New keywords added", only_in_list2)

#             with stats_col2:
#                 st.dataframe(df1, hide_index=True, height = 176)