"""
Generates a polished creator profile avatar image for R.M.Nishant Suman.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_creator_avatar(output_path="static/images/mypic.jpeg"):
    size = 400
    # Create gradient background
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw rounded circle mask
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((10, 10, size - 10, size - 10), fill=255)

    # Gradient from indigo to purple to pink
    gradient = Image.new("RGBA", (size, size))
    g_draw = ImageDraw.Draw(gradient)
    for y in range(size):
        r = int(79 + (236 - 79) * (y / size))
        g = int(70 + (72 - 70) * (y / size))
        b = int(229 + (153 - 229) * (y / size))
        g_draw.line([(0, y), (size, y)], fill=(r, g, b, 255))

    # Apply mask
    avatar = Image.composite(gradient, img, mask)
    a_draw = ImageDraw.Draw(avatar)

    # Outer glowing ring
    a_draw.ellipse((10, 10, size - 10, size - 10), outline=(255, 255, 255, 180), width=6)
    a_draw.ellipse((16, 16, size - 16, size - 16), outline=(99, 102, 241, 150), width=3)

    # Draw stylish Initials "NS" (Nishant Suman)
    try:
        # Try to load a clean truetype font or default
        font_large = ImageFont.truetype("arial.ttf", 110)
        font_small = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw Initials
    a_draw.text((size // 2, size // 2 - 30), "NS", fill=(255, 255, 255, 255), anchor="mm", font=font_large)
    
    # Subtitle pill
    pill_w, pill_h = 240, 42
    px1 = (size - pill_w) // 2
    py1 = size // 2 + 55
    a_draw.rounded_rectangle((px1, py1, px1 + pill_w, py1 + pill_h), radius=16, fill=(15, 23, 42, 220), outline=(255, 255, 255, 100), width=1)
    a_draw.text((size // 2, py1 + pill_h // 2), "AI ENGINEER", fill=(224, 231, 255, 255), anchor="mm", font=font_small)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    avatar.save(output_path, "PNG")
    print(f"Generated creator profile avatar at: {output_path}")


if __name__ == "__main__":
    generate_creator_avatar()
