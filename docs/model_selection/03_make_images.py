"""Step 3: draw synthetic screenshots locally with Pillow (no real brands, no network).

Writes PNG files to images/.

Run:  python 03_make_images.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
os.makedirs(OUT, exist_ok=True)

W, H = 1280, 800


def font(size, bold=False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def browser_chrome(d, address):
    d.rectangle([0, 0, W, 70], fill=(225, 228, 232))
    d.rounded_rectangle([140, 18, W - 140, 52], radius=16, fill="white", outline=(190, 190, 190))
    d.text((160, 25), address, font=font(18), fill=(60, 60, 60))
    for i, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        d.ellipse([20 + i * 30, 25, 40 + i * 30, 45], fill=c)


def login_page():
    im = Image.new("RGB", (W, H), (240, 242, 245))
    d = ImageDraw.Draw(im)
    browser_chrome(d, "secure-account-verify.example.net/signin")
    d.rounded_rectangle([390, 140, 890, 720], radius=12, fill="white", outline=(210, 210, 210))
    d.text((520, 175), "Sign in to your account", font=font(26, True), fill=(20, 20, 20))
    d.text((430, 225), "Your account is locked. Verify your identity to continue.", font=font(15), fill=(180, 30, 30))
    d.text((430, 280), "Email address", font=font(17), fill=(70, 70, 70))
    d.rectangle([430, 308, 850, 356], fill="white", outline=(150, 150, 150), width=2)
    d.text((430, 385), "Password", font=font(17), fill=(70, 70, 70))
    d.rectangle([430, 413, 850, 461], fill="white", outline=(150, 150, 150), width=2)
    d.text((445, 425), "* * * * * * * *", font=font(20), fill=(90, 90, 90))
    d.text((430, 490), "Card number", font=font(17), fill=(70, 70, 70))
    d.rectangle([430, 518, 850, 566], fill="white", outline=(150, 150, 150), width=2)
    d.rounded_rectangle([430, 610, 850, 665], radius=8, fill=(0, 102, 204))
    d.text((600, 625), "Sign in", font=font(22, True), fill="white")
    d.text((560, 685), "Forgot your password?", font=font(15), fill=(0, 102, 204))
    return im


def article_page():
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    browser_chrome(d, "www.example.org/articles/garden-soil-guide")
    d.text((160, 110), "A beginner's guide to garden soil", font=font(38, True), fill=(20, 20, 20))
    d.text((160, 170), "Published 12 March, 6 minute read", font=font(16), fill=(120, 120, 120))
    para = (
        "Healthy soil is the base of every productive garden. Before planting, it helps to know",
        "whether the soil is sandy, silty or heavy with clay, because each type holds water and",
        "nutrients differently. A simple jar test at home gives a rough answer in one afternoon.",
        "",
        "Compost improves almost every soil. It loosens clay, helps sand hold moisture and feeds",
        "the organisms that make nutrients available to roots. Spread a thin layer each season",
        "and let worms do the mixing instead of digging deeply every year.",
        "",
        "Mulch keeps the surface cool and slows evaporation. Straw, leaves and wood chips all",
        "work, as long as the layer is not packed tightly against plant stems.",
        "",
        "Finally, test the acidity every few years. Most vegetables prefer a slightly acidic to",
        "neutral range, and small corrections are easier than large ones.",
    )
    y = 220
    for line in para:
        d.text((160, y), line, font=font(20), fill=(40, 40, 40))
        y += 36
    return im


def sms_page():
    im = Image.new("RGB", (W, H), (245, 245, 247))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 80], fill=(250, 250, 250), outline=(220, 220, 220))
    d.text((560, 26), "Unknown sender", font=font(24, True), fill=(20, 20, 20))
    d.rounded_rectangle([80, 140, 900, 330], radius=24, fill=(229, 229, 234))
    lines = (
        "Your parcel is held at the depot.",
        "Confirm your card details within 24 hours",
        "or it will be returned to sender:",
        "parcel-redelivery.example.net/confirm",
    )
    y = 160
    for line in lines:
        d.text((110, y), line, font=font(24), fill=(20, 20, 20))
        y += 40
    d.rounded_rectangle([700, 400, 1200, 470], radius=24, fill=(0, 122, 255))
    d.text((730, 420), "Who is this?", font=font(24), fill="white")
    d.rounded_rectangle([80, 720, 1200, 775], radius=26, fill="white", outline=(200, 200, 200))
    d.text((110, 735), "Text message", font=font(20), fill=(160, 160, 160))
    return im


def table_page():
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    browser_chrome(d, "www.example.com/timetable")
    d.text((120, 110), "Bus timetable, route 12", font=font(32, True), fill=(20, 20, 20))
    heads = ("Stop", "First bus", "Last bus", "Frequency")
    rows = (
        ("Central station", "05:40", "23:50", "8 min"),
        ("Market street", "05:46", "23:56", "8 min"),
        ("Library", "05:52", "00:02", "8 min"),
        ("Riverside park", "05:59", "00:09", "10 min"),
        ("Hill road", "06:05", "00:15", "10 min"),
        ("Depot", "06:12", "00:22", "10 min"),
    )
    x0, y0, cw, rh = 120, 190, 260, 60
    for c, h in enumerate(heads):
        d.rectangle([x0 + c * cw, y0, x0 + (c + 1) * cw, y0 + rh], fill=(230, 236, 245), outline=(150, 150, 150))
        d.text((x0 + c * cw + 15, y0 + 18), h, font=font(20, True), fill=(20, 20, 20))
    for r, row in enumerate(rows, start=1):
        for c, v in enumerate(row):
            d.rectangle([x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh], outline=(150, 150, 150))
            d.text((x0 + c * cw + 15, y0 + r * rh + 18), v, font=font(20), fill=(40, 40, 40))
    return im


if __name__ == "__main__":
    for name, fn in (
        ("login_form.png", login_page),
        ("article_page.png", article_page),
        ("sms_parcel.png", sms_page),
        ("timetable_page.png", table_page),
    ):
        fn().save(os.path.join(OUT, name))
        print("wrote images/" + name)
