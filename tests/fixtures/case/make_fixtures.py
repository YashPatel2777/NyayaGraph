"""Regenerate the synthetic case fixtures (07 section 8).

Every name, number and event here is fictional; no real person's documents are used.
Run from the project root on Windows (the voice note uses Windows text-to-speech):

    python tests/fixtures/case/make_fixtures.py
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLEAN, NOISY = HERE / "clean", HERE / "noisy"

INVOICE = """TAX INVOICE
QuickKart Retail Private Limited
Order No. QK-55120-8841    Invoice Date: 02/01/2026
Bill to: Ravi Mehta, 14 Lake View Road, Pune 411001

Item: Zenith X2 smartphone, 128 GB, black
Quantity: 1    Unit price: Rs. 18,499.00
Total paid: Rs. 18,499.00 (UPI)

Warranty: 12 months manufacturer warranty from the date of purchase.
Returns: Defective items may be returned within 7 days of delivery."""

LETTER = """Complaint to the seller

To,
The Grievance Officer, QuickKart Retail Private Limited

Subject: Defective Zenith X2 smartphone, Order No. QK-55120-8841

Facts of the purchase

I bought a Zenith X2 smartphone from your platform on 2 January 2026 for Rs. 18,499. It was delivered on 4 January 2026. On 6 January 2026 the display developed vertical green lines and then stopped responding to touch. The phone was never dropped and has no external damage.

Steps already taken

On 7 January 2026 I raised a return request within the 7 day return window. Your delivery partner did not pick up the phone. On 12 January 2026 your seller support told me on WhatsApp that there is a no refund policy. The authorised service centre at Shivaji Nagar inspected the phone on 15 January 2026 and recorded a manufacturing defect in the display panel.

Relief sought

Please refund Rs. 18,499 or replace the phone within 15 days. If I do not hear from you, I will file a complaint before the District Consumer Disputes Redressal Commission.

Ravi Mehta
20 January 2026"""

RECEIPT = [
    "ZENITH AUTHORISED SERVICE CENTRE",
    "Shivaji Nagar, Pune",
    "Job Sheet No. ZSC-20260115-077",
    "Date: 15/01/2026",
    "Customer: Ravi Mehta",
    "Device: Zenith X2  IMEI ending 4471",
    "Complaint: green lines on display, touch not working",
    "Finding: display panel manufacturing defect",
    "Physical damage: none   Liquid damage: none",
    "Estimate if out of warranty: Rs. 6,200",
    "Status: covered under warranty, part on order",
]

VOICE = (
    "This is Ravi Mehta. I am recording what happened with my Zenith X2 phone. "
    "I ordered it from QuickKart on the second of January and paid eighteen thousand "
    "four hundred and ninety nine rupees. Two days after delivery the screen showed green "
    "lines and touch stopped working. I asked for a return on the seventh of January, "
    "but nobody came to collect the phone. The seller told me they have a no refund "
    "policy. The service centre has confirmed it is a manufacturing defect."
)

CHAT = """12/01/2026, 14:30 - Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.
12/01/2026, 14:31 - QuickKart Seller Support: Hello, how can we help you today?
12/01/2026, 14:33 - Ravi Mehta: My Zenith X2 order QK-55120-8841 has a defective screen. Green lines came on 6 January, two days after delivery.
12/01/2026, 14:34 - Ravi Mehta: I raised a return on 7 January but no pickup happened
12/01/2026, 14:36 - QuickKart Seller Support: Sorry for the trouble. As per our policy this item is non-refundable.
12/01/2026, 14:37 - Ravi Mehta: Your website says defective items can be returned within 7 days
12/01/2026, 14:39 - QuickKart Seller Support: No refund policy applies to mobile phones. You can visit the brand service centre.
12/01/2026, 14:40 - Ravi Mehta: <Media omitted>
12/01/2026, 14:41 - Ravi Mehta: This is the photo of the screen.
I want a refund or a replacement.
15/01/2026, 18:05 - Ravi Mehta: Service centre has confirmed a manufacturing defect in the display panel. Job sheet ZSC-20260115-077.
15/01/2026, 18:20 - QuickKart Seller Support: We have noted your concern. Please wait for 5-7 working days.
"""


NOTES = """Notes for my consumer complaint against QuickKart Retail Private Limited

1. I am a consumer under Section 2(7) of the Consumer Protection Act, 2019, because I bought the Zenith X2 phone for my own use.
2. The green lines on the display are a defect under section 2(10), and not collecting the phone for return is a deficiency under Section 2(11).
3. The seller's 'no refund policy' reply on WhatsApp is an unfair trade practice as defined in section 2(47).
4. The complaint can be filed under section 35 before the District Commission, within the time allowed by section 69.
5. A friend said the seller could also be booked under Section 420 IPC. I have not done this.

Ravi Mehta
"""


def make_pdf(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    for text in (INVOICE, LETTER):
        page = doc.new_page()  # A4 portrait
        page.insert_textbox(pymupdf.Rect(56, 56, 540, 790), text, fontsize=10, fontname="helv")
    doc.save(path)


def make_receipt(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    rng = random.Random(7)
    img = Image.new("L", (1240, 900), 245)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 30)
    for i, line in enumerate(RECEIPT):
        draw.text((70, 60 + i * 72), line, fill=25, font=font)
    img = img.rotate(1.5, resample=Image.BICUBIC, fillcolor=235, expand=True)
    noisy = img.point(lambda v: v)  # copy
    px = noisy.load()
    for _ in range(25000):  # speckle, like a cheap phone scan
        x, y = rng.randrange(noisy.width), rng.randrange(noisy.height)
        px[x, y] = rng.choice((90, 140, 200))
    noisy.filter(ImageFilter.GaussianBlur(0.8)).convert("RGB").save(path)


def make_voice(path: Path) -> None:
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.Rate = 0;"
        f"$s.SetOutputToWaveFile('{path}');"
        f"$s.Speak('{VOICE}');"
        "$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)


def main() -> int:
    CLEAN.mkdir(exist_ok=True)
    NOISY.mkdir(exist_ok=True)
    make_pdf(CLEAN / "invoice_and_letter.pdf")
    (CLEAN / "chat_seller.txt").write_text(CHAT, encoding="utf-8")
    (CLEAN / "complaint_notes.txt").write_text(NOTES, encoding="utf-8")
    make_receipt(NOISY / "service_job_sheet.png")
    if sys.platform == "win32":
        make_voice(CLEAN / "voice_note.wav")
    print("fixtures written to", HERE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
