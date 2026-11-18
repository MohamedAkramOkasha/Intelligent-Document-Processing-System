"""Generate synthetic sample documents for demos and tests.

    python scripts/generate_samples.py            # writes into ./samples

Produces a digital PDF invoice, a scanned-style PNG receipt (needs OCR), a DOCX contract,
text invoices (one with deliberately wrong totals, to exercise human review) and a letter.
All companies and people are fictional.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

INVOICE_LINES = [
    ("Cloud hosting - Standard plan", 3, 120.00),
    ("Managed database (monthly)", 1, 250.00),
    ("Support hours", 10, 45.50),
]


def invoice_pdf(path: Path) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Nimbus Cloud Solutions Ltd.", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for line in ("42 Harbour Street, Bristol BS1 4XX, United Kingdom", "billing@nimbus-cloud.example",
                 "VAT No: GB123456789"):
        pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "INVOICE", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for line in ("Invoice Number: INV-2026-0042", "Invoice Date: 2026-10-01", "Due Date: 2026-10-31",
                 "PO Number: PO-7781", "", "Bill To:", "Orbital Analytics GmbH",
                 "Friedrichstrasse 12, 10117 Berlin, Germany"):
        pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 10)
    for text, w in (("Description", 95), ("Qty", 20), ("Unit Price", 35), ("Amount", 35)):
        pdf.cell(w, 7, text, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", size=10)
    subtotal = 0.0
    for desc, qty, unit in INVOICE_LINES:
        amount = qty * unit
        subtotal += amount
        pdf.cell(95, 7, desc, border=1)
        pdf.cell(20, 7, str(qty), border=1)
        pdf.cell(35, 7, f"{unit:,.2f}", border=1)
        pdf.cell(35, 7, f"{amount:,.2f}", border=1)
        pdf.ln()
    tax = round(subtotal * 0.20, 2)
    pdf.ln(4)
    for label, value in (("Subtotal", subtotal), ("VAT (20%)", tax), ("Total (GBP)", subtotal + tax)):
        pdf.cell(150, 6, label, align="R")
        pdf.cell(35, 6, f"{value:,.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.cell(0, 5, "Payment terms: Net 30. Bank transfer to Sort Code 12-34-56, Account 87654321.")
    pdf.output(str(path))


def receipt_png(path: Path) -> None:
    lines = [
        "BEAN & LEAF COFFEE HOUSE", "18 Market Square, Springfield", "Tel: (555) 123-4567", "",
        "Receipt #: R-88213", "Date: 11/02/2026  08:42 AM", "Cashier: Maya", "",
        "Flat White           4.50", "Blueberry Muffin     3.25", "Avocado Toast        8.75",
        "Orange Juice         3.50", "", "Subtotal            20.00", "Tax 8%               1.60",
        "Tip                  2.00", "TOTAL               23.60", "", "VISA ************4821",
        "", "Thank you for shopping with us!",
    ]
    try:
        font = ImageFont.truetype("cour.ttf", 30)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", 30)
        except OSError:
            font = ImageFont.load_default(size=30)
    img = Image.new("RGB", (760, 60 + 44 * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((40, 30 + 44 * i), line, fill=(25, 25, 25), font=font)
    img = img.rotate(0.8, expand=True, fillcolor="white").filter(ImageFilter.GaussianBlur(0.6))
    img.save(path)


def contract_docx(path: Path) -> None:
    import docx

    d = docx.Document()
    d.add_heading("MASTER SERVICES AGREEMENT", 0)
    d.add_paragraph("Agreement No: MSA-2026-117")
    d.add_paragraph(
        "This Master Services Agreement is made between Brightpath Consulting LLC (the \"Service Provider\") "
        "and Riverstone Logistics Inc. (the \"Client\"), effective as of January 15, 2026."
    )
    d.add_heading("1. Term", level=1)
    d.add_paragraph("This Agreement shall commence on the effective date and continue for a term of 12 months, "
                    "expiring on January 14, 2027, unless terminated earlier.")
    d.add_heading("2. Services and Fees", level=1)
    d.add_paragraph("The Service Provider shall deliver supply-chain optimisation consulting services. "
                    "The total contract value is $96,000.00 USD, payable in monthly instalments of $8,000.00 "
                    "within 30 days of invoice.")
    d.add_heading("3. Confidentiality", level=1)
    d.add_paragraph("Each party shall keep confidential all non-public information received from the other party.")
    d.add_heading("4. Termination", level=1)
    d.add_paragraph("Either party may terminate this Agreement with 60 days written notice. The Client shall pay "
                    "for all services rendered up to the termination date.")
    d.add_heading("5. Governing Law", level=1)
    d.add_paragraph("This Agreement shall be governed by the laws of the State of Delaware.")
    d.add_paragraph("IN WITNESS WHEREOF, the parties have executed this Agreement.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Brightpath Consulting LLC", "Riverstone Logistics Inc."
    t.cell(1, 0).text, t.cell(1, 1).text = "Signed: Daniel Okafor, CEO", "Signed: Priya Raman, COO"
    d.save(str(path))


INVOICE_TXT = """Sunrise Office Supplies
77 Elm Avenue, Austin, TX 78701
accounts@sunrise-supplies.example

INVOICE
Invoice No: SOS-10593
Invoice Date: March 3, 2026
Due Date: April 2, 2026

Bill To: Lakeside Dental Clinic

Description            Qty   Unit Price   Amount
A4 Copy Paper (box)     10       $32.00   $320.00
Ink Cartridge Black      4       $45.00   $180.00
Ergonomic Chair          2      $210.00   $420.00

Subtotal: $920.00
Sales Tax (8.25%): $75.90
Total Due: $995.90

Payment terms: Net 30
"""

INVOICE_BAD_TXT = """Quickfix Plumbing Services
Invoice # QF-2207
Date: 2026-09-14
Due Date: 2026-09-01

Bill To: Harper Residence

Pipe replacement        1   $380.00   $380.00
Labour (hours)          3    $65.00   $195.00

Subtotal: $575.00
Tax: $46.00
Total: $650.00
"""

LETTER_TXT = """Greenfield Property Management
Reference: GPM-NOTICE-311
Date: 2026-08-20

To: All Tenants, Building C

Subject: Scheduled water maintenance

Please be advised that water service will be interrupted on August 27, 2026 between 09:00 and 13:00
for scheduled maintenance of the main supply line.
Contact: facilities@greenfield.example
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    invoice_pdf(out / "invoice_nimbus.pdf")
    receipt_png(out / "receipt_bean_leaf.png")
    contract_docx(out / "contract_brightpath_msa.docx")
    (out / "invoice_sunrise.txt").write_text(INVOICE_TXT, encoding="utf-8")
    (out / "invoice_quickfix_errors.txt").write_text(INVOICE_BAD_TXT, encoding="utf-8")
    (out / "letter_greenfield.txt").write_text(LETTER_TXT, encoding="utf-8")
    print(f"Sample documents written to {out.resolve()}")


if __name__ == "__main__":
    main()
