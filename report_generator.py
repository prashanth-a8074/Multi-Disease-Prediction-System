import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Create reports directory
os.makedirs('reports', exist_ok=True)

def generate_pdf_report(patient_name, age, gender, disease_name, risk_percentage, prediction_label, feature_values):
    """
    Generates a professional PDF diagnostic report for the patient.
    
    Parameters:
    - patient_name (str)
    - age (int)
    - gender (str)
    - disease_name (str): display disease name
    - risk_percentage (float)
    - prediction_label (int): 0 or 1
    - feature_values (dict): key-value feature parameters
    
    Returns:
    - file_path (str): path to the saved PDF file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = "".join(x for x in patient_name if x.isalnum() or x in "._- ").strip().replace(" ", "_")
    file_name = f"Health_Report_{clean_name}_{timestamp}.pdf"
    file_path = os.path.join('reports', file_name)
    
    doc = SimpleDocTemplate(file_path, pagesize=letter,
                            rightMargin=54, leftMargin=54,
                            topMargin=54, bottomMargin=54)
    
    styles = getSampleStyleSheet()
    
    # Custom Styles (Premium Branding Colors)
    primary_color = colors.HexColor("#1A365D")   # Deep navy
    secondary_color = colors.HexColor("#3182CE") # Medical blue
    text_color = colors.HexColor("#2D3748")      # Charcoal
    accent_color = colors.HexColor("#E53E3E") if prediction_label == 1 else colors.HexColor("#38A169") # Red / Green
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=primary_color,
        spaceAfter=15,
        alignment=TA_LEFT
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor("#718096"),
        spaceAfter=25
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=secondary_color,
        spaceBefore=15,
        spaceAfter=10,
        borderPadding=4
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,
        leading=14
    )
    
    result_style = ParagraphStyle(
        'ResultText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=accent_color,
        leading=16
    )
    
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.HexColor("#A0AEC0"),
        alignment=TA_CENTER,
        leading=10
    )
    
    story = []
    
    # 1. Header (Banner Line & Title)
    story.append(Paragraph("MULTI-DISEASE DIAGNOSTIC SYSTEM", title_style))
    story.append(Paragraph(f"Clinical Prediction Report | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    
    # Divider line
    d_table = Table([[""]], colWidths=[504])
    d_table.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 2, secondary_color),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(d_table)
    story.append(Spacer(1, 15))
    
    # 2. Patient Profile Table
    story.append(Paragraph("Patient Information", section_heading))
    patient_data = [
        [Paragraph("<b>Full Name:</b>", body_style), Paragraph(patient_name, body_style),
         Paragraph("<b>Date of Birth / Age:</b>", body_style), Paragraph(f"{age} Years", body_style)],
        [Paragraph("<b>Gender:</b>", body_style), Paragraph(gender, body_style),
         Paragraph("<b>Report Status:</b>", body_style), Paragraph("Final Prediction", body_style)]
    ]
    p_table = Table(patient_data, colWidths=[100, 152, 120, 132])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#EDF2F7")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 15))
    
    # 3. Diagnostic Results Block
    story.append(Paragraph("Diagnostic Evaluation", section_heading))
    status_text = "⚠️ AT RISK (POSITIVE PREDICTION)" if prediction_label == 1 else "✅ LOW RISK (NEGATIVE PREDICTION)"
    risk_level = "HIGH RISK" if risk_percentage >= 70 else ("MODERATE RISK" if risk_percentage >= 30 else "LOW RISK")
    
    result_data = [
        [Paragraph("<b>Target Assessment:</b>", body_style), Paragraph(disease_name.upper(), body_style)],
        [Paragraph("<b>System Diagnosis:</b>", body_style), Paragraph(status_text, result_style)],
        [Paragraph("<b>Confidence / Risk Score:</b>", body_style), Paragraph(f"<b>{risk_percentage}%</b> ({risk_level})", result_style)]
    ]
    r_table = Table(result_data, colWidths=[150, 354])
    r_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
        ('PADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(r_table)
    story.append(Spacer(1, 15))
    
    # 4. Input Features Breakdown Table
    story.append(Paragraph("Clinical Metrics Profile", section_heading))
    feat_headers = [Paragraph("<b>Clinical Metric (Feature)</b>", body_style), Paragraph("<b>Recorded Value</b>", body_style)]
    feat_rows = [feat_headers]
    for k, v in feature_values.items():
        # Format key names nicely
        feat_name = k.replace("_", " ").title()
        feat_rows.append([Paragraph(feat_name, body_style), Paragraph(str(v), body_style)])
        
    f_table = Table(feat_rows, colWidths=[280, 224])
    f_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(f_table)
    story.append(Spacer(1, 25))
    
    # 5. Recommendation / Advice Block
    story.append(Paragraph("Clinical Guidance", section_heading))
    if prediction_label == 1:
        advice = ("Based on the clinical parameters provided, the patient exhibits elevated risk indicators. "
                  "We strongly recommend consulting a specialist at the earliest convenience. "
                  "Please review the recommended hospital list in the dashboard interface. "
                  "Avoid self-medication and monitor vital parameters regularly.")
    else:
        advice = ("The diagnostic parameters analyzed suggest a low current risk for this condition. "
                  "Continue maintaining a healthy lifestyle, active physical exercise, and a balanced diet. "
                  "Routine periodic examinations are advised for general wellness surveillance.")
                  
    story.append(Paragraph(advice, body_style))
    story.append(Spacer(1, 35))
    
    # Divider line
    story.append(d_table)
    story.append(Spacer(1, 10))
    
    # 6. Disclaimer footer
    disclaimer = ("DISCLAIMER: This system utilizes machine learning algorithms trained on historical patient cohorts. "
                  "It does not provide definitive medical diagnoses or clinical plans. "
                  "All results should be validated by certified healthcare providers using professional medical procedures.")
    story.append(Paragraph(disclaimer, disclaimer_style))
    
    # Build Document
    doc.build(story)
    return file_path
