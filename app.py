from flask import Flask, request, jsonify, send_from_directory, session, redirect, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
from PIL import Image
import io
import base64
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib import colors
from reportlab.platypus import TableStyle

from src.gradcam import get_gradcam_heatmap, overlay_heatmap

app = Flask(__name__, static_folder="frontend", static_url_path="")
app.secret_key = "supersecretkey"

DOCTOR_CREDENTIALS = {
    "doctor@ai.com": "1234"
}

model = load_model("models/brain_tumor_model.keras")

IMG_SIZE = 224
THRESHOLD = 0.6


# ================= AUTH =================

@app.route("/")
def home():
    if "user" not in session:
        return send_from_directory("frontend", "login.html")
    return send_from_directory("frontend", "index.html")


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    if DOCTOR_CREDENTIALS.get(data.get("email")) == data.get("password"):
        session["user"] = data["email"]
        return jsonify({"success": True})
    return jsonify({"success": False}), 401


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


# ================= PREDICT =================

@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    file = request.files["file"]
    image = Image.open(file).convert("RGB")

    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    img_input = preprocess_input(img_resized)
    img_input = np.expand_dims(img_input, axis=0)

    prediction = float(model.predict(img_input)[0][0])

    tumor_prob = 1 - prediction

    if prediction >= THRESHOLD:
        label = "No Tumor"
        confidence = prediction
    else:
        label = "Tumor Detected"
        confidence = tumor_prob

    heatmap_base64 = None
    try:
        heatmap = get_gradcam_heatmap(model, img_input)
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        cam = overlay_heatmap(heatmap_resized, img_resized)

        _, buffer = cv2.imencode('.jpg', cam)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        print("GradCAM Error:", e)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "tumor_probability": tumor_prob,
        "no_tumor_probability": prediction,
        "heatmap": heatmap_base64
    })


# ================= PDF =================

@app.route("/download-report", methods=["POST"])
def download_report():

    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}

    patient_name = data.get("patientName", "Unknown")
    doctor_notes = data.get("doctorNotes", "")
    pred = data.get("prediction", {})

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    story = []

    # TITLE
    story.append(Paragraph("<b>Brain Tumor Detection Report</b>", styles["Title"]))
    story.append(Spacer(1, 20))

    # ===== GREEN HEADER STYLE FUNCTION =====
    def section(title):
        box = Table([[title]], colWidths=[450])
        box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.teal),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
            ('PADDING', (0,0), (-1,-1), 10),
        ]))
        return box

    # PATIENT INFO
    story.append(section("Patient Information"))
    story.append(Spacer(1, 10))

    info_table = [
        ["Patient Name", patient_name],
        ["Generated On", datetime.now().strftime('%Y-%m-%d %H:%M')],
        ["Prepared by", session.get("user")]
    ]

    table = Table(info_table, colWidths=[150, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))

    story.append(table)
    story.append(Spacer(1, 20))

    # DIAGNOSTIC
    story.append(section("Diagnostic Summary"))
    story.append(Spacer(1, 10))

    result_color = "red" if pred.get("label") == "Tumor Detected" else "green"

    story.append(Paragraph(
        f'<font size=12 color="{result_color}"><b>Result: {pred.get("label")}</b></font>',
        styles["Normal"]
    ))

    story.append(Spacer(1, 15))

    # METRICS
    story.append(section("Prediction Metrics"))
    story.append(Spacer(1, 10))

    table_data = [
        ["Metric", "Value"],
        ["Confidence", f"{pred.get('confidence',0)*100:.2f}%"],
        ["Tumor Probability", f"{pred.get('tumor_probability',0)*100:.2f}%"],
        ["No Tumor Probability", f"{pred.get('no_tumor_probability',0)*100:.2f}%"]
    ]

    table = Table(table_data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))

    story.append(table)
    story.append(Spacer(1, 20))

    # NOTES
    story.append(section("Doctor Notes"))
    story.append(Spacer(1, 10))

    notes = Table([[doctor_notes or "No notes"]], colWidths=[450])
    notes.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
        ('BOX', (0,0), (-1,-1), 1, colors.grey),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))

    story.append(notes)

    doc.build(story)

    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="report.pdf")


# ================= RUN =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)