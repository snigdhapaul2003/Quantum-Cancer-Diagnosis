from __future__ import division, print_function
import sys
import os
import shutil
import glob
import re
from pathlib import Path
import base64
import requests
import tensorflow
from keras.models import Model
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from io import BytesIO
import replicate
import ultralytics
from ultralytics import YOLO
import json
from fastai import *
from fastai.vision import *
from flask import Flask, redirect, url_for, render_template, request,Response, session,  jsonify
from PIL import Image as PILImage
import datetime
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import joblib
import csv
import psycopg2
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from groq import Groq
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from quickstart import gmail_send
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

genai_api = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=genai_api)

# Setup twilio account
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
# Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Define the Replicate API endpoint and your API token
client_groq = Groq(
    api_key=os.getenv("GROQ_API"),
)

# Initialize conversation history
conversation_history = defaultdict(list)

# Define constants
MAX_TOKENS = 2048

geolocator = Nominatim(user_agent="doctor_locator")

extra_doctors = []


def extract_answer(text):
    # Process the text
    doc = nlp(text)

    # Look for "yes" or "no" near the end
    for token in reversed(doc):
        if token.text.lower() in ["yes", "no"]:
            return token.text.lower()

    # Default if not found
    return "unknown"

def encode(img):
    b, g, r = cv2.split(img)
    # Flatten pixel values and concatenate channels
    pixel_values = list(r.flatten()) + list(g.flatten()) + list(b.flatten())
    a = np.array(pixel_values).reshape(-1, 28, 28, 3)
    return a

def detect_b(cls):
    if cls==0:
        dis="Benign-adenosis"
    elif cls==1:
        dis="Benign-fibroadenoma"
    elif cls==2:
        dis="Benign-phyllodes tumor"
    elif cls==3:
        dis="Benign-tubulor adenoma"
    else:
        dis="No Disease Detected"
    return dis

def detect_m(cls):
    if cls==0:
        dis="Malignant-ductal carcinoma"
    elif cls==1:
        dis="Malignant-lobular carcinoma"
    elif cls==2:
        dis="Malignant-mucinous carcinoma"
    elif cls==3:
        dis="Malignant-papillary carcinoma"
    else:
        dis="No Disease Detected"
    return dis

@app.route('/', methods=['GET', "POST"])
def index():
    # Main page
    return render_template('index.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index_bengali')
def index_bengali():
    return render_template('index_bengali.html')

@app.route('/about_bengali')
def about_bengali():
    return render_template('about_bengali.html')

@app.route('/result_bengali')
def result_bengali():
    return render_template('result_bengali.html')

@app.route('/model_bengali')
def model_bengali():
    return render_template('model_bengali.html')

@app.route('/index_hindi')
def index_hindi():
    return render_template('index_hindi.html')

@app.route('/about_hindi')
def about_hindi():
    return render_template('about_hindi.html')

@app.route('/result_hindi')
def result_hindi():
    return render_template('result_hindi.html')

@app.route('/model_hindi')
def model_hindi():
    return render_template('model_hindi.html')

@app.route('/Book_Appointment')
def Book_Appointment():
    return render_template('Book_Appointment.html')


def extract_features(model,img_path):
    """Extract features from an image file."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    print("img_array", img_array)
    features = model.predict(img_array)
    return features.flatten()

def convert_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def pdf():
    # Create a BytesIO buffer to hold the PDF data

    with open('static/text.csv', mode="r") as file:
        # Create a CSV reader object
        loaded_list = json.load(file)

    name = loaded_list[0]
    age = loaded_list[1]
    phone = loaded_list[2]
    email = loaded_list[3]
    model = loaded_list[4]
    disease_class = loaded_list[5]
    current_datetime = datetime.now()

    formatted_datetime = current_datetime.strftime("%d-%m-%Y")

    # Step 1: Read the existing PDF as background
    image_path = 'image.png'  # Path to the JPG image you want to use as the background

    # Step 2: Create a new overlay using reportlab
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(720, 1040))  # Set page size based on your image

    # Step 3: Use the image as the background
    pdf.drawImage(image_path, 0, 0, 720, 1040)

    # Register Arial font
    pdf.setFont("Helvetica", 18)
    pdf.drawString(96, 777, name)
    pdf.drawString(439, 777, age)
    pdf.drawString(115, 742, phone)
    pdf.drawString(452, 742, email)
    pdf.drawString(570, 810, formatted_datetime)

    # Font and size
    font_name = "Helvetica"
    font_size = 18

    # Middle coordinates
    model_x = 195
    disease_x = 533
    y = 578

    # Calculate text widths
    model_width = pdfmetrics.stringWidth(model, font_name, font_size)
    disease_width = pdfmetrics.stringWidth(disease_class, font_name, font_size)

    # Adjust starting positions to align text to the middle
    pdf.drawString(model_x - model_width / 2, y, model)
    pdf.drawString(disease_x - disease_width / 2, y, disease_class)

    pdf.showPage()
    pdf.save()

    buffer.seek(0)

    output_path = 'generated_report.pdf'
    with open(output_path, 'wb') as f:
        f.write(buffer.getvalue())

    pdf_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Return the base64 string as a JSON response
    return pdf_base64, buffer

@app.route('/database')
def database():
    # Database connection parameters
    conn = psycopg2.connect(
        dbname=os.getenv("SQL_DBNAME"),
        user=os.getenv("SQL_USER"),
        password=os.getenv("SQL_PASSWORD"),
        host='localhost',  # assuming it's a local database
        port='5432'        # default PostgreSQL port
    )
    cur = conn.cursor()

    # Initialize serial number
    serial_number = 1

    # Check if the table exists and create it if not
    cur.execute("""
    CREATE TABLE IF NOT EXISTS REPORT_DETAILS (
        serial_number SERIAL PRIMARY KEY,
        name TEXT,
        age INTEGER,
        phone TEXT,
        email TEXT,
        model TEXT,
        disease_class TEXT,
        image TEXT,
        current_datetime TIMESTAMP,
        report TEXT
    )
    """)

    # Get the last serial number
    cur.execute("SELECT MAX(serial_number) FROM REPORT_DETAILS")
    last_serial = cur.fetchone()[0]
    if last_serial is not None:
        serial_number = last_serial + 1

    # Process CSV data
    with open('static/text.csv', mode="r") as file:
        loaded_list = json.load(file)

    name = loaded_list[0]
    age = loaded_list[1]
    phone = loaded_list[2]
    email = loaded_list[3]
    model = loaded_list[4]
    disease_class = loaded_list[5]
    image = convert_image_to_base64(loaded_list[6])
    current_datetime = datetime.now()
    report, buffer = pdf()

    # Insert data into the database
    cur.execute("""
    INSERT INTO REPORT_DETAILS (serial_number, name, age, phone, email, model, disease_class, image, current_datetime, report)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (serial_number, name, age, phone, email, model, disease_class, image, current_datetime, report))

    # Commit the transaction
    conn.commit()

    # Close the cursor and connection
    cur.close()
    conn.close()

    return Response(status=204)

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def check_with_genai():
    # **Specify the correct path to the image file**
    image_path = "static/image.jpeg"  # Update this with the actual image path

    # Create the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_schema": content.Schema(
            type=content.Type.OBJECT,
            properties={
                "response": content.Schema(
                    type=content.Type.BOOLEAN,
                ),
            },
        ),
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
    )

    # TODO Make these files available on the local file system
    # You may need to update the file paths
    files = [
        upload_to_gemini(image_path, mime_type="image/jpeg"),
    ]

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    files[0],
                    "Is this a human?",
                ],
            },
            {
                "role": "model",
                "parts": [
                    "```json\n{\n  \"response\": true\n}\n```",
                ],
            },
        ]
    )

    response = chat_session.send_message("Is this image a histopathological image?")

    # Parse the string into a dictionary
    parsed_json = json.loads(response.text)

    # Fetch the value of "response"
    response_value = parsed_json["response"]
    print("response value", response_value)

    return response_value

def inference(model, name, age, phone, email, uploaded_file):

    # uploaded_file.save('static/image.jpeg')


    img = uploaded_file.read()
    img = PILImage.open(BytesIO(img))


    # Convert image to numpy array
    img_data = np.array(img)

    pil_img = PILImage.fromarray(img_data)

    pil_img.save(f'static/image.jpeg', format="JPEG")

    is_true = check_with_genai()

    if is_true == False:
        return {"error": "The uploaded image is not a valid breast cancer histopathological image."}, None

    # Convert to HSV
    hsv_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2HSV)

    # Save HSV image
    hsv_pil_img = PILImage.fromarray(hsv_img)
    hsv_pil_img.save('static/image_hsv.jpeg', format="JPEG")

    img_path = 'static/image.jpeg'
    img_path_hsv = 'static/image_hsv.jpeg'

    cls = ''
    if model == 'Quantum Classical Stack':
        base_model = DenseNet121(weights='imagenet')
        modelx = Model(inputs=base_model.input, outputs=[base_model.get_layer('avg_pool').output])

        feature_list_rgb = []
        features_rgb = extract_features(modelx, img_path)
        print(features_rgb)
        feature_list_rgb.append(features_rgb.tolist())
        scaler = joblib.load('../models/weight_file/scaler_rgb.joblib')
        print(len(feature_list_rgb[0]))
        features_rgb = scaler.transform(feature_list_rgb)

        feature_list_hsv = []
        features_hsv = extract_features(modelx, img_path_hsv)
        feature_list_hsv.append(features_hsv.tolist())
        feature_list_hsv.append(features_hsv.tolist())
        scaler = joblib.load('../models/weight_file/scaler_hsv.joblib')
        print(len(feature_list_hsv[0]))
        features_hsv = scaler.transform(feature_list_hsv)

        features = np.array([(a + b) / 2 for a, b in zip(features_rgb, features_hsv)])

        pca = joblib.load('../models/weight_file/pca_model_avg.pkl')
        img = pca.transform(features.reshape(1, -1))

        # Load the saved models
        model_stack1 = joblib.load("../models/weight_file/svc_model.pkl")
        model_stack2 = joblib.load("../models/weight_file/qsvc_model.pkl")

        print(img)
        predict1 = model_stack1.predict(img)
        predict2 = model_stack2.predict(img)

        print(predict1)
        print(predict2)

        # Convert predictions to numeric values and stack them
        extra_features = np.array([0 if predict1[0] == 'n' else 1,
                                   0 if predict2[0] == 'n' else 1]).reshape(1, -1)

        # Concatenate extra features to the transformed image
        img = np.hstack((img, extra_features))

        clas = joblib.load('../models/weight_file/lg_final_clf.pkl')
        predict = clas.predict(img)
        print(predict)

        if predict[0] == 0:
            cls = 'n'
        else:
            cls = 't'


    if cls == 'n':
        model_yolo = YOLO("../models/weight_file/benigh_yolov8.pt")
        pred1 = model_yolo(cv2.imread(f'static/image.jpeg'))
        x = pred1[0].probs.top5
        y = pred1[0].probs.top5conf.tolist()
        prob = []
        dis_name = ['Adenosis', 'Fibroadenoma', 'Phyllodes tumor', 'Tubulor adenoma']
        for i in range(4):
            if i in x:
                y_index = x.index(i)
                y_value = y[y_index]
                prob.append(y_value)
            else:
                prob.append(0)
        max_position = prob.index(max(prob))
        cls = detect_b(max_position)

    if cls == 't':
        model_yolo = YOLO("../models/weight_file/malignant_yolov8.pt")
        pred1 = model_yolo(cv2.imread(f'static/image.jpeg'))
        x = pred1[0].probs.top5
        y = pred1[0].probs.top5conf.tolist()
        prob = []
        dis_name = ['Ductal Carcinoma', 'Lobular Carcinoma', 'Mucinous Carcinoma', 'Papillary Carcinoma']
        for i in range(4):
            if i in x:
                y_index = x.index(i)
                y_value = y[y_index]
                prob.append(y_value)
            else:
                prob.append(0)
        max_position = prob.index(max(prob))
        cls = detect_m(max_position)

    result1 = {"class": cls, "probs": prob, "image": img_path, "name": name, "age": age, "phone": phone, "email": email,
               "model": model, "dis_name": dis_name}
    em = []
    em.append(result1['name'])
    em.append(result1['age'])
    em.append(result1['phone'])
    em.append(result1['email'])
    em.append(result1['model'])
    em.append(result1['class'])
    em.append(result1['image'])
    # em.append(result1['probs'])

    return result1, em

@app.route('/upload', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        model = request.form['model']
        name = request.form['name']
        age = request.form['age']
        phone = request.form['phone']
        email = request.form['email']
        uploaded_file = request.files.get('file')

        result1, em = inference(model, name, age, phone, email, uploaded_file)

        if "error" in result1:
            error_message = result1["error"]
            return render_template('index.html', error_message=error_message)

        with open(f'static/text.csv', "w") as file:
            json.dump(em, file)
        return render_template('result.html', result=result1, success=False)
        # return preds
    return 'OK'

@app.route('/upload_beng', methods=["POST", "GET"])
def upload_beng():
    if request.method == 'POST':
        # Get the file from post request
        model = request.form['model']
        name = request.form['name']
        age = request.form['age']
        phone = request.form['phone']
        email = request.form['email']
        uploaded_file = request.files.get('file')

        result1, em = inference(model, name, age, phone, email, uploaded_file)

        with open(f'static/text.csv', "w") as file:
            json.dump(em, file)
        return render_template('result_bengali.html', result=result1, success=False)
        # return preds
    return 'OK'

@app.route('/upload_hindi', methods=["POST", "GET"])
def upload_hindi():
    if request.method == 'POST':
        # Get the file from post request
        model = request.form['model']
        name = request.form['name']
        age = request.form['age']
        phone = request.form['phone']
        email = request.form['email']
        uploaded_file = request.files.get('file')

        result1, em = inference(model, name, age, phone, email, uploaded_file)

        with open(f'static/text.csv', "w") as file:
            json.dump(em, file)
        return render_template('result_hindi.html', result=result1, success=False)
        # return preds
    return 'OK'

@app.route('/generate_pdf')
def generate_pdf():
    # Create a BytesIO buffer to hold the PDF data

    with open('static/text.csv', mode="r") as file:
        # Create a CSV reader object
        loaded_list = json.load(file)

    name = loaded_list[0]
    age = loaded_list[1]
    phone = loaded_list[2]
    email = loaded_list[3]
    model = loaded_list[4]
    disease_class = loaded_list[5]
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%d-%m-%Y")

    report, buffer = pdf()

    whatsapp_message = (
        f"Dear {name},\n\n"
        "Your QCD report has been successfully generated and sent to your email address. "
        "Please check your inbox to access the detailed PDF report.\n\n"
        "Key details from the report:\n"
        f"Date: {formatted_datetime}\n"
        f"Age: {age}\n"
        f"Contact: {phone}\n"
        f"Email-ID: {email}\n"
        f"Detection Model: {model}\n"
        f"Disease Detected: {disease_class}\n\n"
        "Disclaimer: This AI-generated report is preliminary and not a substitute for professional medical advice. "
        "We recommend consulting a qualified physician for a comprehensive evaluation.\n\n"
        "I am a medical assistant, here to assist you with any queries you may have about your health or the report. "
        "Feel free to ask me anything, and I'll do my best to help you.\n\n"
        "If you want to see the report again, type 'generate report' here.\n\n"
        "Thank you for choosing QCD. We are here to assist you in every way possible."
    )

    with open(f"{phone}.txt", "w") as file:
        file.write(whatsapp_message)

    try:
        print(phone)
        print(whatsapp_message)
        print(TWILIO_WHATSAPP_NUMBER)
        client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            to=f'whatsapp:+91{phone}',  # Send to the phone number provided in the CSV
            body=whatsapp_message
        )
        print("Message is sent")
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")

    receiver = email
    sender = 'QCD'
    title = 'AI generated medical report'
    email_body = """
    Dear recipient,

    Please find attached the AI-generated medical report for your recent breast cancer histopathological image analysis.

    This AI-generated report is preliminary and not a substitute for professional medical advice. Consult a healthcare provider for a comprehensive evaluation. Decisions about your health should be made in consultation with a qualified physician.

    Sincerely,
    QCD
        """

    gmail_send(receiver, sender, title, email_body)


    return Response(buffer.getvalue(), mimetype='application/pdf', headers={'Content-Disposition': 'attachment; filename=test_report.pdf'})


@app.route('/send_whatsapp', methods=['POST'])
def send_whatsapp_message():
    try:
        # Load data from the provided JSON file
        with open('static/text.csv', mode="r") as file:
            loaded_list = json.load(file)

        # Extract details from the loaded data
        name = loaded_list[0]
        age = loaded_list[1]
        phone = loaded_list[2]
        email = loaded_list[3]
        model = loaded_list[4]
        disease_class = loaded_list[5]

        # Get the current date and format it
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%d-%m-%Y")

        # Generate WhatsApp message
        whatsapp_message = (
            f"Dear {name},\n\n"
            "Your QCD report has been successfully generated. "
            "Key details from the report:\n\n"
            f"Date: {formatted_datetime}\n"
            f"Age: {age}\n"
            f"Contact: {phone}\n"
            f"Email-ID: {email}\n"
            f"Detection Model: {model}\n"
            f"Disease Detected: {disease_class}\n\n"
            "Disclaimer: This AI-generated report is preliminary and not a substitute for professional medical advice. "
            "We recommend consulting a qualified physician for a comprehensive evaluation.\n\n"
            "Thank you for choosing QCD. We are here to assist you in every way possible."
        )

        print("Whatsapp message is going to send")

        # Send WhatsApp message
        client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            to=f'whatsapp:+91{phone}',  # Send to the provided phone number
            body=whatsapp_message
        )

        print("Whatsapp message is already sent")

        return {"message": "WhatsApp message sent successfully!"}, 200

    except Exception as e:
        return {"error": str(e)}, 500


def add_to_history(from_number, user_input, model_response):
    """
    Add user input and model response to the conversation history.
    """
    disclaimer = (
        "\n\nDisclaimer: This response is for informational purposes only and should "
        "not be considered medical advice. Always consult a licensed healthcare provider."
    )
    conversation_history[from_number].append({"user": user_input, "response": model_response})

def format_conversation_history(from_number):
    """
    Format the conversation history into a single string.
    """
    formatted_history = ""
    for entry in conversation_history[from_number]:
        formatted_history += f"User: {entry['user']}\nBot: {entry['response']}\n"
    return formatted_history

def get_limited_context(from_number):
    """
    Get the conversation context limited by the maximum token count.
    """
    context = format_conversation_history(from_number)
    tokens = context.split()
    if len(tokens) > MAX_TOKENS:
        context = " ".join(tokens[-MAX_TOKENS:])
    return context


def format_doctor_recommendations(df):
    # Remove duplicates based on telephone number
    df_unique = df.drop_duplicates(subset='telephone')

    recommendations = "Here are some gynecologists/oncologists I found for you:\n\n"

    for index, row in df_unique.iterrows():
        recommendations += f"👨‍⚕️ *{row['name']}* (Specialization: {row['speciality']})\n"
        recommendations += f"📍 Location: {row['address']}\n"
        recommendations += f"💰 Fees: ₹{row['fees']}\n"
        recommendations += f"📅 Experience: {row['experience']} years\n"
        recommendations += f"📞 Contact: {row['telephone']}\n"
        recommendations += "----------------------------\n"

    return recommendations

@app.route('/whatsapp', methods=['POST'])
def whatsapp_reply():
    incoming_msg = request.values.get('Body', '').strip().lower()
    from_number = request.values.get('From', '')
    from_number = from_number[-10:]
    print(incoming_msg)

    response = MessagingResponse()
    if incoming_msg == "generate report":
        # Respond with an acknowledgment and trigger the PDF generation
        # response.message("Your report is being generated. You will receive it shortly.")
        send_whatsapp_message()  # Call the function to generate and send the PDF
    else:
        global MAX_TOKENS, conversation_history
        context = get_limited_context(from_number)
        report = open(f'{from_number}.txt', 'r').read()

        # Define the function schema
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "recommend_doctors",
                    "description": "Recommend doctors when user will ask.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the user"},
                            "age": {"type": "integer", "description": "Age of the user"},
                            "phone": {"type": "string", "description": "Phone number of the user"},
                            "email": {"type": "string", "description": "Email address of the user"},
                            "time_slot": {"type": "string", "description": "Preferred time slot (e.g. '10:00-11:00'). If someone tells in am or pm, convert automatically to HH:MM-HH:MM format."},
                            "date": {"type": "string", "description": "Preferred date in YYYY-MM-DD format"},
                            "preferred_location": {"type": "string",
                                                   "description": "Preferred location (area or landmark)"},
                            "city": {"type": "string", "description": "City of the user"},
                            "pincode": {"type": "string", "description": "Pincode of the location"},
                            "old": {"type": "string", "description": "Type of doctor."},
                        },
                        "required": ["name", "age", "phone", "email", "time_slot", "date", "preferred_location", "city",
                                     "pincode", "old"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_whatsapp_message",
                    "description": "Send the report to the user again when asked.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]

        print("From ",from_number)
        system_instructions = (
            "You are a knowledgeable medical assistant. You have access to the following patient's medical report (on breast cancer):\n\n"
            f"{report}\n\n"
            "Give answer in the language in which the user asks to the query. If the user ask in bengali or hindi, please answer in that written language. Otherwise always maintain english language. Provide accurate and reliable answers to queries related to medical topics, medications, possible diets or the provided report. Explain medical terms, medicines, or general medical knowledge when asked. However, do not suggest or recommend any specific medications, dosages, or treatments. If someone asks for a specific medication recommendation, politely inform them to consult a doctor. Ensure every response is unique and clear. Decline to answer any questions that are unrelated to the medical field or some vague questions, encouraging the user to focus on health-related queries instead. Answer in short. Write like a professional whatsapp chat. But every whatsapp chat phrasing will be different. Use emoji. Use tool call if asked by the user. If required information is incomplete to do the tool_call, ask the user for details. In case of tool call type of doctor is mandatory, ask user to choose between oncologist(if the report is malignant) and gynecologist(if the report is benign). You can only recommend doctors, but you can't book a consultation directly. If someone tells yoou to book appointment with a doctor, tell him/her to contact directly through phone number provided, don't repeat the list of doctors. If any error is generated anytime, write it politely, not in technical language. When told something other than doctor recommendation don't do tool call strictly. If anytime the tool call fails for any missing or incorrect data, ask the users for it again, don't be quiet."
        )
        prompt = f"{system_instructions}\n\nPrevious context: {context}\n\nCurrent User Message: {incoming_msg}\nBot:"

        model_response = None
        try:
            chat_completion = client_groq.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": prompt},
                ],
                # model="meta-llama/llama-4-scout-17b-16e-instruct",  # Replace with a domain-specific model if available
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                tools=tools,
                tool_choice="auto"
            )
            if chat_completion.choices[0].message.tool_calls:
                tool_call = chat_completion.choices[0].message.tool_calls[0]
                if tool_call.function.name == "recommend_doctors":
                    args = eval(tool_call.function.arguments)
                    print(args)
                    recommended_doctors, others = recommend_doctors(**args)

                    print(recommended_doctors)

                    # Reply after tool execution
                    formatted_recommendations = format_doctor_recommendations(recommended_doctors)
                    model_response = f"{formatted_recommendations}"
                elif tool_call.function.name == "send_whatsapp_message":
                    args = eval(tool_call.function.arguments)
                    print(args)
                    send_whatsapp_message()
            else:
                model_response = chat_completion.choices[0].message.content
            # model_response = chat_completion.choices[0].message.content
        except Exception as e:
            model_response = f"Error generating response: {e}"

        response.message(model_response)
        add_to_history(from_number, incoming_msg, model_response)

    return str(response)


@app.route("/get_response", methods=["POST"])
def get_response():
    pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. Dont give direct conversation, just answer the question"
    prompt_input = request.form.get("prompt_input")


    allowed_words_file = "allowed_words.txt"
    allowed_words = []
    if os.path.isfile(allowed_words_file):
        with open(allowed_words_file, "r") as file:
            allowed_words = [line.strip() for line in file]


    prompt_words = set(prompt_input.lower().split())  # Split prompt into words

    intersection = any(word.lower() in prompt_words for word in allowed_words)

    if not intersection:
        return jsonify({"response": "Invalid prompt. Please use an allowed word."})

    # Generate LLM response
    output = replicate.run(
        'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
        input={
            "prompt": f"{pre_prompt} {prompt_input} Assistant: ",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_length": 1000,
            "repetition_penalty": 1
        }
    )

    full_response = ""

    for item in output:
        full_response += item

    return jsonify({"response": full_response})


# Function to geocode the address and return latitude and longitude
def geocode_address(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None  # Return None if the address can't be geocoded
    except Exception as e:
        print(f"Error geocoding address {address}: {e}")
        return None, None

# Function to calculate the distance between two lat/lon points
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers


def recommend_doctors(name, age, phone, email, time_slot, date, preferred_location, city, pincode, old=False):
    # # Read the CSV file
    data = pd.read_csv('data.csv')

    # Ensure proper conversion to datetime and time
    data['start_time'] = pd.to_datetime(data['start_time'], format='%H:%M', errors='coerce').dt.time
    data['end_time'] = pd.to_datetime(data['end_time'], format='%H:%M', errors='coerce').dt.time

    # Initialize the geocoder
    geolocator = Nominatim(user_agent="doctor_locator")

    # Take user input for the location
    user_location = preferred_location + " " + city

    # If the user enters coordinates, split them into lat and lon
    user_lat, user_lon = geocode_address(user_location)
    # print(user_lat, user_lon)
    if user_lat is None or user_lon is None:
        print("Could not geocode the provided location.")
        exit()

    # Initialize a list to store distances
    distances = []

    # Loop through each row in the DataFrame and calculate the distance to the user's location
    for index, row in data.iterrows():
        doctor_lat = row['Latitude']
        doctor_lon = row['Longitude']

        # If latitude or longitude is None, append a distance of None
        if doctor_lat is None or doctor_lon is None:
            distances.append(None)
        else:
            distance = calculate_distance(user_lat, user_lon, doctor_lat, doctor_lon)
            distances.append(distance)

    # Add the distances as a new column to the DataFrame
    data['distance_from_user'] = distances

    print(old)
    if old is not None:
        # Ask user for speciality
        selected_speciality = old

        # Filter data based on speciality
        data = data[data['speciality'].str.lower() == selected_speciality]

    # Assuming filtered_data['date'] contains weekday names (e.g., 'Monday', 'Tuesday', etc.)
    weekday_mapping = {
        'monday': 0,
        'tuesday': 1,
        'wednesday': 2,
        'thursday': 3,
        'friday': 4,
        'saturday': 5,
        'sunday': 6
    }

    # Ask user for preferred date and time range
    print(date)
    preferred_date = str(date).strip()

    print(str(time_slot).split('-'))
    preferred_start_time = str(time_slot).split('-')[0].strip()
    preferred_end_time = str(time_slot).split('-')[1].strip()

    # Convert preferred date and times to proper formats
    preferred_start_time = datetime.strptime(preferred_start_time, '%H:%M').time()
    preferred_end_time = datetime.strptime(preferred_end_time, '%H:%M').time()
    preferred_date_obj = datetime.strptime(preferred_date, '%Y-%m-%d')
    preferred_day_of_week = preferred_date_obj.weekday()

    print(type(data['start_time'].iloc[0]))
    print(type(data['end_time'].iloc[0]))
    print(type(preferred_start_time))
    print(type(preferred_end_time))

    # Filter data based on date and time range
    data['day_of_week'] = data['date'].str.lower().map(weekday_mapping)

    # print(data)
    print(preferred_day_of_week)
    # Check for at least 1-hour overlap between user preferred time and doctor's available time
    matched_doctors = data[
        (data['day_of_week'] == preferred_day_of_week) &
        (
            # Condition for the time overlap
                (data['start_time'] < preferred_end_time) &
                (data['end_time'] > preferred_start_time)
        )
        ]

    if len(matched_doctors) >= 10:
        print("\nDoctors matching your preferences:")
        print(len(matched_doctors))
    else:
        print(len(matched_doctors))
        print("\nFewer than 10 doctors match your preferences. Finding additional recommendations...")

        # Exclude matched doctors from KNN data
        unmatched_data = data.drop(matched_doctors.index)

        # Prepare data for k-nearest neighbors
        unmatched_data['day_diff'] = abs(
            unmatched_data['day_of_week'] - preferred_day_of_week) * 1440  # Difference in days in minutes
        unmatched_data['time_diff'] = abs((unmatched_data['start_time'].apply(lambda t: t.hour * 60 + t.minute) -
                                           (preferred_start_time.hour * 60 + preferred_start_time.minute)))

        # K-Nearest Neighbors: Using day_diff and time_diff as features
        knn_data = unmatched_data[['day_diff', 'time_diff']].to_numpy()
        knn = NearestNeighbors(n_neighbors=min(10 - len(matched_doctors),unmatched_data.shape[0]))
        knn.fit(knn_data)

        # Query for closest matches
        distances, indices = knn.kneighbors([[0, 0]])  # Closest to preferred date/time

        # Get additional doctors
        additional_doctors = unmatched_data.iloc[indices[0]]
        # print("\nAdditional recommended doctors:")
        # print(additional_doctors)

    # Combine matched and additional doctors if needed
    final_recommendations = pd.concat([matched_doctors, additional_doctors])
    print("\nFinal doctor recommendations:")
    print(final_recommendations)

    if (len(matched_doctors)) < 10:
        # Convert the relevant data to a NumPy array
        data1 = additional_doctors[['distance_from_user', 'rating', 'experience']].to_numpy()

        # Problem definition
        class DoctorSelectionProblem(Problem):
            def __init__(self):
                super().__init__(n_var=1, n_obj=3, n_constr=0, xl=0, xu=len(data1) - 1, type_var=int)

            def _evaluate(self, x, out, *args, **kwargs):
                idx = x[:, 0].astype(int)
                distances = data1[idx, 0]
                ratings = -data1[idx, 1]  # Maximize by minimizing negative ratings
                experiences = -data1[idx, 2]  # Maximize by minimizing negative experiences
                out["F"] = np.column_stack([distances, ratings, experiences])

        # NSGA-II Algorithm
        algorithm = NSGA2(pop_size=len(additional_doctors))

        # Optimize the problem
        problem = DoctorSelectionProblem()
        res = minimize(problem, algorithm, termination=('n_gen', 50), verbose=False)

        # Handle duplicate outputs
        unique_solutions = np.unique(res.X[:, 0].astype(int))

        # Print selected doctors from the DataFrame
        print("\nSelected Doctors based on Pareto Front optimization:")
        selected_doctors = additional_doctors.iloc[unique_solutions]
        selected_doctors = selected_doctors.sort_values(by='distance_from_user', ascending=True)
        matched_doctors = matched_doctors.sort_values(by='distance_from_user', ascending=True)
        final_doctor_list = pd.concat([matched_doctors, selected_doctors])

    else:
        # Convert the relevant data to a NumPy array
        data1 = matched_doctors[['distance_from_user', 'rating', 'experience']].to_numpy()

        # Problem definition
        class DoctorSelectionProblem(Problem):
            def __init__(self):
                super().__init__(n_var=1, n_obj=3, n_constr=0, xl=0, xu=len(data1) - 1, type_var=int)

            def _evaluate(self, x, out, *args, **kwargs):
                idx = x[:, 0].astype(int)
                distances = data1[idx, 0]
                ratings = -data1[idx, 1]  # Maximize by minimizing negative ratings
                experiences = -data1[idx, 2]  # Maximize by minimizing negative experiences
                out["F"] = np.column_stack([distances, ratings, experiences])

        # NSGA-II Algorithm
        algorithm = NSGA2(pop_size=len(matched_doctors))

        # Optimize the problem
        problem = DoctorSelectionProblem()
        res = minimize(problem, algorithm, termination=('n_gen', 50), verbose=False)

        # Handle duplicate outputs
        unique_solutions = np.unique(res.X[:, 0].astype(int))

        # Print selected doctors from the DataFrame
        print("\nSelected Doctors based on Pareto Front optimization:")
        selected_doctors = matched_doctors.iloc[unique_solutions]
        selected_doctors = selected_doctors.sort_values(by='distance_from_user', ascending=True)
        final_doctor_list = selected_doctors

    print("\nFinal doctor list:")
    return final_doctor_list, data.drop(final_doctor_list.index).sort_values(by='distance_from_user', ascending=True)

# Example route to book an appointment
@app.route('/submit_appointment', methods=['GET', 'POST'])
def submit_appointment():
    global extra_doctors
    # List of doctors (this can be dynamically fetched from a database)

    # If the form is submitted
    if request.method == 'POST':
        # Capture form data
        name = request.form.get('name')
        age = request.form.get('age')
        phone = request.form.get('phone')
        email = request.form.get('email')
        time_slot = request.form.get('time_slot')
        appointment_date = request.form.get('appointment_date')
        preferred_location = request.form.get('preferred_location')
        city = request.form.get('city')
        pincode = request.form.get('pincode')

        print(name, " ", age, " ", phone, " ", email, " ", time_slot, " ", appointment_date, " ", preferred_location, " ", city, " ", pincode)

        # Connect to the database
        conn = psycopg2.connect(
            dbname='QCD',
            user='postgres',
            password='Snigdha@2003',
            host='localhost',
            port='5432'
        )
        cur = conn.cursor()

        # Define the phone number for which to find the latest record
        target_phone_number = "0"+phone  # Replace with the desired phone number

        # Query the latest record for the given phone number based on the current_datetime column
        cur.execute("""
            SELECT * FROM REPORT_DETAILS
            WHERE phone = %s
            ORDER BY current_datetime DESC
            LIMIT 1
        """, (target_phone_number,))

        # Fetch the result
        latest_record = cur.fetchone()

        # Print the result
        if latest_record:
            print("Latest record for phone number", target_phone_number, "is:")
            print(latest_record[6])
        else:
            print("No record found for phone number", target_phone_number)

        old = None
        if latest_record[6] == 'n':
            old = 'gynecologist'
        else:
            old = 'oncologist'

        # Close the cursor and connection
        cur.close()
        conn.close()

    doctors_list, extra_doctors = recommend_doctors(name, age, phone, email, time_slot, appointment_date, preferred_location, city,
                                                    pincode, old)
    doctors = doctors_list[['name', 'specialization', 'address', 'fees', 'date', 'start_time', 'end_time', 'telephone']].to_dict(orient='records')

    extra_doctors = extra_doctors[
        ['name', 'specialization', 'address', 'fees', 'date', 'start_time', 'end_time', 'telephone']].to_dict(
        orient='records')

    # Render the template and pass the doctors list
    print(doctors)
    return render_template('Book_Appointment.html', doctors=doctors)

def serialize_doctor_data(doctors):
    for doctor in doctors:
        doctor['start_time'] = doctor['start_time'].strftime('%H:%M:%S')
        doctor['end_time'] = doctor['end_time'].strftime('%H:%M:%S')
    return doctors

@app.route('/get_extra_doctors', methods=['GET'])
def get_extra_doctors():
    global extra_doctors
    return jsonify(serialize_doctor_data(extra_doctors))

if __name__ == '__main__':
    port = os.environ.get('PORT', 8008)

    if "prepare" not in sys.argv:
        app.run(debug=True, host='0.0.0.0', port=port)
