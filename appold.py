import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
# import spacy
import json
import re
import string
import whisper
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import hashlib
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from typing import Optional
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from langchain_experimental.agents import create_csv_agent
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import cv2
import time
import tensorflow as tf
import numpy as np
import os
import cv2
import mediapipe as mp
from cv2.typing import MatLike
import av
import threading
from queue import Queue
from typing import Optional, Tuple
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from pathlib import Path
import traceback
import warnings
warnings.filterwarnings("ignore")
# Load environment variables
load_dotenv(".env")

gemini_client = genai.Client()

# Configure page
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Dark Theme CSS
st.markdown("""
<style>
    /* Global Styles */
    :root {
        --primary-bg: #0E1117;
        --secondary-bg: #1E1E2E;
        --card-bg: #262730;
        --accent-purple: #9D4EDD;
        --accent-pink: #F72585;
        --accent-blue: #4CC9F0;
        --text-primary: #FFFFFF;
        --text-secondary: #B8B8B8;
        --success: #06FFA5;
        --warning: #FFB627;
        --border-color: #3D3D4E;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Header */
    .main-header {
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-pink) 50%, var(--accent-blue) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 2rem 0;
        padding: 1rem;
        animation: fadeInDown 0.8s ease-out;
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(157, 78, 221, 0.15) 0%, rgba(76, 201, 240, 0.15) 100%);
        backdrop-filter: blur(10px);
        padding: clamp(1rem, 3vw, 2rem);
        border-radius: 20px;
        border: 1px solid rgba(157, 78, 221, 0.3);
        color: var(--text-primary);
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(157, 78, 221, 0.2);
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px 0 rgba(157, 78, 221, 0.4);
        border-color: var(--accent-purple);
    }

    .feature-card h3, .feature-card h4 {
        color: var(--text-primary);
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .feature-card p {
        color: var(--text-secondary);
        line-height: 1.6;
        font-size: clamp(0.9rem, 2vw, 1rem);
    }

    /* Emotion Result Card */
    .emotion-result {
        background: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-pink) 100%);
        padding: clamp(1.5rem, 4vw, 2.5rem);
        border-radius: 25px;
        text-align: center;
        color: var(--text-primary);
        margin: 1rem 0;
        box-shadow: 0 15px 50px rgba(157, 78, 221, 0.4);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 15px 50px rgba(157, 78, 221, 0.4);
        }
        50% {
            box-shadow: 0 15px 60px rgba(157, 78, 221, 0.6);
        }
    }

    .emotion-result h2 {
        font-size: clamp(1.8rem, 4vw, 2.5rem);
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .emotion-result h3 {
        font-size: clamp(1.2rem, 3vw, 1.5rem);
        opacity: 0.9;
    }

    .emotion-result p {
        font-size: clamp(1rem, 2.5vw, 1.3rem);
        opacity: 0.95;
    }

    /* Login Container */
    .login-container {
        background: rgba(30, 30, 46, 0.8);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: clamp(1.5rem, 4vw, 3rem);
        border: 1px solid rgba(157, 78, 221, 0.3);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.5);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E1E2E 0%, #0E1117 100%);
        border-right: 1px solid var(--border-color);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-primary);
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-pink) 100%);
        color: var(--text-primary);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: clamp(0.9rem, 2vw, 1rem);
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(157, 78, 221, 0.3);
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(157, 78, 221, 0.5);
        background: linear-gradient(135deg, var(--accent-pink) 0%, var(--accent-purple) 100%);
    }

    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: var(--secondary-bg);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 0.75rem;
        font-size: clamp(0.9rem, 2vw, 1rem);
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-purple);
        box-shadow: 0 0 0 2px rgba(157, 78, 221, 0.2);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--secondary-bg);
        color: var(--text-secondary);
        border-radius: 12px 12px 0 0;
        padding: 0.75rem 1.5rem;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-pink) 100%);
        color: var(--text-primary);
        border-color: var(--accent-purple);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--text-primary);
        font-size: clamp(1.5rem, 3vw, 2rem);
        font-weight: 700;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-size: clamp(0.9rem, 2vw, 1rem);
    }

    /* DataFrames */
    .stDataFrame {
        background-color: var(--secondary-bg);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Info/Warning/Success Messages */
    .stAlert {
        background-color: var(--secondary-bg);
        border-radius: 12px;
        border-left: 4px solid var(--accent-purple);
        padding: 1rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--secondary-bg);
        color: var(--text-primary);
        border-radius: 12px;
        font-weight: 600;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: var(--secondary-bg);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid var(--border-color);
    }

    /* Spinner */
    .stSpinner > div {
        border-color: var(--accent-purple) transparent transparent transparent;
    }

    /* Radio Buttons */
    .stRadio > div {
        background-color: transparent;
    }

    .stRadio [role="radiogroup"] label {
        background-color: var(--secondary-bg);
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin: 0.25rem 0;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        color: var(--text-primary);
    }

    .stRadio [role="radiogroup"] label:hover {
        border-color: var(--accent-purple);
        background-color: rgba(157, 78, 221, 0.1);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .feature-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }

        .emotion-result {
            padding: 1.5rem;
        }

        .stButton > button {
            padding: 0.6rem 1rem;
        }
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--primary-bg);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink));
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--accent-pink), var(--accent-purple));
    }

    /* Custom Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(157, 78, 221, 0.1) 0%, rgba(76, 201, 240, 0.1) 100%);
        border: 1px solid rgba(157, 78, 221, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(157, 78, 221, 0.3);
    }

    /* Welcome Message */
    .welcome-message {
        background: linear-gradient(135deg, rgba(157, 78, 221, 0.2) 0%, rgba(76, 201, 240, 0.2) 100%);
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(157, 78, 221, 0.3);
        color: var(--text-primary);
    }
</style>
""", unsafe_allow_html=True)

# MongoDB Connection
@st.cache_resource
def init_mongodb():
    try:
        client = MongoClient(os.environ.get("MONGODB_URI"), server_api=ServerApi('1'))
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

client = init_mongodb()

@st.cache_resource
def load_model(model_path="sign_language_model.keras"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

class Landmarker:

    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.75,
        min_tracking_confidence: float = 0.75,
        max_num_hands: int = 1,
    ):
        self.model = mp.solutions.hands.Hands(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=max_num_hands,
        )

    def draw_landmarks(self, image: MatLike):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.model.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not results.multi_hand_landmarks:
            return False, image, None, None, None

        for landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(
                image,
                landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=8, circle_radius=8
                ),
                drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=6, circle_radius=2
                ),
            )

        hand = results.multi_hand_landmarks[0]
        points = self.normalize_points(
            np.array(
                [(landmark.x, landmark.y, landmark.z) for landmark in hand.landmark]
            )
        )
        handedness = results.multi_handedness[0].classification[0].label.lower()

        return (
            True,
            image,
            points,
            (hand.landmark[0].x, hand.landmark[0].y),
            handedness,
        )

    def normalize_points(self, points):
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        for i in range(len(points)):
            points[i][0] = (points[i][0] - min_x) / (max_x - min_x)
            points[i][1] = (points[i][1] - min_y) / (max_y - min_y)

        points = np.expand_dims(points, axis=0)

        return points

class SignLanguageHelper:
    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.75,
        min_tracking_confidence: float = 0.75,
        max_num_hands: int = 1,
    ):
        self.landmarker = Landmarker(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=max_num_hands,
        )
        self.classifier = Classifier()

    def process_frame(self, image):
        success, processed_image, points, hand_position, handedness = \
            self.landmarker.draw_landmarks(image)

        if not success:
            return False, processed_image, None, None, None

        letter, probability = self.classifier.classify(points)

        return True, processed_image, letter, probability, handedness

class Classifier:
    def __init__(self, model_path="sign_language_model.keras"):
        self.model = load_model(model_path)

    def classify(self, points):
        predictions = self.model.predict(points[:, :, :2], verbose=0)
        prediction = np.argmax(predictions, axis=-1)
        probability = predictions[0][prediction[0]]
        letter = LETTERS[prediction[0]]
        return letter, probability

# Configure WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

TEMP_PREDICTIONS_FILE = "temp_sign_predictions.txt"

def save_prediction_to_file(letter):
    try:
        with open(TEMP_PREDICTIONS_FILE, 'a+') as f:
            f.write(letter)
            f.flush()
            os.fsync(f.fileno())
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

def load_predictions_from_file():
    try:
        if os.path.exists(TEMP_PREDICTIONS_FILE):
            with open(TEMP_PREDICTIONS_FILE, 'r') as f:
                text = f.read().strip()
            return text
        return ""
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return ""

def delete_predictions_file():
    try:
        if os.path.exists(TEMP_PREDICTIONS_FILE):
            os.remove(TEMP_PREDICTIONS_FILE)
            return True
    except Exception as e:
        print(f"Error deleting predictions file: {e}")
    return False

def initialize_session_state():
    defaults = {
        'camera_started': False,
        'predictions_history': [],
        'sign_helper': None,
        'last_capture_time': time.time(),
        'capture_interval': 5,
        'current_letter': None,
        'current_confidence': 0.0,
        'show_emotion_result': False,
        'current_emotion3': None,
        'current_text3': "",
        'current_probability3': 0.0,
        'edited_text': "",
        'captured_text': ""
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

class VideoProcessor:
    def __init__(self):
        self.sign_helper = None
        self.last_capture_time = time.time()
        self.capture_interval = 10
        self.last_saved_letter = None

    def recv(self, frame):
        file_path = "charcter_to_word_mapping.json"
        with open(file_path, "r") as json_file:
            charcter_to_word_mapping = json.load(json_file)

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        if self.sign_helper is None:
            try:
                self.sign_helper = SignLanguageHelper(
                    model_complexity=0,
                    min_detection_confidence=0.75,
                    min_tracking_confidence=0.75,
                    max_num_hands=1,
                )
            except Exception as e:
                print("Helper initialization failed:", e)
                traceback.print_exc()
                cv2.putText(img, "Helper initialization failed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            success, processed_frame, letter, probability, handedness = \
                self.sign_helper.process_frame(img)

            letter = charcter_to_word_mapping.get(letter,'') + ' '

            current_time = time.time()
            elapsed_time = current_time - self.last_capture_time
            time_remaining = max(0, self.capture_interval - elapsed_time)

            if elapsed_time >= self.capture_interval and success:
                capture_data = {
                    'letter': letter,
                    'confidence': probability,
                    'hand': handedness,
                    'timestamp': time.strftime("%H:%M:%S")
                }

                if 'predictions_history' not in st.session_state:
                    st.session_state.predictions_history = []
                st.session_state.predictions_history.append(capture_data)

                save_prediction_to_file(letter)

                st.session_state.current_letter = letter
                st.session_state.current_confidence = probability
                self.last_saved_letter = letter

                self.last_capture_time = current_time

            self._draw_overlays(processed_frame, success, letter, probability,
                              handedness, time_remaining)

            return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

        except Exception as e:
            cv2.putText(img, f"Error: {str(e)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _draw_overlays(self, frame, success, letter, probability, hand, time_remaining):
        height, width = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if success:
            text = f"Current: {letter} ({probability:.1%})"
            cv2.putText(frame, text, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            hand_text = f"Hand: {hand}"
            cv2.putText(frame, hand_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            text = "Show hand sign"
            cv2.putText(frame, text, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2)

        timer_text = f"Next: {time_remaining:.1f}s"
        text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(frame, timer_text, (width - text_size[0] - 10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        overlay_bottom = frame.copy()
        cv2.rectangle(overlay_bottom, (0, height - 80), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay_bottom, 0.7, frame, 0.3, 0, frame)

        captured_text = load_predictions_from_file()

        if captured_text:
            if len(captured_text) > 1:
                prev_text = f"Previous: {captured_text[:-1]}"
                cv2.putText(frame, prev_text, (10, height - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

            latest_text = f"Latest Captured: {captured_text[-1]}"
            cv2.putText(frame, latest_text, (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            full_text = f"Text: {captured_text}"
            full_text_size = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(frame, full_text, (width - full_text_size[0] - 10, height - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            no_capture_text = "Captured: (none yet)"
            cv2.putText(frame, no_capture_text, (10, height - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

def emotion_from_sign_language():
    initialize_session_state()

    st.markdown("### 🤟 Emotion Detection from Sign Language")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)

        st.write("Use sign language to express your feelings:")

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("📹 Start Camera", type="primary",
                        disabled=st.session_state.camera_started,
                        key="start_camera_btn"):
                st.session_state.camera_started = True
                st.session_state.predictions_history = []
                st.session_state.show_emotion_result = False
                st.session_state.last_capture_time = time.time()
                st.session_state.captured_text = ""
                st.session_state.edited_text = ""
                delete_predictions_file()
                st.rerun()

        with col_btn2:
            if st.button("🛑 Stop Camera", type="secondary",
                        disabled=not st.session_state.camera_started,
                        key="stop_camera_btn"):
                st.session_state.camera_started = False
                captured_text = load_predictions_from_file()
                st.session_state.captured_text = captured_text
                st.session_state.edited_text = captured_text
                st.rerun()

        if st.session_state.camera_started:
            st.info("📸 Camera is active. Capturing predictions every 5 seconds...")

            webrtc_ctx = webrtc_streamer(
                key="sign-language-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if st.session_state.predictions_history:
                st.markdown("---")
                st.markdown("### 📝 Captured Letters")

                captured_text = ''.join([pred['letter'] for pred in st.session_state.predictions_history])
                st.markdown(f"### **Text so far:** `{captured_text}`")

                if st.session_state.current_letter:
                    st.success(f"✅ Latest: **{st.session_state.current_letter}** ({st.session_state.current_confidence:.1%})")

                with st.expander("📋 View All Predictions", expanded=False):
                    for idx, pred in enumerate(st.session_state.predictions_history):
                        col_p1, col_p2, col_p3 = st.columns([2, 2, 2])

                        with col_p1:
                            st.write(f"**{idx + 1}. {pred['letter']}**")
                        with col_p2:
                            st.write(f"Confidence: {pred['confidence']:.1%}")
                        with col_p3:
                            st.write(f"Time: {pred['timestamp']}")
            else:
                st.info("👆 Show hand signs to start capturing letters...")

        elif not st.session_state.camera_started and st.session_state.captured_text:
            st.markdown("---")
            st.markdown("### ✏️ Review and Edit Captured Text")

            st.success(f"✅ Captured {len(st.session_state.captured_text)} letters successfully!")

            if st.session_state.predictions_history:
                with st.expander("📋 View Captured Details", expanded=False):
                    for idx, pred in enumerate(st.session_state.predictions_history):
                        col_p1, col_p2, col_p3 = st.columns([2, 2, 2])

                        with col_p1:
                            st.write(f"**{idx + 1}. {pred['letter']}**")
                        with col_p2:
                            st.write(f"Confidence: {pred['confidence']:.1%}")
                        with col_p3:
                            st.write(f"Time: {pred['timestamp']}")

            st.markdown("---")

            st.markdown("**Edit the text below if needed:**")
            edited_text = st.text_area(
                label="Captured Text",
                value=st.session_state.edited_text,
                height=120,
                key="sign_text_editor",
                help="You can edit the text before analyzing emotion"
            )

            st.session_state.edited_text = edited_text

            col_analyze, col_restart = st.columns([3, 1])

            with col_analyze:
                if st.button("🔍 Analyze Emotion", type="primary", key="analyze_emotion_btn", use_container_width=True):
                    if edited_text.strip():
                        with st.spinner("Analyzing emotion..."):
                            emotion, probability = detect_emotion(edited_text)

                            st.session_state.current_emotion3 = emotion
                            st.session_state.current_text3 = edited_text
                            st.session_state.current_probability3 = probability
                            st.session_state.show_emotion_result = True
                            st.rerun()
                    else:
                        st.warning("⚠️ Please enter some text to analyze!")

            with col_restart:
                if st.button("🔄 Restart", type="secondary", key="restart_capture_btn", use_container_width=True):
                    st.session_state.predictions_history = []
                    st.session_state.edited_text = ""
                    st.session_state.captured_text = ""
                    st.session_state.camera_started = True
                    st.session_state.show_emotion_result = False
                    delete_predictions_file()
                    st.rerun()

        elif not st.session_state.camera_started and not st.session_state.captured_text:
            captured_text = load_predictions_from_file()
            if captured_text:
                st.session_state.captured_text = captured_text
                st.session_state.edited_text = captured_text
                st.rerun()
            else:
                st.info("👆 Click 'Start Camera' to begin capturing sign language letters.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if st.session_state.show_emotion_result and st.session_state.current_emotion3:
            st.markdown(f'''
            <div class="emotion-result">
                <h3>Detected Emotion</h3>
                <h2>🎭 {st.session_state.current_emotion3.upper()}</h2>
                <p>Confidence: {st.session_state.current_probability3:.1f}%</p>
            </div>
            ''', unsafe_allow_html=True)

    if st.session_state.show_emotion_result and st.session_state.current_emotion3:
        st.markdown("---")
        st.subheader("💭 Share your thoughts")

        comment = st.text_input(
            "Why do you think you're feeling this way?",
            placeholder="Optional: Share what's on your mind...",
            key="sign_comment"
        )

        if st.button("💾 Save to History", type="secondary", key="save_sign_emotion"):
            if save_emotion_data(
                st.session_state.username,
                st.session_state.current_emotion3,
                st.session_state.current_text3,
                comment,
                "sign_language"
            ):
                st.success("✅ Data saved successfully! 🎉")
                delete_predictions_file()
                st.session_state.predictions_history = []
                st.session_state.show_emotion_result = False
                st.session_state.edited_text = ""
                st.session_state.captured_text = ""
                st.session_state.current_letter = None
                st.session_state.current_confidence = 0.0
            else:
                st.error("❌ Error saving data. Please try again.")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    if not client:
        return False

    db = client["EmotionDataDB"]
    collection = db["UserCreds"]

    user = collection.find_one({"username": username})
    if user and user["password"] == hash_password(password):
        return True
    return False

def create_user(username, password, email):
    if not client:
        return False

    db = client["EmotionDataDB"]
    collection = db["UserCreds"]

    if collection.find_one({"username": username}):
        return False

    user_doc = {
        "username": username,
        "password": hash_password(password),
        "email": email,
        "created_date": datetime.now().strftime("%d-%m-%Y")
    }

    try:
        collection.insert_one(user_doc)
        return True
    except Exception as e:
        st.error(f"Error creating user: {e}")
        return False

@st.cache_resource
def load_models():
    return None, None, None

pipeline, label_encoder, nlp = load_models()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if nlp:
        doc = nlp(text)
        processed_tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and token.lemma_ != '-PRON-'
        ]
        return " ".join(processed_tokens).strip()
    return text

def predict_class(text):
    if pipeline and label_encoder:
        processed_text = preprocess_text(text)
        prediction = pipeline.predict([processed_text])
        prediction = label_encoder.inverse_transform(prediction)[0]
        prediction_prob = float(round(np.max(pipeline.predict_proba([processed_text])[0]),3)) * 100
        return prediction, prediction_prob
    return "Error", 0

@st.cache_resource
def load_whisper_model():
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

whisper_model = load_whisper_model()

def save_emotion_data(username, emotion, input_text, comment, data_type):
    if not client:
        return False

    db = client["EmotionDataDB"]
    collection = db["UserHistory"]

    document = {
        "username": username,
        "Detected Emotion": emotion,
        "Input text": input_text,
        "Comment": comment,
        "Date": datetime.now().strftime("%d-%m-%Y"),
        "Type": data_type
    }

    try:
        collection.insert_one(document)
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def get_user_history_data(username):
    if not client:
        return pd.DataFrame()

    db = client["EmotionDataDB"]
    collection = db["UserHistory"]

    cursor = collection.find({"username": username})
    data = list(cursor)

    for doc in data:
        doc.pop("_id", None)

    return pd.DataFrame(data)

class CSVQueryInput(BaseModel):
    query: str = Field(description="Natural language question about the user history data")

class CSVAgentTool(BaseTool):
    name: str = "user_history_query"
    description: str = (
        "Useful for answering any questions based on user history stored in the system. "
        "Can analyze user behavior patterns, emotions, activities, preferences, "
        "and provide insights from historical user data."
    )
    args_schema: Optional[ArgsSchema] = CSVQueryInput
    return_direct: bool = True

    llm: object = Field(exclude=True)
    csv_file_path: str = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            agent_executor = create_csv_agent(
                self.llm,
                self.csv_file_path,
                allow_dangerous_code=True,
                verbose=True
            )

            response = agent_executor.invoke({"input": query})
            return response['output']

        except Exception as e:
            return f"Error processing user history query: {str(e)}"

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(query, run_manager=run_manager.get_sync() if run_manager else None)

@st.cache_resource
def setup_chatbot():
    try:
        tavily_tool = TavilySearch(max_results=5, topic="general")

        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        class ChatState(TypedDict):
            messages: Annotated[list[BaseMessage], add_messages]

        user_history_tool = CSVAgentTool(llm = gemini_llm, csv_file_path="user_data.csv")
        tools = [tavily_tool, user_history_tool]
        llm_with_tools = gemini_llm.bind_tools(tools)

        def chat_node(state: ChatState):
            messages = state['messages']
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        tool_node = ToolNode(tools)

        graph = StateGraph(ChatState)
        graph.add_node("chat_node", chat_node)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "chat_node")
        graph.add_conditional_edges("chat_node", tools_condition)
        graph.add_edge("tools", "chat_node")

        checkpointer = InMemorySaver()
        chatbot = graph.compile(checkpointer=checkpointer)

        return chatbot
    except Exception as e:
        st.error(f"Error setting up chatbot: {e}")
        return None

chatbot = setup_chatbot()

def login_page():
    st.markdown('<h1 class="main-header">🎭 Emotion Detection App</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # st.markdown('<div class="login-container">', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            st.subheader("Welcome Back! 👋")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login", key="login_btn", type="primary", use_container_width=True):
                if username and password:
                    if authenticate_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("Login successful! 🎉")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")

        with tab2:
            st.subheader("Create Account 🆕")
            new_username = st.text_input("Username", key="signup_username")
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")

            if st.button("Sign Up", key="signup_btn", type="primary", use_container_width=True):
                if new_username and new_email and new_password and confirm_password:
                    if new_password == confirm_password:
                        if create_user(new_username, new_password, new_email):
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error("Username already exists or error creating account")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.warning("Please fill all fields")

        st.markdown('</div>', unsafe_allow_html=True)

class EmotionResult(BaseModel):
    emotion: str
    probability: int

def detect_emotion(text: str) -> EmotionResult:
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"Text: {text}",
        config={
            "response_mime_type": "application/json",
            "response_schema": EmotionResult,
            "system_instruction": (
                "You are an emotion detection model. "
                "Classify the main emotion present in the given text into one of: "
                "['anger','happiness','hate','love','sadness','worry'] along with probability score (0-100)"
            ),
        },
    )
    return response.parsed.model_dump()['emotion'], response.parsed.model_dump()['probability']

def emotion_from_text():
    st.markdown("### 📝 Emotion Detection from Text")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.write("Enter your text below and discover the emotional tone:")

        user_text = st.text_area("Your Text", height=150, placeholder="Type your message here...")

        if st.button("🔍 Analyze Emotion", type="primary", use_container_width=True):
            if user_text.strip():
                with st.spinner("Analyzing emotion..."):
                    emotion, probability = detect_emotion(user_text)

                    st.session_state.current_emotion = emotion
                    st.session_state.current_text = user_text
                    st.session_state.current_probability = probability

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if hasattr(st.session_state, 'current_emotion'):
            st.markdown(f'''
            <div class="emotion-result">
                <h3>Detected Emotion</h3>
                <h2>🎭 {st.session_state.current_emotion.upper()}</h2>
                <p>Confidence: {st.session_state.current_probability:.1f}%</p>
            </div>
            ''', unsafe_allow_html=True)

    if hasattr(st.session_state, 'current_emotion'):
        st.markdown("---")
        st.subheader("💭 Share your thoughts")
        comment = st.text_input("Why do you think you're feeling this way?",
                               placeholder="Optional: Share what's on your mind...")

        if st.button("💾 Save to History", type="secondary"):
            if save_emotion_data(st.session_state.username,
                               st.session_state.current_emotion,
                               st.session_state.current_text,
                               comment, "text"):
                st.success("Data saved successfully! 🎉")
            else:
                st.error("Error saving data")

def emotion_from_audio():
    st.markdown("### 🎤 Emotion Detection from Audio")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.write("Record audio to analyze its emotional content:")

        audio_bytes = st.audio_input("Record your audio")

        if audio_bytes and whisper_model:
            if st.button("🎵 Transcribe Audio", type="primary"):
                with st.spinner("Transcribing audio..."):
                    with open("temp_audio.wav", "wb") as f:
                        f.write(audio_bytes.getvalue())

                    result = whisper_model.transcribe("temp_audio.wav")
                    transcribed_text = result["text"]

                    st.session_state.transcribed_text = transcribed_text
                    st.success("Transcription complete!")

                    os.remove("temp_audio.wav")

        if hasattr(st.session_state, 'transcribed_text'):
            st.subheader("Transcribed Text:")
            edited_text = st.text_area("Edit if needed:",
                                     value=st.session_state.transcribed_text,
                                     height=100)

            if st.button("🔍 Analyze Emotion", type="primary"):
                with st.spinner("Analyzing emotion..."):
                    emotion, probability = detect_emotion(edited_text)

                    st.session_state.current_emotion2 = emotion
                    st.session_state.current_text2 = edited_text
                    st.session_state.current_probability2 = probability

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if hasattr(st.session_state, 'current_emotion2'):
            st.markdown(f'''
            <div class="emotion-result">
                <h3>Detected Emotion</h3>
                <h2>🎭 {st.session_state.current_emotion2.upper()}</h2>
                <p>Confidence: {st.session_state.current_probability2:.1f}%</p>
            </div>
            ''', unsafe_allow_html=True)

    if hasattr(st.session_state, 'current_emotion2'):
        st.markdown("---")
        st.subheader("💭 Share your thoughts")
        comment = st.text_input("Why do you think you're feeling this way?",
                               placeholder="Optional: Share what's on your mind...")

        if st.button("💾 Save to History", type="secondary"):
            if save_emotion_data(st.session_state.username,
                               st.session_state.current_emotion2,
                               st.session_state.current_text2,
                               comment, "audio"):
                st.success("Data saved successfully! 🎉")
            else:
                st.error("Error saving data")

def user_analytics():
    st.markdown("### 📊 Your Emotion Analytics")

    df = get_user_history_data(st.session_state.username)

    if df.empty:
        st.info("No data available yet. Start by analyzing some text or audio!")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Entries", len(df))

    with col2:
        most_common = df['Detected Emotion'].value_counts().index[0] if not df.empty else "N/A"
        st.metric("Most Common Emotion", most_common)

    with col3:
        text_count = len(df[df['Type'] == 'text'])
        st.metric("Text Analyses", text_count)

    with col4:
        audio_count = len(df[df['Type'] == 'audio'])
        st.metric("Audio Analyses", audio_count)

    st.subheader("📋 History Table")
    st.dataframe(df, use_container_width=True)

    if len(df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            emotion_counts = df['Detected Emotion'].value_counts()
            fig_pie = px.pie(values=emotion_counts.values,
                           names=emotion_counts.index,
                           title="Emotion Distribution",
                           color_discrete_sequence=px.colors.qualitative.Set3)
            fig_pie.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            type_counts = df['Type'].value_counts()
            fig_bar = px.bar(x=type_counts.index,
                           y=type_counts.values,
                           title="Analysis Type Distribution",
                           color=type_counts.values,
                           color_continuous_scale="purples")
            fig_bar.update_layout(
                height=400,
                xaxis_title="Type",
                yaxis_title="Count",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
            daily_emotions = df.groupby(['Date', 'Detected Emotion']).size().reset_index(name='Count')

            fig_timeline = px.line(daily_emotions,
                                 x='Date',
                                 y='Count',
                                 color='Detected Emotion',
                                 title="Emotion Timeline",
                                 markers=True)
            fig_timeline.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

def analyze_user_emotional_patterns(df):
    if df.empty:
        return "No emotional history available."

    try:
        emotion_counts = df['Detected Emotion'].value_counts()
        most_common_emotion = emotion_counts.index[0]
        least_common_emotion = emotion_counts.index[-1]

        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'], format='%d-%m-%Y', errors='coerce')
        recent_data = df_copy[df_copy['Date'] >= (datetime.now() - pd.Timedelta(days=7))]

        total_entries = len(df)
        recent_entries = len(recent_data)

        recent_emotions = recent_data['Detected Emotion'].tolist() if not recent_data.empty else []

        comments = df['Comment'].dropna().tolist()

        analysis = {
            "total_entries": total_entries,
            "most_common_emotion": most_common_emotion,
            "least_common_emotion": least_common_emotion,
            "recent_entries": recent_entries,
            "recent_emotions": recent_emotions,
            "emotion_distribution": emotion_counts.to_dict(),
            "recent_comments": comments[-5:] if comments else [],
            "analysis_types": df['Type'].value_counts().to_dict()
        }

        return analysis

    except Exception as e:
        return f"Error analyzing emotional patterns: {str(e)}"

def create_emotion_context_prompt(username, user_analysis):

    if isinstance(user_analysis, str):
        return f"""You are an empathetic AI assistant talking to {username}.
        This user has not yet shared their emotional data with the system.
        Be supportive and encourage them to explore their emotions."""

    try:
        prompt = f"""You are an empathetic AI emotional wellness assistant talking to {username}.

IMPORTANT USER CONTEXT:
- Use 'user_history_query' tool if user asks any query related to his or her past data. Note that this tool is capable to answer any complex query in natural language.

Your role is to:
1. Provide personalized emotional support based on their history
2. Recognize patterns in their emotions and gently point them out
3. Offer practical advice for emotional wellness
4. Search the web for relevant coping strategies, mental health tips, or resources when appropriate
5. Be empathetic and non-judgmental
6. Encourage healthy emotional habits

If the user asks about their emotions or needs help with emotional wellness, use web search to find:
- Current mental health resources
- Coping strategies for their specific emotional patterns
- Professional help options in their area
- Evidence-based emotional regulation techniques

Always maintain a warm, supportive, and professional tone."""

        return prompt

    except Exception as e:
        return f"""You are an empathetic AI assistant talking to {username}.
        There was an error processing their emotional history: {str(e)}
        Be supportive and encourage them to explore their emotions."""

def chatbot_interface():
    st.markdown("### 🤖 AI Emotional Wellness Assistant")

    if not chatbot:
        st.error("Chatbot is not available. Please check your API keys.")
        return

    df = get_user_history_data(st.session_state.username)
    df.to_csv("user_data.csv",index = False)
    user_analysis = analyze_user_emotional_patterns(df)
    print( user_analysis)

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("#### 📊 Your Emotion Summary")
        if not df.empty:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(157, 78, 221, 0.3) 0%, rgba(76, 201, 240, 0.3) 100%);
                        padding: 1rem; border-radius: 15px; color: white; margin-bottom: 1rem;
                        border: 1px solid rgba(157, 78, 221, 0.4);">
                <strong>Total Analyses:</strong> {user_analysis['total_entries']}<br>
                <strong>Most Common:</strong> {user_analysis['most_common_emotion']}<br>
                <strong>Recent Activity:</strong> {user_analysis['recent_entries']} (last 7 days)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Start analyzing emotions to get personalized insights!")

    with col1:
        st.markdown('''
        <div class="feature-card">
            <h4>🧠 Personalized Emotional Support</h4>
            <p>Your AI assistant knows your emotional patterns and can provide tailored advice,
            search for relevant resources, and help you understand your emotional journey.</p>
        </div>
        ''', unsafe_allow_html=True)

    CONFIG = {'configurable': {'thread_id': st.session_state.username}}

    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []

        if not df.empty:
            context_prompt = create_emotion_context_prompt(st.session_state.username, user_analysis)
            try:
                chatbot.invoke({'messages': [HumanMessage(content=context_prompt)]}, config=CONFIG)
            except Exception as e:
                print(f"Error initializing chatbot context: {e}")
                

    st.markdown("#### 🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Analyze My Patterns", use_container_width=True):
            if not df.empty:
                pattern_query = "Can you analyze my emotional patterns and provide insights?"
                st.session_state['pending_message'] = pattern_query
            else:
                st.info("No emotion data available yet. Start by analyzing some text or audio!")

    with col2:
        if st.button("🛠️ Coping Strategies", use_container_width=True):
            coping_query = "What coping strategies would work best for my emotional patterns?"
            st.session_state['pending_message'] = coping_query

    with col3:
        if st.button("🌱 Wellness Tips", use_container_width=True):
            wellness_query = "Give me personalized wellness tips based on my emotions"
            st.session_state['pending_message'] = wellness_query

    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.write(message['content'])

    if 'pending_message' in st.session_state:
        user_input = st.session_state['pending_message']
        del st.session_state['pending_message']

        st.session_state['message_history'].append({'role': 'user', 'content': user_input})

        with st.chat_message('user'):
            st.write(user_input)

        with st.chat_message('assistant'):
            with st.spinner("Analyzing your emotions and searching for personalized advice..."):
                try:
                    enriched_prompt = f"""Based on my emotional history: {user_input}

                    Please provide personalized advice and search for relevant resources if needed.
                    Also use user_history_query tool if user asks any question related to his or her past data."""

                    response = chatbot.invoke({'messages': [HumanMessage(content=enriched_prompt)]}, config=CONFIG)
                    ai_message = response['messages'][-1].content
                    st.write(ai_message)

                    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
                except Exception as e:
                    st.error(f"Error getting response: {e}")

        st.rerun()

    user_input = st.chat_input('Ask me about your emotions, request coping strategies, or just chat...')

    if user_input:
        st.session_state['message_history'].append({'role': 'user', 'content': user_input})

        with st.chat_message('user'):
            st.write(user_input)

        with st.chat_message('assistant'):
            with st.spinner("Thinking and searching for the best advice..."):
                try:
                    emotion_keywords = ['emotion', 'feel', 'mood', 'anxiety', 'stress', 'happy', 'sad', 'angry', 'worry', 'cope', 'help']
                    is_emotion_query = any(keyword in user_input.lower() for keyword in emotion_keywords)

                    if is_emotion_query and isinstance(user_analysis, dict):
                        enriched_prompt = f"""{user_input}

                        [Context: User has {user_analysis['total_entries']} emotion analyses, most common emotion is {user_analysis['most_common_emotion']}, recent emotions: {', '.join(user_analysis['recent_emotions']) if user_analysis['recent_emotions'] else 'none'}. Please provide personalized advice and search for relevant mental health resources if appropriate.]"""

                        response = chatbot.invoke({'messages': [HumanMessage(content=enriched_prompt)]}, config=CONFIG)
                    else:
                        response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)

                    ai_message = response['messages'][-1].content
                    st.write(ai_message)

                    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
                except Exception as e:
                    st.error(f"Error getting response: {e}")

    if st.session_state['message_history']:
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state['message_history'] = []
            st.rerun()

def main():
    import base64
    # load image and convert to base64
    with open("bgg.jpg", "rb") as f:
        data = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{data}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
        return

    st.sidebar.markdown(f'''
    <div class="welcome-message">
        <strong>Welcome, {st.session_state.username}! 👋</strong>
    </div>
    ''', unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Choose a feature:",
        ["🏠 Home", "📝 Emotion from Text", "🎤 Emotion from Audio", "🤟 Emotion from Sign Language", "📊 User Analytics", "🤖 Chatbot"],
        index=0
    )

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if page == "🏠 Home":
        st.markdown('<h1 class="main-header">🎭 Emotion Detection Dashboard</h1>', unsafe_allow_html=True)

        st.markdown('''
        <div style="text-align: center; color: var(--text-secondary); margin-bottom: 2rem; font-size: 1.1rem;">
            Discover and understand your emotions through advanced AI-powered analysis
        </div>
        ''', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('''
            <div class="feature-card">
                <h3>📝 Text Analysis</h3>
                <p>Analyze emotions from written text using advanced NLP models.
                Simply type your thoughts and get instant emotion detection with confidence scores.</p>
            </div>
            ''', unsafe_allow_html=True)

            st.markdown('''
            <div class="feature-card">
                <h3>🎤 Audio Analysis</h3>
                <p>Convert speech to text and analyze emotional content from audio recordings.
                Perfect for voice journaling and spoken emotion expression.</p>
            </div>
            ''', unsafe_allow_html=True)

            st.markdown('''
            <div class="feature-card">
                <h3>🤟 Sign Language Detection</h3>
                <p>Express emotions through sign language with real-time detection.
                Capture hand gestures and convert them into emotional insights.</p>
            </div>
            ''', unsafe_allow_html=True)

        with col2:
            st.markdown('''
            <div class="feature-card">
                <h3>📊 Analytics Dashboard</h3>
                <p>View your emotion history with interactive charts and insights.
                Track patterns, trends, and gain deeper understanding of your emotional journey.</p>
            </div>
            ''', unsafe_allow_html=True)

            st.markdown('''
            <div class="feature-card">
                <h3>🤖 AI Wellness Assistant</h3>
                <p>Chat with our empathetic AI assistant for personalized support and guidance.
                Get coping strategies, wellness tips, and emotional advice tailored to you.</p>
            </div>
            ''', unsafe_allow_html=True)

            # Quick stats if user has data
            df = get_user_history_data(st.session_state.username)
            if not df.empty:
                emotion_counts = df['Detected Emotion'].value_counts()
                most_common = emotion_counts.index[0]

                st.markdown(f'''
                <div class="feature-card" style="background: linear-gradient(135deg, rgba(6, 255, 165, 0.15) 0%, rgba(76, 201, 240, 0.15) 100%);">
                    <h3>🎯 Your Stats</h3>
                    <p><strong>Total Analyses:</strong> {len(df)}<br>
                    <strong>Most Common Emotion:</strong> {most_common}<br>
                    <strong>Keep tracking your emotional wellness!</strong></p>
                </div>
                ''', unsafe_allow_html=True)

    elif page == "📝 Emotion from Text":
        emotion_from_text()

    elif page == "🎤 Emotion from Audio":
        emotion_from_audio()

    elif page == "🤟 Emotion from Sign Language":
        emotion_from_sign_language()

    elif page == "📊 User Analytics":
        user_analytics()

    elif page == "🤖 Chatbot":
        chatbot_interface()

if __name__ == "__main__":
    main()
