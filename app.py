import os
from datetime import datetime
from io import StringIO

import pandas as pd
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from google import genai
from transformers import pipeline

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AGRI ROAR | Smart Farming",
    page_icon="🌱",
    layout="wide",
)

MODEL_ID = "kimcomehome/plantvillage-vit-leaf-disease"
GEMINI_MODEL = "gemini-3.5-flash-lite"

# ---------------- GEMINI CONFIGURATION ----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """Create and cache the Gemini client when an API key is configured."""
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


# ---------------- CUSTOM DESIGN ----------------
st.markdown(
    """
    <style>
    .stApp { background: #f5f8f3; }
    section[data-testid="stSidebar"] { background: #e7f2e5; }
    h1, h2, h3 { color: #1b5836; }
    .hero {
        background: linear-gradient(110deg, #174b31, #3d9659);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 20px;
    }
    .hero h1 { color: white; font-size: 38px; }
    .hero p { color: #edf8ef; font-size: 16px; }
    div[data-testid="stMetric"] {
        background: white;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e0e9dc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- LOAD DISEASE MODEL ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline(
        task="image-classification",
        model=MODEL_ID,
        top_k=5,
        device=-1,
    )


# ---------------- SESSION STATE ----------------
if "soil_history" not in st.session_state:
    st.session_state.soil_history = []

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "agri_chat_history" not in st.session_state:
    st.session_state.agri_chat_history = []

# ---------------- HELPERS ----------------
def readable_label(label):
    """Convert model labels into readable text."""
    return label.replace("___", " — ").replace("_", " ").strip()


def disease_guidance(label):
    """Provide cautious, general guidance rather than pesticide prescriptions."""
    if "healthy" in label.lower():
        return [
            "The model classified the image as a healthy leaf.",
            "Continue routine crop monitoring and follow local agricultural guidance.",
        ]

    return [
        "The model detected a disease class that may match the leaf image.",
        "Inspect several leaves and check whether symptoms are spreading.",
        "Avoid making treatment decisions from this prediction alone.",
        "Ask a local agricultural officer or Krishi Vigyan Kendra (KVK) to confirm the diagnosis.",
        "Use only locally registered treatments after the diagnosis is confirmed, following the product label.",
    ]


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🌱 AGRI ROAR")
    st.caption("Your Smart Agriculture Assistant")
    st.divider()
    st.markdown("### Navigation")
    st.markdown(
        """
        - 🏠 Dashboard
        - 🔬 Disease Prediction
        - 🌾 Crop Guide
        - 🧪 Soil Monitor
        - 🏛️ Government Schemes
        - 🤖 AI Farming Assistant
        """
    )
    st.divider()
    st.caption("Student project prototype")

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="hero">
        <h1>🌱 AGRI ROAR</h1>
        <p>Grow better. Harvest smarter.</p>
        <p>Crop disease screening, soil records and crop guidance in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- TABS ----------------
(
    dashboard_tab,
    disease_tab,
    crop_tab,
    soil_tab,
    schemes_tab,
    assistant_tab,
) = st.tabs(
    [
        "🏠 Dashboard",
        "🔬 Disease Prediction",
        "🌾 Crop Guide",
        "🧪 Soil Monitor",
        "🏛️ Government Schemes",
        "🤖 AI Assistant",
    ]
)

# ================= DASHBOARD =================
with dashboard_tab:
    st.subheader("Farm Dashboard")
    st.write("Welcome to AGRI ROAR — your smart farming project.")

    c1, c2, c3 = st.columns(3)
    c1.metric("AI Disease Screening", "Available")
    c2.metric("Soil Records", len(st.session_state.soil_history))
    c3.metric("Images Analyzed", len(st.session_state.prediction_history))

    st.info(
        "Start with Disease Prediction to upload a clear crop-leaf photo. "
        "The AI model supports a defined set of crop and disease classes."
    )

    st.markdown("### How it works")
    st.write("1. Upload a crop-leaf image.")
    st.write("2. The pretrained model compares it with the classes it learned.")
    st.write("3. Review the predicted class and confidence score.")
    st.write("4. Seek expert confirmation before deciding on treatment.")

# ================= DISEASE PREDICTION =================
with disease_tab:
    st.subheader("🌿 Crop Disease Prediction")
    st.write(
        "Upload a clear image showing a crop leaf. "
        "For best results, use good lighting and keep the leaf in focus."
    )

    uploaded_file = st.file_uploader(
        "Upload crop leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        key="disease_upload",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            left, right = st.columns(2)

            with left:
                st.image(image, caption="Uploaded crop image", use_container_width=True)

            with right:
                st.markdown("### Analyze image")
                st.write("The first analysis may take longer while the model loads.")
                analyze = st.button(
                    "🔍 Predict Disease",
                    type="primary",
                    use_container_width=True,
                    key="predict_disease_button",
                )

            if analyze:
                with st.spinner("Loading AI model and analyzing image..."):
                    classifier = load_model()
                    predictions = classifier(image)

                if isinstance(predictions, dict):
                    predictions = [predictions]

                if not predictions:
                    st.error("The model did not return a prediction. Try another image.")
                else:
                    best = predictions[0]
                    label = best["label"]
                    score = float(best["score"])
                    confidence = score * 100
                    display_label = readable_label(label)

                    st.divider()
                    st.markdown("## Prediction Result")

                    if "healthy" in label.lower():
                        st.success(f"🌿 Predicted class: {display_label}")
                    else:
                        st.warning(f"🔎 Possible disease class: {display_label}")

                    st.metric("Model confidence score", f"{confidence:.1f}%")
                    st.progress(min(max(score, 0.0), 1.0))

                    if confidence < 60:
                        st.warning(
                            "The top prediction has a relatively low score. "
                            "The result is uncertain; try a clearer image and seek expert confirmation."
                        )
                    else:
                        st.caption(
                            "This score is the model's output, not a guarantee of diagnostic accuracy."
                        )

                    st.markdown("### Recommended next steps")
                    for item in disease_guidance(label):
                        st.write(f"• {item}")

                    with st.expander("View other model predictions"):
                        for item in predictions:
                            name = readable_label(item["label"])
                            pct = float(item["score"]) * 100
                            st.write(f"**{name}** — {pct:.2f}%")
                            st.progress(min(max(float(item["score"]), 0.0), 1.0))

                    record = {
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Filename": uploaded_file.name,
                        "Prediction": display_label,
                        "Model score (%)": round(confidence, 2),
                    }

                    if (
                        not st.session_state.prediction_history
                        or st.session_state.prediction_history[-1] != record
                    ):
                        st.session_state.prediction_history.append(record)

                    report = pd.DataFrame([record]).to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Prediction Report",
                        data=report,
                        file_name="agri_roar_prediction.csv",
                        mime="text/csv",
                    )

        except Exception as error:
            st.error("The image could not be analyzed.")
            st.write(
                "Check your internet connection, installed packages and model availability, "
                "then try again."
            )
            with st.expander("Technical error details"):
                st.code(str(error))

    st.divider()
    st.caption(
        "Important: This model was trained on PlantVillage classes. "
        "It may misclassify field photos, unfamiliar crops, pests, nutrient problems "
        "or images that do not show a leaf. A high score does not guarantee correctness."
    )

    if st.session_state.prediction_history:
        st.markdown("### Recent predictions")
        st.dataframe(
            pd.DataFrame(st.session_state.prediction_history),
            use_container_width=True,
        )

# ================= CROP GUIDE =================
with crop_tab:
    st.subheader("🌾 Crop Guide")

    crop_values = {
        "Rice": {"water": 50, "days": 5},
        "Wheat": {"water": 60, "days": 21},
        "Maize": {"water": 50, "days": 10},
        "Tomato": {"water": 30, "days": 5},
        "Cotton": {"water": 60, "days": 14},
        "Sugarcane": {"water": 75, "days": 10},
    }

    stage_values = {
        "Seedling": 0.6,
        "Growing": 1.0,
        "Flowering / Fruiting": 1.2,
        "Ripening": 0.5,
    }

    soil_values = {"Sandy": 0.7, "Loamy": 1.0, "Clay": 1.3}

    col1, col2 = st.columns(2)
    with col1:
        crop_name = st.selectbox("Select crop", list(crop_values.keys()))
        area = st.number_input("Farm area (acres)", min_value=0.1, value=1.0, step=0.1)
    with col2:
        stage = st.selectbox("Growth stage", list(stage_values.keys()))
        soil_type = st.selectbox("Soil type", list(soil_values.keys()))

    if st.button("Generate Crop Guide"):
        base = crop_values[crop_name]
        depth = round(base["water"] * stage_values[stage])
        volume = round(depth * 4046.86 * area)
        interval = max(2, round(base["days"] * soil_values[soil_type]))

        st.markdown("### Illustrative results")
        st.metric("Example water depth", f"{depth} mm")
        st.metric("Equivalent water volume", f"{volume:,} litres")
        st.metric("Example interval", f"{interval} days")
        st.warning(
            "These numbers are demo values only, not validated irrigation advice. "
            "Actual water needs depend on weather, soil, crop variety, growth stage "
            "and local practices. Check with an agricultural expert."
        )

# ================= SOIL MONITOR =================
with soil_tab:
    st.subheader("🧪 Soil Monitor")
    st.write("Enter readings from a soil test or sensor.")

    with st.form("soil_form"):
        col1, col2 = st.columns(2)
        with col1:
            ph = st.number_input("Soil pH", min_value=3.0, max_value=10.0, value=6.5, step=0.1)
            moisture = st.number_input("Soil moisture (%)", min_value=0, max_value=100, value=25)
        with col2:
            nitrogen = st.selectbox("Nitrogen (N)", ["Not tested", "Low", "Medium", "High"])
            phosphorus = st.selectbox("Phosphorus (P)", ["Not tested", "Low", "Medium", "High"])
            potassium = st.selectbox("Potassium (K)", ["Not tested", "Low", "Medium", "High"])
        submitted = st.form_submit_button("Save Soil Reading", type="primary")

    if submitted:
        record = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pH": ph,
            "Moisture (%)": moisture,
            "Nitrogen": nitrogen,
            "Phosphorus": phosphorus,
            "Potassium": potassium,
        }
        st.session_state.soil_history.append(record)
        st.success("Soil reading saved for this session.")

        if ph < 5.5:
            st.write("**pH summary:** Acidic reading.")
        elif ph > 7.8:
            st.write("**pH summary:** Alkaline reading.")
        else:
            st.write("**pH summary:** Within the broad demo range.")

        if moisture < 15:
            st.write("**Moisture summary:** Low reading in this demo.")
        elif moisture > 40:
            st.write("**Moisture summary:** High reading in this demo.")
        else:
            st.write("**Moisture summary:** Middle range in this demo.")

        st.caption(
            "These are broad illustrative thresholds. Interpretation depends on "
            "crop, soil, sensor calibration and local conditions."
        )

    if st.session_state.soil_history:
        st.markdown("### Soil History")
        soil_df = pd.DataFrame(st.session_state.soil_history)
        st.dataframe(soil_df, use_container_width=True)
        csv_buffer = StringIO()
        soil_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Soil History",
            data=csv_buffer.getvalue(),
            file_name="agri_roar_soil_history.csv",
            mime="text/csv",
        )

# ================= GOVERNMENT SCHEMES =================
with schemes_tab:
    st.subheader("🏛️ Government Schemes & Farmer Subsidies")
    st.write(
        "Explore selected central and Telangana government programmes that may help "
        "farmers with income support, crop insurance, irrigation, farm equipment and credit."
    )
    st.info(
        "Eligibility, benefits, application windows and subsidy amounts can change. "
        "Confirm current details with the official portal or your local agriculture office before applying."
    )

    schemes = [
        {
            "name": "PM-KISAN — Income Support",
            "category": "Income Support",
            "who": "Eligible landholding farmer families, subject to scheme exclusions.",
            "benefit": "₹6,000 per year, paid in three instalments to eligible beneficiaries.",
            "action": "Check beneficiary status, e-KYC and current application guidance.",
            "url": "https://pmkisan.gov.in/",
            "level": "Central Government",
        },
        {
            "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
            "category": "Crop Insurance",
            "who": "Farmers growing crops notified for the area and season, under current rules.",
            "benefit": "Crop insurance protection for specified risks and notified crops.",
            "action": "Check notified crops, enrolment dates, premium details and claim status.",
            "url": "https://pmfby.gov.in/",
            "level": "Central Government",
        },
        {
            "name": "Pradhan Mantri Krishi Sinchayee Yojana (PMKSY)",
            "category": "Irrigation",
            "who": "Eligible farmers under the relevant component and state implementation rules.",
            "benefit": "Support for improving irrigation and water-use efficiency; assistance varies by component.",
            "action": "Ask the district agriculture or horticulture office about currently open components.",
            "url": "https://pmksy.gov.in/",
            "level": "Central Government",
        },
        {
            "name": "Sub-Mission on Agricultural Mechanization (SMAM)",
            "category": "Machinery Subsidy",
            "who": "Eligible individual farmers or groups under current state guidelines.",
            "benefit": "Financial assistance may be available for eligible farm machinery or custom-hiring facilities.",
            "action": "Check the current application window, eligible equipment and state-specific subsidy rules.",
            "url": "https://agrimachinery.nic.in/",
            "level": "Central Government / State implementation",
        },
        {
            "name": "Telangana Rythu Bharosa",
            "category": "Income Support",
            "who": "Farmers who meet the current Telangana scheme eligibility rules.",
            "benefit": "State agricultural investment support; current amount and eligibility should be verified on the official portal.",
            "action": "Check the latest state guidelines and beneficiary information.",
            "url": "https://rythubharosa.telangana.gov.in/Default_Home.aspx",
            "level": "Telangana Government",
        },
        {
            "name": "Telangana Rythu Bima",
            "category": "Farmer Welfare",
            "who": "Enrolled farmers who meet the current age and other eligibility conditions.",
            "benefit": "Farmer life-insurance support under Telangana's current scheme rules.",
            "action": "Confirm current eligibility and enrolment details with the official state portal or agriculture office.",
            "url": "https://rythubharosa.telangana.gov.in/Default_Home.aspx",
            "level": "Telangana Government",
        },
        {
            "name": "Kisan Credit Card (KCC)",
            "category": "Agricultural Credit",
            "who": "Eligible farmers and other categories permitted by current banking guidelines.",
            "benefit": "Access to agricultural credit for eligible farming needs; loan limits and interest terms depend on rules and lender assessment.",
            "action": "Contact a participating bank or cooperative and ask about KCC eligibility and documents.",
            "url": "https://pmkisan.gov.in/",
            "level": "Central Government / Participating banks",
        },
    ]

    search_text = st.text_input(
        "🔎 Search schemes",
        placeholder="Search by scheme, benefit or category...",
    )
    categories = ["All"] + sorted({item["category"] for item in schemes})
    selected_category = st.selectbox("Filter by support type", categories)

    filtered_schemes = [
        item
        for item in schemes
        if (selected_category == "All" or item["category"] == selected_category)
        and search_text.strip().lower()
        in (
            item["name"] + " " + item["category"] + " " + item["benefit"] + " "
            + item["who"] + " " + item["level"]
        ).lower()
    ]

    st.caption(f"Showing {len(filtered_schemes)} scheme(s)")
    if not filtered_schemes:
        st.warning("No matching schemes found. Try another search or category.")
    else:
        for start in range(0, len(filtered_schemes), 2):
            columns = st.columns(2)
            for column, item in zip(columns, filtered_schemes[start : start + 2]):
                with column:
                    with st.container(border=True):
                        st.markdown(f"### {item['name']}")
                        st.caption(f"{item['level']} · {item['category']}")
                        st.markdown("**Who may qualify**")
                        st.write(item["who"])
                        st.markdown("**Possible benefit**")
                        st.write(item["benefit"])
                        st.markdown("**What to do**")
                        st.write(item["action"])
                        st.link_button(
                            "Visit official portal ↗",
                            item["url"],
                            use_container_width=True,
                        )

    st.divider()
    st.markdown("### Before applying")
    st.write(
        "- Keep Aadhaar, bank details, land or cultivation records and required identity documents ready, as applicable to the scheme.\n"
        "- Never pay an unknown agent who promises guaranteed approval.\n"
        "- For Telangana-specific support, contact your local agriculture officer or MeeSeva centre to confirm the current process.\n"
        "- This section is an information guide, not an application portal; AGRI ROAR does not verify eligibility."
    )

# ================= GEMINI AI FARMING ASSISTANT =================
with assistant_tab:
    st.subheader("🤖 AI Farming Assistant")
    st.write(
        "Ask questions about crop diseases, soil health, irrigation, crop growth "
        "and general farming practices."
    )

    if not GEMINI_API_KEY:
        st.warning(
            "Gemini API key not found. Create a .env file beside app.py and add "
            "GEMINI_API_KEY=your_actual_key. Then restart Streamlit."
        )

    # Display conversation so far
    for message in st.session_state.agri_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask your farming question...", key="agri_chat_input")

    if user_question:
        st.session_state.agri_chat_history.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        client = get_gemini_client()
        if client is None:
            st.error("Please add a valid GEMINI_API_KEY to your .env file and restart the app.")
        else:
            system_instructions = """You are AGRI ROAR AI, a helpful farming assistant.
Answer in simple, clear language. Help with crop diseases, soil health, irrigation,
crop growth and general agricultural practices. Do not claim a disease is confirmed
from a text description alone. Do not invent current government scheme eligibility
or benefits. Do not prescribe specific pesticide dosages. For serious crop problems,
recommend consulting a local agricultural officer or Krishi Vigyan Kendra (KVK).
Clearly state uncertainty when needed."""

            recent_messages = st.session_state.agri_chat_history[-12:]
            conversation = "\n".join(
                f"{'Farmer' if item['role'] == 'user' else 'Assistant'}: {item['content']}"
                for item in recent_messages
            )
            prompt = f"{system_instructions}\n\nConversation:\n{conversation}\n\nReply to the farmer's latest question."

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = client.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=prompt,
                        )
                        answer = response.text or "I couldn't generate a response. Please try again."
                    st.markdown(answer)

                st.session_state.agri_chat_history.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as error:
                st.error(
                    "Unable to get a Gemini response. Check your API key, internet connection, "
                    "API access and model availability."
                )
                with st.expander("Technical error details"):
                    st.code(str(error))

    if st.button("🗑️ Clear Chat History", key="clear_agri_chat"):
        st.session_state.agri_chat_history = []
        st.rerun()

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "AGRI ROAR | Smart Agriculture Student Project | "
    "AI results are preliminary and require verification."
)
