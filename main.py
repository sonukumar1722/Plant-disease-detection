import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import os
from datetime import datetime
from PIL import Image
import io
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Plant Disease Recognition", page_icon="🌿", layout="wide")

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page", 
    ["Disease Prediction", "Weather Forecast", "Home & About"],
    index=0
)

if 'input_mode' not in st.session_state:
    st.session_state.input_mode = "upload"
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'use_front_camera' not in st.session_state:
    st.session_state.use_front_camera = True

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

DISEASE_TREATMENTS = {
    'Apple___Apple_scab': {
        'description': 'Fungal disease causing olive-green to black lesions on leaves and fruit.',
        'treatment': [
            'Apply fungicides containing captan, myclobutanil, or sulfur during early spring',
            'Remove and destroy fallen leaves and infected fruit to reduce fungal spores',
            'Prune trees to improve air circulation',
            'Plant resistant apple varieties like Liberty, Enterprise, or Pristine',
            'Apply lime sulfur spray during dormant season'
        ],
        'prevention': [
            'Maintain proper tree spacing for good air flow',
            'Avoid overhead irrigation',
            'Apply preventive fungicide sprays before rainy periods'
        ]
    },
    'Apple___Black_rot': {
        'description': 'Fungal disease causing fruit rot, leaf spots, and cankers on branches.',
        'treatment': [
            'Prune out all dead wood and cankers during dormant season',
            'Remove mummified fruits from trees and ground',
            'Apply fungicides containing captan or myclobutanil',
            'Spray copper-based fungicides in early spring',
            'Treat wounds and pruning cuts with wound sealant'
        ],
        'prevention': [
            'Maintain tree vigor with proper fertilization',
            'Remove wild or abandoned apple trees nearby',
            'Practice good sanitation by removing all infected material'
        ]
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Fungal disease requiring both apple and cedar/juniper trees to complete lifecycle.',
        'treatment': [
            'Apply fungicides containing myclobutanil, propiconazole, or mancozeb',
            'Begin spraying at pink bud stage and continue through petal fall',
            'Remove nearby cedar and juniper trees within 2-mile radius if possible',
            'Prune out galls from cedar trees in late winter'
        ],
        'prevention': [
            'Plant resistant apple varieties like Redfree, Liberty, or Freedom',
            'Avoid planting apples near cedar or juniper trees',
            'Apply preventive fungicide sprays during wet spring weather'
        ]
    },
    'Apple___healthy': {
        'description': 'Your apple plant appears healthy with no visible disease symptoms.',
        'treatment': [],
        'prevention': [
            'Continue regular monitoring for signs of disease',
            'Maintain proper watering and fertilization schedule',
            'Prune regularly to ensure good air circulation',
            'Apply dormant oil spray in late winter to prevent pests'
        ]
    },
    'Blueberry___healthy': {
        'description': 'Your blueberry plant appears healthy with no visible disease symptoms.',
        'treatment': [],
        'prevention': [
            'Maintain soil pH between 4.5-5.5 for optimal health',
            'Apply mulch to retain moisture and regulate soil temperature',
            'Prune old canes to encourage new growth',
            'Monitor for pests and diseases regularly'
        ]
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'description': 'Fungal disease causing white powdery coating on leaves and shoots.',
        'treatment': [
            'Apply sulfur-based fungicides or potassium bicarbonate sprays',
            'Use fungicides containing myclobutanil or trifloxystrobin',
            'Remove and destroy heavily infected shoots',
            'Apply neem oil as an organic alternative',
            'Spray baking soda solution (1 tbsp per gallon of water)'
        ],
        'prevention': [
            'Improve air circulation through proper pruning',
            'Avoid excessive nitrogen fertilization',
            'Water at soil level, avoiding wetting foliage',
            'Plant in full sun locations'
        ]
    },
    'Cherry_(including_sour)___healthy': {
        'description': 'Your cherry plant appears healthy with no visible disease symptoms.',
        'treatment': [],
        'prevention': [
            'Maintain proper watering schedule',
            'Apply balanced fertilizer in early spring',
            'Prune during dry weather to prevent disease spread',
            'Monitor for pest infestations regularly'
        ]
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': 'Fungal disease causing rectangular gray lesions on corn leaves.',
        'treatment': [
            'Apply foliar fungicides containing strobilurins or triazoles',
            'Use products like azoxystrobin, pyraclostrobin, or propiconazole',
            'Time applications at VT (tasseling) to R2 growth stages',
            'Rotate with non-host crops like soybeans for 1-2 years'
        ],
        'prevention': [
            'Plant resistant corn hybrids',
            'Reduce corn residue through tillage or decomposition',
            'Avoid continuous corn planting in same field',
            'Ensure adequate plant spacing for air movement'
        ]
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Fungal disease causing cinnamon-brown pustules on corn leaves.',
        'treatment': [
            'Apply fungicides containing triazoles or strobilurins',
            'Use products like propiconazole or azoxystrobin',
            'Spray when pustules first appear and conditions favor disease',
            'Scout fields regularly during warm, humid weather'
        ],
        'prevention': [
            'Plant rust-resistant corn varieties',
            'Plant early to avoid peak rust pressure',
            'Monitor weather conditions that favor rust development',
            'Maintain balanced soil fertility'
        ]
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Fungal disease causing long, elliptical gray-green lesions on corn leaves.',
        'treatment': [
            'Apply fungicides at early disease onset (VT to R1 stage)',
            'Use products containing azoxystrobin, propiconazole, or picoxystrobin',
            'Multiple applications may be needed in severe cases',
            'Rotate crops to reduce inoculum levels'
        ],
        'prevention': [
            'Plant resistant corn hybrids with Ht genes',
            'Practice crop rotation with non-host crops',
            'Reduce corn residue through tillage',
            'Avoid planting corn after corn in same field'
        ]
    },
    'Corn_(maize)___healthy': {
        'description': 'Your corn plant appears healthy with no visible disease symptoms.',
        'treatment': [],
        'prevention': [
            'Continue monitoring for pest and disease signs',
            'Maintain proper plant nutrition',
            'Ensure adequate irrigation during dry periods',
            'Scout fields regularly during critical growth stages'
        ]
    },
    'Grape___Black_rot': {
        'description': 'Fungal disease causing brown leaf spots and shriveled, black mummified berries.',
        'treatment': [
            'Apply fungicides containing myclobutanil, mancozeb, or captan',
            'Begin applications at bud break and continue through fruit set',
            'Remove and destroy all mummified berries and infected canes',
            'Apply copper-based sprays during dormant season'
        ],
        'prevention': [
            'Prune vines to improve air circulation',
            'Remove wild grapes from nearby areas',
            'Practice good sanitation by removing all infected material',
            'Maintain open canopy for rapid drying of foliage'
        ]
    },
    'Grape___Esca_(Black_Measles)': {
        'description': 'Fungal disease complex causing tiger-striped leaves and internal wood decay.',
        'treatment': [
            'There is no cure; remove and destroy severely affected vines',
            'Apply wound protectants to pruning cuts',
            'Delay pruning until late in dormant season',
            'Avoid large pruning wounds that expose wood'
        ],
        'prevention': [
            'Use disease-free planting material',
            'Minimize stress on vines through proper irrigation',
            'Protect pruning wounds with fungicide paste',
            'Prune during dry weather conditions'
        ]
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': 'Fungal disease causing brown spots with dark borders on grape leaves.',
        'treatment': [
            'Apply fungicides containing mancozeb or copper hydroxide',
            'Remove and destroy infected leaves',
            'Spray sulfur-based fungicides during growing season',
            'Apply treatments at 7-14 day intervals during wet weather'
        ],
        'prevention': [
            'Improve air circulation through proper canopy management',
            'Avoid overhead irrigation',
            'Remove plant debris from vineyard floor',
            'Maintain balanced vine nutrition'
        ]
    },
    'Grape___healthy': {
        'description': 'Your grape vine appears healthy with no visible disease symptoms.',
        'treatment': [],
        'prevention': [
            'Continue regular monitoring for disease symptoms',
            'Maintain proper pruning and canopy management',
            'Apply preventive fungicide sprays during humid conditions',
            'Ensure good drainage and air circulation'
        ]
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'description': 'Bacterial disease spread by Asian citrus psyllid, causing yellowing and bitter fruit.',
        'treatment': [
            'There is no cure for infected trees',
            'Remove and destroy infected trees to prevent spread',
            'Control Asian citrus psyllid with insecticides (imidacloprid, carbaryl)',
            'Apply foliar nutritional sprays to manage symptoms',
            'Use systemic insecticides to protect healthy trees'
        ],
        'prevention': [
            'Use certified disease-free nursery stock',
            'Monitor and control psyllid populations regularly',
            'Inspect trees frequently for symptoms',
            'Report suspected cases to agricultural authorities',
            'Maintain tree health through proper nutrition'
        ]
    },
    'Peach___Bacterial_spot': {
        'description': 'Bacterial disease causing angular spots on leaves and sunken lesions on fruit.',
        'treatment': [
            'Apply copper-based bactericides during dormant season',
            'Spray oxytetracycline during growing season',
            'Remove and destroy severely infected plant parts',
            'Avoid overhead irrigation to reduce leaf wetness'
        ],
        'prevention': [
            'Plant resistant peach varieties',
            'Provide wind protection to reduce leaf damage',
            'Maintain tree vigor through proper fertilization',
            'Avoid working with wet trees to prevent spread'
        ]
    },
    'Peach___healthy': {
        'description': 'Your peach plant appears healthy with no visible disease symptoms.',
        'treatment': [],
        'prevention': [
            'Continue regular pest and disease monitoring',
            'Apply dormant sprays in late winter',
            'Maintain proper pruning for good air circulation',
            'Protect blossoms from late frost'
        ]
    },
    'Pepper,_bell___Bacterial_spot': {
        'description': 'Bacterial disease causing water-soaked spots on leaves and raised scabs on fruit.',
        'treatment': [
            'Apply copper-based bactericides weekly during wet weather',
            'Use streptomycin sprays if available and permitted',
            'Remove and destroy infected plants and debris',
            'Avoid working with wet plants to prevent spread'
        ],
        'prevention': [
            'Use certified disease-free seeds and transplants',
            'Practice crop rotation (3+ years without peppers/tomatoes)',
            'Avoid overhead irrigation',
            'Provide adequate plant spacing for air circulation'
        ]
    },
    'Pepper,_bell___healthy': {
        'description': 'Your bell pepper plant appears healthy with no visible disease symptoms.',
        'treatment': [],
        'prevention': [
            'Maintain consistent watering to prevent stress',
            'Apply balanced fertilizer regularly',
            'Monitor for pest infestations',
            'Stake plants to improve air circulation'
        ]
    },
    'Potato___Early_blight': {
        'description': 'Fungal disease causing dark concentric rings (target spots) on leaves.',
        'treatment': [
            'Apply fungicides containing chlorothalonil, mancozeb, or azoxystrobin',
            'Begin sprays when plants are 6 inches tall',
            'Repeat applications every 7-14 days as needed',
            'Remove and destroy infected plant debris'
        ],
        'prevention': [
            'Plant certified disease-free seed potatoes',
            'Practice crop rotation (2-3 years)',
            'Avoid overhead irrigation',
            'Maintain adequate plant nutrition, especially nitrogen'
        ]
    },
    'Potato___Late_blight': {
        'description': 'Devastating fungal disease causing water-soaked lesions and white mold on leaves.',
        'treatment': [
            'Apply fungicides containing chlorothalonil, mancozeb, or fluopicolide',
            'Spray immediately upon detection and repeat every 5-7 days',
            'Destroy infected plants and tubers immediately',
            'Do not compost infected material'
        ],
        'prevention': [
            'Plant certified disease-free seed potatoes',
            'Plant resistant varieties when available',
            'Avoid planting near tomatoes',
            'Ensure good air circulation and drainage',
            'Monitor weather for blight-favorable conditions'
        ]
    },
    'Potato___healthy': {
        'description': 'Your potato plant appears healthy with no visible disease symptoms.',
        'treatment': [],
        'prevention': [
            'Continue monitoring for disease symptoms',
            'Maintain proper hilling practices',
            'Water at soil level, avoiding wet foliage',
            'Apply preventive fungicides during humid weather'
        ]
    },
    'Raspberry___healthy': {
        'description': 'Your raspberry plant appears healthy with no visible disease symptoms.',
        'treatment': [],
        'prevention': [
            'Prune out old canes after fruiting',
            'Maintain good air circulation between plants',
            'Apply mulch to retain moisture',
            'Monitor for pests and diseases regularly'
        ]
    },
    'Soybean___healthy': {
        'description': 'Your soybean plant appears healthy with no visible disease symptoms.',
        'treatment': [],
        'prevention': [
            'Continue crop rotation practices',
            'Monitor for pest and disease signs',
            'Maintain proper plant nutrition',
            'Scout fields regularly during critical stages'
        ]
    },
    'Squash___Powdery_mildew': {
        'description': 'Fungal disease causing white powdery coating on leaves and stems.',
        'treatment': [
            'Apply sulfur-based fungicides or potassium bicarbonate',
            'Use neem oil as an organic treatment option',
            'Spray milk solution (40% milk to water) weekly',
            'Apply fungicides containing myclobutanil or trifloxystrobin',
            'Remove heavily infected leaves to slow spread'
        ],
        'prevention': [
            'Plant resistant squash varieties',
            'Space plants widely for good air circulation',
            'Avoid overhead watering',
            'Apply preventive sprays before symptoms appear'
        ]
    },
    'Strawberry___Leaf_scorch': {
        'description': 'Fungal disease causing irregular purple spots that merge into brown scorched areas.',
        'treatment': [
            'Apply fungicides containing captan or myclobutanil',
            'Remove and destroy infected leaves',
            'Renovate beds after harvest by mowing and removing debris',
            'Apply copper-based fungicides in spring'
        ],
        'prevention': [
            'Plant resistant strawberry varieties',
            'Ensure good air circulation between plants',
            'Avoid overhead irrigation',
            'Practice crop rotation (3-4 years)'
        ]
    },
    'Strawberry___healthy': {
        'description': 'Your strawberry plant appears healthy with no visible disease symptoms.',
        'treatment': [],
        'prevention': [
            'Maintain proper spacing between plants',
            'Apply mulch to prevent soil splash',
            'Remove runners to maintain plant vigor',
            'Monitor for pests and diseases regularly'
        ]
    },
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial disease causing small, dark, water-soaked spots on leaves and fruit.',
        'treatment': [
            'Apply copper-based bactericides weekly',
            'Use streptomycin sprays if permitted in your area',
            'Remove and destroy infected plant material',
            'Avoid working with wet plants'
        ],
        'prevention': [
            'Use certified disease-free seeds and transplants',
            'Practice crop rotation (2-3 years)',
            'Avoid overhead irrigation',
            'Provide adequate spacing between plants'
        ]
    },
    'Tomato___Early_blight': {
        'description': 'Fungal disease causing dark concentric rings (target spots) on lower leaves.',
        'treatment': [
            'Apply fungicides containing chlorothalonil or mancozeb',
            'Remove infected lower leaves promptly',
            'Apply copper-based fungicides as organic option',
            'Spray every 7-10 days during favorable conditions'
        ],
        'prevention': [
            'Mulch around plants to prevent soil splash',
            'Stake or cage plants to keep foliage off ground',
            'Practice crop rotation',
            'Remove plant debris at end of season'
        ]
    },
    'Tomato___Late_blight': {
        'description': 'Devastating fungal disease causing water-soaked lesions and white mold.',
        'treatment': [
            'Apply fungicides containing chlorothalonil or mancozeb immediately',
            'Remove and destroy infected plants completely',
            'Spray every 5-7 days during outbreaks',
            'Do not compost infected material - burn or dispose'
        ],
        'prevention': [
            'Plant resistant tomato varieties',
            'Avoid planting near potatoes',
            'Ensure good air circulation',
            'Water at soil level in morning',
            'Monitor weather for blight conditions'
        ]
    },
    'Tomato___Leaf_Mold': {
        'description': 'Fungal disease causing pale green to yellow spots on upper leaves with olive-green mold below.',
        'treatment': [
            'Apply fungicides containing chlorothalonil or mancozeb',
            'Remove and destroy infected leaves',
            'Improve ventilation in greenhouses',
            'Reduce humidity below 85%'
        ],
        'prevention': [
            'Plant resistant tomato varieties',
            'Provide good air circulation',
            'Avoid overhead watering',
            'Maintain relative humidity below 85%',
            'Remove lower leaves to improve airflow'
        ]
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Fungal disease causing small circular spots with gray centers and dark borders.',
        'treatment': [
            'Apply fungicides containing chlorothalonil or copper',
            'Remove and destroy infected lower leaves',
            'Spray every 7-10 days during wet weather',
            'Improve air circulation around plants'
        ],
        'prevention': [
            'Mulch to prevent soil splash',
            'Stake plants to keep foliage off ground',
            'Practice 3-year crop rotation',
            'Remove plant debris at end of season'
        ]
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Tiny pests causing stippling, yellowing, and webbing on tomato leaves.',
        'treatment': [
            'Spray plants with strong water jets to dislodge mites',
            'Apply insecticidal soap or neem oil',
            'Use miticides like abamectin or bifenthrin for severe infestations',
            'Release predatory mites (Phytoseiulus persimilis) as biological control',
            'Remove heavily infested leaves'
        ],
        'prevention': [
            'Maintain adequate plant watering to reduce stress',
            'Avoid excessive nitrogen fertilization',
            'Encourage beneficial insects in garden',
            'Monitor plants regularly, especially during hot, dry weather'
        ]
    },
    'Tomato___Target_Spot': {
        'description': 'Fungal disease causing brown circular lesions with concentric rings on leaves.',
        'treatment': [
            'Apply fungicides containing chlorothalonil or azoxystrobin',
            'Remove and destroy infected leaves',
            'Spray every 7-14 days during favorable conditions',
            'Improve air circulation around plants'
        ],
        'prevention': [
            'Practice crop rotation',
            'Mulch to prevent soil splash',
            'Avoid overhead irrigation',
            'Remove plant debris at end of season'
        ]
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Viral disease spread by whiteflies, causing yellowing, curling, and stunted growth.',
        'treatment': [
            'There is no cure for infected plants',
            'Remove and destroy infected plants immediately',
            'Control whitefly populations with insecticides or sticky traps',
            'Apply systemic insecticides (imidacloprid) to protect healthy plants',
            'Use reflective mulches to repel whiteflies'
        ],
        'prevention': [
            'Plant resistant tomato varieties',
            'Use insect-proof netting in greenhouses',
            'Control whitefly populations proactively',
            'Remove weeds that may harbor whiteflies',
            'Plant early to avoid peak whitefly season'
        ]
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Viral disease causing mottled light and dark green patterns on leaves.',
        'treatment': [
            'There is no cure for infected plants',
            'Remove and destroy infected plants',
            'Disinfect tools with 10% bleach solution',
            'Wash hands thoroughly before handling healthy plants',
            'Do not smoke or use tobacco products near tomatoes'
        ],
        'prevention': [
            'Plant resistant tomato varieties (look for "TMV" resistance)',
            'Use certified disease-free seeds',
            'Disinfect tools and wash hands frequently',
            'Avoid handling plants when wet',
            'Control aphids which can spread the virus'
        ]
    },
    'Tomato___healthy': {
        'description': 'Your tomato plant appears healthy with no visible disease symptoms.',
        'treatment': [],
        'prevention': [
            'Continue regular monitoring for pests and diseases',
            'Maintain consistent watering schedule',
            'Apply balanced fertilizer throughout season',
            'Stake or cage plants for better air circulation'
        ]
    }
}

@st.cache_resource
def load_model():
    model_path = "trained_plant_disease_model.keras"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

def model_prediction(image_data):
    model = load_model()
    if model is None:
        st.error("Model file not found. Please ensure 'trained_plant_disease_model.keras' is in the project directory.")
        return None
    
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    else:
        image = Image.open(image_data)
    
    image = image.convert('RGB')
    image = image.resize((128, 128))
    input_arr = np.array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


if app_mode == "Disease Prediction":
    st.header("🔬 Plant Disease Prediction")
    st.markdown("Upload an image or capture one using your camera to identify plant diseases.")
    
    st.subheader("Choose Input Method")
    input_mode = st.radio(
        "Select how you want to provide the plant image:",
        ["Upload Image", "Use Camera"],
        horizontal=True,
        key="input_method_radio"
    )
    
    current_image = None
    
    if input_mode == "Upload Image":
        st.markdown("---")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the plant leaf you want to analyze"
        )
        
        if uploaded_file is not None:
            current_image = uploaded_file
            st.success("Image uploaded successfully!")
            
            with st.expander("Preview Uploaded Image", expanded=True):
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    else:
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if 'camera_facing' not in st.session_state:
                st.session_state.camera_facing = "Front"
            
            front_btn = st.button("📱 Front Camera", use_container_width=True, 
                                   type="primary" if st.session_state.camera_facing == "Front" else "secondary")
            back_btn = st.button("📷 Back Camera", use_container_width=True,
                                  type="primary" if st.session_state.camera_facing == "Back" else "secondary")
            
            if front_btn:
                st.session_state.camera_facing = "Front"
                st.rerun()
            if back_btn:
                st.session_state.camera_facing = "Back"
                st.rerun()
        
        with col1:
            st.info(f"📸 **Camera Mode** - Position the plant leaf clearly in the frame and click capture.")
            st.caption("Tip: On mobile devices, you can use your browser's camera switcher if available. The camera selection buttons help track your preference.")
        
        camera_key = f"camera_input_{st.session_state.camera_facing.lower()}"
        
        camera_image = st.camera_input(
            "Click the button below to capture an image",
            key=camera_key
        )
        
        if camera_image is not None:
            st.session_state.captured_image = camera_image.getvalue()
            current_image = camera_image
            st.success("Image captured successfully!")
            
            with st.expander("Preview Captured Image", expanded=True):
                st.image(camera_image, caption="Captured Image", use_container_width=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔍 Predict Disease",
            use_container_width=True,
            type="primary",
            disabled=current_image is None
        )
    
    if predict_button:
        if current_image is not None:
            with st.spinner("Analyzing image..."):
                result_index = model_prediction(current_image)
                
                if result_index is not None:
                    disease_name = CLASS_NAMES[result_index]
                    
                    plant_name = disease_name.split("___")[0].replace("_", " ")
                    condition = disease_name.split("___")[1].replace("_", " ")
                    
                    st.markdown("---")
                    st.subheader("Prediction Result")
                    
                    treatment_info = DISEASE_TREATMENTS.get(disease_name, {})
                    
                    if "healthy" in condition.lower():
                        st.success(f"🌱 **Plant:** {plant_name}")
                        st.success(f"✅ **Condition:** Healthy")
                        st.balloons()
                        
                        if treatment_info:
                            st.markdown("---")
                            st.subheader("🛡️ Prevention Tips")
                            st.info(treatment_info.get('description', ''))
                            
                            prevention = treatment_info.get('prevention', [])
                            if prevention:
                                st.markdown("**Keep your plant healthy with these tips:**")
                                for tip in prevention:
                                    st.markdown(f"- {tip}")
                    else:
                        st.warning(f"🌱 **Plant:** {plant_name}")
                        st.error(f"🦠 **Disease Detected:** {condition}")
                        
                        if treatment_info:
                            st.markdown("---")
                            st.subheader("📋 Disease Information")
                            st.info(treatment_info.get('description', ''))
                            
                            st.markdown("---")
                            st.subheader("💊 Recommended Treatment")
                            treatments = treatment_info.get('treatment', [])
                            if treatments:
                                for i, treatment in enumerate(treatments, 1):
                                    st.markdown(f"**{i}.** {treatment}")
                            else:
                                st.warning("No specific treatment available. Consult an agricultural expert.")
                            
                            st.markdown("---")
                            st.subheader("🛡️ Prevention Tips")
                            prevention = treatment_info.get('prevention', [])
                            if prevention:
                                st.markdown("**To prevent future occurrences:**")
                                for tip in prevention:
                                    st.markdown(f"- {tip}")
                        else:
                            st.info("Consider consulting with an agricultural expert for treatment options.")
        else:
            st.error("Please upload an image or capture one using the camera first.")


elif app_mode == "Weather Forecast":
    st.header("🌤️ Weather Forecast for Agriculture")
    st.markdown("Get a 5-day weather forecast with details important for farming and agriculture.")
    
    API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY", "")
    
    if not API_KEY:
        st.warning("OpenWeatherMap API key not configured. Please add your API key to use this feature.")
        st.info("You can get a free API key from https://openweathermap.org/api")
    
    city = st.text_input("🔍 Enter Location", placeholder="e.g., London, New York, Tokyo")
    
    search_button = st.button("Get Weather Forecast", type="primary", use_container_width=True)
    
    if search_button and city:
        if not API_KEY:
            st.error("Please configure your OpenWeatherMap API key first.")
        else:
            try:
                geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
                geo_response = requests.get(geo_url)
                geo_data = geo_response.json()
                
                if not geo_data:
                    st.error(f"Location '{city}' not found. Please check the spelling and try again.")
                else:
                    lat = geo_data[0]["lat"]
                    lon = geo_data[0]["lon"]
                    location_name = geo_data[0].get("name", city)
                    country = geo_data[0].get("country", "")
                    
                    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
                    response = requests.get(forecast_url)
                    data = response.json()
                    
                    if data.get("cod") != "200":
                        st.error(f"Error fetching weather data: {data.get('message', 'Unknown error')}")
                    else:
                        st.success(f"Showing forecast for {location_name}, {country}")
                        
                        daily_data = {}
                        for item in data["list"]:
                            date = item["dt_txt"].split()[0]
                            if date not in daily_data:
                                daily_data[date] = {
                                    "temps": [],
                                    "humidity": [],
                                    "wind_speed": [],
                                    "descriptions": [],
                                    "icons": [],
                                    "pressure": [],
                                    "clouds": [],
                                    "rain": 0,
                                    "feels_like": []
                                }
                            
                            daily_data[date]["temps"].append(item["main"]["temp"])
                            daily_data[date]["humidity"].append(item["main"]["humidity"])
                            daily_data[date]["wind_speed"].append(item["wind"]["speed"])
                            daily_data[date]["descriptions"].append(item["weather"][0]["description"])
                            daily_data[date]["icons"].append(item["weather"][0]["icon"])
                            daily_data[date]["pressure"].append(item["main"]["pressure"])
                            daily_data[date]["clouds"].append(item["clouds"]["all"])
                            daily_data[date]["feels_like"].append(item["main"]["feels_like"])
                            if "rain" in item:
                                daily_data[date]["rain"] += item["rain"].get("3h", 0)
                        
                        days = list(daily_data.keys())[:5]
                        
                        cols_per_row = 5
                        for row_start in range(0, len(days), cols_per_row):
                            row_days = days[row_start:row_start + cols_per_row]
                            cols = st.columns(len(row_days))
                            
                            for idx, day in enumerate(row_days):
                                day_info = daily_data[day]
                                
                                avg_temp = sum(day_info["temps"]) / len(day_info["temps"])
                                max_temp = max(day_info["temps"])
                                min_temp = min(day_info["temps"])
                                avg_humidity = sum(day_info["humidity"]) / len(day_info["humidity"])
                                avg_wind = sum(day_info["wind_speed"]) / len(day_info["wind_speed"])
                                avg_pressure = sum(day_info["pressure"]) / len(day_info["pressure"])
                                avg_clouds = sum(day_info["clouds"]) / len(day_info["clouds"])
                                total_rain = day_info["rain"]
                                avg_feels_like = sum(day_info["feels_like"]) / len(day_info["feels_like"])
                                
                                main_icon = max(set(day_info["icons"]), key=day_info["icons"].count)
                                main_desc = max(set(day_info["descriptions"]), key=day_info["descriptions"].count)
                                
                                date_obj = datetime.strptime(day, "%Y-%m-%d")
                                day_name = date_obj.strftime("%A")
                                formatted_date = date_obj.strftime("%b %d")
                                
                                with cols[idx]:
                                    st.markdown(f"""
                                    <div style="
                                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                        border-radius: 15px;
                                        padding: 15px;
                                        text-align: center;
                                        color: white;
                                        margin-bottom: 10px;
                                    ">
                                        <h4 style="margin: 0; color: white;">{day_name}</h4>
                                        <p style="margin: 0; opacity: 0.8;">{formatted_date}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    icon_url = f"http://openweathermap.org/img/wn/{main_icon}@2x.png"
                                    st.image(icon_url, width=80)
                                    st.caption(main_desc.title())
                                    
                                    st.metric("Temperature", f"{avg_temp:.1f}°C", f"H:{max_temp:.0f}° L:{min_temp:.0f}°")
                                    
                                    st.markdown(f"""
                                    **Agriculture Details:**
                                    - 💧 Humidity: **{avg_humidity:.0f}%**
                                    - 💨 Wind: **{avg_wind:.1f} m/s**
                                    - 🌡️ Feels Like: **{avg_feels_like:.1f}°C**
                                    - 🌧️ Rain: **{total_rain:.1f} mm**
                                    - ☁️ Clouds: **{avg_clouds:.0f}%**
                                    - 📊 Pressure: **{avg_pressure:.0f} hPa**
                                    """)
                                    
                                    if avg_humidity > 80:
                                        st.warning("⚠️ High humidity - disease risk")
                                    if avg_wind > 10:
                                        st.warning("⚠️ High winds - spraying not advised")
                                    if total_rain > 10:
                                        st.info("🌧️ Significant rainfall expected")
                        
                        st.markdown("---")
                        st.subheader("Agricultural Summary")
                        
                        all_temps = [t for day in days for t in daily_data[day]["temps"]]
                        all_humidity = [h for day in days for h in daily_data[day]["humidity"]]
                        total_rain_forecast = sum(daily_data[day]["rain"] for day in days)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Avg Temperature", f"{sum(all_temps)/len(all_temps):.1f}°C")
                        with col2:
                            st.metric("Avg Humidity", f"{sum(all_humidity)/len(all_humidity):.0f}%")
                        with col3:
                            st.metric("Total Rainfall", f"{total_rain_forecast:.1f} mm")
                        with col4:
                            st.metric("Forecast Days", len(days))
                            
            except requests.exceptions.RequestException as e:
                st.error(f"Network error: Unable to fetch weather data. Please check your connection.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    elif search_button and not city:
        st.warning("Please enter a location to search.")


elif app_mode == "Home & About":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
    
    st.markdown("""
    Welcome to the Plant Disease Recognition System!

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Prediction** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Prediction** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)
    
    st.markdown("---")
    
    st.subheader("About the Project")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. 
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes. 
    The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.

    #### Content
    - **Training set:** 70,295 images
    - **Test set:** 33 images  
    - **Validation set:** 17,572 images
    
    #### Supported Plants
    The system can detect diseases in the following plants:
    - Apple, Blueberry, Cherry, Corn (Maize)
    - Grape, Orange, Peach, Pepper (Bell)
    - Potato, Raspberry, Soybean, Squash
    - Strawberry, Tomato
    """)
    
    with st.expander("View All Detectable Conditions"):
        for i, name in enumerate(CLASS_NAMES, 1):
            plant, condition = name.split("___")
            plant = plant.replace("_", " ")
            condition = condition.replace("_", " ")
            st.write(f"{i}. **{plant}** - {condition}")
