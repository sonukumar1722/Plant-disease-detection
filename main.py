import streamlit as st
from streamlit_back_camera_input import back_camera_input
import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import time

pages = st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Weather Forecast"])

# Initialize session state for toggling the camera
if 'camera_button_clicked' not in st.session_state:
    st.session_state.camera_button_clicked = False

if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purposes.

    #### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)
    """)

elif app_mode == "Disease Recognition":
    # TensorFlow Model Prediction
    def model_prediction(test_image):
        model = tf.keras.models.load_model("trained_plant_disease_model.keras")
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Return index of max element

    st.header("Disease Recognition")

    # Upload image section with columns for better layout
    col1, col2 = st.columns(2)
    with col1:
        test_image = st.file_uploader("Choose an Image:")
        if test_image is not None:
            if st.button("Show Uploaded Image"):
                with col2:
                    st.image(test_image, use_column_width=True)


    # Toggle button for camera input
    with col2:
        if st.button("Camera"):
            st.session_state.camera_button_clicked = not st.session_state.camera_button_clicked

    # Display the camera input if the toggle state is True
    if st.session_state.camera_button_clicked:
        with col2:
            camera = back_camera_input("Capture Image")
            if camera is not None:
                with col2:
                    if st.button("Show Image"):
                        with col2:
                            st.image(camera, use_column_width=True)



    # Predict button
    try:
        with col1:
            if st.button("Predict"):
                st.write("Our Prediction")
                with st.spinner("Predicting..."):
                    if test_image:
                        result_index = model_prediction(test_image)
                    elif camera:
                        result_index = model_prediction(camera)
                    time.sleep(2)

                # Reading Labels
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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
                              'Tomato___healthy']
                st.success(f"Model is Predicting it's a {class_name[result_index]}")
    except:
        with col2:
            st.error("Image not found.")

elif app_mode == "Weather Forecast":
    # Get weather forecast for one week
    def get_weather_forecast(city):
        # OpenWeatherMap API key
        API_KEY = "b834d100667d85b9dc5c25d3d4f49894"
        base_url = "http://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": city,
            "appid": API_KEY,
            "units": "metric",
            "cnt": 7 * 8  # 7 days, 3-hour intervals
        }
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while fetching weather data: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    st.header("Weather Forecast")
    city = st.text_input("Enter City Name", "London")
    if st.button("Get Forecast"):
        try:
            weather_data = get_weather_forecast(city)
            for data in weather_data["list"]:
                weather_description = data["weather"][0]["description"]
                icon_id = data["weather"][0]["icon"]
                temperature = data["main"]["temp"]
                feels_like = data["main"]["feels_like"]
                humidity = data["main"]["humidity"]
                wind_speed = data["wind"]["speed"]
                date, time = data["dt_txt"].split()

                # Get weather icon URL based on icon ID
                icon_url = f"http://openweathermap.org/img/wn/{icon_id}@2x.png"
                # Display weather card
                st.write(f"Date: {date} Time: {time}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(icon_url)
                with col2:
                    st.subheader(f"{temperature:.1f}°C")
                    st.caption(f"Feels like: {feels_like:.1f}°C")
                with col3:
                    st.write(weather_description)

                # Display additional weather details in an accordion
                with st.expander("More Details"):
                    details_df = pd.DataFrame({
                        "Description": ["Temperature", "Feels Like", "Humidity", "Wind Speed"],
                        "Value": [f"{temperature:.1f}°C", f"{feels_like:.1f}°C", f"{humidity}%", f"{wind_speed:.1f} m/s"]
                    })
                    st.table(details_df)
        except:
            st.error("Unable to fetch the data!")
