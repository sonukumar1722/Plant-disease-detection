from page import registration as pr
from page import database as db



# Database
def create_connection(host, user, password, database):
    """Creates a connection to the MySQL database."""
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        return connection
    except Error as e:
        print(f"The error '{e}' occurred")
        return None # Indicates connection failure

def create_database(database_name):
    """Creates a database if it doesn't already exist."""

    if connection is not None:  # Check if connection exists before using cursor
        cursor = connection.cursor()
        try:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name};")
            print(f"Database '{database_name}' created successfully")
        except Error as e:
            print(f"The error '{e}' occurred")

def create_table():
    """Creates a table named 'users' if it doesn't already exist."""
      
    if connection is not None: 
        try:
            cursor = connection.cursor()
            query = """
                CREATE TABLE IF NOT EXISTS users (
                id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(40) NOT NULL,
                username VARCHAR(30) UNIQUE,
                password VARCHAR(100) NOT NULL,
                email VARCHAR(40) NOT NULL UNIQUE
            )
            """
            cursor.execute(query)
            # print("Table 'users' created successfully")
        except Error as e:
            print(f"The error '{e}' occurred")

def insert_details(query):
    """Inserts data into a table."""

    if connection is not None:
        cursor = connection.cursor()
        try:
            cursor.execute(query)
            connection.commit()
            print("User added succesfully")
        except Error as e:
            print(f"The error '{e}' occurred")


Registration
def register_user():
    if connection is not None:  # Check if connection exists before using cursor
        cursor = connection.cursor()
        st.header("Sign Up")
        with st.form(key="signup_form", clear_on_submit=True):
            name = st.text_input("Name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            conf_password = st.text_input("Confirm Password", type="password")

            email = st.text_input("Email")
            # print(name, username, conf_password, email)

            if st.form_submit_button("Create Account"):

                # Checks if user already exists
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                if cursor.fetchone():
                    st.error("Username already taken, please choose different username")
                    return

                # checks password and confirm password do match
                elif password != conf_password:
                    st.error("Password and Confirm Password don't match.")
                    return 
                
                # check if any information not provided
                elif not (name and username and password and email):
                    st.error("All fields are required")

                # adds the user details to the database
                try:
                    # update db with users credentials
                    query = f"INSERT INTO users (name, username, password, email) VALUES ('{name}', '{username}', '{hash(password)}', '{email}');"
                    insert_details(query)
                    
                    # Fetch newly created users data (assuming you want to display it)
                    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                    registered_user = cursor.fetchone()  # Fetch the first (and likely only) row
                    
                    if connection is not None:
                        connection.close()
                        print("MYSQL connection is closed")

                    if registered_user:
                        st.success(f"Account created successfully! Welcome, {registered_user[1]}")  # Display username from fetched data
                    else:
                        st.error("An error occurred while fetching users data.")
                except Exception as e:
                    print(f"The error '{e}' occurred")
                    st.error("Some inter error occures!")

        # col1, col2 = st.columns(2)
        # with col1:
        st.write("Already have an account?")
        # with col2:
        if st.button("Login"):
            login_user(connection)

login_form
def login_user():
    if connection is not None:  # Check if connection exists before using cursor
        cursor = connection.cursor()
        st.header("LogIn")
        with st.form(key="login_form", clear_on_submit=True):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.form_submit_button("Login"):
                authentication_status = authenticate_user(username, hash(password))
                if authentication_status:
                    home() #redirect to homepage
                    st.session_state['is_authenticated'] = True
                    st.sidebar.write(f'Welcome, {username}')
                else:
                    st.error('Username/password is incorrect')

        # col1, col2 = st.columns(2)
        # with col1:
        st.write("Don't have an account?")
        # with col2:
        if st.button("Sign Up"):
            register_user()


authenticaton
def hash(password):
    """Hashes the password using bcrypt."""

    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()

def authenticate_user(username, password):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    credential = cursor.fetchone()  # Fetch the first (and likely only) row

    if password == credential[3]:
        return True
    else: 
        return False                    








if __name__ == "__main__":
    """Establish connection outside functions for reusability"""  
    connection =  db.create_connection("localhost", "root", "mysql", "disease")
    # db.create_table()
    pr.register_user(connection)







    pages = st.sidebar.title("Dashboard")
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Weather Forecast", "Login"])

    if app_mode == "Home": 
    
    elif app_mode == "About":
        
        st.header("About")
        st.markdown("""
                    #### About Dataset
                    This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                    This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                    A new directory containing 33 test images is created later for prediction purpose.
                    #### Content
                    1. train (70295 images)
                    2. test (33 images)
                    3. validation (17572 images)

                    """)

    elif app_mode == "Disease Recognition":
        #Tensorflow Model Prediction
        def model_prediction(test_image):
            model = tf.keras.models.load_model("../trained_plant_disease_model.keras")
            image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr]) #convert single image to batch
            predictions = model.predict(input_arr)
            return np.argmax(predictions) #return index of max element


        st.header("Disease Recognition")

        # Upload image section with columns for better layout
        col1, col2 = st.columns(2)
        with col1:
            test_image = st.file_uploader("Choose an Image:")
            if test_image is not None:
                with col2:
                    if st.button("Show Image"):
                        st.image(test_image, width=4, use_column_width=True)
            st.text("OR")
        # with col1:
            camera = st.camera_input("Capture Image", disabled=True)
            if camera:
                with col2:
                    if st.button("Show Image"):
                        st.image(camera, width=4, use_column_width=True)
                
                        
        #Predict button
        try: 
            with col2:
                if(st.button("Predict")):
            
                        
                    st.write("Our Prediction")
                    with st.spinner("Predicting.."):
                        if test_image:
                            result_index = model_prediction(test_image)
                        elif camera:
                            result_index= model_prediction(camera)
                        time.sleep(2)
                    
                    #Reading Labels
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
                whether_data = get_weather_forecast(city) 
                for data in whether_data["list"]:
                    city = city
                    weather_description = data["weather"][0]["description"]
                    icon_id = data["weather"][0]["icon"]
                    temperature = data["main"]["temp"]
                    feels_like = data["main"]["feels_like"]
                    humidity = data["main"]["humidity"]
                    wind_speed = data["wind"]["speed"]
                    date, time = data["dt_txt"].split()

                    # Get weather icon URL based on icon ID
                    icon_url = f"http://openweathermap.org/img/wn/{icon_id}@2x.png"
            except:
                st.error("Unable to fetch the data!!")

            else:
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
            
    else:
        st.error("Page not Found")
