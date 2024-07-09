import streamlit as st
import tensorflow as tf
import numpy as np
import os


# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # Coverting Single Image to batch
    pred = model.predict(input_arr)
    return np.argmax(pred) # return index of max element


#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",['Home', 'About','Prediction'])

# main page
if (app_mode == 'Home'):
    st.header("FRUITS & VEGETATION RECONGIZATIONN SYSTEM")
    image_path = "home_img.jpg"
    st.image(image_path)


# about project
elif (app_mode == 'About'):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text('This dataset encompasses images of various fruits and vegetables, providing a diverse collection for image recognition tasks. The included food items are:')
    st.code("Fruits: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango")
    st.code("Vegetables: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant")
    st.subheader("Content")
    st.text("The dataset is organized into three main folders:")
    st.text("1. Train: Contains 100 images per category.")
    st.text("2. Test: Contains 10 images per category.")
    st.text("3. Validation: Contains 10 images per category.")

# Prediction Page
elif(app_mode == "Prediction"):
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:", accept_multiple_files=True)
    print(test_image)
    if(st.button("Show Image")):
        st.image(test_image, width=4, use_column_width=True)

    # predict button
    if(st.button("Predict")):
        for j in test_image:
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(j)
            # Reading Labels
            with open('labels.txt') as f:
                content = f.readlines()
            label = []
            for i in content:
                label.append(i[:-1])
            st.success("Model is Predicting it's a {} ".format(label[result_index]))

            # Extract the filename without extension
            filename_base = label[result_index]

            # Create a folder for the filename base if it doesn't exist
            folder_path = os.path.join("uploads", filename_base)
            os.makedirs(folder_path, exist_ok=True)

            # Save the file to the corresponeding folder
            file_path = os.path.join(folder_path, j.name)
            with open(file_path, "wb") as f:
                f.write(j.read())

            # print(val)