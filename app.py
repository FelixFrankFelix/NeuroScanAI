import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
#import matplotlib.pyplot as plt
#import io

# Define the path to the saved model

model_directory = r"C:\Users\FRANKFELIXAI\Desktop\OutApp Project\Final year project\Neuroscanaimodel.h5"

model = tf.keras.models.load_model(model_directory)

# Function to make predictions
def predict(image):
    image = img_to_array(image)
    image = image / 255
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    prediction = model.predict(image)
    return prediction

# Load and resize the brain icon image from a file
brain_icon = Image.open(r"C:\Users\FRANKFELIXAI\Desktop\OutApp Project\Final year project\NeuroScanAI_Logo.png")
brain_icon = brain_icon.resize((50, 50))

# Streamlit app
st.image(brain_icon, use_column_width=False, width=50)
st.title("NeuroScanAI")
st.write("Welcome to NeuroScanAI, an AI-powered Brain Tumor Classification tool.")
patient_name = st.text_input("Enter your Name: ")

recommendation_message_PITUITARY = f"""
Dear {patient_name},

I hope this message finds you well. Based on the analysis conducted by NeuroScanAI, it has been identified that you have a PITUITARY brain tumor. While the AI model provides valuable insights, it's essential to remember that this is not a replacement for professional medical advice.

A PITUITARY tumor can have various impacts on your health, and its management may require a personalized treatment plan. I strongly recommend that you schedule an appointment with a medical specialist or neurologist who can conduct a comprehensive evaluation of your condition. They will be able to provide you with a tailored treatment strategy, which may include further diagnostic tests, consultations, or potential interventions.

Please do not hesitate to seek medical attention promptly. Early diagnosis and intervention can significantly influence the outcome and your overall well-being.

Your health is of utmost importance, and a healthcare professional can guide you through this process, addressing any concerns or questions you may have.

Take care, {patient_name}, and prioritize your health.

Warm regards,
NeuroScanAI
"""

recommendation_message_GLIOIMA = f"""
Dear {patient_name},

I hope this message finds you well. NeuroScanAI has identified that you have a GLIOMA brain tumor. While the AI model provides valuable insights, it's crucial to understand that this is not a substitute for professional medical guidance.

GLIOMA tumors vary in severity, and managing them requires personalized medical attention. I strongly advise you to schedule an appointment with a neurologist or oncologist who can conduct a thorough assessment of your condition. They will develop a tailored treatment plan, which may include additional tests, consultations, or potential interventions.

Please prioritize seeking medical care promptly. Early diagnosis and treatment can have a significant impact on your prognosis and overall well-being.

Remember that your health is paramount, and healthcare professionals are here to support you through this journey. If you have any concerns or questions, they are the best resource.

Take good care of yourself, {patient_name}, and stay positive.

Warm regards,
NeuroScanAI
"""

recommendation_message_MENINGIOMA = f"""
Dear {patient_name},

I hope this message finds you well. NeuroScanAI's analysis indicates the presence of a MENINGIOMA brain tumor. While our AI model offers insights, it's vital to remember that this does not replace professional medical advice.

MENINGIOMA tumors can have varying impacts on your health, and their management requires individualized care. I strongly recommend scheduling an appointment with a medical specialist or neurologist who can provide a comprehensive evaluation. They will design a personalized treatment plan, which may involve additional tests, consultations, or potential interventions.

I encourage you to seek medical attention promptly. Early diagnosis and intervention can significantly affect your prognosis and overall well-being.

Your health is paramount, and healthcare professionals are here to assist you with any concerns or questions you may have. 

Take care, {patient_name}, and prioritize your health. 

Warm regards, 
NeuroScanAI
"""

recommendation_message_NoTumor = f"""
Dear {patient_name},

I'm delighted to share that NeuroScanAI's analysis did not detect any brain tumors in your MRI scan. This is great news! 

However, please remember that it's crucial to maintain good brain health. Here are some tips:

1. Stay mentally active: Engage in activities that challenge your brain, such as puzzles, reading, or learning new skills.

2. Eat a balanced diet: Consume foods rich in antioxidants, omega-3 fatty acids, and other nutrients that support brain health.

3. Regular exercise: Physical activity promotes blood flow to the brain and enhances cognitive function.

4. Get enough sleep: Quality sleep is essential for brain recovery and cognitive performance.

5. Stay hydrated: Proper hydration is vital for brain function.

6. Manage stress: Practice stress-reduction techniques like meditation or yoga.

While the absence of a tumor is excellent news, maintaining good brain health is essential. If you have any concerns or questions about brain health, do not hesitate to consult with a healthcare professional.

Take care of your brain, {patient_name}, and prioritize your well-being.

Warm regards,
NeuroScanAI
"""


# Upload an MRI scan image
uploaded_image = st.file_uploader("Upload an MRI scan image, and we will help you classify the results. ", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded MRI Image", use_column_width=True)
    image = load_img(uploaded_image, target_size=(128, 128))

    # Make a prediction
    prediction = predict(image)
    predict_index = np.argmax(prediction)
    tumor_types = ["PITUITARY", "No tumor", "MENINGIOMA", "GLIOMA"]
    
    # Create an empty placeholder for the initial diagnosis
    initial_diagnosis_placeholder = st.empty()
    
    # Display the prediction
    if predict_index == 1:
        initial_diagnosis_placeholder.write(f"Dear {patient_name}, based on the analysis, there is no tumor detected. We recommend consulting a medical professional for confirmation and to ensure your brain health.")
    else:
        initial_diagnosis_placeholder.write(f"Dear {patient_name}, based on the analysis, there is a tumor of type {tumor_types[predict_index]}. We recommend consulting a medical professional for further evaluation and personalized guidance")

    # More Info
    if st.button("More Information"):
        initial_diagnosis_placeholder.empty() 
     # Clear the initial diagnosis
        if predict_index == 0:
            st.write(recommendation_message_PITUITARY)
        if predict_index == 3:
            st.write(recommendation_message_GLIOIMA)
        if predict_index == 2:
            st.write(recommendation_message_MENINGIOMA)
        if predict_index == 1:
            st.write(recommendation_message_NoTumor)


    # Expert Option
    if st.button("Expert Option"):
        st.subheader("Prediction Percentages")
        percentages = [round((score / sum(prediction[0])) * 100, 2) for score in prediction[0]]
        for i, tumor_type in enumerate(tumor_types):
            st.write(f"{tumor_type}: {percentages[i]}%")
        
        if predict_index != 1:
            st.subheader(f"Details on {tumor_types[predict_index]} Tumor:")
            if predict_index == 0:  # PITUITARY
                st.write("A PITUITARY tumor is typically a non-cancerous growth located in the pituitary gland.")
                st.write("Common symptoms include headaches, vision problems, and hormonal imbalances.")
                st.write("Consult a medical specialist for a comprehensive evaluation and treatment plan.")
            elif predict_index == 2:  # MENINGIOMA
                st.write("A MENINGIOMA is usually a slow-growing, non-cancerous tumor of the membranes surrounding the brain.")
                st.write("Symptoms may include headaches, seizures, or neurological issues.")
                st.write("Consult a medical specialist for further diagnosis and guidance.")
            elif predict_index == 3:  # GLIOMA
                st.write("A GLIOMA is a type of brain tumor that can be cancerous and may infiltrate brain tissue.")
                st.write("Symptoms vary depending on the location and size.")
                st.write("Immediate consultation with a neurologist or oncologist is recommended.")


        




    
    # Display the MRI image again with prediction
    #st.image(uploaded_image, caption="MRI Image with Prediction", use_column_width=True)