import streamlit as st
import requests
from PIL import Image
import io
import base64

# Streamlit interface
st.title('Kosmos-2 API Client')
st.sidebar.image('assets/thumbnail.png', use_column_width=True)

# Endpoint URL
api_url = "http://localhost:8000/detect/"

# Default prompt
default_prompt = "<grounding><phrase> a snowman</phrase>"
prompt = st.sidebar.text_input('Enter your prompt:', value=default_prompt)

# File uploader allows user to add their own image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Convert the uploaded image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    img_byte = buffered.getvalue()
    
    files = {
    'file': ('image.jpg', img_byte, 'image/jpeg'),  # 'file' must match the FastAPI parameter name
    }
    data = {
        'custom_prompt': prompt  # Send as form data
    }

    # Post the image and the prompt to the endpoint
    st.write("Sending image and prompt to the API...")
    response = requests.post(api_url, files=files, data=data)  # Use 'data' for the custom prompt

        
    # Check the response
    if response.status_code == 200:
        st.write("Response received from the API!")
        response_data = response.json()
        
        # Display the base64 image
        base64_image = response_data['image_base64']
        st.write("Annotated Image:")
        st.image(base64.b64decode(base64_image), caption='Processed Image.', use_column_width=True)
        
        # Display the other response data
        st.write("Description:", response_data['description'])
        st.sidebar.info("Entities Detected:")
        st.sidebar.json(response_data['entities'])
    else:
        st.error("Failed to get response from the API")
