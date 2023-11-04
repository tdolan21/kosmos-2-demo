from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image, ImageDraw, ImageFont
from typing import Optional
import base64
import json
import io
import os
import random

app = FastAPI()

# Load the model and processor
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

# Add CORS middleware
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8501",
    # Add any other origins from which you want to allow requests
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




def draw_bounding_boxes(image: Image, entities):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Define a color bank of hex codes
    color_bank = [
        "#0AC2FF", "#0AC2FF", "#30D5C8", "#F3C300", 
        "#47FF0A", "#C2FF0A", "#F7CA18", "#D91E18", 
        "#FF0AC2", "#FF0A47", "#DB0A5B", "#1E824C"
    ]
    
    # Use a built-in PIL font at a larger size if arial.ttf is not available
    try:
        # Try to use a specific font
        font_size = 20
        font = ImageFont.truetype("assets/arial.ttf", font_size)
    except IOError:
        # Fall back to the default PIL font at a larger size
        font_size = 20
        font = ImageFont.load_default()

    for entity in entities:
        label, _, boxes = entity
        for box in boxes:
            box_coords = [
                box[0] * width,  # x_min
                box[1] * height, # y_min
                box[2] * width,  # x_max
                box[3] * height  # y_max
            ]
            
            # Randomly choose colors for the outline and text fill
            outline_color = random.choice(color_bank)
            text_fill_color = random.choice(color_bank)
            
            draw.rectangle(box_coords, outline=outline_color, width=4)
            
            # Adjust the position to draw text based on font size
            text_position = (box_coords[0] + 5, box_coords[1] - font_size - 5)
            draw.text(text_position, label, fill=text_fill_color, font=font)

    return image

@app.post("/detect/")
async def detect_and_draw_objects(
    file: UploadFile = File(...),
    custom_prompt: Optional[str] = Form(None)  # Use 'Optional' if the prompt is not mandatory
):
    if file.content_type.startswith('image/'):
        # Read the image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Use the custom prompt if provided, otherwise use the default
        prompt = custom_prompt if custom_prompt else "<grounding><phrase> a snowman</phrase>"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Process the generated text
        processed_text, entities = processor.post_process_generation(generated_text)

        # Draw bounding boxes on the image
        annotated_image = draw_bounding_boxes(image, entities)

        # Convert the annotated image to base64
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Prepare the response data
        response_data = {
            "description": processed_text,
            "entities": entities,
            "image_base64": img_str
        }

        # Return the JSON response with the image and detection data
        return JSONResponse(content=response_data)
    else:
        raise HTTPException(status_code=400, detail="File type not supported")

