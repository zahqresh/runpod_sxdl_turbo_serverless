!pip install diffusers transformers accelerate googletrans==3.1.0a0 numba --upgrade
import torch
import requests
import os
import uuid
from PIL import Image

gpu = torch.cuda.is_available()

from IPython.display import clear_output
from diffusers import AutoPipelineForText2Image
from googletrans import Translator
import runpod
clear_output()

# Load the pipeline based on GPU availability
try:
    if gpu:
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.to("cuda")
        clear_output()
        print("GPU usage")
        print("OK!")
    else:
        print(
            "GPU was not found. Switching to CPU usage. The number of steps is automatically set to 1 for higher performance"
        )
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
        clear_output()
        print("CPU usage")
        print("OK!")
except Exception as e:
    print(f"Error initializing pipeline: {e}")


# your_handler.py
def handler(job):
    job_input = job["input"]

    Prompt = job_input.get("prompt")

    if not Prompt:
        return {
            "error": "Prompt is missing!"
        }

    # Set the prompt and steps
    Steps = 4  # @param {type:"number"}

    try:
        # Translate the prompt to English
        translator = Translator()
        translation = translator.translate(Prompt)
        PromptEN = translation.text
    except Exception as e:
        return {"status": "failed", "message": f"Error translating prompt: {e}"}

    try:
        image_links = []  # Array to hold image links

        for _ in range(4):  # Generate 4 different variations
            # Generate the image based on the available hardware
            if not gpu:
                image = pipe(prompt=PromptEN, num_inference_steps=1, guidance_scale=0.0).images[0]
            else:
                image = pipe(prompt=PromptEN, num_inference_steps=Steps, guidance_scale=0.0).images[0]

            # Generate a unique filename for the image
            unique_filename = f"generated_image_{uuid.uuid4().hex}.png"

            # Save the image to a file
            image.save(unique_filename)

            # Upload the image to file.io
            with open(unique_filename, "rb") as f:
                response = requests.post(
                    "https://file.io/",
                    files={"file": (unique_filename, f, "image/png")}
                )

            # Check if the upload was successful
            if response.status_code == 200:
                file_url = response.json().get("link")
                image_links.append(file_url)  # Append the link to the array
            else:
                image_links.append(f"Error uploading image: {response.text}")

            # Delete the image after uploading
            os.remove(unique_filename)

        return {"status": "success", "message": "Your job results", "image_urls": image_links}

    except Exception as e:
        return {"status": "failed", "message": f"Error generating or uploading images: {e}"}


runpod.serverless.start({"handler": handler})
