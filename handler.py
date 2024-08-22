import torch
import boto3
import os
import uuid
from PIL import Image
from IPython.display import clear_output
from diffusers import StableDiffusion3Pipeline
from googletrans import Translator
import runpod

clear_output()

# Load the pipeline based on GPU availability
gpu = torch.cuda.is_available()

try:
    if gpu:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=torch.float16
        ).to("cuda")
        print("GPU usage")
        print("OK!")
    else:
        print(
            "GPU was not found. Switching to CPU usage. The number of steps is automatically set to 1 for higher performance"
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=torch.float16
        ).to("cuda")
        clear_output()
        print("CPU usage")
        print("OK!")
except Exception as e:
    print(f"Error initializing pipeline: {e}")

# your_handler.py
def handler(job):
    job_input = job["input"]
    Prompt = job_input.get("prompt")
    aws_id = job_input.get("aws_id")
    aws_secret = job_input.get("aws_secret")
    bucket_name = job_input.get("s3_bucket")

    if not Prompt:
        return {
            "error": "Prompt is missing!"
        }
    #setup s3
    aws_access_key_id = aws_id
    aws_secret_access_key = aws_secret

    # Initialize boto3 
    boto3_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # Create an S3 client 
    s3_client = boto3_session.client('s3')

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
        presigned_urls = []  # Array to hold presigned URLs

        for _ in range(4):  # Generate 4 different variations
            # Generate the image based on the available hardware
            if not gpu:
                image = pipe(
                    prompt=PromptEN, 
                    num_inference_steps=28, 
                    guidance_scale=7.0, 
                    height=576, 
                    width=1024
                ).images[0]
            else:
                image = pipe(
                    prompt=PromptEN, 
                    num_inference_steps=28, 
                    guidance_scale=7.0, 
                    height=576, 
                    width=1024
                ).images[0]

            # Generate a unique filename for the image
            unique_filename = f"generated_image_{uuid.uuid4().hex}.png"
            
            # Save the image to a file
            image.save(unique_filename)
            
            # Upload the image to S3
            s3_key = f"images/{unique_filename}"
            s3_client.upload_file(unique_filename, bucket_name, s3_key)
            
            # Generate a presigned URL
            presigned_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': s3_key},
                ExpiresIn=3600  # Set expiration to 1 hour
            )
            presigned_urls.append(presigned_url)  # Append the URL to the array
            
            # Delete the image after uploading
            os.remove(unique_filename)

        return {"status": "success", "message": "Your job results", "image_urls": presigned_urls}

    except Exception as e:
        return {"status": "failed", "message": f"Error generating or uploading images: {e}"}

runpod.serverless.start({"handler": handler})
