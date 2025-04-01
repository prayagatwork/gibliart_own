from openai import OpenAI
from diffusers import StableDiffusionPipeline
import torch

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=apikey,
)

# Get image description using OpenAI model
completion = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-Title": "<YOUR_SITE_NAME>",
    },
    extra_body={},
    model="google/gemini-2.5-pro-exp-03-25:free",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    },
                },
            ],
        }
    ],
)

# Extract the description for the next model
# Debugging: Print full API response
print("API Response:", completion)

# Handle cases where API response is None
if not completion or not completion.choices or not completion.choices[0].message.content:
    print("⚠️ API response is empty! Check API key or request format.")
    exit()

prompt = completion.choices[0].message.content.strip()
print(f"Using prompt: {prompt}")

# Load Ghibli-Diffusion Model
model_id = "nitrosocke/Ghibli-Diffusion"

# Choose device (MPS for Mac, CUDA for Nvidia, CPU fallback)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if device == "mps" else torch.float16

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe.to(device)

# Generate Ghibli-style image
image = pipe(prompt).images[0]
image.save("./ghibli_output.png")
print("Image saved as ghibli_output.png")
