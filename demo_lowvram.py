from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
import torch
from diffusers.models import AutoencoderKL

torch.cuda.empty_cache()

# Load the VAE and pipeline models
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
model_ckpt = "stabilityai/sdxl-turbo"
pipe = DemoFusionSDXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16, vae=vae)
pipe = pipe.to("cuda")
# Define your prompts
prompt = "Envision a portrait of a teacher, his face thoughtful, framed by round glasses. His eyes, blue. His attire, simple yet dignified."
negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

# Generate images using the pipeline
images = pipe(prompt, negative_prompt=negative_prompt,
              height=4096, width=4096, view_batch_size=12, stride=96,
              num_inference_steps=20, guidance_scale=1.5,
              cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
              multi_decoder=True, show_image=True, lowvram=True
             )

# Save the generated images
for i, image in enumerate(images):
    image.save(f'image_{i}.png')
    
print("Image resolution step 1: ", images[0].size)
print("Image resolution step 2: ", images[1].size)
print("Image resolution step 3: ", images[2].size)
