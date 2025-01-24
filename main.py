import os
import sys
import imageio
import requests
from telegram.ext import Application, MessageHandler, filters
import torch
from PIL import Image, UnidentifiedImageError
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline,
    FaceReducer,
    FloaterRemover,
    DegenerateFaceRemover,
)
from hy3dgen.texgen import Hunyuan3DPaintPipeline
import trimesh
import numpy as np


os.environ["PYOPENGL_PLATFORM"] = "egl"  # Use EGL for headless rendering
os.environ["PYGLET_HEADLESS"] = "1"

model_path = "tencent/Hunyuan3D-2"
pipeline1 = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
pipeline2 = Hunyuan3DPaintPipeline.from_pretrained(model_path)

TOKEN = "7293609338:AAEU4yi_QV-rWNhBuTxQAldR6zeTIKtgvk8"


async def convert_glb_to_gif(glb_path):
    # Load the GLB file
    original_mesh = trimesh.load(glb_path)

    frames = []
    for angle in range(0, 360, 10):
        # Create a fresh copy of the mesh for this angle
        mesh = original_mesh.copy()

        # Create rotation matrix for this absolute angle
        rotation = trimesh.transformations.rotation_matrix(
            angle * np.pi / 180.0, [0, 1, 0]
        )

        # Apply rotation to the fresh copy
        mesh = mesh.apply_transform(rotation)

        # Create scene with rotated mesh
        scene = trimesh.Scene([mesh])
        scene.camera.resolution = [256, 256]
        scene.camera_transform = scene.camera.look_at(
            points=mesh.bounds, center=mesh.centroid, distance=mesh.scale * 2.0
        )

        # Export to PNG
        temp_file = f"frame_{angle}.png"
        with open(temp_file, "wb") as f:
            f.write(scene.save_image(resolution=[256, 256]))

        if os.path.exists(temp_file):
            frames.append(np.array(Image.open(temp_file)))
            os.remove(temp_file)  # Clean up temp files

    if frames:
        imageio.mimsave("output.gif", frames, fps=15)
        return "output.gif"
    return None


async def process_image(image_path):
    rembg = BackgroundRemover()
    image = Image.open(image_path)
    image = image.resize((1024, 1024))
    if image.mode == "RGB":
        image = rembg(image)
    mesh = pipeline1(
        image=image,
        num_inference_steps=30,
        mc_algo="mc",
        generator=torch.manual_seed(2025),
    )[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    mesh = pipeline2(mesh, image=image)
    mesh.export("texture.glb")
    return "texture.glb"


async def handle_image(update, context):
    if not update.message.photo:
        await context.bot.send_message(
            chat_id=update.message.chat_id, text="Please send an image"
        )
        return

    photo_file = await update.message.photo[-1].get_file()
    file_url = photo_file.file_path
    response = requests.get(file_url)
    response.raise_for_status()

    with open("input.png", "wb") as file:
        file.write(response.content)
    await update.message.reply_text("Photo downloaded successfully as 'input.png'.")

    try:
        image = Image.open("input.png")
        image.verify()
        await update.message.reply_text("The image file is valid.")
    except UnidentifiedImageError:
        await update.message.reply_text("The downloaded file is not a valid image.")
        return

    await context.bot.send_message(chat_id=update.message.chat_id, text="Processing...")
    glb_file = await process_image("input.png")

    # Convert GLB to GIF
    gif_file = await convert_glb_to_gif(glb_file)

    # Send both GLB and GIF
    with open(glb_file, "rb") as file:
        doc_message = await context.bot.send_document(
            chat_id=update.message.chat_id, document=file
        )
        file_id = doc_message.document.file_id
        file = await context.bot.get_file(file_id)
        file_url = file.file_path
        await context.bot.send_message(
            chat_id=update.message.chat_id, text=f"GLB File URL: {file_url}"
        )

    # Send the GIF
    with open(gif_file, "rb") as file:
        await context.bot.send_animation(
            chat_id=update.message.chat_id,
            animation=file,
            caption="Rotating view of the 3D model",
        )


app = Application.builder().token(TOKEN).build()
app.add_handler(MessageHandler(filters.PHOTO, handle_image))
app.run_polling()
