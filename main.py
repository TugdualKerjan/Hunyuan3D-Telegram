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

model_path = "tencent/Hunyuan3D-2"

pipeline1 = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
pipeline2 = Hunyuan3DPaintPipeline.from_pretrained(model_path)


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
    # mesh.export("output.glb")

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

    # Generate the file URL
    file_url = photo_file.file_path
    # Download the file using requests
    response = requests.get(file_url)
    response.raise_for_status()  # Check for HTTP request errors

    # Save the file locally
    with open("input.png", "wb") as file:
        file.write(response.content)
    await update.message.reply_text("Photo downloaded successfully as 'input.png'.")

    # Verify the downloaded file is a valid image
    try:
        image = Image.open("input.png")
        image.verify()  # Verify that it is an actual image
        await update.message.reply_text("The image file is valid.")
    except UnidentifiedImageError:
        await update.message.reply_text("The downloaded file is not a valid image.")
        return

    await context.bot.send_message(chat_id=update.message.chat_id, text="Processing...")
    glb_file = await process_image("input.png")

    with open(glb_file, "rb") as file:
        doc_message = await context.bot.send_document(
            chat_id=update.message.chat_id, document=file
        )

        file_id = doc_message.document.file_id
        file = await context.bot.get_file(file_id)
        file_url = file.file_path
        await context.bot.send_message(
            chat_id=update.message.chat_id, text=f"File URL: {file_url}"
        )


app = Application.builder().token(TOKEN).build()
app.add_handler(MessageHandler(filters.PHOTO, handle_image))
app.run_polling()
