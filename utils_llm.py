from api_resource import *
from openai import OpenAI
from io import BytesIO
import anthropic
import google.generativeai as genai
from utils import load_image



def get_gpt_model():
    model = OpenAI(api_key=openai_api)
    return model



def prompt_gpt4o(model, prompt, image_path=None, demonstrations=None):
    """
    Generates a response from the GPT-4o model using optional image input.

    Args:
        model: The GPT-4o model instance.
        prompt (str): The text prompt for the model.
        image_path (str, optional): The file path of the image. Defaults to None.

    Returns:
        str: The model's response text.
    """
    try:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]

        if image_path:
            # Process the image into Base64 format
            with open(image_path, "rb") as image_file:
                base64_image = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

            # Append image information to the message content
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                }
            )

        # Generate response from the model
        response = model.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.00,
            max_tokens=4096,
        )

        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"An error occurred while querying the GPT-4o model: {e}")

def prompt_gpt4(model, prompt):
    try:
        response = model.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.01,
            max_tokens=4096,
        )
        res = response.choices[0].message.content
        return res
    except Exception as e:
        print(prompt)
        return ""

def get_claude_model():
    client = anthropic.Anthropic(api_key=claude_api)
    return client


import base64
import io
import time
from PIL import Image

def prompt_claude(client, prompt, image_path=None, demonstrations=None):
    try:
        # Prepare image data if image_path is provided
        image_data = None
        media_type = None

        if image_path:
            if isinstance(image_path, str):
                with Image.open(image_path) as img:
                    # Supported formats and their MIME types
                    format_to_mime = {
                        "JPEG": "image/jpeg",
                        "PNG": "image/png",
                        "WEBP": "image/jpeg",  # Convert WEBP to JPEG
                    }

                    image_format = img.format.upper()
                    if image_format not in format_to_mime:
                        raise ValueError(f"Unsupported image format: {image_format}")

                    # Convert WEBP to JPEG if necessary
                    if image_format == "WEBP":
                        img = img.convert("RGB")  # Convert WEBP to RGB for JPEG compatibility
                        image_format = "JPEG"  # Update format for saving

                    media_type = format_to_mime[image_format]

                    # Save the image to a buffer and encode it in Base64
                    buffered = io.BytesIO()
                    img.save(buffered, format=image_format)
                    image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
            else:
                img_byte_arr = BytesIO()
                image_path.save(img_byte_arr, format="PNG")
                image_path = img_byte_arr.getvalue()
                image_data = base64.b64encode(image_path).decode("utf-8")
                media_type = "image/png"



        # Construct the API message with few-shot demonstrations
        messages = []

        # Add demonstrations as context
        if demonstrations:
            for demo_pair in demonstrations:
                for demo in demo_pair:
                    demo_image_path, demo_prompt, demo_answer = demo

                    # Prepare demo image data
                    demo_image_data = None
                    demo_media_type = None

                    if demo_image_path:
                        with Image.open(demo_image_path) as demo_img:
                            demo_image_format = demo_img.format.upper()
                            if demo_image_format not in format_to_mime:
                                raise ValueError(f"Unsupported image format: {demo_image_format}")

                            if demo_image_format == "WEBP":
                                demo_img = demo_img.convert("RGB")
                                demo_image_format = "JPEG"

                            demo_media_type = format_to_mime[demo_image_format]

                            demo_buffered = io.BytesIO()
                            demo_img.save(demo_buffered, format=demo_image_format)
                            demo_image_data = base64.b64encode(demo_buffered.getvalue()).decode("utf-8")

                    # Add demo image and prompt to messages
                    if demo_image_data and demo_media_type:
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": demo_media_type,
                                        "data": demo_image_data,
                                    },
                                },
                                {"type": "text", "text": demo_prompt},
                            ]
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": demo_prompt}]
                        })

                    # Add demo answer to messages
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": demo_answer}]
                    })

        # Add the actual user prompt and image
        user_message = {"role": "user", "content": []}

        if image_data and media_type:
            user_message["content"].append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                }
            )

        user_message["content"].append({"type": "text", "text": prompt})
        messages.append(user_message)

        # Query the API
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Specify your model version
            max_tokens=4096,
            messages=messages,
        )
        time.sleep(1)
        return response.content[0].text

    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")


def get_open_model(model_name):
    return None

def prompt_open_model(model_name):
    return None


# def prompt_claude(client, prompt, image_id):
#     # Determine the media type based on the file extension
#     media_type, _ = mimetypes.guess_type(image_id)
#     print(media_type)
#     if not media_type:
#         raise ValueError("Unsupported file type. Ensure the file is a valid JPEG or PNG image.")
#
#     # Open and encode the image
#     with open(image_id, "rb") as image_file:
#         image_data = base64.b64encode(image_file.read()).decode("utf-8")
#
#     # Create a message with the image and text content
#     message = client.messages.create(
#         model="claude-3-5-sonnet-20241022",  # Specify your model version
#         max_tokens=1024,
#         messages=[
#             {
#                 "role": "user",
#                 "content": {
#                     "type": "composite",
#                     "parts": [
#                         {
#                             "type": "image",
#                             "source": {
#                                 "type": "base64",
#                                 "media_type": media_type,
#                                 "data": image_data,
#                             },
#                         },
#                         {
#                             "type": "text",
#                             "text": prompt,
#                         },
#                     ],
#                 },
#             },
#         ],
#     )
#
#     return message["content"][0]["text"]

def get_gemini_model():
    genai.configure(api_key=gemini_api)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model



def prompt_gemini(client, prompt, image_path=None, demonstrations=None):
    """
    Generates a response from the Gemini model using optional image input and few-shot demonstrations.

    Args:
        client: The Gemini model client.
        prompt (str): The text prompt for the model.
        image_path (str, optional): The path to the image to load. Defaults to None.
        demonstrations (list, optional): A list of few-shot demonstrations. Defaults to None.

    Returns:
        str: The model's response text.
    """
    try:
        # Prepare the inputs as a list of parts
        parts = []

        # Add demonstrations as context
        if demonstrations:
            for demo_pair in demonstrations:
                for demo in demo_pair:
                    demo_image_path, demo_prompt, demo_answer = demo

                    # Add demo image if image path is provided
                    if demo_image_path:
                        demo_image_data = load_image(demo_image_path)
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",  # Adjust mime type if necessary
                                "data": demo_image_data
                            }
                        })

                    # Add demo prompt
                    parts.append({"text": f"Example: {demo_prompt}"})

                    # Add demo answer
                    parts.append({"text": f"{demo_answer}"})

        # Add the actual user prompt and image
        if image_path:
            if isinstance(image_path, str):
                image_data = load_image(image_path)
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",  # Adjust mime type if necessary
                        "data": image_data
                    }
                })
            elif not isinstance(image_path, bytes):
                img_byte_arr = BytesIO()
                image_path.save(img_byte_arr, format="PNG")
                image_path = img_byte_arr.getvalue()
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",  # Adjust mime type if necessary
                        "data": image_path
                    }
                })
            else:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",  # Adjust mime type if necessary
                        "data": image_path
                    }
                })

        # Add the user prompt
        parts.append({"text": prompt})

        # Generate content using the provided parts
        response = client.generate_content({"parts": parts})
        return response.text

    except Exception as e:
        raise RuntimeError(f"An error occurred while querying the Gemini model: {e}")


def get_commercial_model(model_name):
    model_map = {
        "gpt4o": get_gpt_model,
        "claude": get_claude_model,
        "gemini": get_gemini_model,
    }
    if model_name in model_map:
        return model_map[model_name]()
    raise ValueError(f"Unknown model name: {model_name}")

def prompt_commercial_model(client, model_name, prompt, image_id, demonstrations=None):
    print(prompt)
    prompt_map = {
        "gpt4o": prompt_gpt4o,
        "claude": prompt_claude,
        "gemini": prompt_gemini,
    }
    if model_name in prompt_map:
        try:
            return prompt_map[model_name](client, prompt, image_id, demonstrations)
        except Exception as e:
            print(image_id, str(e))
            return ""
    raise ValueError(f"Unknown model name: {model_name}")

if __name__ == "__main__":


    for item in ["gpt4o", "claude", "gemini"]:
        client = get_commercial_model(item)
        print(prompt_commercial_model(client, item, "what is in this image?", "assets/view.jpg"))

