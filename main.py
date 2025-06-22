import time
import os
import uuid
import subprocess
import logging


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from google import genai
from google.genai import types

logger = logging.getLogger("uvicorn")

# Initialize Google GenAI client (assumes GOOGLE_API_KEY is in env)
client = genai.Client()

app = FastAPI()


class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate-video/")
async def generate_video(request: PromptRequest):
    prompt = request.prompt
    uid = str(uuid.uuid4())[:8]  # Unique ID for filenames

    logger.info("Video generation started")
    try:
        operation = client.models.generate_videos(
            model="veo-2.0-generate-001",
            prompt=prompt,
            config=types.GenerateVideosConfig(
                person_generation="dont_allow",
                aspect_ratio="16:9",
            ),
        )

        # Poll for completion
        while not operation.done:
            time.sleep(10)
            operation = client.operations.get(operation)

        logger.info("Video generation completed")
        # Get first video
        video = operation.response.generated_videos[0]
        video_file = client.files.download(file=video.video)

        mp4_path = f"{uid}_video.mp4"
        gif_path = f"{uid}_preview.gif"

        # Save MP4
        with open(mp4_path, "wb") as f:
            f.write(video_file)

        # Convert MP4 to GIF using ffmpeg
        convert_command = [
            "ffmpeg", "-y", "-i", mp4_path,
            "-vf", "fps=10,scale=320:-1:flags=lanczos",
            "-t", "3", gif_path
        ]
        subprocess.run(convert_command, check=True)

        return FileResponse(gif_path, media_type="image/gif", filename="preview.gif")

    except Exception as e:
        logger.info(f"Video generation {e}")
        raise HTTPException(status_code=500, detail=str(e))
