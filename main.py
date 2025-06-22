import cv2
import torch
import os
import subprocess
import time
import copy
import shutil
import trimesh
import numpy as np
import open3d as o3d
import tempfile
import logging
import uvicorn
import uuid
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse

from google import genai
from google.genai import types

# Assuming fast3r library is installed and its components are available
# These imports are based on your provided script
# from fast3r.viz.video_utils import extract_frames_from_video
from fast3r.dust3r.viz import cat_meshes, pts3d_to_trimesh
from fast3r.dust3r.utils.image import load_images, rgb
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

logger = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)

# --- Initialize Google GenAI client (assumes GOOGLE_API_KEY is in env) ---
# It's good practice to handle potential missing keys gracefully.
try:
    client = genai.Client()
except Exception as e:
    logger.error(f"Failed to initialize Google GenAI Client: {e}")
    client = None

# --- FastAPI App Initialization ---

app = FastAPI(
    title="Fast3R 3D Reconstruction API",
    description="Generate a video from a prompt, then reconstruct it into a 3D mesh.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Global Variables for Model, Device, and Video Cache ---
model = None
lit_module = None
device = None
# Simple in-memory cache to store the path of the last generated video.
# For a production system, a more robust cache like Redis would be better.
VIDEO_CACHE = {"latest_video_path": None, "latest_video_uid": None}


# --- Model Loading on Startup ---

@app.on_event("startup")
async def startup_event():
    """
    Loads the Fast3R model into memory when the API server starts.
    """
    global model, lit_module, device
    logger.info("--- Loading Model ---")
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model from Hugging Face Hub
    try:
        model_name = "jedyang97/Fast3R_ViT_Large_512"
        model = Fast3R.from_pretrained(model_name)
        model = model.to(device)
        
        # Wrap the model in the lightning module for helper functions
        lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
        
        # Set to evaluation mode
        model.eval()
        lit_module.eval()
        
        logger.info(f"--- Model Loaded Successfully on {device} ---")
    except Exception as e:
        logger.error(f"--- Failed to load model: {e} ---")
        # You might want to prevent the app from starting if the model fails to load.
        model = None
        lit_module = None


# --- Helper Functions ---

def create_mesh_from_preds(
    preds, 
    views,
    export_ply_path,
    min_conf_thr_percentile=30,
    flip_axes=True
):
    """
    Generates and exports a combined 3D mesh from model predictions.
    """
    logger.info("--- Generating 3D Mesh ---")
    meshes = []
    for i, pred in enumerate(preds):
        pts3d = pred['pts3d_in_other_view'].cpu().numpy().squeeze()
        img_rgb = views[i]['img'].cpu().numpy().squeeze().transpose(1, 2, 0)
        conf = pred['conf'].cpu().numpy().squeeze()
        conf_thr = np.percentile(conf, min_conf_thr_percentile)
        mask = conf > conf_thr
        img_rgb_uint8 = ((img_rgb + 1) * 127.5).astype(np.uint8).clip(0, 255)
        mesh_dict = pts3d_to_trimesh(img_rgb_uint8, pts3d, valid=mask)
        if mesh_dict['vertices'].shape[0] > 0:
            meshes.append(mesh_dict)

    if not meshes:
        raise ValueError("No valid mesh parts could be generated.")

    combined_mesh = trimesh.Trimesh(**cat_meshes(meshes))

    if flip_axes:
        combined_mesh.vertices[:, [1, 2]] = combined_mesh.vertices[:, [2, 1]]
        combined_mesh.vertices[:, 2] = -combined_mesh.vertices[:, 2]

    combined_mesh.export(export_ply_path)
    logger.info(f"--- Mesh successfully exported to {export_ply_path} ---")


def extract_frames_from_video(video_path, output_dir):
    """
    Extracts frames from a video file.
    """
    saved_frames = []
    
    # Use FFmpeg which is generally more robust
    try:
        logger.info(f"Extracting frames using FFmpeg from: {video_path}")
        frame_pattern = os.path.join(output_dir, "frame_%04d.jpg")
        ffmpeg_cmd = ["ffmpeg", "-i", video_path, "-vf", "fps=1", "-q:v", "2", frame_pattern]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        for file in sorted(os.listdir(output_dir)):
            if file.startswith("frame_") and file.endswith(".jpg"):
                saved_frames.append(os.path.join(output_dir, file))
        
        if not saved_frames:
             raise RuntimeError("FFmpeg ran but extracted no frames.")
        
        logger.info(f"Successfully extracted {len(saved_frames)} frames using FFmpeg.")

    except Exception as e:
        logger.error(f"FFmpeg extraction failed: {e}. Falling back to OpenCV.")
        # Fallback to OpenCV if FFmpeg fails
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps / 1)) # Aim for ~1 FPS
            frame_count = 0
            saved_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                    cv2.imwrite(output_path, frame)
                    saved_frames.append(output_path)
                    saved_count += 1
                frame_count += 1
            cap.release()
            logger.info(f"Extracted {len(saved_frames)} frames using OpenCV.")
        except Exception as cv_e:
            logger.error(f"OpenCV frame extraction also failed: {cv_e}")
            raise RuntimeError("Both FFmpeg and OpenCV failed to extract frames.") from cv_e
            
    return saved_frames

# --- API Endpoint for Video Generation ---

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate-video/")
async def generate_video(request: PromptRequest):
    """
    Generates a video from a text prompt using Google's VEO model,
    saves it locally, and returns a GIF preview.
    The path to the full MP4 video is cached on the server.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Google GenAI Client is not available.")

    prompt = request.prompt
    uid = str(uuid.uuid4())[:8]
    output_dir = tempfile.mkdtemp(prefix=f"{uid}_")
    mp4_path = os.path.join(output_dir, f"{uid}_video.mp4")
    gif_path = os.path.join(output_dir, f"{uid}_preview.gif")

    logger.info(f"Starting video generation for prompt: '{prompt}'")
    try:
        operation = client.models.generate_videos(
            model="veo-2.0-generate-001",
            prompt=prompt,
            config=types.GenerateVideosConfig(person_generation="dont_allow", aspect_ratio="16:9"),
        )
        while not operation.done:
            time.sleep(10)
            operation = client.operations.get(operation)

        logger.info("Video generation completed.")
        video = operation.response.generated_videos[0]
        video_file = client.files.download(file=video.video)

        with open(mp4_path, "wb") as f:
            f.write(video_file)

        # Cache the path for the reconstruction endpoint
        VIDEO_CACHE["latest_video_path"] = mp4_path
        VIDEO_CACHE["latest_video_uid"] = uid
        logger.info(f"Video saved to {mp4_path} and cached.")

        # Convert to GIF for preview
        convert_command = ["ffmpeg", "-y", "-i", mp4_path, "-vf", "fps=10,scale=320:-1:flags=lanczos", "-t", "3", gif_path]
        subprocess.run(convert_command, check=True)

        # The client needs to know the UID to make a follow-up request,
        # so we return it along with the GIF.
        # A better way is to return a JSON object with the UID and a URL to the GIF.
        # For now, let's return the GIF directly and the UID in a custom header.
        return FileResponse(
            gif_path, 
            media_type="image/gif", 
            filename=f"{uid}_preview.gif",
            headers={"X-Video-UID": uid} # Send UID back to client
        )

    except Exception as e:
        logger.exception("Video generation failed")
        shutil.rmtree(output_dir) # Clean up on failure
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")


# --- API Endpoint for Video Reconstruction ---

@app.post("/reconstruct-video/", response_class=FileResponse)
async def reconstruct_video_to_mesh():
    """
    Takes the most recently generated video from the local cache,
    performs 3D reconstruction, and returns a .ply mesh file.
    This endpoint should be called after a successful call to /generate-video/.
    """
    if not model or not lit_module:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please wait and try again.")
    
    # 1. Retrieve the video path from the cache
    video_path = VIDEO_CACHE.get("latest_video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="No cached video found. Please call /generate-video/ first.")
    
    logger.info(f"--- Starting reconstruction for cached video: {video_path} ---")

    # Create a temporary directory for processing this reconstruction
    temp_dir = tempfile.mkdtemp(prefix=f"recon_{VIDEO_CACHE.get('latest_video_uid', 'vid')}_")
    frames_dir = os.path.join(temp_dir, "frames")
    output_ply_path = os.path.join(temp_dir, "reconstruction.ply")
    os.makedirs(frames_dir, exist_ok=True)
    
    try:
        # 2. Extract frames from the video
        logger.info("--- Extracting frames from video ---")
        filelist = extract_frames_from_video(video_path, frames_dir)
        
        if not filelist:
            raise HTTPException(status_code=400, detail="Could not extract any frames from the video.")
        
        logger.info(f"--- Extracted {len(filelist)} frames ---")

        # 3. Load images for the model
        images = load_images(filelist, size=512, verbose=False)

        # 4. Run model inference
        logger.info("--- Running model inference ---")
        start_time = time.time()
        output_dict, profiling_info = inference(
            images,
            model,
            device,
            dtype=torch.float32,
            verbose=True,
            profiling=True,
        )
        logger.info(f"--- Inference completed in {time.time() - start_time:.2f} seconds ---")

        # 5. Align point clouds
        logger.info("--- Aligning point clouds ---")
        lit_module.align_local_pts3d_to_global(
            preds=output_dict['preds'], 
            views=output_dict['views'], 
            min_conf_thr_percentile=85
        )

        # 6. Generate and export the final mesh
        create_mesh_from_preds(
            preds=output_dict['preds'],
            views=output_dict['views'],
            export_ply_path=output_ply_path,
            min_conf_thr_percentile=30,
            flip_axes=True
        )

        # 7. Return the generated .ply file
        # Note: The temp_dir will not be cleaned up automatically.
        # A background task could be used for cleanup after the response is sent.
        return FileResponse(
            path=output_ply_path,
            media_type="application/octet-stream",
            filename="reconstruction.ply"
        )

    except Exception as e:
        logger.exception("An error occurred during reconstruction")
        shutil.rmtree(temp_dir) # Clean up on failure
        raise HTTPException(status_code=500, detail=str(e))


# --- Main entry point to run the API server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
