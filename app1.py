import os
import io
import base64
import torch
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from read import text_recognizer
from model import Model
from utils import CTCLabelConverter
from ultralytics import YOLO

app = FastAPI()

# Enable CORS (Allows Frontend to Access API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def read_root():
    print("[INFO] Root endpoint accessed.")
    return {"message": "Welcome to the Urdu OCR API! Use /predict to upload an image."}

# Load Urdu glyphs
try:
    with open("UrduGlyphs.txt", "r", encoding="utf-8") as file:
        content = file.read().strip().replace("\n", "") + " "
except FileNotFoundError:
    print("[ERROR] UrduGlyphs.txt not found!")
    content = ""

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
converter = CTCLabelConverter(content)
recognition_model = Model(num_class=len(converter.character), device=device).to(device)
recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
recognition_model.eval()

detection_model = YOLO("yolov8m_UrduDoc.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("[INFO] Received image for OCR processing.")

        # Load and convert image
        input_image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Detection
        detection_results = detection_model.predict(
            source=input_image, conf=0.2, imgsz=1280, save=False, nms=True, device=device
        )
        bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
        bounding_boxes.sort(key=lambda x: x[1])  # Sort bounding boxes by y-coordinate

        # Draw boxes and extract text
        draw = ImageDraw.Draw(input_image)
        texts = []
        for box in bounding_boxes:
            draw.rectangle(box, outline="red", width=2)
            cropped_img = input_image.crop(box)
            text = text_recognizer(cropped_img, recognition_model, converter, device)
            texts.append(text)

        print("[INFO] Recognized Text:", texts)

        # Convert image to base64 for response
        buffer = io.BytesIO()
        input_image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        print("[INFO] OCR processing complete.")
        return JSONResponse(content={"text": "\n".join(texts), "image": img_str}, media_type="application/json")

    except Exception as e:
        print("[ERROR] An error occurred:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the app with Google Cloud Run compatible PORT
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # Use Cloud Run's $PORT or default to 8080
    print(f"[INFO] Starting Urdu OCR API on port {port}...")
    uvicorn.run("app1:app", host="0.0.0.0", port=port)
