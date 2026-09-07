from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile
import os
import base64

app = FastAPI(
    title="AquaScan API",
    description="AI-powered underwater sonar detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
   allow_origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://aquascan-frontend.vercel.app"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")

print("Loading AquaScan YOLO model...")
model = YOLO(MODEL_PATH)
print("YOLO model loaded successfully.")


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "AquaScan API is running",
        "model": "YOLO shipwreck segmentation"
    }


@app.get("/model")
def model_info():
    return {
        "model": "AquaScan YOLO",
        "classes": model.names
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.filename:
        return {
            "success": False,
            "error": "No file uploaded"
        }

    suffix = os.path.splitext(file.filename)[1] or ".png"
    temp_path = None

    try:
        # Save uploaded image
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name

        # Run YOLO segmentation
        results = model.predict(
            source=temp_path,
            conf=0.25,
            save=False,
            verbose=False
        )

        result = results[0]

        detections = []

        # Read detections
        if result.boxes is not None:

            for i in range(len(result.boxes)):

                confidence = float(result.boxes.conf[i])
                class_id = int(result.boxes.cls[i])

                class_name = model.names[class_id]

                if confidence >= 0.75:
                    risk = "High"
                elif confidence >= 0.50:
                    risk = "Medium"
                else:
                    risk = "Low"

                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "risk": risk
                })

        # ---------------------------------------------
        # CREATE YOLO ANNOTATED IMAGE
        # ---------------------------------------------

        annotated = result.plot()

        # Encode image as JPEG
        import cv2

        success, encoded_image = cv2.imencode(
            ".jpg",
            annotated
        )

        if not success:
            raise RuntimeError("Could not encode annotated image")

        image_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode("utf-8")

        # ---------------------------------------------
        # RESPONSE
        # ---------------------------------------------

        return {
            "success": True,
            "filename": file.filename,
            "detections": detections,
            "count": len(detections),
            "annotated_image": f"data:image/jpeg;base64,{image_base64}"
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)