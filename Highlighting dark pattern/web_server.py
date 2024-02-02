from fastapi import FastAPI, Form, UploadFile, File
from PIL import Image
from joblib import load
import json
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

#Fast API + html
# Mount the 'static' directory to serve static files like the HTML file
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Serve the HTML file
    return FileResponse("static/index.html")

with open("D:\Project\Dark_Pattern_Detectin_in_e-commerce_websites-master\Dark_Pattern_Detectin_in_e-commerce_websites-master\Path\paths.json") as f:
    file_paths = json.load(f)

# Load the trained model
model = load(file_paths["MODEL_JOBLIB"])

# Load the trained vectorizer
vectorizer = load(file_paths["VECTORIZER"])

#Fast API + html
@app.post("/predict")
def predict_dark_pattern(text: str = Form(...)):
    line = text

    # Transform the input text using the loaded vectorizer
    X = vectorizer.transform([line])

    # Use the loaded model to predict if the line is a dark pattern
    prediction = model.predict(X)
    
    prediction = int(prediction[0])

    result = {
        0: "No Dark Patterns were found",
        1: "Dark Patterns are found",
    }

    return {"prediction": result.get(int(prediction))}

def extract_text_using_tesseract(image_file: UploadFile) -> str:
    # Save the uploaded image to a temporary file
    with open("temp_image.png", "wb") as temp_image:
        temp_image.write(image_file.file.read())

    # Use pytesseract to extract text from the image
    extracted_text = pytesseract.image_to_string(Image.open("temp_image.png"))

    return extracted_text

# In your FastAPI endpoint
@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    extracted_text = extract_text_using_tesseract(file)
    return {"text": extracted_text}