from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import pickle
import numpy as np
from sklearn.datasets import load_digits

# Load trained SVM model
with open("svm_rps_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load digits dataset (for sample images)
digits = load_digits()

app = FastAPI()

# HTML with digit images as buttons
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Digit Recognition Classifier</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 30px; }
        img { width: 80px; height: 80px; margin: 10px; cursor: pointer; border: 2px solid #ccc; border-radius: 8px; }
        #result { margin-top: 30px; font-size: 22px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Digit Recognition Classifier</h1>
    <p>Click an image to classify:</p>
    <div>
        <!-- Display digits 0–9 as clickable images -->
        <img src="/digit/0" onclick="predict(0)">
        <img src="/digit/1" onclick="predict(1)">
        <img src="/digit/2" onclick="predict(2)">
        <img src="/digit/3" onclick="predict(3)">
        <img src="/digit/4" onclick="predict(4)">
        <img src="/digit/5" onclick="predict(5)">
        <img src="/digit/6" onclick="predict(6)">
        <img src="/digit/7" onclick="predict(7)">
        <img src="/digit/8" onclick="predict(8)">
        <img src="/digit/9" onclick="predict(9)">
    </div>
    <div id="result"></div>

    <script>
        async function predict(choice) {
            const response = await fetch(`/predict?digit=${choice}`);
            const data = await response.json();
            document.getElementById("result").innerText = "Prediction: " + data.prediction;
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return html_content

@app.get("/digit/{label}")
async def get_digit_image(label: int):
    """
    Return a digit image (PNG) for the given label.
    We'll just use the first example of each digit from the dataset.
    """
    import matplotlib.pyplot as plt
    import io
    from fastapi.responses import StreamingResponse

    # Find first index of this digit
    idx = np.where(digits.target == label)[0][0]
    fig, ax = plt.subplots()
    ax.imshow(digits.images[idx], cmap="cividis")
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/predict")
async def predict(digit: int):
    """
    Predict the digit using the trained model.
    We'll take the first sample of that digit from the dataset.
    """
    idx = np.where(digits.target == digit)[0][0]
    sample = digits.images[idx].reshape(1, -1)
    pred = model.predict(sample)[0]
    return {"prediction": str(pred)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)