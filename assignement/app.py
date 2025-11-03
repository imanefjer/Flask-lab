from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
import pandas as pd
import uvicorn

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    wheelbase: float = Form(...),
    carlength: float = Form(...),
    carwidth: float = Form(...),
    carheight: float = Form(...),
    curbweight: float = Form(...),
    enginesize: float = Form(...),
    horsepower: float = Form(...),
    peakrpm: float = Form(...),
    citympg: float = Form(...),
    highwaympg: float = Form(...)
):
    input_features = np.array([[wheelbase, carlength, carwidth, carheight,
                                curbweight, enginesize, horsepower, peakrpm,
                                citympg, highwaympg]])
    
    input_scaled = scaler.transform(input_features)
    
    prediction = model.predict(input_scaled)[0]
    output = round(prediction, 2)
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction_text": f"Predicted Car Price: $ {output}"}
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)