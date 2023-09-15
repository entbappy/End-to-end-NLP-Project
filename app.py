from textClassification.pipeline.prediction import PredictionPipeline
from fastapi import FastAPI
import uvicorn
import os
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse


text:str = "What is NLP?"

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@app.get("/train")
async def training():
    try:
        os.system("python main.py")

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    


@app.post("/predict")
async def predict_route(text):
    try:

        obj = PredictionPipeline()
        text = obj.predict(text)
        return text
    except Exception as e:
        raise e


if __name__=="__main__":
    APP_HOST = "0.0.0.0"
    APP_PORT = 8080
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
