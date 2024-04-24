# from fastapi import APIRouter, UploadFile, File
# from fastapi.responses import HTMLResponse

# import predict

# router = APIRouter()

# @router.post('/', tags=["prediction"])
# async def predicttion(file: UploadFile = File(...)):
#     try:
#         with open(file.filename, "wb") as f:
#             f.write(await file.read())
#     except Exception: 
#         return {"message": "Error"}
    
#     return predict.depressionDetection(message="message")

# @router.get("/", tags=["prediction"], response_class=HTMLResponse)
# async def get_home_page():
#     with open("neural_network/site.html", "r") as file:
#         html_content = file.read()
#     return HTMLResponse(content=html_content)

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import HTMLResponse
import predict
import moviepy.editor as mp

router = APIRouter()

@router.post('/', tags=["prediction"])
async def prediction(file: UploadFile = File(...)):
    try:
        content = await file.read()
        with open("temp_video.mp4", "wb") as f:
            f.write(content)
        video = mp.VideoFileClip("temp_video.mp4")
        predictions = predict.PredictionService
        html_content = predictions.start_prediction(predictions, video)
        video.close()
    except Exception as e: 
        return {"message": str(e)}
    
    return HTMLResponse(content=html_content)

@router.get("/", tags=["prediction"], response_class=HTMLResponse)
async def get_home_page():
    with open("neural_network/site.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)