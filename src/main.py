from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import os
from services.house import load_model, get_red_channel_mean, run_model

app = FastAPI()

class TempData(BaseModel):
    camera_temp: List[List[str]]
    image_path: str
    age: int
    gender: int

#모델 로드
model_path = os.path.join(os.path.dirname(__file__), 'models', 'final_model.sav')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = load_model(model_path)

@app.post('/predict/')
async def predict_temp(data: TempData) -> float:
    """예측 모델을 실행하여 결과를 반환"""
    try:
        red_channel_mean = get_red_channel_mean(data.image_path)
        temp = await run_model(data.camera_temp, red_channel_mean, data.age, data.gender, model)
        return temp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
