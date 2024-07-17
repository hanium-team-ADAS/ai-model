import joblib
import pandas as pd
import cv2
import numpy as np
from typing import List
import os

def load_model(model_path: str):
    """모델 로드"""
    return joblib.load(model_path)

def get_red_channel_mean(image_path: str) -> float:
    """이미지에서 RED 채널의 평균값 계산"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지를 로드할 수 없습니다.")
    red_channel_mean = np.mean(image[:, :, 2])
    return red_channel_mean
    
def get_age_category(age: int):
    """연령대 카테고리를 반환"""
    if age <= 12:
        return "child"
    elif 13 <= age <= 24:
        return "youth"
    elif 25 <= age <= 64:
        return "middle_age"
    else:
        return "old_age"

async def run_model(camera_temp: List[List[str]], red_channel_mean: float, age: int, gender: int, model) -> float:
    """모델 입력 데이터를 준비하고 예측을 수행"""
    all_values = [float(value) for sublist in camera_temp for value in sublist]
    camera_temp = max(all_values)
    age_category = get_age_category(age)
    # 테스트를 위해 일부 변수는 임의로 지정
    dic = {
        "outer_exists_True": 0,
        "outer_exists_False": 1,
        "camera_temp": camera_temp,
        "age_child": 1 if age_category == "child" else 0,
        "age_youth": 1 if age_category == "youth" else 0,
        "age_middle-age": 1 if age_category == "middle_age" else 0,   
        "age_old-age": 1 if age_category == "old_age" else 0,         
        "gender_male": 1 if gender == 1 else 0,          
        "gender_female": 1 if gender == 0 else 0,          
        "mask_exists_True": 0,        
        "mask_exists_False": 1,
        "pixel_val": red_channel_mean
    }

    input_df = pd.DataFrame([dic])
    print("Input DataFrame:", input_df) 
    prediction = model.predict(input_df)
    return float(prediction[0]) #온도 보정값 반환