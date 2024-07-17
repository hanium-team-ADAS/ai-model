import joblib
import pandas as pd
from typing import List
import os

# 현재 파일의 디렉토리를 기준으로 모델 파일의 경로를 설정합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '../models/final_model.sav')
loaded_model = joblib.load(model_path)

async def runModel(outer: bool) -> float:
    try:
        # outer 값에 따라 입력 데이터를 동적으로 변경합니다.
        dic = {
            "outer_exists_True": 1 if outer else 0,
            "outer_exists_False": 0 if outer else 1,
            "camera_temp": 34.1,
            "age_youth": 0,
            "age_old-age": 0,
            "age_child": 0,
            "age_middle-age": 1,
            "gender_male": 0,
            "gender_female": 1,
            "mask_exists_True": 1,
            "mask_exists_False": 0,
            "pixel_val": 232.2617188
        }

        # dictionary 형태를 DataFrame 형태로 변환합니다.
        input_df = pd.DataFrame([dic])

        print("Input DataFrame:", input_df)  # 디버깅 정보를 출력합니다.

        # input 값을 이용해서 예측값을 만들고, z에 대입합니다.
        z = loaded_model.predict(input_df)

        # 변수 z의 타입이 numpy이기 때문에 list로 바꿔줍니다.
        result: List[float] = z.tolist()  # json 파일은 list만 받음

        return result[0]
    except Exception as e:
        print(f"Error during model prediction: {e}")
        raise e  # 예외를 다시 발생시켜 FastAPI 로그에 기록되도록 합니다.
