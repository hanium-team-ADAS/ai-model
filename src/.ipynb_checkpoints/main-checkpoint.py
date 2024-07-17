from fastapi import FastAPI # fastapi 패키지를 불러온다.
from services.house import runModel
app = FastAPI() # FastAPI 클래스를 app이라는 변수로 인스턴스 선언을 해준다.

@app.get('/')
async def predict_temp(outer: bool) -> float:
    try:
        temp = await runModel(outer)  # async와 await를 같이 써야함.
        return temp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")



