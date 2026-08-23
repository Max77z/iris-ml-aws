from fastapi import FastAPI

from pydantic import BaseModel

import joblib
 
 
app = FastAPI()
 
model = joblib.load("model/iris_model.joblib")
 
 
class IrisRequest(BaseModel):

    sepal_length: float

    sepal_width: float

    petal_length: float

    petal_width: float
 
 
@app.get("/")

def root():

    return {

        "message": "Iris ML API  v2 is running"

    }
 
 
@app.get("/health")

def health():

    return {

        "status": "healthy"

    }
 
 
@app.post("/predict")

def predict(request: IrisRequest):
 
    features = [[

        request.sepal_length,

        request.sepal_width,

        request.petal_length,

        request.petal_width

    ]]
 
    prediction = model.predict(features)[0]
 
    flowers = [

        "setosa",

        "versicolor",

        "virginica"

    ]
 
    return {

        "prediction": int(prediction),

        "flower": flowers[prediction]

    }
 