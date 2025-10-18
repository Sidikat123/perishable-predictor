from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from clean.preprocess import clean_data
import pandas as pd
import pickle
from typing import List, Dict, Any
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title = 'Perishable Goods Prediction API', version = '1.0')

class Item(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ..., 
        example = [
            {
                "Wastage_Units": 100,
                "Product_Name": "Whole Wheat Bread 800g",
                "Product_Category": "Bakery",
                "Shelf_Life_Days ": 3,
                "Price": 2.5,
                "Cold_Storage_Capacity": 500,
                "Store_Size": 1500,
                "Rainfall": 20.5,
                "Avg_Temperature": 22.3,
                "Region": "North"
            }
        ]
    )

@app.post("/predict")
def predict(req: Item):
    try:
        data = pd.DataFrame(req.records)
        cleaned_data = clean_data(data)

        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'random_forest_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        pred = model.predict(cleaned_data)

        return {"predictions": pred.tolist()}

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Data cleaning error: {str(e)}")
    
if __name__ == "__main__":
    print(f"Server is on port {os.getenv('port', 3000)}")
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv('port', 3000)))