from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from clean.preprocess import clean_data
import pandas as pd
import pickle
import json
import traceback
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
                "Cold_Storage_Capacity": 3788,
                "Shelf_Life_Days": 4,
                "Marketing_Spend": 670.37,
                "Product_ID": 1,
                "Store_ID": 1,
                "Week_Number": "2024-W01",
                "Wastage_Units": 718,
                "Product_Category": "Meat", 
                "Product_Name": "Whole Wheat Bread 800g",
                "Price": 2.46,
                "Rainfall": 23,
                "Avg_Temperature": 22.3,
                "Region": "London",
                "Store_Size": 12000
            }
        ]
    )


@app.post("/predict")
def predict(req: Item):
    try:
        data = pd.DataFrame(req.records)
        cleaned_data = clean_data(data)

        # Load model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'rf_randomsearchcv_model.pkl')
        with open(model_path, 'rb') as file:
            bundle = pickle.load(file)
            
        # Access components from the dictionary
        model = bundle['model']
        feature_scaler = bundle['feature_scaler']
        target_scaler = bundle['target_scaler']
        params = bundle['best_params']

        # Training feature schema
        schema_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'schema.json')
        with open(schema_path, 'r') as f:
            feature = json.load(f)
            main_features = feature['features']

        # Clean data
        cleaned_data = cleaned_data.reindex(columns = main_features, fill_value=0)

        print("Final Features Used for Prediction:")
        print(cleaned_data.columns.tolist())

        print("Loaded Schema Features from JSON:")
        print(main_features)

        # Scale the input features
        scaled_features = feature_scaler.transform(cleaned_data)

        # Convert back to DataFrame with correct column names
        scaled_features = pd.DataFrame(scaled_features, columns=main_features)

        # Predict on scaled features (model outputs scaled target)
        pred_scaled = model.predict(scaled_features)

        # Inverse transform the predictions to original scale
        pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        # SReturn predictions as list
        return {"predictions": pred.tolist()}
    
    except Exception as e:
        print(f"Error during prediction:"), 
        traceback.print_exc()  # This shows full traceback in the terminal
        raise HTTPException(status_code=500, detail=f"Data cleaning error: {str(e)}")

    
if __name__ == "__main__":
    print(f"Server is on port {os.getenv('port', 3000)}")
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv('port', 3000)))