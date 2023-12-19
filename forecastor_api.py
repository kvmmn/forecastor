import os
import pandas as pd
from dotenv import load_dotenv
from pmdarima import auto_arima
from sqlalchemy import create_engine
from fastapi import FastAPI, HTTPException
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the environment variables from the .env file.
load_dotenv()

# Initialize FastAPI app.
app = FastAPI()


def get_data():
    """
    Connects to  PostgreSQL database and retrieves data from the 'Order' table.

    Returns:
        pd.DataFrame: DataFrame containing the retrieved data.
    """
    # Database connection parameters.
    db_params = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

    # Constructing the connection string.
    connection_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"

    # Create a database engine.
    engine = create_engine(connection_string)

    try:
        # Executing the SQL query using pandas.
        query = """SELECT "Order"."createdAt", "Order"."FinalPrice", "Order"."vendorId" 
                   FROM "Order";"""
        df = pd.read_sql(query, engine)
    except Exception as error:
        print("Error while connecting to PostgreSQL:", error)
        # Return an empty DataFrame in case of an error.
        return pd.DataFrame()
    finally:
        # Dispose of the engine to release the resources.
        engine.dispose()
    return df


def preprocess_data(df, vendor_id):
    """
    Preprocesses the time series data for a given vendor_id.

    Args:
        df (pd.DataFrame): The raw data.
        vendor_id (str): The VendorID to filter the data.
        Default value is "274" :: Total Snappstore sales.

    Returns:
        pd.DataFrame: DataFrame with processed data ready for time series analysis.
    """
    # Filter the data based on vendor_id.
    if vendor_id != "274":
        df = df[df["vendorId"] == vendor_id]

    # Rename columns and convert types.
    df_processed = df.rename(
        columns={"createdAt": "date", "FinalPrice": "price", "vendorId": "vendor_id"}
    )
    df_processed["date"] = pd.to_datetime(df_processed["date"])
    df_processed["price"] = df_processed["price"].fillna(0).astype(int)

    # Resample data by day and sum the values.
    df_processed = df_processed.set_index("date").resample("D").sum()
    df_processed = df_processed[:-1]
    return df_processed


def fit_sarimax(df):
    """
    Fits a SARIMAX model to the given time series data.

    Args:
        df (pd.DataFrame): Preprocessed time series data.

    Returns:
        SARIMAXResults: Fitted SARIMAX model, or None if fitting fails.
    """
    try:
        # Automatic model selection using auto_arima.
        auto_model = auto_arima(
            df["price"],
            seasonal=True,
            m=7,
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )

        # Extract the best model parameters.
        best_order = auto_model.order
        best_seasonal_order = auto_model.seasonal_order

        # Fit the SARIMAX model.
        model = SARIMAX(
            df["price"], order=best_order, seasonal_order=best_seasonal_order
        )
        results = model.fit(disp=False)
    except Exception as e:
        print(f"Model fitting error: {e}")
        return None
    return results


def predict_next_period_sales(df, steps, period_name):
    """
    Predicts sales for the specified future period using the SARIMAX model.

    Args:
        df (pd.DataFrame): Preprocessed time series data.
        steps (int): Number of future steps to predict.
        period_name (str): Description of the prediction period.

    Returns:
        tuple: Tuple containing forecasted values and total sales for the period.
    """
    results = fit_sarimax(df)
    if results:
        # Generate forecast.
        forecast = results.get_forecast(steps=steps).predicted_mean
        total_sales = int(forecast.sum())
        print(f"Predicted sales for the next {period_name}: {total_sales:,}")
        return forecast, total_sales
    else:
        print("Model fitting was unsuccessful.")
        return None, None


@app.get("/get_data/")
async def api_get_data():
    """
    API endpoint to retrieve data from the database.
    Returns a JSON representation of the data.
    """
    data = get_data()
    if data.empty:
        raise HTTPException(status_code=404, detail="Data not found")
    return data.to_dict()


@app.get("/preprocess_data/{vendor_id}")
async def api_preprocess_data(vendor_id: str):
    """
    API endpoint to preprocess data for a given vendor_id.
    Args:
        vendor_id (str): Vendor ID to filter the data.

    Returns:
        JSON: Preprocessed data.
    """
    df = get_data()
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")
    processed_data = preprocess_data(df, vendor_id)
    return processed_data.to_dict()


@app.get("/predict_sales/{vendor_id}/{steps}/{period_name}")
async def api_predict_sales(vendor_id: str, steps: int, period_name: str):
    """
    API endpoint to predict sales for the next specified period.
    Args:
        vendor_id (str): Vendor ID for which the prediction is to be made.
        steps (int): Number of steps (days) to predict.
        period_name (str): Description of the prediction period.

    Returns:
        JSON: Forecasted values and total sales.
    """
    df = get_data()
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")
    df_processed = preprocess_data(df, vendor_id)
    forecast, total_sales = predict_next_period_sales(df_processed, steps, period_name)
    if forecast is None:
        raise HTTPException(status_code=500, detail="Model fitting was unsuccessful")
    return {"forecast": forecast.to_dict(), "total_sales": total_sales}


@app.get("/")
async def root():
    return {"root message": "Welcome to Snappstore\'s Daily Sales Forecast App :: Forecastor"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=274)
