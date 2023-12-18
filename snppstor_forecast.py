import pandas as pd
import streamlit as st
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
# import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load the environment variables from .env file
load_dotenv()


def get_data():
    """
    Connects to a PostgreSQL database and retrieves data from the 'Order' table.

    Returns:
        DataFrame: The data retrieved from the database.
    """
    db_params = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD")
    }

    # connection = None

    # try:
    #     connection = psycopg2.connect(**db_params)
    #     query = """SELECT "Order"."createdAt", "Order"."FinalPrice", "Order"."vendorId" 
    #                FROM "Order";"""
    #     df = pd.read_sql(query, connection)
    # except (Exception, psycopg2.Error) as error:
    #     print("Error while connecting to PostgreSQL:", error)
    #     return pd.DataFrame()  # Returns an empty DataFrame in case of error
    # finally:
    #     if connection:
    #         connection.close()
    #         print("PostgreSQL connection is closed")
    
    # Constructing the connection string
    connection_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    
    # Creating an engine
    engine = create_engine(connection_string)

    try:
        # Executing query using pandas read_sql function
        query = """SELECT "Order"."createdAt", "Order"."FinalPrice", "Order"."vendorId" 
                   FROM "Order";"""
        df = pd.read_sql(query, engine)
    except Exception as error:
        print("Error while connecting to PostgreSQL:", error)
        return pd.DataFrame()  # Returns an empty DataFrame in case of error
    finally:
        engine.dispose()  # Disposing the engine

    return df


def preprocess_data(df, vendor_id):
    """
    Preprocesses the time series data based on the given vendor_id.

    Args:
        df (DataFrame): The raw data.
        vendor_id (int): The VendorID to filter the data.

    Returns:
        DataFrame: Processed data ready for time series analysis.
    """
    if vendor_id != "274":
        df = df[df["vendorId"] == vendor_id]

    df_processed = df.rename(
        columns={"createdAt": "date", "FinalPrice": "price", "vendorId": "vendor_id"}
    )
    df_processed["date"] = pd.to_datetime(df_processed["date"])
    df_processed["price"] = df_processed["price"].fillna(0).astype(int)
    df_processed = df_processed.set_index("date").resample("D").sum()
    df_processed = df_processed[:-1]

    return df_processed


def fit_sarimax(df):
    """
    Fits a SARIMAX model to the time series data.

    Args:
        df (DataFrame): The preprocessed time series data.

    Returns:
        SARIMAXResults: The fitted SARIMAX model.
    """
    try:
        auto_model = auto_arima(
            df["price"],
            seasonal=True,
            m=7,
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        best_order = auto_model.order
        best_seasonal_order = auto_model.seasonal_order

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
    Predicts sales for the next specified period using the SARIMAX model.

    Args:
        df (DataFrame): The preprocessed time series data.
        steps (int): Number of steps to predict.
        period_name (str): Description of the prediction period.

    Returns:
        tuple: Forecasted values and total sales for the period.
    """
    results = fit_sarimax(df)
    if results:
        forecast = results.get_forecast(steps=steps).predicted_mean
        total_sales = int(forecast.sum())
        print(f"Predicted sales for the next {period_name}: {total_sales:,}")
        return forecast, total_sales
    else:
        print("Model fitting was unsuccessful.")
        return None, None


def run_streamlit_app():
    st.title("Vendor Sales Forecast")
    
    st.info("Syncing with database...")
    df = get_data()
    st.info("Synced!")

    vendor_id = st.text_input("Enter VendorId:", "274")
    if not df.empty:
        df_processed = preprocess_data(df, vendor_id)

        if st.button("Forecast Sales"):
            with st.spinner("Calculating forecasts..."):
                next_week_forecast, total_week_sales = predict_next_period_sales(
                    df_processed, steps=7, period_name="week"
                )

                if next_week_forecast is not None:
                    st.write(
                        f"Total forecasted sales for the upcoming week: {total_week_sales:,.2f}"
                    )
                    st.write("Forecasted sales by day for the upcoming week:")
                    st.write(next_week_forecast.to_frame(name="Sales"))
                else:
                    st.error(
                        "Forecasting was unsuccessful. Please check the data or model parameters."
                    )


if __name__ == "__main__":
    run_streamlit_app()
