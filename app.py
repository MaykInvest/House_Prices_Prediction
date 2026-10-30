import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Prediction", page_icon="🏡")

SALE_YEAR = 2014   # the model was trained on 2014 sales


@st.cache_resource          # load the model once, not on every click
def load_model():
    return joblib.load('house_price_model.pkl')


@st.cache_data
def load_locations():
    return pd.read_csv('locations.csv')


model = load_model()
locations = load_locations()

st.title("🏡 House Price Prediction")
st.caption("Seattle area, Washington. Prices are in 2014 dollars.")

# --- Inputs ---
col1, col2 = st.columns(2)

with col1:
    cities = sorted(locations['city'].unique())
    city = st.selectbox("City", cities, index=cities.index('Seattle'))
    zips = locations.loc[locations['city'] == city, 'statezip']
    statezip = st.selectbox("ZIP code", zips)
    bedrooms  = st.number_input("Bedrooms", 0, 10, 3)
    bathrooms = st.number_input("Bathrooms", 0.0, 8.0, 2.0, step=0.25)
    floors    = st.number_input("Floors", 1.0, 3.5, 1.0, step=0.5)

with col2:
    sqft_above    = st.number_input("Above-ground area (sq ft)", 300, 10000, 1800, step=50)
    sqft_basement = st.number_input("Basement (sq ft)", 0, 5000, 0, step=50)
    sqft_lot      = st.number_input("Lot size (sq ft)", 500, 1_000_000, 5000, step=100)
    yr_built      = st.number_input("Year built", 1900, SALE_YEAR, 1990)
    yr_renovated  = st.number_input("Year renovated (0 = never)", 0, SALE_YEAR, 0)

view       = st.slider("View (0 = none, 4 = excellent)", 0, 4, 0)
condition  = st.slider("Condition (1 = poor, 5 = excellent)", 1, 5, 3)
waterfront = st.checkbox("Waterfront")

# --- Prediction ---
if st.button("Predict price", type="primary"):
    if 0 < yr_renovated < yr_built:
        st.error("The renovation year can't be before the year the house was built.")
        st.stop()

    # Same feature engineering as the notebook (it is not inside the saved pipeline)
    house = pd.DataFrame([{
        'bedrooms':           bedrooms,
        'bathrooms':          bathrooms,
        'sqft_living':        sqft_above + sqft_basement,
        'sqft_lot':           sqft_lot,
        'floors':             floors,
        'waterfront':         int(waterfront),
        'view':               view,
        'condition':          condition,
        'sqft_above':         sqft_above,
        'sqft_basement':      sqft_basement,
        'city':               city,
        'statezip':           statezip,
        'house_age':          SALE_YEAR - yr_built,
        'was_renovated':      int(yr_renovated > 0),
        'years_since_update': SALE_YEAR - (yr_renovated if yr_renovated > 0 else yr_built),
    }])

    price = model.predict(house)[0]
    st.metric("Predicted price", f"${price:,.0f}")
    st.caption("On unseen houses the model is off by about $94k (16%) on average.")