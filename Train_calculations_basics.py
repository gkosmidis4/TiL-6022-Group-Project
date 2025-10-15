import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('data\TrainDistancesDelft_NSPrices_NSTravelTime.csv')
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={
  'Distance (km)': 'distance_km',
  'NS 2nd class fare (€) (no subscription, e-ticket)': 'ns_fare_eur',
  'travel time (min)': 'travel_time_min'
})
df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce')
df['ns_fare_eur'] = pd.to_numeric(df['ns_fare_eur'], errors='coerce')
df['travel_time_min'] = pd.to_numeric(df['travel_time_min'], errors='coerce')