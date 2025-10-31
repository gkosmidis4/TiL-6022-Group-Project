import matplotlib.pyplot as plt
import numpy as np
import math

import pandas as pd
# use forward slashes to avoid escape-sequence warnings on Windows
df = pd.read_csv('data/TrainDistancesDelft_NSPrices_NSTravelTime.csv')
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={
  'Distance (km)': 'distance_km',
  'NS 2nd class fare (€) (no subscription, e-ticket)': 'ns_fare_eur',
  'travel time (min)': 'travel_time_min'
})
# ensure City header is normalized to lowercase 'city'
df = df.rename(columns={'City': 'city'})
df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce')
df['ns_fare_eur'] = pd.to_numeric(df['ns_fare_eur'], errors='coerce')
df['travel_time_min'] = pd.to_numeric(df['travel_time_min'], errors='coerce')


# Print city plus the three train columns (distance, fare, travel time)
print("\nTrain columns (city, distance_km, ns_fare_eur, travel_time_min):")
wanted = ['city', 'distance_km', 'ns_fare_eur', 'travel_time_min']
missing = [c for c in wanted if c not in df.columns]
if missing:
    print(f"Missing columns, can't print requested fields: {missing}")
    print("Available columns:", list(df.columns))
else:
    print(df[wanted].head(50).to_string(index=False))


def train_cost_per_city(df: 'pd.DataFrame') -> 'pd.Series':
  """Return train cost (NS 2nd class fare) per city as a Series indexed by city.

  The function expects the DataFrame to contain a 'city' column and 'ns_fare_eur'.
  """
  if 'city' not in df.columns or 'ns_fare_eur' not in df.columns:
    raise KeyError("DataFrame must contain 'city' and 'ns_fare_eur' columns")
  # Return a copy to avoid accidental modifications
  return df.set_index('city')['ns_fare_eur'].copy()


def Train_Travel_Time_per_city(df: 'pd.DataFrame') -> 'pd.Series':
  """Return total travel time per city (train travel time + 10 minutes bike ride).

  Expects 'city' and 'travel_time_min' columns. Returns minutes as numeric.
  """
  if 'city' not in df.columns or 'travel_time_min' not in df.columns:
    raise KeyError("DataFrame must contain 'city' and 'travel_time_min' columns")
  # return raw train travel time (minutes) without additional bike time
  total = pd.to_numeric(df['travel_time_min'], errors='coerce')
  return pd.Series(total.values, index=df['city'].values, name='train_travel_time_min')


# Print outputs of the two functions for inspection
print("\nTrain cost per city (EUR):")
try:
  costs = train_cost_per_city(df)
  print(costs.to_string())
except Exception as e:
  print("Could not compute train cost per city:", e)

print("\nTrain travel time per city (minutes, raw train time):")
try:
  times = Train_Travel_Time_per_city(df)
  print(times.to_string())
except Exception as e:
  print("Could not compute train travel time per city:", e)


# --- simplified plotting -------------------------------------------------
import os

FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)
# Set to True if you want the script to show plots interactively after saving.
# In some environments (headless servers, or certain IDE terminals) showing
# may not work; set to False to only save files.
SHOW_PLOTS = False

# days per month (average working days to Delft); changeable variable
DAYS_PER_MONTH = 18


def compute_ticks(minv, maxv, target_ticks=8):
  """Compute 'nice' ticks between minv and maxv aiming for ~target_ticks ticks.

  This is the same logic used for scatter plots but exposed at module level so
  bar charts can also use sensible tick steps.
  """
  if math.isnan(minv) or math.isnan(maxv):
    return []
  if minv == maxv:
    if minv == 0:
      minv, maxv = 0, 1
    else:
      minv = minv - abs(minv) * 0.1
      maxv = maxv + abs(maxv) * 0.1
  rng = maxv - minv
  if rng == 0:
    rng = abs(maxv) if maxv != 0 else 1
  raw_step = rng / max(1, target_ticks - 1)
  exp = math.floor(math.log10(raw_step))
  base = 10 ** exp
  lead = raw_step / base
  if lead <= 1.5:
    nice = 1 * base
  elif lead <= 3:
    nice = 2 * base
  elif lead <= 7:
    nice = 5 * base
  else:
    nice = 10 * base
  start = math.floor(minv / nice) * nice
  stop = math.ceil(maxv / nice) * nice
  ticks = np.arange(start, stop + nice / 2, nice)
  return ticks


def simple_scatter(x, y, labels, xlabel, ylabel, title, fname):
  """Create a simple scatter plot with city name labels and grid ticks every 10."""
  plt.figure(figsize=(7, 5))
  plt.scatter(x, y, c='C0')
  # annotate points with city names (simple offset)
  # stagger labels slightly to reduce overlap: alternate up/down offsets
  for i, (xi, yi, lab) in enumerate(zip(x, y, labels)):
    yoff = 0.5 if i % 2 == 0 else -0.5
    xoff = 0.5
    plt.text(xi + xoff, yi + yoff, lab, fontsize=7)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  # set sensible tick steps based on axis ranges
  def compute_ticks(minv, maxv, target_ticks=8):
    """Compute 'nice' ticks between minv and maxv aiming for ~target_ticks ticks."""
    if math.isnan(minv) or math.isnan(maxv):
      return []
    if minv == maxv:
      # single value: create small symmetric range
      if minv == 0:
        minv, maxv = 0, 1
      else:
        minv = minv - abs(minv) * 0.1
        maxv = maxv + abs(maxv) * 0.1
    rng = maxv - minv
    if rng == 0:
      rng = abs(maxv) if maxv != 0 else 1
    raw_step = rng / max(1, target_ticks - 1)
    # nice step: 1, 2, 5 times power of ten
    exp = math.floor(math.log10(raw_step))
    base = 10 ** exp
    lead = raw_step / base
    if lead <= 1.5:
      nice = 1 * base
    elif lead <= 3:
      nice = 2 * base
    elif lead <= 7:
      nice = 5 * base
    else:
      nice = 10 * base
    # compute tick start/stop
    start = math.floor(minv / nice) * nice
    stop = math.ceil(maxv / nice) * nice
    # create ticks
    ticks = np.arange(start, stop + nice / 2, nice)
    return ticks

  xmin, xmax = plt.xlim()
  ymin, ymax = plt.ylim()
  xticks = compute_ticks(xmin, xmax, target_ticks=8)
  yticks = compute_ticks(ymin, ymax, target_ticks=8)
  plt.xticks(xticks)
  plt.yticks(yticks)
  plt.grid(True)
  path = os.path.join(FIG_DIR, fname)
  plt.tight_layout()
  plt.savefig(path)
  print(f"Saved: {path}")
  if SHOW_PLOTS:
    try:
      plt.show()
    except Exception:
      # If interactive display fails, continue without raising
      pass
  plt.close()


# Prepare simple data for plotting (drop rows with missing values)
plot_df = df[['city', 'distance_km', 'ns_fare_eur', 'travel_time_min']].dropna()
cities = plot_df['city'].astype(str).values
distance = plot_df['distance_km'].values
cost = plot_df['ns_fare_eur'].values
time = plot_df['travel_time_min'].values

# Print functions (very simple) so output appears in the terminal
'''
def print_cost_vs_distance(df):
  print('\nCost vs Distance (city, distance_km, ns_fare_eur):')
  for city, dist, c in df[['city', 'distance_km', 'ns_fare_eur']].itertuples(index=False):
    print(f"{city:20} {dist:6.0f} km   {c:5.2f} EUR")
'''
'''
def print_cost_vs_time(df):
  print('\nCost vs Time (city, travel_time_min, ns_fare_eur):')
  for city, t, c in df[['city', 'travel_time_min', 'ns_fare_eur']].itertuples(index=False):
    print(f"{city:20} {t:6.0f} min  {c:5.2f} EUR")
'''
'''
def print_distance_vs_time(df):
  print('\nDistance vs Time (city, travel_time_min, distance_km):')
  for city, t, dist in df[['city', 'travel_time_min', 'distance_km']].itertuples(index=False):
    print(f"{city:20} {t:6.0f} min  {dist:6.0f} km")
'''


# 1) Cost vs Distance (cost on y-axis, distance on x-axis)
simple_scatter(distance, cost, cities, 'Distance (km)', 'Train cost (EUR)',
               'Cost vs Distance (per city per trip to Delft)', 'cost_vs_distance.png')

# 2) Cost vs Time (cost on y-axis, time on x-axis)
simple_scatter(time, cost, cities, 'Train travel time (min)', 'Train cost (EUR)',
               'Cost vs Time (per city per trip to Delft)', 'cost_vs_time.png')

# 3) Distance vs Time (distance on y-axis, time on x-axis)
simple_scatter(time, distance, cities, 'Train travel time (min)', 'Distance (km)',
               'Distance vs Time (per city per trip to Delft)', 'distance_vs_time.png')

# Create bar charts with cities on x-axis
def simple_bar(values, city_names, ylabel, title, fname):
  """Create a simple bar chart with cities on x-axis."""
  plt.figure(figsize=(10, 5))
  bars = plt.bar(city_names, values)
  plt.xticks(rotation=45, ha='right')
  plt.ylabel(ylabel)
  plt.title(title)
  plt.grid(True, axis='y')
  # set sensible y ticks for bar charts
  ymin, ymax = plt.ylim()
  yticks = compute_ticks(ymin, ymax, target_ticks=6)
  if len(yticks) > 0:
    plt.yticks(yticks)

  # add numeric labels above bars
  # choose formatting based on ylabel
  fmt = '{:.0f}'
  if 'EUR' in ylabel or 'Cost' in title:
    fmt = '{:.2f}'
  elif 'minutes' in ylabel or 'time' in ylabel.lower():
    fmt = '{:.0f}'
  elif 'km' in ylabel or 'Distance' in title:
    fmt = '{:.0f}'

  for bar in bars:
    h = bar.get_height()
    if np.isnan(h):
      label = ''
    else:
      try:
        label = fmt.format(h)
      except Exception:
        label = str(h)
    plt.text(bar.get_x() + bar.get_width() / 2, h + (ymax - ymin) * 0.01, label,
         ha='center', va='bottom', fontsize=8)

  path = os.path.join(FIG_DIR, fname)
  plt.tight_layout()
  plt.savefig(path)
  print(f"Saved: {path}")
  if SHOW_PLOTS:
    try:
      plt.show()
    except Exception:
      pass
  plt.close()


print("\nGenerating bar charts...")

# Multiply values by 2 to represent round-trip (to and from Delft)
time_rt = time * 2
distance_rt = distance * 2
cost_rt = cost * 2

# 1. Time bar chart (roundtrip)
simple_bar(time_rt, cities, 'Travel time (minutes)',
          'Train Travel Time by City (per roundtrip to and from Delft)', 'time_bars_roundtrip.png')

# 2. Distance bar chart (roundtrip)
simple_bar(distance_rt, cities, 'Distance (km)',
          'Distance by City (per roundtrip to and from Delft)', 'distance_bars_roundtrip.png')

# 3. Cost bar chart (roundtrip)
simple_bar(cost_rt, cities, 'Cost (EUR)',
          'Train Cost by City (per roundtrip to and from Delft)', 'cost_bars_roundtrip.png')


# --- Monthly bar charts (readable y-axis) --------------------------------
# (monthly bar charts will be generated after computing monthly values below)


# --- Monthly calculations and plots -------------------------------------
# Compute monthly values based on round-trip per day multiplied by DAYS_PER_MONTH
time_month = time_rt * DAYS_PER_MONTH
distance_month = distance_rt * DAYS_PER_MONTH
cost_month = cost_rt * DAYS_PER_MONTH

# Generate monthly bar charts (readable y-axis)
print('\nGenerating monthly bar charts...')
simple_bar(time_month, cities, 'Travel time per month (minutes)',
           'Train Travel Time per Month by City', 'time_bars_month.png')

simple_bar(distance_month, cities, 'Distance per month (km)',
           'Distance per Month by City', 'distance_bars_month.png')

simple_bar(cost_month, cities, 'Cost per month (EUR)',
           'Cost per Month by City', 'cost_bars_month.png')

# 1) Train cost per month (y) vs travel time per day (x)
simple_scatter(time, cost_month, cities,
               'Train travel time per day (min)', 'Train cost per month (EUR)',
               'Cost per month vs time per day', 'cost_month_vs_time_day.png')

# 2) Train cost per month (y) vs travel time per month (x)
simple_scatter(time_month, cost_month, cities,
               'Train travel time per month (min)', 'Train cost per month (EUR)',
               'Cost per month vs time per month', 'cost_month_vs_time_month.png')

# 3) Distance per month (y) vs travel time per month (x)
simple_scatter(time_month, distance_month, cities,
               'Train travel time per month (min)', 'Distance per month (km)',
               'Distance per month vs time per month', 'distance_month_vs_time_month.png')

# 4) Cost per month (y) vs distance per month (x)
simple_scatter(distance_month, cost_month, cities,
               'Distance per month (km)', 'Train cost per month (EUR)',
               'Cost per month vs distance per month', 'cost_month_vs_distance_month.png')



