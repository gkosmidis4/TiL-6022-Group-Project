import matplotlib.pyplot as plt
import numpy as np
import math

import pandas as pd
import os
# Resolve data file path relative to this script so the script works
# regardless of the current working directory when invoked.
base_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(base_dir, 'data', 'TrainDistancesDelft_NSPrices_NSTravelTime.csv')
if not os.path.exists(data_file):
  raise FileNotFoundError(f"Required data file not found: {data_file}\n" \
              "Make sure you run the script from the project folder or that the file exists.")
df = pd.read_csv(data_file)
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
  """Return train travel time per city (minutes).

  Adds 10 minutes to the reported train travel time to represent the bike
  transfer from Delft station to campus (one-way). Returns minutes as numeric.
  """
  if 'city' not in df.columns or 'travel_time_min' not in df.columns:
    raise KeyError("DataFrame must contain 'city' and 'travel_time_min' columns")
  base = pd.to_numeric(df['travel_time_min'], errors='coerce')
  # add 10 minutes one-way for the bike transfer to/from campus
  total = base + 10
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

FIG_DIR = os.path.join('figures', 'train')
os.makedirs(FIG_DIR, exist_ok=True)
# Set to True if you want the script to show plots interactively after saving.
# In some environments (headless servers, or certain IDE terminals) showing
# may not work; set to False to only save files.
SHOW_PLOTS = False

# days per month (average working days to Delft); changeable variable
DAYS_PER_MONTH = 18

# --- Subscriptions (toggles only) --------------------------------------
# These variables allow toggling subscription scenarios for future analysis.
# They are declared here and NOT applied to the current cost calculations
# unless you explicitly enable them in code below.
# 1) daluren (off-peak) subscription: 40% discount outside peak hours, costs 5.90 EUR/month
DALUREN_ENABLED = False
DALUREN_COST_PER_MONTH = 5.90
DALUREN_DISCOUNT = 0.40

# 2) altijd korting subscription: 20% discount during peak hours (we consider only the 20%)
# costs 28.50 EUR/month
ALTIJD_KORTING_ENABLED = False
ALTIJD_COST_PER_MONTH = 28.50
ALTIJD_PEAK_DISCOUNT = 0.20

# NS corporate yellow for plots (Nederlandse Spoorwegen)
NS_YELLOW = '#FFD400'

# current graphs. The current plots are labelled as peak-hour scenarios.


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
  # use NS yellow for points with a thin black edge for legibility
  plt.scatter(x, y, c=NS_YELLOW, edgecolors='k')
  # annotate points with city names (simple offset)
  # stagger labels slightly to reduce overlap: alternate up/down offsets
  # improved label placement: if points are close, shift labels vertically
  # compute data ranges to set a proximity threshold
  try:
    xr = float(np.nanmax(x) - np.nanmin(x))
    yr = float(np.nanmax(y) - np.nanmin(y))
  except Exception:
    xr = 1.0
    yr = 1.0
  if xr == 0:
    xr = 1.0
  if yr == 0:
    yr = 1.0
  # proximity threshold: use a larger fraction (6%) so nearby city labels
  # like Rotterdam/The Hague/Utrecht get separated more aggressively
  threshold = math.hypot(xr * 0.06, yr * 0.06)
  placed = []
  for i, (xi, yi, lab) in enumerate(zip(x, y, labels)):
    # count how many already-placed points are within threshold
    close_count = 0
    for (xj, yj) in placed:
      if math.hypot(xi - xj, yi - yj) < threshold:
        close_count += 1
    # vertical offset based on number of close points (alternate sign)
    # increase vertical step for dense clusters
    vstep = yr * 0.04
    if close_count == 0:
      yoff = 0
    else:
      # alternate up/down and increase magnitude when multiple overlap
      sign = 1 if (close_count % 2) == 1 else -1
      yoff = sign * ( (close_count + 1) // 2 ) * vstep
    # slightly larger horizontal offset for dense clusters to avoid vertical-only stacking
    xoff = xr * 0.015
    plt.text(xi + xoff, yi + yoff, lab, fontsize=7)
    placed.append((xi, yi))
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
# add 10 minutes one-way to represent bike transfer station <-> campus
time = plot_df['travel_time_min'].values + 10
# --- Basic bar charts --------------------------------------------------

# Create bar charts with cities on x-axis
def simple_bar(values, city_names, ylabel, title, fname):
  """Create a simple bar chart with cities on x-axis."""
  plt.figure(figsize=(10, 5))
  # NS yellow bars with black edge
  bars = plt.bar(city_names, values, color=NS_YELLOW, edgecolor='k')
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
          'Train Travel Time by City (per roundtrip to and from Delft) [peak hour]', 'time_bars_roundtrip_peak.png')

# 2. Cost bar chart (roundtrip)
simple_bar(cost_rt, cities, 'Cost (EUR)',
          'Train Cost by City (per roundtrip to and from Delft) (EUR) [peak hour]', 'cost_bars_roundtrip_peak.png')


# --- Monthly bar charts (readable y-axis) --------------------------------
# (monthly bar charts will be generated after computing monthly values below)


# --- Monthly calculations and plots -------------------------------------
# Compute monthly values based on round-trip per day multiplied by DAYS_PER_MONTH
time_month = time_rt * DAYS_PER_MONTH
distance_month = distance_rt * DAYS_PER_MONTH

# add monthly bike cost: 17.90 EUR per month -> convert to per day when adding to daily
MONTHLY_BIKE_COST = 17.90
DAILY_BIKE_COST = MONTHLY_BIKE_COST / 30.0
# add daily bike cost once per day to the roundtrip cost (daily addition)
cost_rt = cost_rt + DAILY_BIKE_COST
cost_month = cost_rt * DAYS_PER_MONTH

# Generate monthly bar charts (readable y-axis)
print('\nGenerating monthly bar charts...')
simple_bar(time_month, cities, 'Travel time per month (minutes)',
           'Train Travel Time per Month by City [peak hour]', 'time_bars_month_peak.png')

simple_bar(cost_month, cities, 'Cost per month (EUR)',
           'Cost per Month by City (EUR) [peak hour]', 'cost_bars_month_peak.png')

# 1) Train cost per day (y) vs travel time per day (x)
simple_scatter(time, cost_rt, cities,
               'Train travel time per day (min)', 'Train cost per day (EUR)',
               'Cost per day vs time per day [peak hour]', 'cost_day_vs_time_day_peak.png')

# 2) Train cost per month (y) vs travel time per month (x)
simple_scatter(time_month, cost_month, cities,
               'Train travel time per month (min)', 'Train cost per month (EUR)',
               'Cost per month vs time per month [peak hour]', 'cost_month_vs_time_month_peak.png')

# --- Export summary CSV -------------------------------------------------
# Build a small DataFrame with daily and monthly times and costs
# Create simplified summary with only the requested columns:
# city, time_per_month_min, cost_per_month_eur
summary_df = pd.DataFrame({
  'city': cities,
  'time_per_month_min': time_month,
  'cost_per_month_eur': cost_month
})

out_file = os.path.join(base_dir, 'data', 'train_monthly_summary.csv')
summary_df.to_csv(out_file, index=False)
print(f"Saved simplified monthly summary CSV: {out_file}")
 



