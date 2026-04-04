## WHERE TO FIND FRED IDs
### Method 1: FRED Website

1) Go to: https://fred.stlouisfed.org/
2) Search for: "corn price", "wheat", "fertilizer", etc.
3) Click on the series you are interested in
4) Check the URL: The ID is located there
    - Example: https://fred.stlouisfed.org/series/PMAIZMTUSDM
    - ID = PMAIZMTUSDM

### Method 2: API Search

``` Python

from fredapi import Fred

fred = Fred(api_key='tu_key')

results = fred.search('corn price')

for series in results:
    print(f"ID: {series['id']}")
    print(f"Title: {series['title']}")
    print(f"Frequency: {series['frequency']}")
    print("---")

info = fred.get_series_info('PMAIZMTUSDM')

print(f"Title: {info['title']}")
print(f"Units: {info['units']}")
print(f"Frequency: {info['frequency']}") 
print(f"Seasonal: {info['seasonal_adjustment']}")

``` 

## How to Add New Series
### Simply add to the SERIES dictionary:
```json
SERIES = {
    # ... existing series ...
    
    # New series
    'coffee': {
        'id': 'PCOFFOTMUSDM',  # ← Search on FRED
        'name': 'Coffee Price Index',
        'unit': 'Index 2010=100',
        'category': 'crop',
        'frequency': 'monthly'  # ← Check on FRED
    }
}
```