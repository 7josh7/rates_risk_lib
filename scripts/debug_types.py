"""Debug option types."""
import pandas as pd
positions_df = pd.read_csv('data/sample_book/positions.csv', comment='#')
pos = positions_df.iloc[10]  # SWAPTION
print('expiry_date:')
print(f'  value: {pos["expiry_date"]}')
print(f'  type: {type(pos["expiry_date"])}')
print(f'  pd.isna: {pd.isna(pos["expiry_date"])}')
print()
print('expiry_tenor:')
print(f'  value: {pos.get("expiry_tenor")}')
print(f'  type: {type(pos.get("expiry_tenor"))}')
print(f'  is None: {pos.get("expiry_tenor") is None}')
