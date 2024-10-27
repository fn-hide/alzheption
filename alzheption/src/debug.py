import plotly.express as px
import pandas as pd


# --- prove that Sharpness and UIL in specific order
df = pd.read_csv('alzheption/result/img_attributes_normalized.csv')

# df = df[df.ID == 'i228869']
df = df[df.ID == 'i248520']

df = df.sort_values(by=['Class', 'ID', 'Name'])

fig = px.line(df, x='Name', y='Sharpness', title='Sharpness')
fig.write_html('alzheption/result/Sharpness.html')

fig = px.line(df, x='Name', y='Brightness', title='Brightness')
fig.write_html('alzheption/result/Brightness.html')

fig = px.line(df, x='Name', y='Contrast', title='Contrast')
fig.write_html('alzheption/result/Contrast.html')

fig = px.line(df, x='Name', y='UIL', title='UIL')
fig.write_html('alzheption/result/UIL.html')

fig = px.line(df, x='Name', y='Shadow', title='Shadow')
fig.write_html('alzheption/result/Shadow.html')

fig = px.line(df, x='Name', y='Specularity', title='Specularity')
fig.write_html('alzheption/result/Specularity.html')

fig = px.line(df, x='Name', y='BU', title='BU')
fig.write_html('alzheption/result/BU.html')

print(df)

