import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import os
import warnings

import matplotlib.pyplot as plt 
import folium
from streamlit_folium import folium_static
import streamlit_echarts
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Data Science", page_icon=":bar_chart:",layout="wide")
st.title(" :bar_chart: Startups")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

df=pd.read_csv("XL_cleaned.csv", encoding = "ISO-8859-1")
if df is not None:

    st.write(df,use_container_width=True)
col1, col2 = st.columns((2))


startYear = df["Funding Year"].min()
endYear = df["Funding Year"].max()

with col1:
    date1 = st.number_input("Start Year", startYear)

with col2:
    date2 = st.number_input("End Year",min_value=2018, max_value=2022, value=2022)

df = df[(df["Funding Year"] >= date1) & (df["Funding Year"] <= date2)].copy()



with col1:
    st.subheader("Amount funded in stages")
    fig = px.pie(df, values= "Amount",names="Stage", hole = 0.4)
    st.plotly_chart(fig,use_container_width=True, height = 200)


with col2:
    st.subheader("Sector wise Amount invested")
    fig = px.bar(df, x = "Sector", y = "Amount",color_discrete_map = {'FINANCE': '#7FD4C1', 'AI': '#30BFDD', 'EV': '#8690FF', 
                                   'OTHERS': '#ACD0F4', 'FOOD': '#F7C0BB', 'HEALTH': '#AB63FA','ENTERTAINMENT': '#FF97FF'})
    st.plotly_chart(fig,use_container_width=True, height = 200)
    
    



cl1, cl2 = st.columns((2))
with cl1:
    with st.expander("Amt vs Stages ViewData"):
        st.write(df.style.background_gradient(cmap="Blues"))
        csv = df.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data = csv, file_name = "AmtvsStages.csv", mime = "text/csv",
                            help = 'Click here to download the data as a CSV file')

with cl2:
    with st.expander("Sector_ViewData"):
        Sect = df.groupby(by = "Sector", as_index = False)["Amount"].sum()
        Sect=Sect.sort_values("Amount",ascending=False)
        st.write(Sect.style.background_gradient(cmap="Oranges"))
        csv = Sect.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data = csv, file_name = "SectorwiseAmount.csv", mime = "text/csv",
                        help = 'Click here to download the data as a CSV file')

st.subheader("COVID ANALYSIS")
df2 = df[(df["Funding Year"] >= 2020) & (df["Funding Year"] <= 2021)].copy()

st.write("Summary statistics:", df2['Amount'].describe()) 
df2['Percentage Change'] = df2['Amount'].pct_change() * 100
st.line_chart(df2['Percentage Change']) 
window_size = 7 
df2['SMA'] = df2['Amount'].rolling(window=window_size).mean()

from scipy.stats import zscore

df2['Z-Score'] = zscore(df2['Amount'])


anomaly_threshold = 0.98 

df2['Anomaly'] = abs(df2['Z-Score']) > anomaly_threshold

st.write("During the COVID-19 pandemic, funding in this dataset showed the following trends and anomalies:")
if df2['Percentage Change'].iloc[-1] > 0:
    st.write("Funding has increased by {:.2f}% from the previous period.".format(df2['Percentage Change'].iloc[-1]))
else:
    st.write("Funding has decreased by {:.2f}% from the previous period.".format(df2['Percentage Change'].iloc[-1]))

selected_sector = st.selectbox("Select a sector", ["OTHERS","ENTERTAINMENT",'FOOD',"AI", 'EV', 'HEALTH', 'FINANCE'])
sector_data = df2[df2['Sector'] == selected_sector]

percentage_change = (sector_data['Amount'].iloc[-1] - sector_data['Amount'].iloc[0]) / sector_data['Amount'].iloc[0] * 100

average_funding_covid = sector_data['Amount'].mean()


total_funding_covid = sector_data['Amount'].sum()


st.write(f"The {selected_sector} sector experienced a {percentage_change:.2f}% increase in funding during COVID.")
st.write(f"The average funding during COVID was ${average_funding_covid:,.2f}.")
st.write(f"The total funding during COVID was ${total_funding_covid:,.2f}.")





st.subheader("STATE WISE ANALYSIS")
selected_year = st.selectbox("Select year", [2018,2019,2020,2021,2022])
gj = r"india_state.geojson"
m = folium.Map(
    tiles="CartoDB positron", name="Light Map", zoom_start=2, attr="My Data attribution"
)
startup_data = pd.read_csv("lIST OF INCREASING STartups.csv", encoding="latin-1")
folium.Choropleth(
    geo_data=gj,
    name="chropleth",
    data=startup_data,
    columns=["States", f"{selected_year}"],
    key_on="feature.properties.NAME_1",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.1,
    legend_name=f"{selected_year}",
    html=f'<div style="font-size:14pt;color:green">{selected_year}</div>'
).add_to(m)
folium.features.GeoJson(
    r"india_state.geojson",
    name="States",
    popup=folium.features.GeoJsonPopup(fields=["NAME_1","ID_1"], aliases=["State","Startups"]),
).add_to(m)
folium_static(m,width=1600,height=950)

data1=pd.read_csv("ev sector.csv")





data=pd.read_csv("fintech_modified (1).csv")

st.title('Funding Evolution by Sector')
col1, col2 = st.columns((2))
with col1:
    selected_sector = st.selectbox('Select a Sector', df['Sector'].unique())
    start_year = st.number_input('Select Start Year', min_value=2018, max_value=2022, value=2018)

with col2:
    end_year = st.number_input('Select End Year', min_value=2018, max_value=2022, value=2022)


filtered_data = df[(df['Sector'] == selected_sector) & (df['Funding Year'] >= start_year) & (df['Funding Year'] <= end_year)]
st.subheader('Filtered Data')
st.write(filtered_data)



# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# df = pd.read_csv("fintech_modified 1.csv", encoding="latin-1")

# top10 = df.nlargest(10, "Revenue")
# plt.plot(top10["Company Name"], top10["Revenue"])
# plt.title("Top 10 largest companies by Revenue")
# plt.tight_layout()
# plt.xlabel("Company")
# plt.ylabel("Revenue")
# plt.show()

# city_count = {
#     "Location": list(df["Location"].unique()),
#     "Count": [df[df["Location"] == i].shape[0] for i in df["Location"].unique()],
# }
# l = [df[df["Location"] == i].shape[0] for i in df["Location"].unique()]
# city_count = pd.DataFrame(city_count)
# # city_count["Count"] = pd.to_numeric(city_count["Count"])
# city_count1 = cit_count.nlargest(5, "Count")
# # print(l)
# plt.plot(city_count1["Location"], city_count1["Count"])
# plt.show()

# year_fund = [
#     df["Funding Year"].unique().tolist(),
#     [
#         df[df["Funding Year"] == i]["Amount"].sum()
#         for i in df["Funding Year"].unique().tolist()
#     ],
# ]
# plt.plot(year_fund[0], year_fund[1])
# plt.xlabel("Year")
# plt.ylabel("Funding")
# plt.show()

# most_funded = df.nlargest(10, "Amount")
# plt.plot(most_funded["Company Name"], most_funded["Amount"])
# plt.xlabel("Company")
# plt.ylabel("Funding")
# plt.show()




st.title('Identifying Stagnating or Decreasing Sectors')
years_to_compare = st.multiselect(
    'Select Years to Compare',
    [2018, 2019, 2020, 2021,2022])

data_filtered = df[df['Funding Year'].isin(years_to_compare)]

data_filtered['Growth Rate'] = data_filtered.groupby(by='Sector', axis=0)['Amount'].pct_change().fillna(0) * 100

stagnating_sectors = data_filtered[data_filtered['Growth Rate'] < 0]

if not stagnating_sectors.empty:
    st.subheader(f'Stagnating or Decreasing Sectors ({min(years_to_compare)} - {max(years_to_compare)})')
    st.write(stagnating_sectors[['Sector', 'Growth Rate']])
else:
    st.subheader('No Stagnating or Decreasing Sectors Found')
    

