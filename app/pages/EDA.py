import streamlit as st


st.set_page_config(layout="wide")

st.title("EDA")


st.subheader("Overview Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Total Requests",
    value= 2934932
)

col2.metric(
    label="Average Daily Hits",
    value= 56441
)

col3.metric(
    label="Total traffic",
    value= f"{53.07}GB"
    )

col4.metric(
    label="Error Rate",
    value= f"{9.97}%"
)

st.subheader("Traffic Size Across Different Time Aggregation Intervals")
interval= st.selectbox("Select the interval", ("1min", "5min", "15min", "60min", "combined"))
if st.button("Submit"):
    if interval == "combined":
        st.image("eda/img/overview.png")

    elif interval == "1min":
        st.image("eda/img/size_over_time_1_min.png")
    elif interval == "5min":
        st.image("eda/img/size_over_time_5_min.png")

    elif interval == "15min":
        st.image("eda/img/size_over_time_15_min.png")

    elif interval == "60min":
        st.image("eda/img/size_over_time_60_min.png")

st.subheader("Daily Traffic & Reliability Patterns")
st.image("eda/img/avg.png")

st.subheader("Request Intensity by Hour and Day")
st.image("eda/img/heatmap.png")

st.subheader("Traffic Composition")
col1, col2= st.columns(2)
col1.markdown("Top 10 Domains")
col1.image("eda/img/top-10-domain.png",use_container_width=True)

col2.markdown("Top 10 File types")
col2.image("eda/img/top-10-file.png", use_container_width=True)





