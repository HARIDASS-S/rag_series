import streamlit as st
st.title("streamlite")
st.write("hd hello")
st.write(100)

st.markdown("# Streamlit")
st.markdown("**Interactive Python Applications**")
import streamlit as st

st.caption("Data updated on September 8, 2026")

code = """
x = 10
y = 20
print(x + y)
"""

st.code(code, language="python")

import pandas as pd

data = {
    "Name": ["Aman", "Riya", "Karan"],
    "Marks": [85, 92, 78]
}

df = pd.DataFrame(data)

st.dataframe(df)
st.dataframe(data)
if st.button("clik me "):
    st.write("buton is clicked :")