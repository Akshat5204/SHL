import streamlit as st
from embeddings.recommender import recommend

st.set_page_config(page_title="SHL Recommender", layout="centered")

st.title("SHL Assessment Recommendation System")

query = st.text_area("Enter Job Description or Query")

if st.button("Recommend"):
    if query.strip() == "":
        st.warning("Please enter a query")
    else:
        with st.spinner("Finding best assessments..."):
            results = recommend(query, k=5)

        st.subheader("Recommended Assessments")
        for i, r in enumerate(results, start=1):
            st.write(f"{i}. {r}")
