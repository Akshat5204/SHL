import streamlit as st
from embeddings.recommender import recommend

st.set_page_config(page_title="SHL Recommender")

st.title("SHL Assessment Recommendation System")

query = st.text_area("Enter Job Description or Query")

if st.button("Recommend"):
    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Analyzing job requirements..."):
            results = recommend(query, k=5)

        st.subheader("Recommended Assessments")

        for r in results:
            st.markdown(f"**{r['name']}**")
            st.markdown(f"[Open Assessment]({r['url']})")
            st.markdown("---")
