import streamlit as st
import sqlite3
import json
import pandas as pd
import os

st.set_page_config(page_title="AgentQuant — Research Trail", layout="wide", initial_sidebar_state="expanded")

st.title("⚡ AgentQuant — Quantitative Strategy Research Trail")

db_path = "memory/agentquant.db"

if not os.path.exists(db_path):
    st.warning("No research database found yet. Run `python graph.py` to launch research runs and record metrics.")
else:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM attempts ORDER BY id DESC", conn)
    conn.close()

    if df.empty:
        st.info("Database initialized but no attempts logged yet.")
    else:
        tickers = list(df["ticker"].unique())
        selected_ticker = st.sidebar.selectbox("Select Asset Ticker", tickers)
        
        subset = df[df["ticker"] == selected_ticker].sort_values("iteration")
        
        st.subheader(f"🎯 Final Status for {selected_ticker}")
        accepted = subset[subset["decision"] == "accept"]
        if not accepted.empty:
            last_accepted = accepted.iloc[-1]
            st.success("Accepted Strategy Found!")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Strategy Proposal")
                hypo_data = json.loads(last_accepted["hypothesis"])
                st.markdown(f"**Rule Name:** {hypo_data.get('rule_name')}")
                st.markdown(f"**Signal Type:** `{hypo_data.get('signal_type')}`")
                st.markdown("**Parameters:**")
                st.json(hypo_data.get("params", {}))
                st.markdown(f"**Reasoning:** {hypo_data.get('reasoning')}")
            with col2:
                st.markdown("#### Validated Metrics")
                st.json(json.loads(last_accepted["metrics"]))
        else:
            st.warning("No strategy passed critique within iteration limit.")

        st.markdown("---")
        st.subheader("📜 Full Iteration Trail & Reasoning")
        for _, row in subset.iterrows():
            decision_label = "✅ ACCEPT" if row['decision'] == 'accept' else "🔄 REFINE"
            with st.expander(f"Iteration {row['iteration']} — {decision_label}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Hypothesis:**")
                    hypo_data = json.loads(row["hypothesis"])
                    st.markdown(f"**Rule Name:** {hypo_data.get('rule_name')}")
                    st.markdown(f"**Signal Type:** `{hypo_data.get('signal_type')}`")
                    st.json(hypo_data)
                with c2:
                    st.markdown("**Metrics:**")
                    st.json(json.loads(row["metrics"]))
                st.markdown(f"**Critique Reasoning:** {row['reasoning']}")
