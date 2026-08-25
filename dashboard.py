import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from stats_engine.core import check_cuped_test, get_ttest_strat_pvalue, get_ttest_pvalue

st.set_page_config(page_title="A/B Testing Platform", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_pvalue(p):
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"

@st.cache_data
def load_experiment_data():
    conn = sqlite3.connect("ab_platform.db")
    df = pd.read_sql("SELECT * FROM mart_ab_test_ready", conn)
    conn.close()
    return df

@st.cache_data
def run_all_methods(_df):
    df_control = _df[_df['pilot'] == 0]
    df_pilot = _df[_df['pilot'] == 1]

    # Naive
    naive_pvalue = get_ttest_pvalue(df_control['metric'], df_pilot['metric'])
    naive_delta = df_pilot['metric'].mean() - df_control['metric'].mean()

    # CUPED
    cuped_pvalue, cuped_delta = check_cuped_test(df_control, df_pilot, 'cov')

    # Stratification
    country_map = {c: i for i, c in enumerate(_df['country'].unique())}
    df_strat = _df.copy()
    df_strat['strat'] = df_strat['country'].map(country_map)
    a_strat = df_strat[df_strat['pilot'] == 0][['metric', 'strat']].values
    b_strat = df_strat[df_strat['pilot'] == 1][['metric', 'strat']].values
    strat_pvalue, strat_delta = get_ttest_strat_pvalue(a_strat, b_strat)

    results = pd.DataFrame({
        "Method": ["Naive t-test", "CUPED", "Stratification"],
        "P-value (raw)": [naive_pvalue, cuped_pvalue, strat_pvalue],
        "Delta": [naive_delta, cuped_delta, strat_delta],
    })
    return results, cuped_pvalue, cuped_delta

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("A/B Testing Platform")
page = st.sidebar.radio(
    "Navigate to:",
    ["Experiment Overview", "Peeking Problem Demo"]
)

# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

df = load_experiment_data()
methods_df, cuped_pvalue, cuped_delta = run_all_methods(df)

# ---------------------------------------------------------------------------
# PAGE 1: Experiment Overview
# ---------------------------------------------------------------------------

if page == "Experiment Overview":
    st.title("Experiment: FX Fee Reduction")

    st.markdown("""
    **Hypothesis:** Reducing the currency exchange fee from 0.5% to 0.2% 
    increases FX revenue per user through higher exchange frequency.

    **Design:** 50/50 split via double hashing, stratified by country, 
    CUPED covariate = historical FX revenue (4 weeks pre-experiment).

    **Statistical test:** t-test on the CUPED-adjusted metric, alpha = 0.05.
    """)

    st.divider()
    st.subheader("Headline result (CUPED-adjusted)")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "P-value",
        format_pvalue(cuped_pvalue),
        delta="Significant" if cuped_pvalue < 0.05 else "Not significant",
        delta_color="normal"
    )
    col2.metric(
        "Delta (revenue per user)",
        f"{cuped_delta:.4f}",
        delta=f"{cuped_delta:.4f}",
        delta_color="normal"
    )
    col3.metric("Users in experiment", f"{len(df):,}")

    st.info(
        "**Interpretation:** the effect is statistically significant but "
        "negative -- the fee cut reduces total FX revenue despite the "
        "increase in exchange frequency. Recommendation: do not ship this "
        "change."
    )

    st.divider()
    st.subheader("Method comparison")
    st.caption("Same experiment, evaluated four different ways.")

    display_df = methods_df.copy()
    display_df["P-value"] = display_df["P-value (raw)"].apply(format_pvalue)
    display_df["Delta"] = display_df["Delta"].round(4)
    display_df = display_df[["Method", "P-value", "Delta"]]

    linearization_row = pd.DataFrame({
        "Method": ["Linearization"],
        "P-value": ["7.19e-104"],
        "Delta": [-1.2881],
    })
    display_df = pd.concat([display_df, linearization_row], ignore_index=True)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.warning(
        "**Note on Linearization:** standard linearization assumes a "
        "stable revenue/transaction-count ratio across groups, estimated "
        "from control. Since the fee itself differs by group here (it's "
        "the treatment), this assumption is violated -- linearization "
        "gives a different (larger magnitude) delta than CUPED/"
        "stratification. For this specific experiment design, CUPED and "
        "stratification are more reliable."
    )

    st.divider()
    st.subheader("Raw data sample")
    st.caption(
        "Output of the dbt mart `mart_ab_test_ready`: one row per user, "
        "with pilot/control assignment, country, cumulative revenue "
        "(`metric`), and the CUPED covariate (`cov`)."
    )
    st.dataframe(df.head(10), use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 2: Peeking Problem Demo
# ---------------------------------------------------------------------------

else:
    st.title("The Peeking Problem")

    st.markdown("""
    **Question:** what happens if you check the p-value every day and stop 
    the experiment as soon as it looks significant?

    **Answer:** the true Type I error rate ends up far higher than the 
    stated alpha. Below is a simulation comparing two scenarios on the 
    same FX fee metric.
    """)

    st.divider()

    peeking_aa = pd.read_csv("peeking_aa_test.csv")
    peeking_effect = pd.read_csv("peeking_with_effect.csv")
    peeking_aa["scenario"] = "AA-test (no real effect)"
    peeking_effect["scenario"] = "Real FX experiment (effect exists)"
    combined = pd.concat([peeking_aa, peeking_effect])

    fig = px.line(
        combined, x="day", y="naive_pvalue", color="scenario", markers=True,
        title="Naive p-value by day of experiment",
        labels={"naive_pvalue": "p-value", "day": "Day"},
    )
    fig.add_hline(
        y=0.05, line_dash="dash", line_color="red",
        annotation_text="alpha = 0.05", annotation_position="top left",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Notice the AA-test line dips below the alpha threshold on days "
        "1-5, purely by chance. Stopping the experiment early on any of "
        "those days would have produced a false positive, even though no "
        "real effect exists. This is why the evaluation date must be fixed "
        "before the experiment starts, not chosen based on how results "
        "look mid-way."
    ) 

    st.divider()
    st.subheader("Raw simulation data")
    tab1, tab2 = st.tabs(["AA-test", "Real effect"])
    with tab1:
        st.dataframe(peeking_aa, use_container_width=True)
    with tab2:
        st.dataframe(peeking_effect, use_container_width=True)