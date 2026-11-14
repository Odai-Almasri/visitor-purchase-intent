import joblib
import pandas as pd
import streamlit as st
import sklearn

from features import RAW_FEATURES, recommend_action

st.set_page_config(page_title="Will This Visitor Buy?", page_icon="🛒", layout="centered")


@st.cache_resource
def load_artifacts():
    return joblib.load("model.joblib")


try:
    art = load_artifacts()
except Exception as e:  # missing or incompatible model file
    st.error("Could not load model.joblib. Run the notebook first and keep the file next to app.py.")
    st.caption(f"Details: {e}")
    st.stop()

pipe, d, rng = art["pipeline"], art["defaults"], art["ranges"]
t_coupon, t_chat = art["t_coupon"], art["t_chat"]

st.title("🛒 Will This Visitor Buy?")
st.write("Enter what a visitor did in the current session. The model estimates the chance of a purchase "
         "and suggests an on-site action.")

if art.get("sklearn_version") != sklearn.__version__:
    st.warning(f"Model trained with scikit-learn {art.get('sklearn_version')}, running {sklearn.__version__}. "
               "Results may differ; match the version in requirements.txt.")

ACTIONS = {
    "chat": ("💬 Show the chat assistant", "High purchase intent: offer help to close the sale. No discount needed."),
    "coupon": ("🎟️ Show a coupon", "On the fence: a discount may tip the decision."),
    "none": ("⏸️ No action", "Low purchase intent: do not spend discount budget on this visitor."),
}


def show_result(proba: float):
    st.metric("Purchase probability", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))
    title, why = ACTIONS[recommend_action(proba, t_coupon, t_chat)]
    box = {"chat": st.success, "coupon": st.warning, "none": st.info}[recommend_action(proba, t_coupon, t_chat)]
    box(f"**{title}**\n\n{why}")


tab1, tab2, tab3 = st.tabs(["Single session", "Batch (CSV)", "About the model"])

with tab1:
    with st.form("session"):
        c1, c2 = st.columns(2)
        admin = c1.number_input("Account/admin pages viewed", 0, 500, int(d["Administrative"]))
        admin_d = c2.number_input("Seconds on admin pages", 0.0, 100000.0, float(d["Administrative_Duration"]))
        info = c1.number_input("Info pages viewed", 0, 500, int(d["Informational"]))
        info_d = c2.number_input("Seconds on info pages", 0.0, 100000.0, float(d["Informational_Duration"]))
        prod = c1.number_input("Product pages viewed", 0, 2000, int(d["ProductRelated"]))
        prod_d = c2.number_input("Seconds on product pages", 0.0, 200000.0, float(d["ProductRelated_Duration"]))
        bounce = c1.slider("Bounce rate", 0.0, 0.2, min(float(d["BounceRates"]), 0.2), 0.001)
        exit_r = c2.slider("Exit rate", 0.0, 0.2, min(float(d["ExitRates"]), 0.2), 0.001)
        page_val = c1.number_input("Page value", 0.0, 500.0, float(d["PageValues"]))
        special = c2.select_slider("Closeness to a special day", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.0)
        month = c1.selectbox("Month", art["months"])
        visitor = c2.selectbox("Visitor type", art["visitor_types"])
        weekend = st.checkbox("Weekend session")
        with st.expander("Technical details (optional)"):
            opts = art["cat_options"]
            osys = st.selectbox("Operating system code", opts["OperatingSystems"],
                                index=opts["OperatingSystems"].index(d["OperatingSystems"]))
            browser = st.selectbox("Browser code", opts["Browser"], index=opts["Browser"].index(d["Browser"]))
            region = st.selectbox("Region code", opts["Region"], index=opts["Region"].index(d["Region"]))
            traffic = st.selectbox("Traffic type code", opts["TrafficType"],
                                   index=opts["TrafficType"].index(d["TrafficType"]))
        go = st.form_submit_button("Predict")

    if go:
        if prod == 0 and prod_d > 0 or admin == 0 and admin_d > 0 or info == 0 and info_d > 0:
            st.warning("Time spent is above 0 but the page count is 0. Please check the inputs.")
        row = pd.DataFrame([{
            "Administrative": admin, "Administrative_Duration": admin_d,
            "Informational": info, "Informational_Duration": info_d,
            "ProductRelated": prod, "ProductRelated_Duration": prod_d,
            "BounceRates": bounce, "ExitRates": exit_r, "PageValues": page_val, "SpecialDay": special,
            "Month": month, "OperatingSystems": osys, "Browser": browser, "Region": region,
            "TrafficType": traffic, "VisitorType": visitor, "Weekend": weekend,
        }])[RAW_FEATURES]
        try:
            show_result(float(pipe.predict_proba(row)[0, 1]))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    st.write("Upload a CSV with these columns: " + ", ".join(RAW_FEATURES))
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            missing = [c for c in RAW_FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                proba = pipe.predict_proba(df[RAW_FEATURES])[:, 1]
                out = df.copy()
                out["purchase_probability"] = proba.round(4)
                out["suggested_action"] = [recommend_action(p, t_coupon, t_chat) for p in proba]
                st.dataframe(out.head(200))
                st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process the file: {e}")

with tab3:
    m = art["test_metrics_at_t_chat"]
    st.write(f"**Model:** {art['model_name']} (trained on {art['n_train']:,} sessions)")
    st.write("**Held-out test results at the chat threshold:**")
    st.table(pd.DataFrame([m]).rename(columns=str.upper))
    st.write(f"Action thresholds: coupon from **{t_coupon:.2f}**, chat from **{t_chat:.2f}**.")
    st.caption("Data: Online Shoppers Purchasing Intention (UCI). Predictions are probabilities, not guarantees, "
               "and do not prove a coupon causes a purchase.")
