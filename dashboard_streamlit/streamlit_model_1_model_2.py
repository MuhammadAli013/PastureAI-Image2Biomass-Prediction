"""
PastureAI - Pasture Biomass Dashboard
Run: streamlit run app.py
pip install streamlit torch torchvision pillow folium streamlit-folium piexif numpy pandas scikit-learn joblib
"""

import io
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib

st.set_page_config(page_title="PastureAI", page_icon="🌿", layout="wide")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("🌿 PastureAI")
st.sidebar.markdown("---")

model_choice = st.sidebar.radio(
    "Choose Model",
    ["Only Image (Model 1)", "Image + Sensor (Model 2)"],
)
st.sidebar.markdown("---")

if model_choice == "Image + Sensor (Model 2)":
    ndvi   = st.sidebar.slider("NDVI", min_value=0.16, max_value=0.91, value=0.5, step=0.01)
    height = st.sidebar.slider("Canopy Height (cm)", min_value=1.0, max_value=70.0, value=20.0, step=0.5)
    st.sidebar.markdown("---")

threshold = st.sidebar.slider("Grazing Threshold (g)", 20, 100, 45, step=5)
st.sidebar.markdown("---")

st.sidebar.markdown("**Biomass Class Thresholds (g)**")
t_very_low = st.sidebar.number_input("Very Low upper limit", value=10, step=5)
t_low      = st.sidebar.number_input("Low upper limit",      value=25, step=5)
t_moderate = st.sidebar.number_input("Moderate upper limit", value=40, step=5)
t_good     = st.sidebar.number_input("Good upper limit",     value=60, step=5)

st.sidebar.markdown("---")
if model_choice == "Only Image (Model 1)":
    st.sidebar.caption("Model 1 — Image Only | ResNet-34")
else:
    st.sidebar.caption("Model 2 — Image + Sensors | ResNet-34")

# ─────────────────────────────────────────────
# BIOMASS CLASSES (dynamic)
# ─────────────────────────────────────────────
BIOMASS_CLASSES = [
    {"label": "Very Low",  "min": 0,          "max": t_very_low, "color": "red",        "action": "Do not graze — critical rest needed"},
    {"label": "Low",       "min": t_very_low,  "max": t_low,      "color": "orange",     "action": "Not ready — allow 2–3 more weeks"},
    {"label": "Moderate",  "min": t_low,       "max": t_moderate, "color": "yellow",     "action": "Approaching ready — monitor closely"},
    {"label": "Good",      "min": t_moderate,  "max": t_good,     "color": "lightgreen", "action": "Ready to graze — good feed quality"},
    {"label": "Excellent", "min": t_good,      "max": 9999,       "color": "darkgreen",  "action": "Excellent — high biomass, prioritize this zone"},
]

for cls in BIOMASS_CLASSES:
    st.sidebar.markdown(f"**{cls['label']}**: {cls['min']}–{'∞' if cls['max']==9999 else cls['max']}g")

TARGETS = ["Dry_Green_g", "GDM_g", "Dry_Total_g"]

def classify_biomass(total_g):
    for cls in BIOMASS_CLASSES:
        if cls["min"] <= total_g < cls["max"]:
            return cls
    return BIOMASS_CLASSES[-1]


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet34(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),         nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3),           nn.ReLU()
        )
    def forward(self, x):
        return self.head(self.backbone(x))


class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet34(weights=None)
        img_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_branch = backbone

        self.tab_branch = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(img_features + 256, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),                nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3),                  nn.ReLU()
        )

    def forward(self, x, tab):
        img_emb = self.image_branch(x)
        tab_emb = self.tab_branch(tab)
        return self.head(torch.cat([img_emb, tab_emb], dim=1))


@st.cache_resource
def load_model1(path="model1_best.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model1().to(device)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        st.sidebar.success("✅ Model 1 loaded")
    else:
        st.sidebar.warning("⚠️ model1_best.pth not found")
    model.eval()
    return model, device


@st.cache_resource
def load_model2(path="model2_best.pth", scaler_path="model2_scaler.pkl"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model2().to(device)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        st.sidebar.success("✅ Model 2 loaded")
    else:
        st.sidebar.warning("⚠️ model2_best.pth not found")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    if scaler is None:
        st.sidebar.warning("⚠️ model2_scaler.pkl not found")
    model.eval()
    return model, device, scaler


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict_m1(model, device, pil_image):
    tensor = transform(pil_image).unsqueeze(0).to(device)
    preds  = model(tensor).cpu().numpy()[0]
    return {t: round(max(0.0, float(v)), 2) for t, v in zip(TARGETS, preds)}


@torch.no_grad()
def predict_m2(model, device, scaler, pil_image, ndvi_val, height_val):
    tensor  = transform(pil_image).unsqueeze(0).to(device)
    tab_raw = np.array([[ndvi_val, height_val]])
    if scaler:
        tab_raw = scaler.transform(tab_raw)
    tab   = torch.tensor(tab_raw, dtype=torch.float32).to(device)
    preds = model(tensor, tab).cpu().numpy()[0]
    st.write("Raw tabular input:", ndvi_val, height_val)
    st.write("After scaler:", tab_raw)
    return {t: round(max(0.0, float(v)), 2) for t, v in zip(TARGETS, preds)}


# ─────────────────────────────────────────────
# GPS
# ─────────────────────────────────────────────
def read_gps(raw_bytes):
    try:
        import piexif
        exif_dict = piexif.load(raw_bytes)
        gps = exif_dict.get("GPS", {})
        if not gps:
            return None, None

        def dms_to_dec(dms, ref):
            d = dms[0][0] / dms[0][1]
            m = dms[1][0] / dms[1][1]
            s = dms[2][0] / dms[2][1]
            val = d + m / 60 + s / 3600
            if ref in [b"S", b"W"]:
                val = -val
            return val

        lat = dms_to_dec(gps[piexif.GPSIFD.GPSLatitude],
                         gps[piexif.GPSIFD.GPSLatitudeRef])
        lon = dms_to_dec(gps[piexif.GPSIFD.GPSLongitude],
                         gps[piexif.GPSIFD.GPSLongitudeRef])
        return lat, lon
    except Exception:
        return None, None


# ─────────────────────────────────────────────
# MAP HELPER
# ─────────────────────────────────────────────
def make_map(lat, lon, bclass, preds):
    m = folium.Map(location=[lat, lon], zoom_start=17)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite"
    ).add_to(m)
    folium.CircleMarker(
        location=[lat, lon],
        radius=15, color="white", weight=2,
        fill=True, fill_color=bclass["color"], fill_opacity=0.85,
        tooltip=f"{bclass['label']}: {preds['Dry_Total_g']}g",
        popup=f"Green: {preds['Dry_Green_g']}g | GDM: {preds['GDM_g']}g | Total: {preds['Dry_Total_g']}g"
    ).add_to(m)
    return m


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.title("🌿 PastureAI — Pasture Biomass Estimator")

# ── MODEL 2 ───────────────────────────────────
if model_choice == "Image + Sensor (Model 2)":
    model, device, scaler = load_model2()

    uploaded = st.file_uploader(
        "Upload a pasture image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="m2_uploader"
    )

    if not uploaded:
        st.info("Upload a pasture image. Set NDVI and canopy height in the sidebar.")
    else:
        raw_bytes = uploaded.read()
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        lat, lon  = read_gps(raw_bytes)
        preds     = predict_m2(model, device, scaler, pil_image, ndvi, height)
        bclass    = classify_biomass(preds["Dry_Total_g"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Image")
            st.image(pil_image, use_container_width=True)
            if lat and lon:
                st.caption(f"📍 GPS: {lat:.5f}°N, {abs(lon):.5f}°W")

            st.subheader("Sensor Inputs")
            st.write(f"**NDVI:** {ndvi}")
            st.write(f"**Canopy Height:** {height} cm")

            st.subheader("Predictions")
            st.metric("🌿 Green Biomass", f"{preds['Dry_Green_g']} g")
            st.metric("🌱 GDM",           f"{preds['GDM_g']} g")
            st.metric("📦 Total Biomass", f"{preds['Dry_Total_g']} g")

            st.subheader("Grazing Decision")
            if preds["Dry_Total_g"] >= threshold:
                st.success(f"✅ Ready to Graze — {bclass['label']} | {bclass['action']}")
            else:
                deficit = threshold - preds["Dry_Total_g"]
                st.warning(f"⚠️ Not Ready — {deficit:.1f}g below threshold | {bclass['action']}")

        with col2:
            st.subheader("Field Map")
            if lat and lon:
                st_folium(make_map(lat, lon, bclass, preds),
                          width="100%", height=450, returned_objects=[])
            else:
                st.info("No GPS data in this image.")

# ── MODEL 1 ───────────────────────────────────
else:
    model, device = load_model1()

    uploaded_files = st.file_uploader(
        "Upload one or multiple pasture images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="m1_uploader"
    )

    if not uploaded_files:
        st.info("Upload one image for detailed analysis, or multiple images for paddock-wide survey.")

    elif len(uploaded_files) == 1:
        f         = uploaded_files[0]
        raw_bytes = f.read()
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        lat, lon  = read_gps(raw_bytes)
        preds     = predict_m1(model, device, pil_image)
        bclass    = classify_biomass(preds["Dry_Total_g"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Image")
            st.image(pil_image, use_container_width=True)
            if lat and lon:
                st.caption(f"📍 GPS: {lat:.5f}°N, {abs(lon):.5f}°W")

            st.subheader("Predictions")
            st.metric("🌿 Green Biomass", f"{preds['Dry_Green_g']} g")
            st.metric("🌱 GDM",           f"{preds['GDM_g']} g")
            st.metric("📦 Total Biomass", f"{preds['Dry_Total_g']} g")

            st.subheader("Grazing Decision")
            if preds["Dry_Total_g"] >= threshold:
                st.success(f"✅ Ready to Graze — {bclass['label']} | {bclass['action']}")
            else:
                deficit = threshold - preds["Dry_Total_g"]
                st.warning(f"⚠️ Not Ready — {deficit:.1f}g below threshold | {bclass['action']}")

        with col2:
            st.subheader("Field Map")
            if lat and lon:
                st_folium(make_map(lat, lon, bclass, preds),
                          width="100%", height=450, returned_objects=[])
            else:
                st.info("No GPS data in this image. Upload a GPS-tagged image to see the map.")

    else:
        results  = []
        progress = st.progress(0, text="Analysing images...")

        for i, f in enumerate(uploaded_files):
            raw_bytes = f.read()
            pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            lat, lon  = read_gps(raw_bytes)
            preds     = predict_m1(model, device, pil_image)
            bclass    = classify_biomass(preds["Dry_Total_g"])
            results.append({
                "image_name":    f.name,
                "lat":           lat,
                "lon":           lon,
                "Green_Biomass": preds["Dry_Green_g"],
                "GDM":           preds["GDM_g"],
                "Total_Biomass": preds["Dry_Total_g"],
                "Class":         bclass["label"],
                "Comment":       bclass["action"],
                "color":         bclass["color"],
            })
            progress.progress((i + 1) / len(uploaded_files),
                              text=f"Analysing {f.name}...")

        progress.empty()
        df = pd.DataFrame(results)

        # Map
        st.subheader("📍 Field Map")
        gps_df = df[df["lat"].notna() & df["lon"].notna()]
        if not gps_df.empty:
            m = folium.Map(location=[gps_df["lat"].mean(), gps_df["lon"].mean()], zoom_start=17)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", name="Satellite"
            ).add_to(m)
            for _, row in gps_df.iterrows():
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=12, color="white", weight=2,
                    fill=True, fill_color=row["color"], fill_opacity=0.85,
                    tooltip=f"{row['image_name']} | {row['Class']}: {row['Total_Biomass']}g",
                    popup=(f"<b>{row['image_name']}</b><br>"
                           f"Green: {row['Green_Biomass']}g<br>"
                           f"GDM: {row['GDM']}g<br>"
                           f"Total: {row['Total_Biomass']}g<br>"
                           f"{row['Comment']}")
                ).add_to(m)
            st_folium(m, width="100%", height=500, returned_objects=[])
        else:
            st.info("No GPS data found in uploaded images.")

        # Summary
        st.subheader("📊 Overall Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📷 Images",            len(df))
        c2.metric("🌿 Avg Green Biomass", f"{df['Green_Biomass'].mean():.1f} g")
        c3.metric("🌱 Avg GDM",           f"{df['GDM'].mean():.1f} g")
        c4.metric("📦 Avg Total Biomass", f"{df['Total_Biomass'].mean():.1f} g")
        ready = (df["Total_Biomass"] >= threshold).sum()
        st.markdown(f"**{ready} of {len(df)} zones ready to graze** (threshold: {threshold}g)")

        # Table
        st.subheader("📋 Detailed Results")
        display_df = df[["image_name", "Green_Biomass", "GDM", "Total_Biomass", "Class", "Comment"]].copy()
        avg_row = pd.DataFrame([{
            "image_name":    "AVERAGE",
            "Green_Biomass": round(df["Green_Biomass"].mean(), 2),
            "GDM":           round(df["GDM"].mean(), 2),
            "Total_Biomass": round(df["Total_Biomass"].mean(), 2),
            "Class":         classify_biomass(df["Total_Biomass"].mean())["label"],
            "Comment":       classify_biomass(df["Total_Biomass"].mean())["action"],
        }])
        display_df = pd.concat([display_df, avg_row], ignore_index=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv,
                           file_name="pasture_biomass_results.csv", mime="text/csv")