# dashboard/app.py
#
# PURPOSE:
#   Streamlit dashboard for retail shelf monitoring.
#   Features: detection, heatmap, shelf share, OCR, batch upload,
#   before/after comparison, live camera feed.

import io
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
from PIL import Image, ImageDraw


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=cfg.DASHBOARD_TITLE,
    page_icon="🛒",
    layout="wide",
)

API_URL = f"http://localhost:{cfg.API_PORT}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_available() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def call_api_detect(image_bytes: bytes) -> dict:
    r = requests.post(
        f"{API_URL}/detect",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def call_api_batch(file_list) -> dict:
    files = [("files", (f.name, f.getvalue(), "image/jpeg")) for f in file_list]
    r = requests.post(f"{API_URL}/detect/batch", files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def call_api_ocr(image_bytes: bytes) -> dict:
    r = requests.post(
        f"{API_URL}/analytics/ocr",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def call_api_identify_products(image_bytes: bytes) -> dict:
    r = requests.post(
        f"{API_URL}/analytics/identify-products",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def severity_badge(severity: str) -> str:
    return {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(severity, "⚪")


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_detections_on_image(pil_image: Image.Image, detections: list) -> Image.Image:
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    colors = {"product": (0, 200, 100), "default": (255, 128, 0)}

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cls = det["class_name"]
        conf = det["confidence"]
        color = colors.get(cls, colors["default"])

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{cls} {conf:.2f}"
        draw.rectangle([x1, y1 - 16, x1 + len(label) * 7, y1], fill=color)
        draw.text((x1 + 2, y1 - 14), label, fill=(255, 255, 255))

    return img


def generate_heatmap_overlay(pil_image: Image.Image, detections: list) -> Image.Image:
    """Generate heatmap overlay on image."""
    from src.analytics.heatmap import generate_heatmap, overlay_heatmap
    import cv2

    img_array = np.array(pil_image)
    heat = generate_heatmap(
        detections,
        img_array.shape[1],
        img_array.shape[0],
    )
    blended = overlay_heatmap(img_array, heat)
    return Image.fromarray(blended)


# ── Render components ─────────────────────────────────────────────────────────

def render_header():
    st.title("🛒 Retail Shelf Intelligence")
    st.markdown("Upload a shelf image to detect products, count stock, and identify anomalies.")
    st.divider()


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("Connection")
        status = api_available()
        st.success("API connected ✅") if status else st.error("API offline ❌")

        st.divider()
        st.subheader("Detection")
        conf = st.slider("Confidence threshold", 0.1, 0.9, cfg.CONFIDENCE_THRESHOLD, 0.05)

        st.divider()
        st.subheader("Anomaly thresholds")
        empty_thresh = st.slider("Empty shelf (max products)", 0, 10, cfg.EMPTY_SHELF_MAX_PRODUCTS)
        low_thresh = st.slider("Low stock (max products)", 1, 20, cfg.LOW_STOCK_MAX_PRODUCTS)

        st.divider()
        st.subheader("Model info")
        st.caption(f"Model: `{cfg.MODEL_NAME}`")
        st.caption(f"Mode: `{cfg.DETECTION_MODE}`")
        st.caption(f"Device: `{cfg.DEVICE}`")

        return conf, empty_thresh, low_thresh


def render_metrics(data: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total products", data["total_products"])
    col2.metric("Avg confidence", f"{data['avg_confidence']:.2f}")
    col3.metric("Anomalies found", len(data["anomalies"]))
    col4.metric("Processing time", f"{data.get('processing_time_ms', 0):.0f} ms")


def render_detection_image(pil_image: Image.Image, detections: list):
    annotated = draw_detections_on_image(pil_image, detections)
    st.image(annotated, caption="Detected products", use_container_width=True)


def render_heatmap(pil_image: Image.Image, detections: list):
    """Render heatmap + gap map side by side."""
    from src.analytics.heatmap import generate_heatmap, overlay_heatmap, generate_gap_map

    img_array = np.array(pil_image)
    h, w = img_array.shape[:2]

    heat = generate_heatmap(detections, w, h)
    heat_img = overlay_heatmap(img_array, heat)

    gap = generate_gap_map(detections, w, h)
    import cv2
    gap_color = cv2.applyColorMap((gap * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    gap_color = cv2.cvtColor(gap_color, cv2.COLOR_BGR2RGB)
    gap_img = cv2.addWeighted(img_array, 0.4, gap_color, 0.6, 0)

    col1, col2 = st.columns(2)
    with col1:
        st.image(heat_img, caption="Product density heatmap", use_container_width=True)
    with col2:
        st.image(gap_img, caption="Shelf gap map (empty areas)", use_container_width=True)


def render_shelf_share(data: dict):
    """Render shelf share analysis."""
    from src.analytics.shelf_share import calculate_shelf_share

    share = calculate_shelf_share(
        data["detections"],
        data["image_width"],
        data["image_height"],
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        # Occupancy gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(share.occupancy_rate * 100, 1),
            title={"text": "Shelf Occupancy"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1D9E75"},
                "steps": [
                    {"range": [0, 30], "color": "#FFE0E0"},
                    {"range": [30, 70], "color": "#FFF3E0"},
                    {"range": [70, 100], "color": "#E0FFE8"},
                ],
            },
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if share.share_by_class:
            fig = go.Figure(go.Pie(
                labels=list(share.share_by_class.keys()),
                values=list(share.share_by_class.values()),
                hole=0.4,
                textinfo="label+percent",
            ))
            fig.update_layout(
                title="Shelf share by product",
                height=250,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)


def render_product_chart(data: dict):
    """Render product counts — uses OCR-identified names if available, else class names."""
    inv = data.get("product_inventory", {})
    counts_by_name = inv.get("counts_by_name", {})

    if counts_by_name:
        # Sort descending by count
        sorted_items = sorted(counts_by_name.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        fig = go.Figure(go.Bar(
            x=counts,
            y=names,
            orientation="h",
            marker_color="#534AB7",
            text=counts,
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Products by name ({inv.get('unique_products', 0)} unique)",
            xaxis_title="Count",
            height=max(300, len(names) * 28 + 100),
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show identification stats
        identified = inv.get("total_identified", 0)
        unidentified = inv.get("total_unidentified", 0)
        total = identified + unidentified
        if total > 0:
            st.caption(
                f"📊 {identified}/{total} products identified by name "
                f"({identified/total*100:.0f}%)"
            )
    else:
        # Fallback to generic class counts
        counts_by_class = data.get("counts_by_class", {})
        if not counts_by_class:
            st.info("No products detected.")
            return

        fig = go.Figure(go.Bar(
            x=list(counts_by_class.keys()),
            y=list(counts_by_class.values()),
            marker_color="#1D9E75",
            text=list(counts_by_class.values()),
            textposition="outside",
        ))
        fig.update_layout(
            title="Products by class",
            xaxis_title="Class", yaxis_title="Count",
            height=300, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_zone_chart(zones: list):
    if not zones:
        return

    zone_ids = [f"Zone {z['zone_id']}" for z in zones]
    counts = [z["count"] for z in zones]
    colors = []
    for c in counts:
        if c <= cfg.EMPTY_SHELF_MAX_PRODUCTS:
            colors.append("#E24B4A")
        elif c <= cfg.LOW_STOCK_MAX_PRODUCTS:
            colors.append("#EF9F27")
        else:
            colors.append("#1D9E75")

    fig = go.Figure(go.Bar(
        x=zone_ids, y=counts,
        marker_color=colors,
        text=counts, textposition="outside",
    ))
    fig.update_layout(
        title="Products by shelf zone",
        xaxis_title="Zone", yaxis_title="Count",
        height=300, margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_anomalies(anomalies: list):
    if not anomalies:
        st.success("✅ No anomalies detected. Shelf looks healthy.")
        return

    st.warning(f"⚠️ {len(anomalies)} anomaly/anomalies detected")

    grouped = {}
    for a in anomalies:
        grouped.setdefault(a["type"], []).append(a)

    type_info = {
        "empty_shelf": ("🔴", "Empty Shelf"),
        "low_stock": ("🟠", "Low Stock"),
        "misplaced": ("🟡", "Misplaced"),
        "ml_anomaly": ("🟣", "ML Anomaly"),
    }

    for atype, items in grouped.items():
        icon, label = type_info.get(atype, ("⚪", atype))
        severity = items[0].get("severity", "low").upper()
        st.markdown(f"**{icon} {label}** — {len(items)} detected &nbsp; `{severity}`")

        zone_ids = sorted(set(str(a.get("zone_id", "?")) for a in items))
        if zone_ids != ["-1"]:
            st.caption(f"Zones affected: {', '.join(zone_ids)}")
        else:
            st.caption("Detected across shelf")

    with st.expander("📋 View all details"):
        for a in anomalies:
            icon = severity_badge(a["severity"])
            st.markdown(f"{icon} **[{a['type']}]** {a['description']}")


def render_confidence_histogram(detections: list):
    if not detections:
        return
    confs = [d["confidence"] for d in detections]
    fig = px.histogram(
        x=confs, nbins=20,
        labels={"x": "Confidence", "y": "Count"},
        title="Detection confidence distribution",
        color_discrete_sequence=["#534AB7"],
    )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ── Page: Single Image Analysis ──────────────────────────────────────────────

def page_single_image():
    """Main single-image analysis page."""
    uploaded = st.file_uploader(
        "Upload a shelf image",
        type=["jpg", "jpeg", "png"],
        help="Upload a retail shelf photo to analyse",
    )

    if uploaded is None:
        st.info("👆 Upload an image to get started.")
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                **What this dashboard shows:**
                - 🔍 Detected products with bounding boxes
                - 🔥 Product density heatmaps
                - 📊 Shelf share & occupancy analysis
                - 🚨 Rule-based + ML anomaly alerts
                - 📝 OCR text extraction (prices, labels)
                """)
        return

    pil_image = Image.open(uploaded).convert("RGB")
    image_bytes = uploaded.getvalue()

    with st.spinner("Analysing shelf — detecting products & reading labels (this may take a moment)..."):
        try:
            data = call_api_detect(image_bytes)
        except Exception as e:
            st.error(f"Detection failed: {e}")
            return

    # Metrics
    st.subheader("Summary")
    render_metrics(data)
    st.divider()

    # Detection + anomalies
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Shelf view")

        view_mode = st.radio(
            "View mode",
            ["Detections", "Heatmap", "Gap map"],
            horizontal=True,
        )

        if view_mode == "Detections":
            render_detection_image(pil_image, data["detections"])
        elif view_mode == "Heatmap":
            heatmap_img = generate_heatmap_overlay(pil_image, data["detections"])
            st.image(heatmap_img, caption="Product density heatmap", use_container_width=True)
        elif view_mode == "Gap map":
            from src.analytics.heatmap import generate_gap_map
            import cv2
            img_array = np.array(pil_image)
            gap = generate_gap_map(data["detections"], img_array.shape[1], img_array.shape[0])
            gap_color = cv2.applyColorMap((gap * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            gap_color = cv2.cvtColor(gap_color, cv2.COLOR_BGR2RGB)
            gap_img = cv2.addWeighted(img_array, 0.4, gap_color, 0.6, 0)
            st.image(gap_img, caption="Shelf gaps (bright = empty)", use_container_width=True)

    with right:
        st.subheader("Anomalies")
        render_anomalies(data["anomalies"])

    st.divider()

    # Shelf share
    st.subheader("📊 Shelf share analysis")
    render_shelf_share(data)

    st.divider()

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        render_product_chart(data)
    with col2:
        render_zone_chart(data["zones"])

    st.divider()
    render_confidence_histogram(data["detections"])

    # Product inventory summary
    inv = data.get("product_inventory", {})
    if inv.get("counts_by_name"):
        st.divider()
        st.subheader("🏷️ Product inventory")
        col1, col2, col3 = st.columns(3)
        col1.metric("Unique products", inv.get("unique_products", 0))
        col2.metric("Identified", inv.get("total_identified", 0))
        col3.metric("Unidentified", inv.get("total_unidentified", 0))

        with st.expander("📋 View all identified products"):
            for det in data["detections"]:
                name = det.get("product_name", "Unknown")
                if name != "Unknown":
                    st.markdown(
                        f"**{name}** — conf: {det['confidence']:.2f}"
                    )

    # Raw JSON
    with st.expander("Raw response JSON"):
        st.json(data)


# ── Page: Batch Upload ────────────────────────────────────────────────────────

def page_batch_upload():
    """Multi-image batch analysis page."""
    st.subheader("📦 Batch Image Upload")
    st.markdown("Upload multiple shelf images for batch analysis.")

    uploaded_files = st.file_uploader(
        "Upload shelf images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("👆 Upload multiple images to get started.")
        return

    st.info(f"📷 {len(uploaded_files)} images selected")

    if st.button("🚀 Analyse all", type="primary"):
        with st.spinner(f"Processing {len(uploaded_files)} images..."):
            try:
                batch_data = call_api_batch(uploaded_files)
            except Exception as e:
                st.error(f"Batch detection failed: {e}")
                return

        results = batch_data["results"]

        # Summary metrics
        total_products = sum(r.get("total_products", 0) for r in results if "error" not in r)
        total_anomalies = sum(len(r.get("anomalies", [])) for r in results if "error" not in r)
        avg_conf = np.mean([
            r.get("avg_confidence", 0) for r in results if "error" not in r
        ]) if results else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Images processed", len(results))
        col2.metric("Total products", total_products)
        col3.metric("Total anomalies", total_anomalies)
        col4.metric("Avg confidence", f"{avg_conf:.2f}")

        st.divider()

        # Per-image results
        for i, (file, result) in enumerate(zip(uploaded_files, results)):
            if "error" in result:
                st.error(f"❌ {file.name}: {result['error']}")
                continue

            with st.expander(
                f"📷 {result.get('filename', file.name)} — "
                f"{result['total_products']} products, "
                f"{len(result['anomalies'])} anomalies"
            ):
                col1, col2 = st.columns([2, 1])
                with col1:
                    pil = Image.open(file).convert("RGB")
                    annotated = draw_detections_on_image(pil, result["detections"])
                    st.image(annotated, use_container_width=True)
                with col2:
                    st.metric("Products", result["total_products"])
                    st.metric("Confidence", f"{result['avg_confidence']:.2f}")
                    if result["anomalies"]:
                        for a in result["anomalies"][:3]:
                            st.markdown(
                                f"{severity_badge(a['severity'])} "
                                f"**{a['type']}**: {a['description'][:60]}..."
                            )
                    else:
                        st.success("✅ Healthy")

        # Comparison chart
        st.divider()
        st.subheader("Cross-image comparison")
        names = [r.get("filename", f"img_{i}") for i, r in enumerate(results) if "error" not in r]
        counts = [r["total_products"] for r in results if "error" not in r]

        fig = go.Figure(go.Bar(x=names, y=counts, marker_color="#1D9E75"))
        fig.update_layout(
            title="Products per image",
            xaxis_title="Image", yaxis_title="Count",
            height=300, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Before/After Comparison ─────────────────────────────────────────────

def page_before_after():
    """Side-by-side comparison of two shelf images."""
    st.subheader("🔄 Before / After Comparison")
    st.markdown("Compare two shelf images side by side to track changes.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before** (e.g. before restocking)")
        before_file = st.file_uploader("Upload BEFORE image", type=["jpg", "jpeg", "png"], key="before")
    with col2:
        st.markdown("**After** (e.g. after restocking)")
        after_file = st.file_uploader("Upload AFTER image", type=["jpg", "jpeg", "png"], key="after")

    if not before_file or not after_file:
        st.info("👆 Upload both images to compare.")
        return

    if st.button("🔍 Compare", type="primary"):
        with st.spinner("Analysing both images..."):
            try:
                before_data = call_api_detect(before_file.getvalue())
                after_data = call_api_detect(after_file.getvalue())
            except Exception as e:
                st.error(f"Detection failed: {e}")
                return

        # Delta metrics
        st.subheader("📊 Changes detected")
        col1, col2, col3, col4 = st.columns(4)

        delta_products = after_data["total_products"] - before_data["total_products"]
        delta_anomalies = len(after_data["anomalies"]) - len(before_data["anomalies"])
        delta_conf = after_data["avg_confidence"] - before_data["avg_confidence"]

        col1.metric("Before products", before_data["total_products"])
        col2.metric("After products", after_data["total_products"], delta=delta_products)
        col3.metric("Anomaly change", len(after_data["anomalies"]),
                     delta=delta_anomalies, delta_color="inverse")
        col4.metric("Confidence change", f"{after_data['avg_confidence']:.2f}",
                     delta=f"{delta_conf:+.2f}")

        st.divider()

        # Side-by-side images
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before**")
            pil_before = Image.open(before_file).convert("RGB")
            annotated_before = draw_detections_on_image(pil_before, before_data["detections"])
            st.image(annotated_before, use_container_width=True)

        with col2:
            st.markdown("**After**")
            pil_after = Image.open(after_file).convert("RGB")
            annotated_after = draw_detections_on_image(pil_after, after_data["detections"])
            st.image(annotated_after, use_container_width=True)

        # Zone comparison
        st.divider()
        st.subheader("Zone-by-zone comparison")

        before_zones = {z["zone_id"]: z["count"] for z in before_data.get("zones", [])}
        after_zones = {z["zone_id"]: z["count"] for z in after_data.get("zones", [])}
        all_zone_ids = sorted(set(list(before_zones.keys()) + list(after_zones.keys())))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Before",
            x=[f"Zone {z}" for z in all_zone_ids],
            y=[before_zones.get(z, 0) for z in all_zone_ids],
            marker_color="#E24B4A",
        ))
        fig.add_trace(go.Bar(
            name="After",
            x=[f"Zone {z}" for z in all_zone_ids],
            y=[after_zones.get(z, 0) for z in all_zone_ids],
            marker_color="#1D9E75",
        ))
        fig.update_layout(
            barmode="group",
            title="Products per zone: Before vs After",
            height=300, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Live Camera Feed ────────────────────────────────────────────────────

def page_live_camera():
    """Live camera feed with periodic detection."""
    st.subheader("📹 Live Camera Feed")
    st.markdown("Capture from your webcam and run detection in real-time.")

    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        import av

        st.info("Click **START** to begin camera capture, then click **📸 Capture & Analyse** to detect products.")

        webrtc_ctx = webrtc_streamer(
            key="live-cam",
            mode=WebRtcMode.SENDONLY,
            media_stream_constraints={"video": True, "audio": False},
        )

        if st.button("📸 Capture & Analyse"):
            if webrtc_ctx.video_receiver:
                try:
                    frame = webrtc_ctx.video_receiver.get_frame(timeout=5)
                    img = frame.to_ndarray(format="rgb24")
                    pil_img = Image.fromarray(img)

                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG")

                    with st.spinner("Analysing captured frame..."):
                        data = call_api_detect(buf.getvalue())

                    render_metrics(data)
                    annotated = draw_detections_on_image(pil_img, data["detections"])
                    st.image(annotated, caption="Live capture detection", use_container_width=True)
                    render_anomalies(data["anomalies"])
                except Exception as e:
                    st.error(f"Capture failed: {e}")
            else:
                st.warning("Camera not started. Click START first.")

    except ImportError:
        st.warning("📹 Live camera requires `streamlit-webrtc`. Install with:\n```\npip install streamlit-webrtc\n```")

    st.divider()

    # Manual image capture fallback
    st.subheader("📷 Manual capture")
    camera_input = st.camera_input("Take a photo")
    if camera_input:
        with st.spinner("Analysing..."):
            try:
                data = call_api_detect(camera_input.getvalue())
                pil_img = Image.open(camera_input).convert("RGB")

                render_metrics(data)
                st.divider()
                annotated = draw_detections_on_image(pil_img, data["detections"])
                st.image(annotated, caption="Camera capture", use_container_width=True)
                render_anomalies(data["anomalies"])
            except Exception as e:
                st.error(f"Detection failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    render_header()
    conf, empty_thresh, low_thresh = render_sidebar()

    cfg.CONFIDENCE_THRESHOLD = conf
    cfg.EMPTY_SHELF_MAX_PRODUCTS = empty_thresh
    cfg.LOW_STOCK_MAX_PRODUCTS = low_thresh

    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Single Image",
        "📦 Batch Upload",
        "🔄 Before / After",
        "📹 Live Camera",
    ])

    with tab1:
        page_single_image()
    with tab2:
        page_batch_upload()
    with tab3:
        page_before_after()
    with tab4:
        page_live_camera()


if __name__ == "__main__":
    main()
