import requests, json

print("Testing integrated OCR product identification...")
r = requests.post(
    "http://localhost:8000/detect",
    files={"file": ("val_14.jpg", open("data/processed/images/val/val_14.jpg", "rb"), "image/jpeg")},
    timeout=120,
)
data = r.json()

print(f"\nStatus: {r.status_code}")
print(f"Total products detected: {data['total_products']}")
print(f"Processing time: {data['processing_time_ms']:.0f} ms")

inv = data.get("product_inventory", {})
print(f"\n=== Product Inventory (OCR) ===")
print(f"Unique products: {inv.get('unique_products', 0)}")
print(f"Identified: {inv.get('total_identified', 0)}")
print(f"Unidentified: {inv.get('total_unidentified', 0)}")

counts = inv.get("counts_by_name", {})
if counts:
    print(f"\nProducts by name:")
    for name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {count}")
else:
    print("\nNo products identified by name.")

# Check that detections have product_name field
sample = data["detections"][:3]
print(f"\nSample detections with names:")
for d in sample:
    print(f"  {d.get('product_name', 'N/A')} (conf: {d['confidence']:.2f})")
