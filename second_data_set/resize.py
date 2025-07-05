import os
import cv2

# Dimensions
orig_w, orig_h = 480, 300
target_size = 416
scale_x = target_size / orig_w
scale_y = target_size / orig_h

# Paths
image_dir = "images"
output_image_dir = "images_resized"
os.makedirs(output_image_dir, exist_ok=True)

label_splits = ["train", "val", "trainval"]
label_input_root = "new_labels"
label_output_root = "resized_labels"

# Track processed images
processed_images = set()

for split in label_splits:
    input_label_dir = os.path.join(label_input_root, split)
    output_label_dir = os.path.join(label_output_root, split)
    os.makedirs(output_label_dir, exist_ok=True)

    for label_file in os.listdir(input_label_dir):
        if not label_file.endswith(".txt"):
            continue

        # Image name
        image_name = os.path.splitext(label_file)[0] + ".jpg"
        input_image_path = os.path.join(image_dir, image_name)
        output_image_path = os.path.join(output_image_dir, image_name)

        # Resize image only once
        if image_name not in processed_images:
            if not os.path.exists(input_image_path):
                print(f"⚠️ Missing image: {image_name}")
                continue
            img = cv2.imread(input_image_path)
            if img is None:
                print(f"⚠️ Failed to load: {image_name}")
                continue

            resized_img = cv2.resize(img, (target_size, target_size))
            cv2.imwrite(output_image_path, resized_img)
            processed_images.add(image_name)

        # Adjust labels
        input_label_path = os.path.join(input_label_dir, label_file)
        output_label_path = os.path.join(output_label_dir, label_file)

        with open(input_label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, x, y, w, h = map(float, parts)

            # Unnormalize from 480x300
            x_abs = x * orig_w
            y_abs = y * orig_h
            w_abs = w * orig_w
            h_abs = h * orig_h

            # Rescale
            x_new = (x_abs * scale_x) / target_size
            y_new = (y_abs * scale_y) / target_size
            w_new = (w_abs * scale_x) / target_size
            h_new = (h_abs * scale_y) / target_size

            new_lines.append(f"{int(cls)} {x_new:.6f} {y_new:.6f} {w_new:.6f} {h_new:.6f}")

        with open(output_label_path, "w") as f:
            f.write("\n".join(new_lines))
