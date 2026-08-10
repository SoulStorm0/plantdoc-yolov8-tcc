from pathlib import Path

from PIL import Image, ImageDraw


def main():
    root = Path("datasets/smoke").resolve()
    for split, count in (("train", 4), ("valid", 2), ("test", 2)):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            class_id = index % 2
            image = Image.new("RGB", (96, 96), (235, 235, 225))
            draw = ImageDraw.Draw(image)
            color = (30, 150, 50) if class_id == 0 else (145, 75, 35)
            draw.rectangle((24, 24, 72, 72), fill=color)
            image.save(image_dir / f"sample_{index}.jpg")
            (label_dir / f"sample_{index}.txt").write_text(
                f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
            )
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames: [healthy, diseased]\n",
        encoding="utf-8",
    )
    print(root / "data.yaml")


if __name__ == "__main__":
    main()

