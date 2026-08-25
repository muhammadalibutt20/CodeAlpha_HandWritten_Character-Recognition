"""Train a CNN and recognize handwritten sentences drawn in a window."""

from pathlib import Path
import argparse
import gzip
import random
import tkinter as tk

import kagglehub
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFilter
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


SEED = 42
IMAGE_SIZE = 28
CLASS_COUNT = 62  # EMNIST ByClass: digits, uppercase letters, lowercase letters.
BATCH_SIZE = 128
EPOCHS = 10
MAX_SAMPLES = None  # Set to an integer for a faster local experiment.
MODEL_FILE = Path("emnist_character_cnn.pt")
MODEL_VERSION = 3
LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def seed_everything(seed: int = SEED) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def find_csv(dataset_dir: Path, split: str) -> Path:
	candidates = sorted(
		file for file in dataset_dir.rglob("*")
		if file.is_file() and "byclass" in file.name.lower()
		and split in file.name.lower() and file.suffix.lower() in {".csv", ".gz"}
	)
	if not candidates:
		raise FileNotFoundError(f"Could not find an EMNIST ByClass {split} CSV under {dataset_dir}.")
	return candidates[0]


def preprocess_images(pixels: np.ndarray) -> np.ndarray:
	images = pixels.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
	# EMNIST stores images rotated 90 degrees and mirrored; a plain transpose
	# is the correct fix (equivalent to the standard fliplr-then-rot90 recipe).
	images = images.transpose(0, 2, 1).copy()
	processed = np.empty_like(images)
	for index, image in enumerate(images):
		processed[index] = np.asarray(
			Image.fromarray(image).filter(ImageFilter.MedianFilter(size=3))
		)
	images = processed.astype(np.float32) / 255.0
	return ((images - 0.1307) / 0.3081)[:, None, :, :]


def load_emnist_images(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
	opener = gzip.open if file_path.suffix.lower() == ".gz" else open
	with opener(file_path, "rt", newline="") as data_file:
		frame = pd.read_csv(data_file, header=None)
	labels = frame.iloc[:, 0].to_numpy(dtype=np.int64)
	pixels = frame.iloc[:, 1:].to_numpy(dtype=np.uint8)
	if pixels.shape[1] != IMAGE_SIZE * IMAGE_SIZE:
		raise ValueError(f"Expected 784 pixels per row, found {pixels.shape[1]}.")
	if MAX_SAMPLES is not None:
		pixels, labels = pixels[:MAX_SAMPLES], labels[:MAX_SAMPLES]
	return preprocess_images(pixels), labels


class CharacterCNN(nn.Module):
	def __init__(self, class_count: int = CLASS_COUNT) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
			nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
			nn.Dropout(0.25),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5),
			nn.Linear(128, class_count),
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.classifier(self.features(inputs))


def make_loaders(images: np.ndarray, labels: np.ndarray) -> tuple[DataLoader, DataLoader]:
	train_images, validation_images, train_labels, validation_labels = train_test_split(
		images, labels, test_size=0.1, random_state=SEED, stratify=labels
	)
	train_set = AugmentedCharacterDataset(train_images, train_labels)
	validation_set = TensorDataset(torch.from_numpy(validation_images), torch.from_numpy(validation_labels))
	return (
		DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
		DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
	)


class AugmentedCharacterDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
	"""Add canvas-like position and stroke-width variation during training."""

	def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
		self.images = torch.from_numpy(images)
		self.labels = torch.from_numpy(labels)

	def __len__(self) -> int:
		return len(self.labels)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
		image = self.images[index]
		if random.random() < 0.8:
			image = torch.roll(
				image, shifts=(random.randint(-2, 2), random.randint(-2, 2)), dims=(1, 2)
			)
		if random.random() < 0.5:
			image = torch.nn.functional.max_pool2d(
				image[None], kernel_size=3, stride=1, padding=1
			)[0]
		return image, self.labels[index]


def train_model(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, device: torch.device) -> None:
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
	best_accuracy = 0.0
	for epoch in range(1, EPOCHS + 1):
		model.train()
		for images, labels in train_loader:
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()
			criterion(model(images), labels).backward()
			optimizer.step()
		model.eval()
		correct = total = 0
		with torch.no_grad():
			for images, labels in validation_loader:
				predictions = model(images.to(device)).argmax(dim=1)
				correct += (predictions == labels.to(device)).sum().item()
				total += labels.size(0)
		accuracy = correct / total
		print(f"Epoch {epoch}/{EPOCHS} - validation accuracy: {accuracy:.2%}")
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			torch.save({"version": MODEL_VERSION, "state_dict": model.state_dict()}, MODEL_FILE)


def normalize_character(crop: np.ndarray) -> np.ndarray:
	image = Image.fromarray(crop.astype(np.uint8))
	box = image.getbbox()
	if box is None:
		return np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
	image = image.crop(box)
	image.thumbnail((20, 20), Image.Resampling.LANCZOS)
	canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
	canvas.paste(image, ((IMAGE_SIZE - image.width) // 2, (IMAGE_SIZE - image.height) // 2))
	canvas = canvas.filter(ImageFilter.MedianFilter(size=3))
	values = np.asarray(canvas, dtype=np.float32) / 255.0
	return ((values - 0.1307) / 0.3081)[None, :, :]


def segment_sentence(image: Image.Image) -> list[np.ndarray]:
	pixels = np.asarray(image)
	ink = pixels > 40
	row_indices = np.flatnonzero(ink.any(axis=1))
	if not len(row_indices):
		return []
	line_groups = np.split(row_indices, np.where(np.diff(row_indices) > 8)[0] + 1)
	characters: list[np.ndarray] = []
	for rows in line_groups:
		columns = np.flatnonzero(ink[rows[0]:rows[-1] + 1].any(axis=0))
		gaps = np.diff(columns)
		# Character spacing is relative to the writing, not a fixed canvas size:
		# use the median width of continuous ink strokes (not the gaps between them).
		stroke_breaks = np.where(gaps > 1)[0] + 1
		stroke_widths = [run[-1] - run[0] + 1 for run in np.split(columns, stroke_breaks)]
		character_gap = max(8, min(24, int(np.median(stroke_widths) * 2.5))) if stroke_widths else 12
		column_groups = np.split(columns, np.where(gaps > character_gap)[0] + 1)
		for group_index, columns_for_character in enumerate(column_groups):
			if group_index and columns_for_character[0] - column_groups[group_index - 1][-1] > character_gap * 2.5:
				characters.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
			crop = pixels[rows[0]:rows[-1] + 1, columns_for_character[0]:columns_for_character[-1] + 1]
			characters.append(crop)
	return characters


def start_recognizer(model: nn.Module, device: torch.device) -> None:
	window = tk.Tk()
	window.title("Handwritten Sentence Recognizer")
	window.configure(bg="#202124")
	canvas = tk.Canvas(window, width=1000, height=300, bg="black", highlightthickness=0)
	canvas.pack(padx=12, pady=12)
	image = Image.new("L", (1000, 300), 0)
	drawing = ImageDraw.Draw(image)
	last_point: list[tuple[int, int] | None] = [None]

	def draw(event: tk.Event) -> None:
		point = (event.x, event.y)
		if last_point[0] is None:
			canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill="white", outline="white")
			drawing.ellipse((event.x - 5, event.y - 5, event.x + 5, event.y + 5), fill=255)
		else:
			canvas.create_line(*last_point[0], *point, fill="white", width=10, capstyle=tk.ROUND, smooth=True)
			drawing.line([last_point[0], point], fill=255, width=10)
		last_point[0] = point

	def stop_drawing(_: tk.Event) -> None:
		last_point[0] = None

	def recognize() -> None:
		crops = segment_sentence(image)
		if not crops:
			result.set("Write a sentence first.")
			return
		batch = torch.from_numpy(np.stack([normalize_character(crop) for crop in crops])).to(device)
		with torch.no_grad():
			probabilities = torch.softmax(model(batch), dim=1)
			confidences, indices = probabilities.max(dim=1)
		indices = indices.cpu().tolist()
		confidences = confidences.cpu().tolist()
		text = "".join(
			LABELS[index] if crop.any() and confidence >= 0.35 else "?"
			for index, confidence, crop in zip(indices, confidences, crops)
		)
		result.set(text)

	def clear() -> None:
		canvas.delete("all")
		drawing.rectangle((0, 0, 1000, 300), fill=0)
		result.set("")

	canvas.bind("<B1-Motion>", draw)
	canvas.bind("<ButtonRelease-1>", stop_drawing)
	controls = tk.Frame(window, bg="#202124")
	controls.pack(fill="x", padx=12, pady=(0, 12))
	result = tk.StringVar()
	tk.Button(controls, text="Recognize", command=recognize).pack(side="left")
	tk.Button(controls, text="Clear", command=clear).pack(side="left", padx=8)
	tk.Label(controls, textvariable=result, bg="#202124", fg="white", font=("Segoe UI", 18)).pack(side="left", padx=12)
	window.mainloop()


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", action="store_true", help="Retrain the CNN before opening the recognizer.")
	args = parser.parse_args()
	seed_everything()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = CharacterCNN().to(device)
	checkpoint = torch.load(MODEL_FILE, map_location=device) if MODEL_FILE.exists() else None
	if args.train or checkpoint is None or checkpoint.get("version") != MODEL_VERSION:
		dataset_path = Path(kagglehub.dataset_download("crawford/emnist"))
		train_file = find_csv(dataset_path, "train")
		images, labels = load_emnist_images(train_file)
		train_loader, validation_loader = make_loaders(images, labels)
		print(f"Loaded {len(labels):,} ByClass images; training on {device}.")
		train_model(model, train_loader, validation_loader, device)
	checkpoint = torch.load(MODEL_FILE, map_location=device)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()
	start_recognizer(model, device)


if __name__ == "__main__":
	main()

