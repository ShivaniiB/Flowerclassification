import os
from src.data import create_datasets
from src.model import build_model

def main():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(ROOT_DIR, "data")
    model_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    train_ds, test_ds, class_names = create_datasets(data_dir)

    model = build_model(num_classes=len(class_names))
    model.summary()

    print("\nTraining started...\n")
    model.fit(
        train_ds,
        epochs=15,
        validation_data=test_ds
    )

    model_path = os.path.join(model_dir, "flower_cnn.keras")
    model.save(model_path)
    print(f"\nModel saved at: {model_path}")

if __name__ == "__main__":
    main()
