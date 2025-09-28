import pandas as pd
import torch

from DataFormattage import DataFormattage
from ModelTraining import HandPoseClassifier


def main():
    print("Hello from sign-language-translator!")

    poses = ["hello", "thank_you", "i_love_you", "yes", "no", "please", "albania"]
    #     poses = ["hello", "i_love_you", "yes", "no", "please"]
    formatter = DataFormattage(poses=poses, path="./data")
    df = formatter.all_metadata()

    input_dim = df.drop(columns=["pose"]).shape[1]
    hidden_dim = 100
    num_classes = df["pose"].nunique()

    classifier = HandPoseClassifier(
        input_dim, hidden_dim, num_classes, data=df, test_rate=0.2, y_label="pose"
    )

    classifier.Train(epochs=250)
    acc = classifier.Test()

    print(f"Test accuracy: {acc}")

    torch.save(classifier.model.state_dict(), "model.pth")
    print("Model and label classes saved.")


if __name__ == "__main__":
    main()
