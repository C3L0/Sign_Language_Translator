from DataFormattage import DataFormattage
from ModelTraining import HandPoseClassifier

import pandas as pd


def main():
    print("Hello from sign-language-translator!")

    formatter = DataFormattage("./data")
    df = formatter.all_metadata()

    input_dim = df.drop(columns=["pose"]).shape[1]
    hidden_dim = 256
    num_classes = df["pose"].nunique()
    classifier = HandPoseClassifier(
        input_dim, hidden_dim, num_classes, data=df, test_rate=0.2, y_label="pose"
    )

    classifier.Train(epochs=50)
    acc = classifier.Test()

    print(f"Test accuracy: {acc}")


if __name__ == "__main__":
    main()
