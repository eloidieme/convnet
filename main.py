from cnnClassifier.models.predict import predict

model_paths = [
    'models/best_with_compensating/F1', 
    'models/best_with_compensating/F2', 
    'models/best_with_compensating/W'
    ]

if __name__ == "__main__":
    name = input("Enter your name: ")
    prediction = predict([name], *model_paths)
    print(f"Your name is {prediction}!")