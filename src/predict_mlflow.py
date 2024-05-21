import mlflow


# Fetch the model
model_name = "fasttext"
version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{version}")


list_libs = ["COIFFEUR", "coiffeur, & 98789"]

results = model.predict(list_libs, params={"k": 1})
print(results)

