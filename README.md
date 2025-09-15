# Hands-On Lab

This lab demonstrates how to integrate Launchable with existing simple Java project. You’ll be able to understand Launchable thorough editing CI configurations and adding a new feature to this project.

# Table of contens

- [Hands-on 1. Setup repository](HANDSON1.md)
- [Hands-on 2. Introduce Launchable command](HANDSON2.md)
- [Hands-on 3. Run test with predictive test selection](HANDSON3.md)

# MLOps  Pipeline – Read Me

# Overview:
This MLOps pipeline will illustrate the steps in creating a simple classification model to classify the species of IRIS flower from the given IRIS flower features.

# Pipeline Steps:
1.	Train the ML Model (using Scikit-learn’s logistic_regression technique.
2.	Evaluate the model
3.	Deploy the model

# Pipeline project structure:
<img width="207" height="132" alt="image" src="https://github.com/user-attachments/assets/18cf7e93-fb58-44e9-920b-db7c3315de30" />


# Pipeline execution:
1.	Create a Python virtual environment
2.	Install the required libraries by running the following command:
pip install -r requirements.txt
3.	Train the model using the following command:
python src/train.py --config config.yaml
4.	Evaluate the model using the command:
python src/evaluate.py --config config.yaml
5.	Deploy the model using the command:
uvicorn src.serve:app --host 0.0.0.0 --port 8000 –reload
6.	Model prediction using the following command or use REST Client like Postman
curl -X POST http://localhost:8000/predict \
-H 'Content-Type: application/json' \
-d '[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}]'

 
<img width="468" height="647" alt="image" src="https://github.com/user-attachments/assets/1d8aa4a3-aca6-4447-a86f-84f4e245e26d" />









