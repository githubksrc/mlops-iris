from fastapi.testclient import TestClient
from main import app

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    # test values are changed
    payload = {
        "sepal_length": 7.2,
        "sepal_width": 3.2,
        "petal_length": 6.3,
        "petal_width": 3.1,
        
      }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Virginica"}


# test to check if Iris Setosa is classified correctly
def test_pred_Setosa():
    # defining a sample payload for the testcase
    # test values are changed
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        
      }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Setosa"}

# test to check if Iris Versicolour is classified correctly
def test_pred_Versicolour():
    # defining a sample payload for the testcase
    # test values are changed
    payload = {
        "sepal_length": 4.2,
        "sepal_width": 2.2,
        "petal_length": 3.5,
        "petal_width": 2.1,
        
      }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Versicolour"}
