# 🖥️ Backend – Flask Application

This folder contains the Flask-based backend for the **TrafficTelligence** project, which estimates traffic volume using trained machine learning models. It handles API routing, user inputs, model predictions, and serves results to the frontend UI.

---

## 📄 File: `app.py`

- Loads a trained traffic volume prediction model (`traffic_model.pkl`)
- Accepts user input via web form (e.g., date, time, weather conditions, etc.)
- Preprocesses and encodes the input features
- Makes a prediction using the trained model
- Returns the predicted traffic volume and displays it on the result page

---

## 🧰 Libraries Used

| Library            | Purpose                                         |
|--------------------|-------------------------------------------------|
| **Flask**          | Lightweight web framework for backend routing   |
| **NumPy**          | Numerical array processing                      |
| **Pandas**         | Handling structured tabular data                |
| **Scikit-learn**   | Model training, preprocessing, and prediction   |
| **Pickle**         | Serialization of the trained model              |
| **Werkzeug**       | Secure file and form handling (if applicable)   |

---

## ▶️ How to Run the App

Make sure you have Python 3.7+ and install the necessary dependencies.

### 🔧 Step 1: Install dependencies

```bash
pip install -r requirements.txt
python app.py
Backend (Flask)/
├── app.py
├── model/
│   └── traffic_model.pkl
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── (optional CSS/JS/images)
├── requirements.txt

Let me know if you want:

- A `requirements.txt` file generated automatically  
- A diagram of request flow  
- Code sample for `app.py` used in this project  
- Instructions for deploying this app online (e.g., on Render, Railway, or Heroku)

