# AggriApp - Seed Recognition  

AggriApp is a **smart agriculture** application designed to assist farmers with plant disease detection, seed identification, and best gardening practices. This repository focuses on the **Seed Recognition** module, which classifies seed types using machine learning and provides a web interface for real-time testing.  

## Features  

**Machine Learning Model** – Trained to recognize different seed types based on physical attributes (size, shape, texture).  
**Web Interface** – User-friendly UI to upload seed images or data and receive instant predictions.  
**Extensible** – Can be adapted to recognize more seed types or integrate with other AggriApp modules.  

## Installation  

### Prerequisites  
Ensure you have Python 3.6+ installed along with the necessary dependencies:  

```bash
pip install -r requirements.txt
```

### Model Training  

1. Clone the repository:  
   ```bash
   git clone https://github.com/utachicodes/aggriapp-seed-recognition.git
   cd aggriapp-seed-recognition
   ```

2. Prepare your dataset (ensure seed images or feature data are labeled correctly).  

3. Train the model:  
   ```bash
   python train_model.py
   ```

4. The trained model (`seed_model.h5`) will be saved and used for predictions.  

### Web Interface  

1. Install additional dependencies:  
   ```bash
   pip install -r web_requirements.txt
   ```

2. Run the web application:  
   ```bash
   python app.py
   ```

3. Open `http://127.0.0.1:5000/` in your browser and upload seed data for classification.  

## Usage  

### Model Prediction (CLI)  

```python
from seed_classifier import model

# Load the trained model
model = model.load_model('seed_model.h5')

# Example seed data
seed_data = [5.0, 2.3, 1.4]  # Feature vector

# Get prediction
prediction = model.predict(seed_data)
print(f"Predicted seed type: {prediction}")
```

### Web Interface  

1. Open the web app (`http://127.0.0.1:5000/`).  
2. Upload an image or a dataset of seed features.  
3. Click "Predict" to classify the seed type.  

## Contribution  

We welcome contributions to improve the Seed Recognition module!  

1. Fork the repository.  
2. Create a new branch (`git checkout -b feature-name`).  
3. Implement your changes and commit (`git commit -m 'Added new feature'`).  
4. Push to your branch (`git push origin feature-name`).  
5. Open a Pull Request.  

## License  

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  



