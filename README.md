File Structure : 

chatbot/
│── dataset/
│   ├── emergency_dataset.json    # Your dataset with responses & emergency contacts
│── models/
│   ├── intent_classifier.pkl     # Trained intent classification model (if used)
│   ├── response_generator.pkl    # Trained chatbot response model
│── src/
│   ├── chatbot.py                # Main chatbot logic
│   ├── train_model.py            # Script to train intent recognition model
│   ├── preprocess.py             # Data preprocessing and feature extraction
│── utils/
│   ├── text_processing.py        # Helper functions for text handling
│   ├── emergency_utils.py        # Utility functions to fetch emergency contacts
│── app.py                         # Flask/FastAPI app to serve chatbot (optional)
│── requirements.txt               # Dependencies
│── README.md                      # Project documentation
