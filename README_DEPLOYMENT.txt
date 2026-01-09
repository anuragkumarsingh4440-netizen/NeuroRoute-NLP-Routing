Deployment Artifacts:
1. final_neural_model.keras -> Trained neural network model
2. label_encoder.pkl -> Label decoding for predictions
3. model_results.pkl -> Stored metrics and reports
4. model_comparison.csv -> Model comparison table
5. X_validation.pkl, y_validation.pkl -> Validation dataset

Usage:
- Load model using keras.models.load_model
- Load label_encoder for inverse_transform
- Use same preprocessing + embeddings pipeline
