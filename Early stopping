# EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss',    # Monitor validation loss
                               patience=3,            # Number of epochs to wait before stopping
                               restore_best_weights=True,  # Restore the best weights once stopped
                               verbose=1)

# Create model and fit with early stopping
model = create_model(learning_rate=0.0001)
model.fit(train_data, validation_data=val_data, epochs=50, callbacks=[early_stopping], verbose=1)
