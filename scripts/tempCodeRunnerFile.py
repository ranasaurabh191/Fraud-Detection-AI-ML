cnn = Sequential()

# Change kernel_size to 1
cnn.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, X_train.shape[1])))
cnn.add(Flatten())
cnn.add(Dropout(0.3))
cnn.add(Dense(1, activation='sigmoid'))

cnn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
cnn.fit(X_train_rnn, y_train, epochs=5, batch_size=32, validation_data=(X_test_rnn, y_test))

# Save the model
cnn.save('./models/cnn_model.h5')
print("✅ CNN trained successfully!")


# ✅ Train RNN model
rnn = Sequential()
rnn.add(SimpleRNN(50, activation='relu', input_shape=(1, X_train.shape[1])))
rnn.add(Dense(1, activation='sigmoid'))
rnn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
rnn.fit(X_train_rnn, y_train, epochs=5, batch_size=32, validation_data=(X_test_rnn, y_test))
rnn.save('./models/rnn_model.h5')
print("✅ RNN trained!")

# ✅ Save models
import joblib
joblib.dump(gbm, './models/gbm_model.pkl')
joblib.dump(svm, './models/svm_model.pkl')

print("✅ Models saved successfully!")
