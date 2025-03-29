# DeepLearning_AHRS
AHRS (accelerometer, gyroscope & magnetometer sensor fusion) using a LSTM model

An AHRS (Attitude and Heading Reference System) is a sensor fusion algorithm use to predict the attitude of an IMU (Inertial Measurement Unit).
Traditionnal algorithms like the Extended Kalman Filter, madgwick Filter, Manhony Filter or the simple complementary filter all work upon the same principle:
    1) Measure: The accelerometer measure the acceleration of the drone, the gravity vector and some noise. On the short term it is way too noisy to be usable. But over the long term the noise and the accelerations of the drone cancels out and the remaining gravity vector can be use to determine the attitude of the drone.
    The magnetometer measure the magnetic vector, so the yaw can be computed from it.
    2) Prediction: The gyroscope measure angular velocity with very little noise. This angular velocity can be integrated over time to compute the attitude (if we know the starting attitude). This is very precise on the short term but suffer from drift over the long term.
    3) Fusion: A low pass filter is applied to the attitude computed from the accelerometer + magnetometer and a high pass filter is applied to the attitude computed from the gyroscope.
Thoses algorithms are robust, widely used and suitable for low power embedded electronics. But the non-linearity of a drone movement make thoses algorithms less precise in some situations.

Since IMU data is sequential and attitude evolves over time, this is a time-series regression problem, making recurrent neural networks (RNNs), particularly LSTMs, a natural fit.

LSTM (Long Short Term Memory):
    The LSTM is deep learning neural network of the RNN (Recurent Neural Network) familly, suitable for handling temporal series. It's main advantage in this application is it's short term and long term memory, making it able to keep track of the IMU attitude over time.

LSTM in stateful mode
    LSTM models can work in stateful or stateless mode. In stateless mode, the model's context is limited to the current batch only, which is not ideal to keep track of the attitude over time. In stateful mode, the current batch keep in context the previous one wich help keeping track of the attitude over time.

Loss function:
    Angular error



