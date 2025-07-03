# 📈 LSTM Forecasting for Time Series

##  Context

As part of my internship, I explored the use of **machine learning for time series forecasting**. Specifically, I implemented a **Long Short-Term Memory (LSTM)** model, which is particularly well-suited for sequential and temporal data.

##  What is an LSTM?

Imagine you're watching a movie: to understand the current scene, you need to remember what happened before. Similarly, an LSTM network uses **short- and long-term memory** to decide what information to retain or forget over time, making it ideal for modeling **temporal dependencies** in time series data.

##  Data

Due to privacy constraints, the dataset used in this project cannot be shared.  
However, the main goal was to build a model capable of predicting multiple months into the future.

To do this, a **recursive (autoregressive) forecasting** approach was used:  
→ each predicted value is fed back into the model to forecast the next time step.



##  Challenges

1. **Sparse data**: the dataset had discontinuities that complicated the modeling process.
2. **Seasonality**: the data exhibited strong cyclic patterns (e.g., January to December), requiring frequency encoding.

##  Preprocessing Pipeline

- Data cleaning
- Cyclical encoding of months (`sin` and `cos`)
- Sequence generation using  sliding window
- Train/test split
- Recursive multi-step prediction

> Each sequence includes 3 features:  
> • `value` (target)  
> • `month_sin`, `month_cos` (cyclical encoding)

##  Model Architecture

To better capture **seasonality**, several experiments were conducted.  
Adding **sine and cosine projections** of the month to represent it on a **trigonometric circle** improved the model's ability to learn temporal patterns.

### Final architecture:
- 1 LSTM layer (200 units)
- 1 Dense output layer

##  Results

The model was evaluated using a 12-month recursive forecast strategy.

| Metric     | Before Improvements | After Improvements |
|------------|---------------------|--------------------|
| **MAPE**   | 0.2881              | **0.1441**         |
| **R²**     | 0.3741              | **0.9332**         |

> The inclusion of cyclical encoding (`sin`, `cos`) and a **12-month window** significantly improved accuracy by helping the model capture the underlying seasonal structure.

These results demonstrate a **clear improvement** in forecasting performance.

##  Tech Stack

- **Language**: Python
- **Libraries**:
  - TensorFlow / Keras
  - NumPy
  - Pandas
  - Matplotlib / Seaborn
- **Model**: Multivariate Recursive LSTM
- **Forecasting method**: Multi-step recursive prediction
