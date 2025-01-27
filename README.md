# DATA CELLAR Unsupervised Anomaly Detection on Energy Production Data

Unsupervised framework to perform Anomaly Detection on Energy Production Data. 

Several models have been tested, three that perform reconstruction on the input time series and two that perform forecast of the next timestamp in the input sequence. The Linear Autoencoder has proven to be the best one due to its performance.

The Unsupervised approach is necessary as there are not public datasets representing energy production data which are also annotated specifically for the task of anomaly detection.

## Dataset

The dataset used for training and testing the models is publicly available on [Open Power System Data](https://data.open-power-system-data.org/time_series/): we considered the hourly-frequency timeseries, and considered mainly the columns "[country]_solar_generation_actual", as it consists of several timeseries depicting the energy production of different countries, or areas.

The processed dataset is available [here](/data/production_ts.csv).

## Training

To train the models launch the following command:

```
python train.py --model_type [choose_the_preferred_model] --dataset [dataset_dir] --train_window [xx] --BATCH_SIZE [xx] --N_EPOCHS [xx] --hidden_size [xx] --checkpoint_path [xx] --model_type [xx] --dataset [xx] --train_window [xx] --BATCH_SIZE [xx] --N_EPOCHS [xx] --hidden_size [xx] --checkpoint_dir [xx] --threshold_method [xx] --percentile [xx] --synth_gen [T/F] --contamination [xx] --period [xx] --anom_amplitude_factor [xx]
```

## Testing

To test the capability of the models in performing the anomaly detection task, an evaluation framework based on the synthetic generation of anomalous data points was set up.

The synthetic anomalies constitute the ground truth anomalies to which the abnormal points found by the models are going to be compared to.

To test the models launch the following command:

```
python test.py --model_type [choose_the_preferred_model] 
```