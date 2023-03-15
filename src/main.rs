extern crate csv;
extern crate gnuplot;
extern crate statrs;
extern crate ndarray;
extern crate rand;
extern crate tch;

use plotters::prelude::*;
use gnuplot::{AxesCommon, Figure};
use rand::prelude::*;
use statrs::distribution::{Normal, Univariate};
use tch::{nn, Device, Kind, Tensor};

use std::error::Error;
use std::fs::File;
use std::io;
use std::path::Path;

const SEQUENCE_LENGTH: usize = 12;
const EPOCHS: usize = 500;
const BATCH_SIZE: usize = 16;
const LEARNING_RATE: f64 = 0.01;

fn main() -> Result<(), Box<dyn Error>> {
    let data = load_data("Electric_Production.csv")?;

    let (train_data, test_data) = split_data(&data, 0.8);

    // LR
    let linear_model = train_linear_regression(&train_data)?;
    let linear_predictions = predict(&linear_model, &test_data);

    // RNN
    let rnn_model = train_rnn(&train_data)?;
    let rnn_predictions = predict_rnn(&rnn_model, &test_data);

    // plot
    plot_results(&test_data, &linear_predictions, &rnn_predictions);

    Ok(())
}

fn load_data<P: AsRef<Path>>(filename: P) -> Result<Vec<f64>, io::Error> {
    let mut data = Vec::new();
    let file = File::open(filename)?;
    let mut rdr = csv::Reader::from_reader(file);
    for result in rdr.records() {
        let record = result?;
        let value = record[1].parse::<f64>().unwrap();
        data.push(value);
    }
    Ok(data)
}

fn split_data(data: &[f64], split_ratio: f64) -> (Vec<f64>, Vec<f64>) {
    let split_index = (data.len() as f64 * split_ratio) as usize;
    let train_data = data[..split_index].to_vec();
    let test_data = data[split_index..].to_vec();
    (train_data, test_data)
}

fn train_linear_regression(train_data: &[f64]) -> Result<(f64, f64), Box<dyn Error>> {
    let n = train_data.len();
    let x_mean = (0..n).map(|i| i as f64).sum::<f64>() / n as f64;
    let y_mean = train_data.iter().sum::<f64>() / n as f64;

    let slope = train_data
        .iter()
        .enumerate()
        .map(|(i, y)| (i as f64 - x_mean) * (y - y_mean))
        .sum::<f64>()
        / train_data
            .iter()
            .enumerate()
            .map(|(i, _)| (i as f64 - x_mean).powi(2))
            .sum::<f64>();

    let intercept = y_mean - slope * x_mean;

    Ok((slope, intercept))
}

fn predict_linear_regression((slope, intercept): &(f64, f64), test_data: &[f64]) -> Vec<f64> {
    test_data.iter().enumerate().map(|(i, _)| slope * i as f64 + intercept).collect()
}

fn plot_results(test_data: &[f64], linear_preds: &[f64], rnn_preds: &[f64]) {
    let root = BitMapBackend::new("output.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let x: Vec<f64> = (0..test_data.len()).map(|i| i as f64).collect();

    let linear_data = linear_preds
        .iter()
        .enumerate()
        .map(|(i, &y)| (x[i], y))
        .collect::<Vec<_>>();

    let rnn_data = rnn_preds
        .iter()
        .enumerate()
        .map(|(i, &y)| (x[i], y))
        .collect::<Vec<_>>();

    let test_data = test_data
        .iter()
        .enumerate()
        .map(|(i, &y)| (x[i], y))
        .collect::<Vec<_>>();

    let mut chart = ChartBuilder::on(&root)
        .caption("Electric Production", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..test_data.len() as f64, 0f64..600f64)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(test_data, &RED))
        .unwrap()
        .label("Actual");

    chart
        .draw_series(LineSeries::new(linear_data, &BLUE))
        .unwrap()
        .label("Linear Regression");

    chart
        .draw_series(LineSeries::new(rnn_data, &GREEN))
        .unwrap()
        .label("RNN");

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
}

fn train_rnn(data: &[f64]) -> Result<nn::Path, Box<dyn Error>> {
    // vector de datos a matriz de secuencias
    let seq_len = 12;
    let num_seqs = data.len() - seq_len;
    let mut seqs = Vec::with_capacity(num_seqs);
    for i in 0..num_seqs {
        let seq = data[i..i + seq_len].to_vec();
        seqs.push(seq);
    }

    // conjunto entrenamiento y validación
    let num_train = (num_seqs as f64 * 0.8) as usize;
    let (train_seqs, val_seqs) = seqs.split_at(num_train);

    //tensores de Torch
    let train_x = Tensor::of_slice(train_seqs);
    let train_y = train_x.clone();
    let val_x = Tensor::of_slice(val_seqs);
    let val_y = val_x.clone();

    // def red neuronal
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let input_size = 1;
    let hidden_size = 64;
    let output_size = 1;
    let num_layers = 2;
    let dropout = 0.2;
    let rnn = nn::lstm(&vs.root(), input_size, hidden_size, num_layers, dropout);
    let linear = nn::linear(&vs.root(), hidden_size, output_size, Default::default());

    // función de pérdida y el optimizador
    let loss_fn = nn::MSELoss::new(VarStore::new(Device::cuda_if_available()));
    let mut optimizer = nn::Adam::default().build(&vs, 1e-3)?;

    // training NN
    let batch_size = 64;
    let num_epochs = 100;
    for epoch in 0..num_epochs {
        let mut train_loss = 0.0;
        let mut val_loss = 0.0;
        let mut train_batches = 0;
        let mut val_batches = 0;
        for batch in train_x.chunk(batch_size) {
            optimizer.zero_grad();
            let output = batch.apply_t(&rnn).apply(&linear);
            let target = batch.clone();
            let loss = loss_fn.forward(&output, &target);
            loss.backward();
            optimizer.step();
            train_loss += f64::from(loss);
            train_batches += 1;
        }
        for batch in val_x.chunk(batch_size) {
            let output = batch.apply_t(&rnn).apply(&linear);
            let target = batch.clone();
            let loss = loss_fn.forward(&output, &target);
            val_loss += f64::from(loss);
            val_batches += 1;
        }
        println!("Epoch {} - Train Loss: {:.4}, Val Loss: {:.4}", epoch + 1, train_loss / train_batches as f64, val_loss / val_batches as f64);
    }

    // red neuronal entrenada
    Ok(vs.root())
}