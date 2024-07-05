/* Calculate the Silhouette score for Generate a dataset of clusters based on the input configuration.

Example run:
cargo run -- ~/data/algohertz/clustering/datasets/clusters_100_5.parquet

Copyright © 2024 AlgoHertz. License: MIT. */

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::fs::File;
use std::path::Path;
use ndarray::Array2;
use std::collections::HashMap;

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

fn main() {
    let path = std::env::args().nth(1).expect("Please provide a Parquet file path.");
    let file = File::open(&Path::new(&path)).expect("Failed to open Parquet file");
    let reader = SerializedFileReader::new(file).expect("Failed to create Parquet reader");
    let iter = reader.get_row_iter(None).expect("Failed to get row iterator");

    let mut points = Vec::new();
    let mut clusters = Vec::new();
    dbg!("Processing the rows...");
    for row_result in iter {
        let row = row_result.unwrap();
        let cluster = match row.get_long(0) {
            Ok(cluster) => cluster as i32,
            Err(_) => row.get_int(0).unwrap(),
        };
        let mut point = Vec::new();
        for i in 1..row.len() {
            point.push(row.get_double(i).unwrap());
        }
        clusters.push(cluster);
        points.push(point);
    }

    let n = points.len();
    let d = points[0].len();

    let data = Array2::from_shape_vec((n, d), points.into_iter().flatten().collect()).unwrap();
    let clusters = Array2::from_shape_vec((n, 1), clusters).unwrap();
    dbg!("Calculating the Silhouette score...");
    let silhouette_score = calculate_silhouette_score(&data, &clusters);
    println!("Silhouette Score: {}", silhouette_score);
}

fn calculate_silhouette_score(data: &Array2<f64>, clusters: &Array2<i32>) -> f64 {
    let n = data.shape()[0];

    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut s = vec![0.0; n];

    let mut cluster_map: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, &cluster) in clusters.column(0).iter().enumerate() {
        cluster_map.entry(cluster).or_insert(Vec::new()).push(i);
    }

    data.axis_chunks_iter(ndarray::Axis(0), 1).enumerate().for_each(|(i, point)| {
        let cluster = clusters[(i, 0)];
        let point = point.row(0).to_vec();
        
        let same_cluster_points: Vec<_> = cluster_map[&cluster].iter().filter(|&&index| index != i).collect();
        let other_cluster_points: Vec<_> = cluster_map.iter().filter(|(&c, _)| c != cluster).flat_map(|(_, indices)| indices).collect();

        if !same_cluster_points.is_empty() {
            a[i] = same_cluster_points.iter().map(|&&index| euclidean_distance(&point, &data.row(index).to_vec())).sum::<f64>() / same_cluster_points.len() as f64;
        }

        if !other_cluster_points.is_empty() {
            b[i] = other_cluster_points.iter().map(|&&index| euclidean_distance(&point, &data.row(index).to_vec())).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        }

        s[i] = (b[i] - a[i]) / a[i].max(b[i]);
    });

    s.iter().sum::<f64>() / n as f64
}

