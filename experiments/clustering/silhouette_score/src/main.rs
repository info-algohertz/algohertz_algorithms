/* Calculate the Silhouette score for a dataset of clusters specified by the input path.

Example run:
cargo run -- ~/data/algohertz/clustering/datasets/clusters_100_5.parquet

References:
https://en.wikipedia.org/wiki/Silhouette_(clustering)

Copyright © 2024 AlgoHertz. License: MIT. */

use polars::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Result;

fn compute_centroid(points: &Vec<Vec<f64>>) -> Vec<f64> {
    if points.len() == 0 {
        panic!("Cannot compute a centroid of an empty cluster.");
    }
    let mut centroid = vec![];
    for i in 0..points.get(0).unwrap().len() {
        let median = compute_median(&points[i]);
        centroid.push(median);        
    }
    return centroid;
}

fn compute_median(values: &Vec<f64>) -> f64 {
    let mut sorted_values = values.clone();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let len = sorted_values.len();
    if len == 0 {
        panic!("Cannot compute median of an empty vector");
    }

    if len % 2 == 1 {
        sorted_values[len / 2]
    } else {
        let mid1 = sorted_values[len / 2 - 1];
        let mid2 = sorted_values[len / 2];
        (mid1 + mid2) / 2.0
    }
}

fn compute_mean(values: &Vec<f64>) -> f64 {
    let mut sum: f64 = 0.0;
    for v in values {
        sum += v;
    }
    let mean: f64 = sum / values.len() as f64;
    return mean;
}

fn euclidean_metric(point_a: &Vec<f64>, point_b: &Vec<f64>) -> f64 {
    if point_a.len() != point_b.len() {
        panic!("Points must have the same dimension");
    }

    let sum_of_squares: f64 = point_a
        .iter()
        .zip(point_b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    sum_of_squares.sqrt()
}

fn compute_silhouette_score(
    clusters: &HashMap<i64, Vec<Vec<f64>>>,
    dist_metric: fn(&Vec<f64>, &Vec<f64>) -> f64,
) -> f64 {
    let mut scores: Vec<f64> = vec![];
    let cluster_numbers = clusters.keys();
    let mut centroids: HashMap<i64, Vec<f64>> = HashMap::with_capacity(cluster_numbers.len());
    for cluster_number in cluster_numbers.clone() {
        centroids.insert(*cluster_number, compute_centroid(clusters.get(&cluster_number).unwrap()));
    }
    
    for cluster_number in cluster_numbers.clone() {
        let points = clusters.get(&cluster_number).unwrap();
        let ratio = points.len() as f64 / (points.len() as f64 - 1.0);
        let cluster_centroid = centroids.get(cluster_number).unwrap();
        for point in points {
            let d = dist_metric(&point, &cluster_centroid);
            let a = ratio * d;

            let mut b = f64::INFINITY;
            for cluster_number2 in cluster_numbers.clone() {
                if cluster_number == cluster_number2 {
                    continue;
                }
                let cluster_centroid2 = centroids.get(cluster_number2).unwrap();
                let d = dist_metric(&point, &cluster_centroid2);
                if d < b {
                    b = d;
                }
            }
            let s = (b - a) / a.max(b);
            scores.push(s);
        }
    }
    let score = compute_mean(&scores);
    return score;
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("Please provide a Parquet file path to the cluster dataset.");

    let df = ParquetReader::new(File::open(path)?)
        .finish()
        .expect("Failed to read Parquet file");
    assert!(df.get_column_names()[0] == "cluster");

    let binding = df.column("cluster").unwrap().unique().unwrap();
    let mut cluster_numbers: Vec<i64> = vec![];
    for some_cluster in binding.i64().unwrap() {
        let cluster = some_cluster.unwrap();
        cluster_numbers.push(cluster);
    }

    let mut clusters: HashMap<i64, Vec<Vec<f64>>> = HashMap::with_capacity(cluster_numbers.len());
    for cluster_number in cluster_numbers {
        clusters.insert(cluster_number, Vec::new());
    }

    for idx in 0..df.height() {
        let row = df.get_row(idx);
        let result = &row.unwrap().0;
        let cluster = match result[0] {
            AnyValue::Int64(val) => Some(val),
            AnyValue::UInt32(val) => Some(val as i64),
            _ => None,
        }
        .unwrap();
        let raw_point = result[1..].to_vec();
        let mut point: Vec<f64> = vec![];
        for value in raw_point {
            let v = match value {
                AnyValue::Float64(val) => Some(val),
                _ => None,
            }
            .unwrap();
            point.push(v);
        }
        clusters.get_mut(&cluster).unwrap().push(point);
    }

    let silhouette_score = compute_silhouette_score(&clusters, euclidean_metric);
    println!("{:?}", silhouette_score);

    Ok(())
}
