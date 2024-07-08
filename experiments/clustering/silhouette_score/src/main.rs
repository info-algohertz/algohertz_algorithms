/* Calculate the Silhouette score for Generate a dataset of clusters based on the input configuration.

Example run:
cargo run -- ~/data/algohertz/clustering/datasets/clusters_100_5.parquet

Copyright © 2024 AlgoHertz. License: MIT. */

use polars::prelude::*;
use std::io::Result;
use std::fs::File;
use std::collections::HashMap;

fn compute_mean(values: &Vec<f64>) -> f64 {
    let mut sum: f64 = 0.0;
    for v in values {
        sum += v;
    }
    let mean: f64 = sum/values.len() as f64;
    return mean;
}

fn euclidean_metric(point_a: &Vec<f64>, point_b: &Vec<f64>) -> f64 {
    if point_a.len() != point_b.len() {
        panic!("Points must have the same dimension");
    }

    let sum_of_squares: f64 = point_a.iter()
        .zip(point_b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    sum_of_squares.sqrt()
}

fn compute_mean_distance(points: &Vec<Vec<f64>>, point: &Vec<f64>, dist_metric: fn(&Vec<f64>, &Vec<f64>) -> f64) -> f64 {
    let mut distances: Vec<f64> = vec![];
    for point2 in points {
        let d = dist_metric(point, point2);
        distances.push(d);
    }
    return compute_mean(&distances);
}

fn compute_silhouette_score(dim_count: &usize, clusters: &HashMap<i64, Vec<Vec<f64>>>) -> f64 {
    let mut scores: Vec<f64> = vec![];
    let cluster_numbers = clusters.keys();
    for cluster_number in cluster_numbers.clone() {
        let points = clusters.get(&cluster_number).unwrap();
        let ratio = points.len() as f64/(points.len() as f64 - 1.0);
        for point in points {
            let d = compute_mean_distance(&points, &point, euclidean_metric);
            let a = ratio*d;
            
            //let distances: Vec<f64> = vec![];
            let mut b = f64::INFINITY;
            for cluster_number2 in cluster_numbers.clone() {
                if cluster_number == cluster_number2 {
                    continue;
                }
                let points2 = clusters.get(&cluster_number2).unwrap();
                let d = compute_mean_distance(&points2, &point, euclidean_metric);
                if d < b {
                    b = d;
                }
            }
            let s = (b - a)/a.max(b);
            scores.push(s);
        }
    }
    let score = compute_mean(&scores);
    return score;
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("Please provide a Parquet file path to the cluster dataset.");
    
    let df = ParquetReader::new(File::open(path)?)
        .finish()
        .expect("Failed to read Parquet file");    
    let dim_count = df.shape().1 - 1;
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
        }.unwrap();
        let raw_point = result[1..].to_vec();
        let mut point: Vec<f64> = vec![];
        for value in raw_point {
            let v = match value {
                AnyValue::Float64(val) => Some(val),
                _ => None
            }.unwrap();
            point.push(v);
        }
        clusters.get_mut(&cluster).unwrap().push(point);
    }
    

    let silhouette_score = compute_silhouette_score(&dim_count, &clusters);
    println!("{:?}", silhouette_score);
    
    Ok(())
}

