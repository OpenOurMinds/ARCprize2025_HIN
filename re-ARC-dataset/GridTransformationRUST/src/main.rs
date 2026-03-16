use anyhow::Result;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use ndarray::{s, Array2, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

// Type alias for a grid, which is a 2D vector of unsigned 8-bit integers.
// This is the type we will serialize to JSON.
type Grid = Vec<Vec<u8>>;

//=========================================================
// 1. DATA STRUCTURES FOR JSON PARSING
//=========================================================

/// Represents a single input-output pair from a task.
#[derive(Debug, Deserialize, Serialize, Clone)]
struct TaskPair {
    input: Grid,
    output: Grid,
}

/// Represents a full task with training and testing pairs.
/// The original Python script only processes the first `train` pair.
#[derive(Debug, Deserialize)]
struct FullTask {
    train: Vec<TaskPair>,
}

/// An enum to handle the different possible JSON structures in the dataset.
/// `serde(untagged)` allows it to try deserializing into each variant.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TaskFileContent {
    Full(FullTask),
    PairList(Vec<TaskPair>),
    SinglePair(TaskPair),
}

impl TaskFileContent {
    /// Extracts a list of task pairs to be processed, regardless of the original file structure.
    fn into_task_pairs(self) -> Vec<TaskPair> {
        match self {
            // If the file has `train`, take the first pair from the training examples.
            TaskFileContent::Full(task) => {
                if task.train.is_empty() {
                    vec![]
                } else {
                    vec![task.train[0].clone()]
                }
            }
            TaskFileContent::PairList(pairs) => {
                if pairs.is_empty() {
                    vec![]
                } else {
                    vec![pairs[0].clone()]
                }
            }
            TaskFileContent::SinglePair(pair) => vec![pair],
        }
    }
}

//=========================================================
// 2. GRID TRANSFORMATION FUNCTIONS
//=========================================================

/// Converts an ndarray::Array2<u8> into a Grid (Vec<Vec<u8>>).
/// This is necessary for serialization with serde_json.
fn array_to_grid(arr: Array2<u8>) -> Grid {
    arr.outer_iter().map(|row| row.to_vec()).collect()
}

/// Inverts the colors of a grid (0 becomes 9, 1 becomes 8, etc.).
/// This function now returns a `Grid` type.
fn invert_grid(grid: &ArrayView2<u8>) -> Grid {
    let inverted = 9 - grid;
    array_to_grid(inverted.to_owned())
}

/// Rotates a grid by a given angle (90, 180, or 270 degrees).
/// This function now returns a `Grid` type.
fn get_rotated_grid(grid: &ArrayView2<u8>, angle: u32) -> Grid {
    let rotated = match angle {
        90 => grid.reversed_axes().t().to_owned(),
        180 => grid.slice(s![..;-1, ..;-1]).to_owned(),
        270 => grid.reversed_axes().slice(s![..;-1, ..]).to_owned(),
        _ => grid.to_owned(),
    };
    array_to_grid(rotated)
}

//=========================================================
// 3. FILE PROCESSING LOGIC
//=========================================================

fn process_and_save_file(file_path: &Path, output_dir_path: &Path) -> Result<()> {
    let file = File::open(file_path)?;
    let content: TaskFileContent = serde_json::from_reader(file)?;
    let tasks_to_process = content.into_task_pairs();

    // Only process if there's at least one valid task pair.
    if let Some(task) = tasks_to_process.get(0) {
        let input_grid_arr =
            Array2::from_shape_vec((task.input.len(), task.input[0].len()), task.input.clone().into_iter().flatten().collect())?;
        let output_grid_arr =
            Array2::from_shape_vec((task.output.len(), task.output[0].len()), task.output.clone().into_iter().flatten().collect())?;

        // Use a boxed trait object to create a uniform type for all closures.
        let transformations: Vec<(&str, Box<dyn Fn(&ArrayView2<u8>) -> Grid>)> = vec![
            ("original", Box::new(|g: &ArrayView2<u8>| array_to_grid(g.to_owned()))),
            ("inverted", Box::new(|g| invert_grid(g))),
            ("rotated_90", Box::new(|g| get_rotated_grid(g, 90))),
            ("rotated_180", Box::new(|g| get_rotated_grid(g, 180))),
            ("rotated_270", Box::new(|g| get_rotated_grid(g, 270))),
        ];

        let mut transformed_grids = HashMap::new();
        for (name, func) in &transformations {
            transformed_grids.insert(name.to_string(), func(&input_grid_arr.view()));
        }

        let mut transformed_output_grids = HashMap::new();
        for (name, func) in &transformations {
            transformed_output_grids.insert(name.to_string(), func(&output_grid_arr.view()));
        }

        let combined_data = serde_json::json!({
            "input_transformations": transformed_grids,
            "output_transformations": transformed_output_grids,
        });

        // Save the transformed data.
        let file_stem = file_path.file_stem().ok_or_else(|| anyhow::anyhow!("Invalid file name"))?;
        let output_file_path = output_dir_path.join(format!("{}.json", file_stem.to_string_lossy()));
        let output_file = File::create(&output_file_path)?;
        let writer = BufWriter::new(output_file);
        serde_json::to_writer_pretty(writer, &combined_data)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    // IMPORTANT: Adjust this path to the correct location of your JSON files.
    let base_data_path = Path::new("/Users/seungwonlee/ARCprize2025_HIN/re-ARC-dataset/tasks");
    if !base_data_path.exists() {
        println!("Error: The provided path does not exist. Please update the path in main.rs.");
        return Ok(());
    }

    let path_pattern = base_data_path.join("*.json").to_str().unwrap().to_string();
    let challenge_files: Vec<PathBuf> = glob::glob(&path_pattern)?.filter_map(Result::ok).collect();
    let output_dir_path = base_data_path.join("GridTransitionDataset_Rust");

    if challenge_files.is_empty() {
        println!("No JSON files found at '{}'. Please check the path.", path_pattern);
        return Ok(());
    }
    
    // Ensure the output directory exists
    fs::create_dir_all(&output_dir_path)?;

    println!("Processing {} files...", challenge_files.len());
    
    // Setup progress bar
    let pb = ProgressBar::new(challenge_files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
        .progress_chars("#>-"));
    
    // Process files in parallel using Rayon
    challenge_files
        .par_iter()
        .progress_with(pb)
        .for_each(|file_path| {
            if let Err(e) = process_and_save_file(file_path, &output_dir_path) {
                eprintln!("Failed to process file {:?}: {}", file_path, e);
            }
        });
        
    println!("\nFinished processing files. Transformed data saved to {:?}", output_dir_path);
    Ok(())
}
