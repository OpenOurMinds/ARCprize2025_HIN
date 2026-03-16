use anyhow::{Context, Result};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use uuid::Uuid;

// Data structures for JSON serialization/deserialization.
// This handles the flexible nature of the input JSON, where 'task_id' might exist or not.
#[derive(Debug, Deserialize, Serialize)]
struct Task {
    task_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    original_task_id: Option<String>,
    #[serde(flatten)]
    rest: serde_json::Value,
}

fn process_and_save_file(file_path: &Path, output_dir_path: &Path) -> Result<()> {
    // Determine the transformation method from the parent directory name.
    let conversion_method = file_path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .context("Could not determine conversion method from parent directory name.")?;

    // Read and parse the JSON file.
    let file_content = fs::read_to_string(file_path)?;
    let mut task: serde_json::Value = serde_json::from_str(&file_content)?;

    // Generate a new, unique UUID for the task ID.
    let new_task_id = Uuid::new_v4().to_string();

    // Extract the original task ID for traceability.
    let original_task_id = file_path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string());

    // Update the task JSON with the new IDs.
    if task.get("task_id").is_some() {
        task["original_task_id"] = task.get("task_id").cloned().unwrap_or_default();
    } else if let Some(orig_id) = &original_task_id {
        task["original_task_id"] = serde_json::Value::String(orig_id.to_string());
    }
    task["task_id"] = serde_json::Value::String(new_task_id.clone());

    // Define the new subdirectory and file path for the output.
    let new_subdir_path = output_dir_path.join(conversion_method);
    fs::create_dir_all(&new_subdir_path)?;
    let new_file_path = new_subdir_path.join(format!("{}.json", new_task_id));

    // Save the new JSON file.
    let file = File::create(&new_file_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &task)?;

    Ok(())
}

fn main() -> Result<()> {
    // UPDATED: This is the directory where your JSON files are located.
    let input_directory = Path::new("/Users/seungwonlee/ARCprize2025_HIN/re-ARC-dataset/tasks/GridTransitionDataset_Rust");
    
    // This is the new directory where the corrected and sorted files will be saved.
    let output_directory = input_directory.join("transformed_output");

    println!("Starting process to correct and sort task IDs...");

    // Get a list of all JSON files in the input directory, searching recursively.
    let file_paths: Vec<PathBuf> = glob::glob(input_directory.join("**/*.json").to_str().unwrap())?
        .filter_map(Result::ok)
        .collect();

    if file_paths.is_empty() {
        println!("No JSON files found in the input directory: {:?}", input_directory);
        return Ok(());
    }

    println!("Found {} files to process.", file_paths.len());

    // Process files in parallel using Rayon and a progress bar.
    let pb = ProgressBar::new(file_paths.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    file_paths
        .par_iter()
        .progress_with(pb)
        .for_each(|file_path| {
            if let Err(e) = process_and_save_file(file_path, &output_directory) {
                eprintln!("\nError processing file {:?}: {:?}", file_path, e);
            }
        });

    println!("\nProcess complete. All augmented files now have unique task IDs and are sorted by transformation method.");

    Ok(())
}
