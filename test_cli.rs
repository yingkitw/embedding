use std::process::Command;

fn main() {
    println!("=== Embedding Trainer Test Script ===");
    
    // Test 1: Basic compilation
    println!("\n1. Testing basic compilation...");
    let output = Command::new("cargo")
        .args(&["check"])
        .output()
        .expect("Failed to execute cargo check");
    
    if output.status.success() {
        println!("✓ Compilation successful");
    } else {
        println!("✗ Compilation failed:");
        println!("{}", String::from_utf8_lossy(&output.stderr));
        return;
    }
    
    // Test 2: Run tests
    println!("\n2. Running tests...");
    let output = Command::new("cargo")
        .args(&["test"])
        .output()
        .expect("Failed to run tests");
    
    if output.status.success() {
        println!("✓ Tests passed");
    } else {
        println!("✗ Some tests failed:");
        println!("{}", String::from_utf8_lossy(&output.stderr));
    }
    
    // Test 3: Build CLI tool
    println!("\n3. Building CLI tool...");
    let output = Command::new("cargo")
        .args(&["build", "--release"])
        .output()
        .expect("Failed to build CLI tool");
    
    if output.status.success() {
        println!("✓ CLI tool built successfully");
    } else {
        println!("✗ Failed to build CLI tool:");
        println!("{}", String::from_utf8_lossy(&output.stderr));
        return;
    }
    
    // Test 4: Check if CLI tool exists
    let cli_path = std::path::Path::new("target/release/embedding-train");
    if cli_path.exists() {
        println!("✓ CLI tool found at: {}", cli_path.display());
        
        // Test 5: Show help
        println!("\n4. Testing CLI help...");
        let output = Command::new(cli_path)
            .args(&["--help"])
            .output()
            .expect("Failed to get CLI help");
        
        if output.status.success() {
            println!("✓ CLI help works");
            println!("CLI Help (first few lines):");
            let help_text = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = help_text.lines().collect();
            for line in lines.iter().take(10) {
                println!("  {}", line);
            }
            if lines.len() > 10 {
                println!("  ... (truncated)");
            }
        } else {
            println!("✗ CLI help failed:");
            println!("{}", String::from_utf8_lossy(&output.stderr));
        }
    } else {
        println!("✗ CLI tool not found at: {}", cli_path.display());
    }
    
    // Test 6: Check example data
    println!("\n5. Checking example data...");
    let example_data = std::path::Path::new("example_data.txt");
    if example_data.exists() {
        println!("✓ Example data file found");
        if let Ok(content) = std::fs::read_to_string(example_data) {
            let lines: Vec<&str> = content.lines().collect();
            println!("  Example data contains {} sentences", lines.len());
            for (i, line) in lines.iter().take(3).enumerate() {
                println!("    {}. {}", i + 1, line);
            }
        }
    } else {
        println!("✗ Example data file not found");
    }
    
    println!("\n=== Test Script Complete ===");
    println!("To test the full functionality, run:");
    println!("  cargo run -- train --input example_data.txt --output model.json --embeddings embeddings.txt --dim 100 --epochs 5");
}