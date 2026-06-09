use clap::Parser;
use embedding::cli::{Cli, run};

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    
    run(Cli::parse());
}