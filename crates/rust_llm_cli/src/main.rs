//! `rust-llm`: RubyLLM's Rails generators for Loco + Inertia + React apps.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cwd = match std::env::current_dir() {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("cannot read the current directory: {e}");
            std::process::exit(1);
        }
    };
    std::process::exit(rust_llm_cli::run(&args, &cwd));
}
