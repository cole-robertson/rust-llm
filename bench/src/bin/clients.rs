//! Diagnostic: RSS cost of N `reqwest::Client`s (rust_llm builds one per `Chat`), without any requests.
fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)?.parse().ok()))
        .unwrap_or(0)
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|n| n.parse().ok()).unwrap_or(1000);
    let before = rss_kib();
    let clients: Vec<_> = (0..n).map(|_| reqwest::Client::builder().build().expect("client")).collect();
    let after = rss_kib();
    println!("{{\"clients\":{n},\"kib_per_client\":{:.1}}}", (after - before) as f64 / n as f64);
    drop(clients);
}
