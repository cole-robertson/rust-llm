//! Ports of RubyLLM 2.1's `support/process_cache_spec.rb` against `rust_llm::support::ProcessCache`.
//! Ruby compares values with `be` (object identity); here each value is an `Arc` compared with
//! `Arc::ptr_eq`. Fibers are tokio tasks.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use rust_llm::support::ProcessCache;

type Value = Arc<()>;

struct Subject {
    cache: ProcessCache<&'static str, Value>,
    builds: Mutex<Vec<&'static str>>,
}

impl Subject {
    fn new() -> Self {
        Subject {
            cache: ProcessCache::new(2),
            builds: Mutex::new(Vec::new()),
        }
    }

    fn fetch(&self, key: &'static str) -> Value {
        self.cache.fetch(key, || {
            self.builds.lock().unwrap().push(key);
            Arc::new(())
        })
    }

    fn builds(&self) -> Vec<&'static str> {
        self.builds.lock().unwrap().clone()
    }
}

// spec: support/process_cache_spec.rb:17 builds one value per key and hands it out again
#[test]
fn builds_one_value_per_key_and_hands_it_out_again() {
    let cache = Subject::new();
    let first = cache.fetch("openai");
    assert!(Arc::ptr_eq(&cache.fetch("openai"), &first));
    assert!(!Arc::ptr_eq(&cache.fetch("anthropic"), &first));
    assert_eq!(cache.builds(), ["openai", "anthropic"]);
}

// spec: support/process_cache_spec.rb:25 forgets the least recently used value beyond its limit
#[test]
fn forgets_the_least_recently_used_value_beyond_its_limit() {
    let cache = Subject::new();
    cache.fetch("openai");
    cache.fetch("anthropic");
    cache.fetch("openai");
    cache.fetch("gemini");

    cache.fetch("openai");
    cache.fetch("anthropic");

    assert_eq!(
        cache.builds(),
        ["openai", "anthropic", "gemini", "anthropic"]
    );
}

// spec: support/process_cache_spec.rb:37 forgets every value when cleared
#[test]
fn forgets_every_value_when_cleared() {
    let cache = Subject::new();
    let first = cache.fetch("openai");
    cache.cache.clear();
    assert!(!Arc::ptr_eq(&cache.fetch("openai"), &first));
}

// spec: support/process_cache_spec.rb:44 forgets one value when deleted
#[test]
fn forgets_one_value_when_deleted() {
    let cache = Subject::new();
    let openai = cache.fetch("openai");
    let anthropic = cache.fetch("anthropic");
    cache.cache.delete(&"openai");
    assert!(!Arc::ptr_eq(&cache.fetch("openai"), &openai));
    assert!(Arc::ptr_eq(&cache.fetch("anthropic"), &anthropic));
}

// spec: support/process_cache_spec.rb:53 hands every caller that races to build a value the first one stored
#[test]
fn hands_every_caller_that_races_to_build_a_value_the_first_one_stored() {
    let cache = Arc::new(Subject::new());
    let values: Vec<Value> = (0..8)
        .map(|_| {
            let cache = cache.clone();
            std::thread::spawn(move || {
                cache.cache.fetch("openai", || {
                    std::thread::sleep(Duration::from_millis(50));
                    Arc::new(())
                })
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect();
    assert!(values.iter().all(|v| Arc::ptr_eq(v, &values[0])));
    assert!(Arc::ptr_eq(&cache.fetch("openai"), &values[0]));
}

// spec: support/process_cache_spec.rb:67 lets other threads through while a value is being built
#[test]
fn lets_other_threads_through_while_a_value_is_being_built() {
    let cache = Arc::new(Subject::new());
    let order = Arc::new(Mutex::new(Vec::new()));
    let (building_tx, building_rx) = std::sync::mpsc::channel();
    let slow = {
        let (cache, order) = (cache.clone(), order.clone());
        std::thread::spawn(move || {
            cache.cache.fetch("slow", || {
                building_tx.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(200));
                order.lock().unwrap().push("slow_built");
                Arc::new(())
            })
        })
    };
    building_rx.recv().unwrap();
    cache.fetch("fast");
    order.lock().unwrap().push("fast_returned");
    slow.join().unwrap();
    assert_eq!(*order.lock().unwrap(), ["fast_returned", "slow_built"]);
}

// spec: support/process_cache_spec.rb:86 lets other fibers through while a value is being built
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lets_other_tasks_through_while_a_value_is_being_built() {
    let cache = Arc::new(Subject::new());
    let order = Arc::new(Mutex::new(Vec::new()));
    let slow = {
        let (cache, order) = (cache.clone(), order.clone());
        tokio::spawn(async move {
            cache.cache.fetch("slow", || {
                std::thread::sleep(Duration::from_millis(50));
                order.lock().unwrap().push("slow_built");
                Arc::new(())
            });
        })
    };
    tokio::time::sleep(Duration::from_millis(10)).await;
    let fast = cache.clone();
    tokio::spawn(async move { fast.fetch("fast") })
        .await
        .unwrap();
    order.lock().unwrap().push("fast_returned");
    slow.await.unwrap();
    assert_eq!(*order.lock().unwrap(), ["fast_returned", "slow_built"]);
}
