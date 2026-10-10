//! Port of `lib/ruby_llm/support/process_cache.rb`.

use std::sync::Mutex;

/// `Support::ProcessCache`: a small least-recently-used cache shared by every caller in the
/// process. Values are built outside the lock, so a slow build never stalls other threads; when
/// two callers race to build the same key, the first value stored wins and both get it.
///
/// Ruby also forgets every entry in a forked child, whose inherited sockets belong to the parent.
/// The port has no fork; the connection cache keys its entries by tokio runtime instead (see
/// `transport::Connection`), since pooled connections belong to the runtime that opened them.
pub struct ProcessCache<K, V> {
    limit: usize,
    /// Least recently used first.
    entries: Mutex<Vec<(K, V)>>,
}

impl<K: PartialEq, V: Clone> ProcessCache<K, V> {
    /// `ProcessCache::LIMIT`.
    pub const LIMIT: usize = 64;

    pub const fn new(limit: usize) -> Self {
        ProcessCache {
            limit,
            entries: Mutex::new(Vec::new()),
        }
    }

    /// `#fetch(key) { build }`: the stored value, or `build`'s, stored for the next caller.
    pub fn fetch(&self, key: K, build: impl FnOnce() -> V) -> V {
        match self.try_fetch(key, || Ok::<V, std::convert::Infallible>(build())) {
            Ok(value) => value,
            Err(never) => match never {},
        }
    }

    /// `fetch` with a build that can fail; a failed build stores nothing.
    pub fn try_fetch<E>(&self, key: K, build: impl FnOnce() -> Result<V, E>) -> Result<V, E> {
        if let Some(value) = Self::touch(&mut self.lock(), &key) {
            return Ok(value);
        }
        let value = build()?;
        let mut entries = self.lock();
        if let Some(first) = Self::touch(&mut entries, &key) {
            return Ok(first);
        }
        entries.push((key, value.clone()));
        let excess = entries.len().saturating_sub(self.limit);
        entries.drain(..excess);
        Ok(value)
    }

    /// `#clear`.
    pub fn clear(&self) {
        self.lock().clear();
    }

    /// `#delete(key)`.
    pub fn delete(&self, key: &K) {
        self.lock().retain(|(k, _)| k != key);
    }

    /// Moves `key` to the most recently used end and returns its value.
    fn touch(entries: &mut Vec<(K, V)>, key: &K) -> Option<V> {
        let index = entries.iter().position(|(k, _)| k == key)?;
        let entry = entries.remove(index);
        let value = entry.1.clone();
        entries.push(entry);
        Some(value)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(K, V)>> {
        // A panicking build runs outside the lock, so a poisoned lock still holds whole entries.
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl<K: PartialEq, V: Clone> Default for ProcessCache<K, V> {
    fn default() -> Self {
        Self::new(Self::LIMIT)
    }
}
