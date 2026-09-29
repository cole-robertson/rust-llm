//! Port of `lib/ruby_llm/mcp/collection.rb`: the MCP servers connected to a chat, readable by
//! name (`chat.mcp().get("linear")`).

use super::Mcp;

#[derive(Clone, Default)]
pub struct Collection {
    servers: Vec<Mcp>,
}

impl Collection {
    /// `<<`: a server with the same name replaces the earlier one.
    pub(crate) fn push(&mut self, server: Mcp) {
        let name = server.name();
        self.servers.retain(|s| s.name() != name);
        self.servers.push(server);
    }

    /// `[name]`.
    pub fn get(&self, name: &str) -> Option<&Mcp> {
        self.servers.iter().find(|s| s.name() == name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Mcp> {
        self.servers.iter()
    }

    /// `empty?`.
    pub fn is_empty(&self) -> bool {
        self.servers.is_empty()
    }

    pub fn len(&self) -> usize {
        self.servers.len()
    }
}

impl std::fmt::Debug for Collection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.servers.iter().map(|s| s.name()))
            .finish()
    }
}
