//! Shared by the generator tests: a minimal Loco + Inertia app skeleton.

use std::fs;

/// The parts of the starter kit the generators read or inject into, verbatim where it matters
/// (anchors: `inject-above`, `pub struct Migrator`, `AppRoutes::empty()`, `fn connect_workers`,
/// `fn initializers`, `// scaffold:paths`, `// scaffold:routes`, `// scaffold:nav`).
pub fn app() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let files: &[(&str, &str)] = &[
        (
            "Cargo.toml",
            "[package]\nname = \"app\"\n\n[dependencies]\nloco-rs = { workspace = true }\nmigration = { path = \"migration\" }\n\n[dev-dependencies]\nrstest = \"0.25\"\n",
        ),
        (
            "migration/Cargo.toml",
            "[package]\nname = \"migration\"\n\n[dependencies]\nloco-rs = { workspace = true }\n\n[dependencies.sea-orm-migration]\nversion = \"2.0\"\n",
        ),
        (
            "migration/src/lib.rs",
            "pub use sea_orm_migration::prelude::*;\nmod m20220101_000001_users;\n\npub struct Migrator;\n\nimpl MigratorTrait for Migrator {\n    fn migrations() -> Vec<Box<dyn MigrationTrait>> {\n        vec![\n            Box::new(m20220101_000001_users::Migration),\n            // inject-above (do not remove this comment)\n        ]\n    }\n}\n",
        ),
        (
            "src/lib.rs",
            "pub mod app;\npub mod controllers;\npub mod initializers;\npub mod models;\npub mod route_table;\npub mod workers;\n",
        ),
        (
            "src/app.rs",
            "impl Hooks for App {\n    async fn initializers(_ctx: &AppContext) -> Result<Vec<Box<dyn loco_rs::app::Initializer>>> {\n        Ok(vec![Box::new(crate::inertia::ssr::SsrSupervisor)])\n    }\n    fn routes(ctx: &AppContext) -> AppRoutes {\n        let routes = AppRoutes::empty()\n            .add_route(controllers::home::routes());\n        routes\n    }\n    async fn connect_workers(ctx: &AppContext, queue: &Queue) -> Result<()> {\n        Ok(())\n    }\n}\n",
        ),
        (
            "src/models/mod.rs",
            "pub mod _entities;\n#[cfg(feature = \"bench\")]\npub mod bench_events;\npub mod sessions;\npub mod users;\n",
        ),
        (
            "src/controllers/mod.rs",
            "use std::sync::Arc;\n\n#[cfg(feature = \"bench\")]\npub mod bench;\npub mod dashboard;\npub mod users;\n\npub fn settings() {}\n",
        ),
        ("src/workers/mod.rs", "//! Background workers.\n"),
        ("src/initializers/mod.rs", "\n"),
        (
            "src/route_table.rs",
            "pub const ROOT: &str = \"/\";\n// scaffold:paths (above this line)\n\npub const ROUTES: &[RouteDef] = {\n    &[\n        route(\"home.index\", Get, ROOT, None),\n        // scaffold:routes (above this line)\n    ]\n};\n",
        ),
        (
            "frontend/components/app-sidebar.tsx",
            "import { Link } from \"@inertiajs/react\"\nimport { BookOpen, Folder, LayoutGrid } from \"lucide-react\"\n\nimport { dashboard } from \"@/routes\"\n\nconst mainNavItems: NavItem[] = [\n  {\n    title: \"Dashboard\",\n  },\n  // scaffold:nav\n]\n",
        ),
    ];
    for (path, content) in files {
        let path = dir.path().join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }
    dir
}
