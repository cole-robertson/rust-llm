//! SeaORM entities for the tables `migrations` creates: `RubyLLM::ActiveRecord::Model`,
//! `ToolCall`, `Usage`, and the app's `Chat`/`Message` (`acts_as_chat`/`acts_as_message`), plus
//! `rust_llm_attachments`, which stands in for Active Storage.

pub mod chats {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "chats")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        pub rust_llm_model_id: i32,
        pub cancelled: bool,
        pub created_at: DateTimeWithTimeZone,
        pub updated_at: DateTimeWithTimeZone,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(has_many = "super::messages::Entity")]
        Messages,
        #[sea_orm(
            belongs_to = "super::rust_llm_models::Entity",
            from = "Column::RustLlmModelId",
            to = "super::rust_llm_models::Column::Id"
        )]
        RustLlmModel,
    }

    impl Related<super::messages::Entity> for Entity {
        fn to() -> RelationDef {
            Relation::Messages.def()
        }
    }

    impl Related<super::rust_llm_models::Entity> for Entity {
        fn to() -> RelationDef {
            Relation::RustLlmModel.def()
        }
    }

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod messages {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "messages")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        pub chat_id: i32,
        pub role: String,
        pub content: Option<String>,
        pub cache_until_here: bool,
        pub thinking_text: Option<String>,
        pub thinking_signature: Option<String>,
        pub citations: Option<Json>,
        pub server_tool_calls: Option<Json>,
        pub raw_content: Option<Json>,
        pub raw_reasoning: Option<Json>,
        pub finish_reason: Option<String>,
        pub created_at: DateTimeWithTimeZone,
        pub updated_at: DateTimeWithTimeZone,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(belongs_to = "super::chats::Entity", from = "Column::ChatId", to = "super::chats::Column::Id")]
        Chat,
    }

    impl Related<super::chats::Entity> for Entity {
        fn to() -> RelationDef {
            Relation::Chat.def()
        }
    }

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod rust_llm_models {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "rust_llm_models")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        pub model_id: String,
        pub name: String,
        pub provider: String,
        pub family: Option<String>,
        pub model_created_at: Option<DateTimeWithTimeZone>,
        pub context_window: Option<i32>,
        pub max_output_tokens: Option<i32>,
        pub knowledge_cutoff: Option<Date>,
        pub unlisted_at: Option<DateTimeWithTimeZone>,
        pub modalities: Option<Json>,
        pub capabilities: Option<Json>,
        pub pricing: Option<Json>,
        pub metadata: Option<Json>,
        pub created_at: DateTimeWithTimeZone,
        pub updated_at: DateTimeWithTimeZone,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod rust_llm_tool_calls {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "rust_llm_tool_calls")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        pub message_type: String,
        pub message_id: i64,
        pub result_type: Option<String>,
        pub result_id: Option<i64>,
        pub tool_call_id: String,
        pub name: String,
        pub thought_signature: Option<String>,
        /// `"approved"` / `"denied"` / `NULL`.
        pub approval: Option<String>,
        pub remote: bool,
        pub arguments: Option<Json>,
        pub pending_input: Option<Json>,
        pub created_at: DateTimeWithTimeZone,
        pub updated_at: DateTimeWithTimeZone,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod rust_llm_usages {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "rust_llm_usages")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        pub chat_type: String,
        pub chat_id: i64,
        pub message_type: Option<String>,
        pub message_id: Option<i64>,
        pub operation: String,
        pub provider: String,
        pub model: String,
        pub status: String,
        pub input_tokens: Option<i32>,
        pub output_tokens: Option<i32>,
        pub cache_read_tokens: Option<i32>,
        pub cache_write_tokens: Option<i32>,
        pub thinking_tokens: Option<i32>,
        pub input_cost: Option<f64>,
        pub output_cost: Option<f64>,
        pub cache_read_cost: Option<f64>,
        pub cache_write_cost: Option<f64>,
        pub thinking_cost: Option<f64>,
        pub total_cost: Option<f64>,
        pub created_at: DateTimeWithTimeZone,
        pub updated_at: DateTimeWithTimeZone,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod rust_llm_attachments {
    use sea_orm::entity::prelude::*;

    /// A message's file: the bytes and the Active Storage blob attributes RubyLLM reads back.
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "rust_llm_attachments")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        pub message_type: String,
        pub message_id: i64,
        pub filename: String,
        pub content_type: String,
        pub byte_size: i64,
        /// `{ "resolution": "high" }`, like the blob metadata RubyLLM writes.
        pub metadata: Option<Json>,
        pub data: Vec<u8>,
        pub created_at: DateTimeWithTimeZone,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}
