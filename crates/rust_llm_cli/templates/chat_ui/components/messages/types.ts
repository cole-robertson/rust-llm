// Page props for the chat UI, as the Rust controllers build them (src/models/messages.rs).

export interface ToolCall {
  id: number
  tool_call_id: string
  name: string
  arguments: Record<string, unknown>
}

export interface Message {
  id: number
  role: "system" | "user" | "assistant" | "tool"
  content: string | null
  created_at: string
  tool_calls: ToolCall[]
  // The tool a `tool` message answers (`message.parent_tool_call.name`).
  tool_name: string | null
  tool_error_message: string | null
}

export interface Chat {
  id: number
  model_label: string | null
  created_at: string
}

export interface ChatSummary extends Chat {
  message_count: number
}

export interface ChatModel {
  id: string
  name: string
  provider: string
  provider_name: string
  label: string
  // "provider:id", what the model picker submits.
  value: string
  context_window: number | null
  max_output_tokens: number | null
  capabilities: string[]
  input_price: number | null
  output_price: number | null
}

export interface ToolCallProps {
  message: Message
  toolCall: ToolCall
}

export interface ToolResultProps {
  message: Message
}
