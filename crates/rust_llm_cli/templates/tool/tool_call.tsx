import { formatArguments, formatTime } from "../format"
import type { ToolCallProps } from "../types"

const label = "{{display_name}} Call"

// `{{tool_name}}` ({{class_name}}), like RubyLLM's tool_calls/_{{tool_name}} partial.
export default function ToolCall({ message, toolCall }: ToolCallProps) {
  return (
    <div
      id={`message_tool_call_${toolCall.id}`}
      className="border-muted-foreground bg-muted/50 rounded-md border-l-4 p-3"
    >
      <div className="mb-1 font-semibold">{label}</div>
      <pre className="m-0 text-sm whitespace-pre-wrap">
        {toolCall.name}({formatArguments(toolCall.arguments)})
      </pre>
      <div className="text-muted-foreground mt-1 text-xs">
        {formatTime(message.created_at)}
      </div>
    </div>
  )
}
