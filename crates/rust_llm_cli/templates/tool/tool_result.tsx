import { ErrorMessage } from "../error"
import { formatTime } from "../format"
import type { ToolResultProps } from "../types"

const label = "{{display_name}} Result"

// `{{tool_name}}` ({{class_name}}), like RubyLLM's tool_results/_{{tool_name}} partial.
export default function ToolResult({ message }: ToolResultProps) {
  if (message.tool_error_message) {
    return (
      <ErrorMessage
        message={message}
        title="Tool Result Error"
        errorMessage={message.tool_error_message}
      />
    )
  }

  return (
    <div
      id={`message_${message.id}`}
      className="border-muted-foreground bg-muted/50 rounded-md border-l-4 p-3"
    >
      <div className="mb-1 font-semibold">{label}</div>
      <pre className="m-0 text-sm whitespace-pre-wrap">
        {message.content?.trim() ? message.content : "(no output)"}
      </pre>
      <div className="text-muted-foreground mt-1 text-xs">
        {formatTime(message.created_at)}
      </div>
    </div>
  )
}
