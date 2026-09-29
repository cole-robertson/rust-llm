import { ErrorMessage } from "../error"
import { formatTime } from "../format"
import type { ToolResultProps } from "../types"

// messages/tool_results/_default: any tool without its own component.
export default function DefaultToolResult({ message }: ToolResultProps) {
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
      <div className="mb-1 font-semibold">Tool</div>
      <pre className="m-0 text-sm whitespace-pre-wrap">
        {message.content?.trim() ? message.content : "(no output)"}
      </pre>
      <div className="text-muted-foreground mt-1 text-xs">
        {formatTime(message.created_at)}
      </div>
    </div>
  )
}
