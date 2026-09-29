import { formatArguments, formatTime } from "../format"
import type { ToolCallProps } from "../types"

// messages/tool_calls/_default: any tool without its own component.
export default function DefaultToolCall({ message, toolCall }: ToolCallProps) {
  return (
    <div
      id={`message_tool_call_${toolCall.id}`}
      className="border-muted-foreground bg-muted/50 rounded-md border-l-4 p-3"
    >
      <div className="mb-1 font-semibold">Tool Call</div>
      <pre className="m-0 text-sm whitespace-pre-wrap">
        {toolCall.name}({formatArguments(toolCall.arguments)})
      </pre>
      <div className="text-muted-foreground mt-1 text-xs">
        {formatTime(message.created_at)}
      </div>
    </div>
  )
}
