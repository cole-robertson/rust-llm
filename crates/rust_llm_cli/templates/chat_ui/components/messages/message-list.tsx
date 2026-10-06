import { type ComponentType, createElement } from "react"

import { formatTime } from "./format"
import DefaultToolCall from "./tool_calls/default"
import DefaultToolResult from "./tool_results/default"
import type { Message, ToolCallProps, ToolResultProps } from "./types"

// `rust-llm generate tool NAME` adds tool_calls/<name>.tsx and tool_results/<name>.tsx; the
// helper's `tool_call_partial`/`tool_result_partial` lookup, falling back to `default`.
function byToolName<P>(
  modules: Record<string, { default: ComponentType<P> }>,
): Map<string, ComponentType<P>> {
  return new Map(
    Object.entries(modules).map(([path, module]) => [
      path.replace(/^.*\/(.*)\.tsx$/, "$1"),
      module.default,
    ]),
  )
}

const toolCallComponents = byToolName(
  import.meta.glob<{ default: ComponentType<ToolCallProps> }>(
    "./tool_calls/*.tsx",
    { eager: true },
  ),
)
const toolResultComponents = byToolName(
  import.meta.glob<{ default: ComponentType<ToolResultProps> }>(
    "./tool_results/*.tsx",
    { eager: true },
  ),
)

function normalize(name: string | null) {
  return (name ?? "").replace(/-/g, "_")
}

function ToolCallItem(props: ToolCallProps) {
  return createElement(
    toolCallComponents.get(normalize(props.toolCall.name)) ?? DefaultToolCall,
    props,
  )
}

function ToolResultItem(props: ToolResultProps) {
  return createElement(
    toolResultComponents.get(normalize(props.message.tool_name)) ??
      DefaultToolResult,
    props,
  )
}

export function Bubble({
  id,
  label,
  content,
  createdAt,
  className,
}: {
  id: number
  label: string
  content: string | null
  createdAt?: string
  className: string
}) {
  return (
    <div
      id={`message_${id}`}
      className={`rounded-md border-l-4 p-3 ${className}`}
    >
      <div className="mb-1 font-semibold">{label}</div>
      <div id={`message_${id}_content`} className="whitespace-pre-wrap">
        {content}
      </div>
      {createdAt && (
        <div className="text-muted-foreground mt-1 text-xs">
          {formatTime(createdAt)}
        </div>
      )}
    </div>
  )
}

function MessageBubble({
  message,
  label,
  className,
}: {
  message: Message
  label: string
  className: string
}) {
  return (
    <Bubble
      id={message.id}
      label={label}
      content={message.content}
      createdAt={message.created_at}
      className={className}
    />
  )
}

// `message.to_partial_path`: tool_calls, tool, or the role.
export function MessageItem({ message }: { message: Message }) {
  if (message.tool_calls.length > 0) {
    return (
      <div id={`message_${message.id}`} className="space-y-2">
        {message.content && (
          <MessageBubble
            message={message}
            label="Assistant"
            className="border-green-600"
          />
        )}
        {message.tool_calls.map((toolCall) => (
          <ToolCallItem
            key={toolCall.id}
            message={message}
            toolCall={toolCall}
          />
        ))}
      </div>
    )
  }

  switch (message.role) {
    case "tool":
      return <ToolResultItem message={message} />
    case "user":
      return (
        <MessageBubble
          message={message}
          label="User"
          className="border-blue-600"
        />
      )
    case "system":
      return (
        <MessageBubble
          message={message}
          label="System"
          className="border-muted-foreground bg-muted/50"
        />
      )
    default:
      return (
        <MessageBubble
          message={message}
          label="Assistant"
          className="border-green-600"
        />
      )
  }
}

export function MessageList({ messages }: { messages: Message[] }) {
  return (
    <div id="messages" className="space-y-4">
      {messages.map((message) => (
        <MessageItem key={message.id} message={message} />
      ))}
    </div>
  )
}
