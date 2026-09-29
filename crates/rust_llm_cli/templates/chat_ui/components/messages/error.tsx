import { formatTime } from "./format"
import type { Message } from "./types"

// messages/_error
export function ErrorMessage({
  message,
  title,
  errorMessage,
}: {
  message: Message
  title?: string
  errorMessage: string
}) {
  return (
    <div
      id={`message_${message.id}`}
      className="border-destructive bg-destructive/10 rounded-md border-l-4 p-3"
    >
      <div className="text-destructive mb-1 font-semibold">
        {title ?? "Error"}
      </div>
      <pre className="text-destructive m-0 text-sm whitespace-pre-wrap">
        {errorMessage}
      </pre>
      <div className="text-destructive/80 mt-1 text-xs">
        {formatTime(message.created_at)}
      </div>
    </div>
  )
}
