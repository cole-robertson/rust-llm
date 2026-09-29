// `created_at&.strftime("%I:%M %p")`
export function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  })
}

// `tool_call.arguments.map { |k, v| "#{k}: #{v.inspect}" }.join(", ")`
export function formatArguments(args: Record<string, unknown>): string {
  return Object.entries(args)
    .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
    .join(", ")
}
