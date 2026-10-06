import { Head, Link, router, usePage } from "@inertiajs/react"
import { type FormEvent, useRef, useState } from "react"

import AppLogoIcon from "@/components/app-logo-icon"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Spinner } from "@/components/ui/spinner"
import { Textarea } from "@/components/ui/textarea"
import { send } from "@/lib/live"
import { home, publicChat, sessions } from "@/routes"

// `rust-llm generate public_chat`: a chat anyone can try without signing in. The reply to each
// message streams back in the POST's own response (Server-Sent Events), so it only ever reaches
// the browser that asked. The conversation lives on the server for this browser session.

interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

interface Limits {
  max_input_chars: number
  max_turns: number
  turns_left: number
}

type Frame =
  | { type: "chunk"; content: string }
  | { type: "end"; content: string }
  | { type: "error"; message: string }

// Each `data:` line of a Server-Sent Events body, as it arrives.
async function* frames(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<Frame> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  for (;;) {
    const { value, done } = await reader.read()
    if (done) return
    buffer += decoder.decode(value, { stream: true })
    let end
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const event = buffer.slice(0, end)
      buffer = buffer.slice(end + 2)
      const data = event
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n")
      if (data) yield JSON.parse(data) as Frame
    }
  }
}

export default function PublicChat({
  messages: initial,
  model_label,
  limits,
}: {
  messages: ChatMessage[]
  model_label: string
  limits: Limits
}) {
  const { auth } = usePage().props
  const [messages, setMessages] = useState(initial)
  const [draft, setDraft] = useState("")
  const [reply, setReply] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [turnsLeft, setTurnsLeft] = useState(limits.turns_left)
  const bottom = useRef<HTMLDivElement>(null)

  const busy = reply !== null
  const tooLong = draft.length > limits.max_input_chars

  async function submit(event: FormEvent) {
    event.preventDefault()
    const content = draft.trim()
    if (!content || busy || tooLong) return
    setError(null)
    setDraft("")
    setMessages((m) => [...m, { role: "user", content }])
    setReply("")
    let text = ""
    try {
      const res = await send(publicChat.messages().url, "POST", { content })
      if (!res.ok || !res.body) {
        const body = (await res.json().catch(() => ({}))) as { error?: string }
        throw new Error(body.error ?? "Something went wrong. Please try again.")
      }
      for await (const frame of frames(res.body)) {
        if (frame.type === "chunk") {
          text += frame.content
          setReply(text)
          bottom.current?.scrollIntoView({ block: "end" })
        } else if (frame.type === "end") {
          text = frame.content
        } else {
          throw new Error(frame.message)
        }
      }
      setMessages((m) => [...m, { role: "assistant", content: text }])
      setTurnsLeft((n) => Math.max(0, n - 1))
    } catch (e) {
      // The message wasn't kept: take it back off the list and give it back to edit.
      setMessages((m) => m.slice(0, -1))
      setDraft(content)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setReply(null)
    }
  }

  return (
    <>
      <Head title="Chat" />
      <div className="bg-background flex min-h-screen flex-col items-center p-4 lg:p-8">
        <header className="mb-6 flex w-full max-w-3xl items-center gap-4 text-sm">
          <Link
            href={home.index()}
            className="flex items-center gap-2 font-medium"
          >
            <AppLogoIcon className="size-6" />
            {import.meta.env.VITE_APP_NAME ?? "Chat"}
          </Link>
          <span className="text-muted-foreground mr-auto">{model_label}</span>
          {!auth.user && (
            <Link
              href={sessions.new()}
              className="underline-offset-4 hover:underline"
            >
              Sign in
            </Link>
          )}
        </header>

        <main className="flex w-full max-w-3xl flex-1 flex-col gap-4">
          {messages.length === 0 && !busy && (
            <p className="text-muted-foreground py-12 text-center">
              Ask anything. Replies stream in as they are written.
            </p>
          )}
          {messages.map((message, i) => (
            <Card
              key={i}
              className={
                message.role === "user" ? "bg-muted/50 ml-12" : "mr-12"
              }
            >
              <CardContent className="whitespace-pre-wrap">
                {message.content}
              </CardContent>
            </Card>
          ))}
          {busy && (
            <Card className="mr-12" aria-live="polite">
              <CardContent className="whitespace-pre-wrap">
                {reply || <Spinner />}
              </CardContent>
            </Card>
          )}
          <div ref={bottom} />

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <form
            onSubmit={(e) => void submit(e)}
            className="flex flex-col gap-2"
          >
            <Textarea
              name="content"
              value={draft}
              placeholder="Message…"
              rows={3}
              autoFocus
              disabled={turnsLeft === 0}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) void submit(e)
              }}
            />
            <div className="text-muted-foreground flex items-center gap-4 text-xs">
              <span className={tooLong ? "text-destructive" : undefined}>
                {draft.length.toLocaleString()} /{" "}
                {limits.max_input_chars.toLocaleString()}
              </span>
              <span className="mr-auto">
                {turnsLeft} of {limits.max_turns} messages left
              </span>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                disabled={busy}
                onClick={() => router.delete(publicChat.destroy().url)}
              >
                Start over
              </Button>
              <Button
                type="submit"
                size="sm"
                disabled={busy || tooLong || !draft.trim() || turnsLeft === 0}
              >
                {busy && <Spinner />}
                Send
              </Button>
            </div>
          </form>
        </main>
      </div>
    </>
  )
}
