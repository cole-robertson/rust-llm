import { Form, Head, Link, router } from "@inertiajs/react"
import { useState } from "react"

import { Bubble, MessageList } from "@/components/messages/message-list"
import type {
  Chat,
  ChatEvent,
  Message,
  StreamingMessage,
} from "@/components/messages/types"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Field, FieldError } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Spinner } from "@/components/ui/spinner"
import { useCurrentAccount } from "@/hooks/use-current-account"
import AppLayout from "@/layouts/app-layout"
import { useChannel } from "@/lib/live"
import { chatMessages, chats as routes } from "@/routes"
import type { BreadcrumbItem } from "@/types"

export default function ChatShow({
  chat,
  messages,
  awaiting_response,
  default_model_label,
}: {
  chat: Chat
  messages: Message[]
  awaiting_response: boolean
  default_model_label: string
}) {
  const { slug: accountSlug } = useCurrentAccount()
  const at = { accountSlug, id: chat.id }
  const breadcrumbs: BreadcrumbItem[] = [
    { title: "Chats", href: routes.index(accountSlug).url },
    { title: `Chat ${chat.id}`, href: routes.show(at).url },
  ]

  // RubyLLM appends each chunk over Turbo; here the worker broadcasts on ChatChannel. The
  // reply grows in `streaming` until its row is written, then `messages` is reloaded.
  const [streaming, setStreaming] = useState<StreamingMessage | null>(null)
  const [error, setError] = useState<string | null>(null)
  useChannel<ChatEvent>("ChatChannel", at, (event) => {
    switch (event.type) {
      case "message_start":
        setError(null)
        setStreaming(
          event.role === "assistant"
            ? { id: event.message_id, content: "" }
            : null,
        )
        break
      case "chunk":
        setStreaming((current) =>
          current?.id === event.message_id
            ? { ...current, content: current.content + event.content }
            : { id: event.message_id, content: event.content },
        )
        break
      case "message_end":
        router.reload({
          only: ["messages", "awaiting_response"],
          onSuccess: () =>
            setStreaming((current) =>
              current?.id === event.message_id ? null : current,
            ),
        })
        break
      case "error":
        setStreaming(null)
        setError(event.message)
        router.reload({ only: ["messages", "awaiting_response"] })
        break
    }
  })

  // The streamed text replaces the (still empty) row it is being written into.
  const shown = streaming
    ? messages.filter((message) => message.id !== streaming.id)
    : messages

  return (
    <AppLayout breadcrumbs={breadcrumbs}>
      <Head title={`Chat ${chat.id}`} />
      <div className="max-w-3xl space-y-6 p-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">
            Chat {chat.id}
          </h1>
          <p className="text-muted-foreground text-sm">
            Using <strong>{chat.model_label ?? default_model_label}</strong>
          </p>
        </div>

        <MessageList messages={shown} />

        {streaming && (
          <Bubble
            id={streaming.id}
            label="Assistant"
            content={streaming.content}
            className="border-green-600"
          />
        )}

        {awaiting_response && !streaming && !error && (
          <p className="text-muted-foreground flex items-center gap-2 text-sm">
            <Spinner /> Waiting for the assistant…
          </p>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <Form
          action={chatMessages.create({ accountSlug, chatId: chat.id })}
          resetOnSuccess
          options={{ preserveScroll: true }}
          className="flex items-start gap-2"
        >
          {({ processing, errors }) => (
            <>
              <Field className="flex-1">
                <Input
                  id="content"
                  name="content"
                  placeholder="Message..."
                  autoFocus
                />
                <FieldError
                  errors={errors.content?.map((message) => ({ message }))}
                />
              </Field>
              <Button type="submit" disabled={processing}>
                {processing && <Spinner />}
                Send message
              </Button>
            </>
          )}
        </Form>

        <Button variant="ghost" asChild>
          <Link href={routes.index(accountSlug)}>Back to chats</Link>
        </Button>
      </div>
    </AppLayout>
  )
}
