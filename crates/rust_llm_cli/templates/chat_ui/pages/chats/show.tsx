import { Form, Head, Link, usePoll } from "@inertiajs/react"
import { useEffect } from "react"

import { MessageList } from "@/components/messages/message-list"
import type { Chat, Message } from "@/components/messages/types"
import { Button } from "@/components/ui/button"
import { Field, FieldError } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Spinner } from "@/components/ui/spinner"
import AppLayout from "@/layouts/app-layout"
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
  const breadcrumbs: BreadcrumbItem[] = [
    { title: "Chats", href: routes.index().url },
    { title: `Chat ${chat.id}`, href: routes.show(chat.id).url },
  ]

  // RubyLLM streams chunks over Turbo. Here the worker persists each message as it lands and
  // the page polls for them (a partial reload of `messages`) while a response is pending.
  const { start, stop } = usePoll(
    1000,
    { only: ["messages", "awaiting_response"] },
    { autoStart: false },
  )
  useEffect(() => {
    if (awaiting_response) {
      start()
    } else {
      stop()
    }
  }, [awaiting_response, start, stop])

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

        <MessageList messages={messages} />

        {awaiting_response && (
          <p className="text-muted-foreground flex items-center gap-2 text-sm">
            <Spinner /> Waiting for the assistant…
          </p>
        )}

        <Form
          action={chatMessages.create(chat.id)}
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
          <Link href={routes.index()}>Back to chats</Link>
        </Button>
      </div>
    </AppLayout>
  )
}
