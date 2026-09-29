import { Head, Link } from "@inertiajs/react"

import Heading from "@/components/heading"
import type { ChatSummary } from "@/components/messages/types"
import { Button } from "@/components/ui/button"
import AppLayout from "@/layouts/app-layout"
import { chats as routes, models } from "@/routes"
import type { BreadcrumbItem } from "@/types"

const breadcrumbs: BreadcrumbItem[] = [
  { title: "Chats", href: routes.index().url },
]

export default function ChatIndex({ chats }: { chats: ChatSummary[] }) {
  return (
    <AppLayout breadcrumbs={breadcrumbs}>
      <Head title="Chats" />
      <div className="space-y-6 p-4">
        <div className="flex items-start justify-between gap-4">
          <Heading title="Chats" />
          <div className="flex gap-2">
            <Button variant="outline" asChild>
              <Link href={models.index()}>Models</Link>
            </Button>
            <Button asChild>
              <Link href={routes.new()}>New chat</Link>
            </Button>
          </div>
        </div>

        {chats.length === 0 ? (
          <p className="text-muted-foreground text-sm">No chats found.</p>
        ) : (
          <ul className="divide-y rounded-lg border">
            {chats.map((chat) => (
              <li
                key={chat.id}
                id={`chat_${chat.id}`}
                className="flex items-center justify-between gap-4 p-4"
              >
                <div className="space-y-1">
                  <Link
                    href={routes.show(chat.id)}
                    className="font-medium underline-offset-4 hover:underline"
                  >
                    Chat {chat.id}
                  </Link>
                  <p className="text-muted-foreground text-sm">
                    {chat.model_label ?? "Default model"} · {chat.message_count}{" "}
                    messages · {new Date(chat.created_at).toLocaleString()}
                  </p>
                </div>
                <Button variant="destructive" size="sm" asChild>
                  <Link
                    href={routes.destroy(chat.id)}
                    as="button"
                    onBefore={() => confirm("Are you sure?")}
                  >
                    Delete
                  </Link>
                </Button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </AppLayout>
  )
}
