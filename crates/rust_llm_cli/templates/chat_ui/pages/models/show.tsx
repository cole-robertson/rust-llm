import { Head, Link } from "@inertiajs/react"

import Heading from "@/components/heading"
import type { ChatModel } from "@/components/messages/types"
import { Button } from "@/components/ui/button"
import AppLayout from "@/layouts/app-layout"
import { chats, models as routes } from "@/routes"
import type { BreadcrumbItem } from "@/types"

export default function ModelShow({ model }: { model: ChatModel }) {
  const breadcrumbs: BreadcrumbItem[] = [
    { title: "Models", href: routes.index().url },
    { title: model.name, href: routes.show(model.id).url },
  ]

  return (
    <AppLayout breadcrumbs={breadcrumbs}>
      <Head title={model.name} />
      <div className="max-w-2xl space-y-6 p-4">
        <Heading title={model.name} />

        <dl className="divide-y rounded-lg border text-sm">
          <div className="grid grid-cols-3 gap-4 p-4">
            <dt className="text-muted-foreground font-medium">ID</dt>
            <dd className="col-span-2">{model.id}</dd>
          </div>
          <div className="grid grid-cols-3 gap-4 p-4">
            <dt className="text-muted-foreground font-medium">Provider</dt>
            <dd className="col-span-2">{model.provider_name}</dd>
          </div>
          <div className="grid grid-cols-3 gap-4 p-4">
            <dt className="text-muted-foreground font-medium">
              Context Window
            </dt>
            <dd className="col-span-2">
              {model.context_window?.toLocaleString() ?? "—"} tokens
            </dd>
          </div>
          <div className="grid grid-cols-3 gap-4 p-4">
            <dt className="text-muted-foreground font-medium">Max Output</dt>
            <dd className="col-span-2">
              {model.max_output_tokens?.toLocaleString() ?? "—"} tokens
            </dd>
          </div>
          {model.capabilities.length > 0 && (
            <div className="grid grid-cols-3 gap-4 p-4">
              <dt className="text-muted-foreground font-medium">
                Capabilities
              </dt>
              <dd className="col-span-2">{model.capabilities.join(", ")}</dd>
            </div>
          )}
        </dl>

        <div className="flex gap-2">
          <Button asChild>
            <Link href={chats.new({ query: { model: model.value } })}>
              Start chat with this model
            </Link>
          </Button>
          <Button variant="ghost" asChild>
            <Link href={routes.index()}>All models</Link>
          </Button>
        </div>
      </div>
    </AppLayout>
  )
}
