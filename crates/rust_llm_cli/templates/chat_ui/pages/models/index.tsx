import { Head, Link, router } from "@inertiajs/react"

import Heading from "@/components/heading"
import type { ChatModel } from "@/components/messages/types"
import { Button } from "@/components/ui/button"
import AppLayout from "@/layouts/app-layout"
import { chats, models as routes } from "@/routes"
import type { BreadcrumbItem } from "@/types"

const breadcrumbs: BreadcrumbItem[] = [
  { title: "Models", href: routes.index().url },
]

function price(model: ChatModel) {
  if (model.input_price == null || model.output_price == null) return null
  return `$${model.input_price.toFixed(2)} / $${model.output_price.toFixed(2)}`
}

export default function ModelIndex({ models }: { models: ChatModel[] }) {
  return (
    <AppLayout breadcrumbs={breadcrumbs}>
      <Head title="Models" />
      <div className="space-y-6 p-4">
        <div className="flex items-start justify-between gap-4">
          <Heading title="Models" />
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => router.post(routes.refresh().url)}>
              Refresh
            </Button>
            <Button variant="outline" asChild>
              <Link href={chats.index()}>Chats</Link>
            </Button>
          </div>
        </div>

        {models.length === 0 ? (
          <p className="text-muted-foreground text-sm">No chat models found.</p>
        ) : (
          <div className="overflow-x-auto rounded-lg border">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 text-left">
                <tr>
                  <th className="p-3">Provider</th>
                  <th className="p-3">Model</th>
                  <th className="p-3">Context Window</th>
                  <th className="p-3">$/1M tokens (In/Out)</th>
                  <th className="p-3"></th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {models.map((model) => (
                  <tr key={model.value}>
                    <td className="p-3">{model.provider_name}</td>
                    <td className="p-3">
                      <Link
                        href={routes.show(model.id, {
                          query: { provider: model.provider },
                        })}
                        className="underline-offset-4 hover:underline"
                      >
                        {model.name}
                      </Link>
                    </td>
                    <td className="p-3">
                      {model.context_window?.toLocaleString()}
                    </td>
                    <td className="p-3">{price(model)}</td>
                    <td className="p-3">
                      <Link
                        href={chats.new({ query: { model: model.value } })}
                        className="underline-offset-4 hover:underline"
                      >
                        Start chat
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </AppLayout>
  )
}
