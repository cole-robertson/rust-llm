import { Form, Head } from "@inertiajs/react"

import Heading from "@/components/heading"
import {
  ModelPicker,
  withDefaultModel,
} from "@/components/messages/model-picker"
import type { ChatModel } from "@/components/messages/types"
import { Button } from "@/components/ui/button"
import {
  Field,
  FieldError,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Spinner } from "@/components/ui/spinner"
import { useCurrentAccount } from "@/hooks/use-current-account"
import AppLayout from "@/layouts/app-layout"
import { chats as routes } from "@/routes"
import type { BreadcrumbItem } from "@/types"

export default function ChatNew({
  chat_models,
  selected_model,
  default_model_label,
}: {
  chat_models: ChatModel[]
  selected_model: string | null
  default_model_label: string
}) {
  const { slug: accountSlug } = useCurrentAccount()
  const breadcrumbs: BreadcrumbItem[] = [
    { title: "Chats", href: routes.index(accountSlug).url },
    { title: "New chat", href: routes.new(accountSlug).url },
  ]

  return (
    <AppLayout breadcrumbs={breadcrumbs}>
      <Head title="New chat" />
      <div className="max-w-2xl p-4">
        <Heading title="New chat" />
        <Form
          action={routes.create(accountSlug)}
          transform={withDefaultModel}
          disableWhileProcessing
          className="flex flex-col gap-6"
        >
          {({ processing, errors }) => (
            <>
              <FieldGroup>
                <Field>
                  <FieldLabel htmlFor="model">Select AI model:</FieldLabel>
                  <ModelPicker
                    models={chat_models}
                    selected={selected_model}
                    defaultLabel={default_model_label}
                  />
                  <FieldError
                    errors={errors.model?.map((message) => ({ message }))}
                  />
                </Field>
                <Field>
                  <FieldLabel htmlFor="prompt">Prompt</FieldLabel>
                  <Input
                    id="prompt"
                    name="prompt"
                    placeholder="What would you like to discuss?"
                    autoFocus
                  />
                  <FieldError
                    errors={errors.prompt?.map((message) => ({ message }))}
                  />
                </Field>
              </FieldGroup>
              <div>
                <Button type="submit" disabled={processing}>
                  {processing && <Spinner />}
                  Start new chat
                </Button>
              </div>
            </>
          )}
        </Form>
      </div>
    </AppLayout>
  )
}
