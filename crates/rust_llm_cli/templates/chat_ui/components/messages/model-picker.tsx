import type { FormDataConvertible } from "@inertiajs/core"

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

import type { ChatModel } from "./types"

// Radix Select cannot hold an empty value, so the default model is this sentinel; the
// hidden input submits "" for it, which the controller reads as "use the default model".
const DEFAULT = "__default__"

// The chat form's model select (`form.select :model`).
export function ModelPicker({
  models,
  selected,
  defaultLabel,
}: {
  models: ChatModel[]
  selected: string | null
  defaultLabel: string
}) {
  return (
    <Select name="model" defaultValue={selected ?? DEFAULT}>
      <SelectTrigger id="model" className="w-full max-w-xl">
        <SelectValue placeholder={defaultLabel} />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={DEFAULT}>{defaultLabel}</SelectItem>
        {models.map((model) => (
          <SelectItem key={model.value} value={model.value}>
            {model.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

// Inertia's <Form transform>: maps the sentinel back to "".
export function withDefaultModel(
  data: Record<string, FormDataConvertible>,
): Record<string, FormDataConvertible> {
  return { ...data, model: data.model === DEFAULT ? "" : data.model }
}
