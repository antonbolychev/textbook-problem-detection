import { defineConfig } from "@hey-api/openapi-ts"

export default defineConfig({
  input: "./openapi.json",
  output: "./src/client",
  plugins: [
    "legacy/axios",
    {
      name: "@hey-api/sdk",
      asClass: true,
      operationId: true,
      classNameBuilder: (operation: any) => {
        const service = operation.service && operation.service.length > 0 ? operation.service : "Default"
        return `${service}Service`
      },
      methodNameBuilder: (operation: any) => {
        let name = operation.name ?? ""
        const service = operation.service ?? ""
        if (service && name.toLowerCase().startsWith(service.toLowerCase())) {
          name = name.slice(service.length)
        }
        if (!name) {
          return "request"
        }
        return name.charAt(0).toLowerCase() + name.slice(1)
      },
    },
    {
      name: "@hey-api/schemas",
      type: "json",
    },
  ],
})
