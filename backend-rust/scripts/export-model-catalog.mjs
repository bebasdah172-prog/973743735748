import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { PROVIDER_MODELS, PROVIDER_ID_TO_ALIAS } from "../../open-sse/config/providerModels.js";
import REGISTRY from "../../open-sse/providers/registry/index.js";
import { getCapabilitiesForModel } from "../../open-sse/providers/capabilities.js";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const backendDir = path.resolve(scriptDir, "..");
const providerDefinitions = Object.fromEntries(REGISTRY.map((provider) => [provider.id, provider]));
const providerIdByAlias = Object.fromEntries(
  Object.entries(PROVIDER_ID_TO_ALIAS).map(([id, alias]) => [alias, id]),
);
const modelKind = (model) => {
  const kind = model?.kind || model?.type;
  return kind === "imageToText" ? "llm" : (kind || "llm");
};

const providers = Object.entries(PROVIDER_MODELS).map(([alias, models]) => {
  const id = providerIdByAlias[alias] || alias;
  const definition = providerDefinitions[id] || {};
  return {
    id,
    alias,
    aliases: definition.aliases || [],
    category: definition.category || null,
    displayName: definition.display?.name || alias,
    noAuth: definition.noAuth === true,
    authModes: definition.authModes
      || (definition.category === "apikey" ? ["apikey"] : []),
    serviceKinds: definition.serviceKinds || ["llm"],
    models: models
      .filter((model) => modelKind(model) === "llm")
      .map((model) => ({
        id: model.id,
        capabilities: getCapabilitiesForModel(id, model.id),
      })),
  };
});

const outputPath = path.join(backendDir, "models.json");
fs.writeFileSync(outputPath, `${JSON.stringify({ providers }, null, 2)}\n`);
console.log(`Exported ${providers.length} providers to ${outputPath}`);
