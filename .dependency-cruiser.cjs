/** @type {import('dependency-cruiser').IConfiguration} */
module.exports = {
  forbidden: [
    // shared cannot import from any layer above it
    {
      name: "fsd-shared-no-upward",
      severity: "error",
      comment: "shared/ must not import from entities, features, widgets, views, or app",
      from: { path: "^src/shared/" },
      to: { path: "^src/(entities|features|widgets|views|app)/" },
    },
    // entities can only import from shared
    {
      name: "fsd-entities-no-upward",
      severity: "error",
      comment: "entities/ can only import from shared/",
      from: { path: "^src/entities/" },
      to: { path: "^src/(features|widgets|views|app)/" },
    },
    // features can only import from entities and shared
    {
      name: "fsd-features-no-upward",
      severity: "error",
      comment: "features/ can only import from entities/ and shared/",
      from: { path: "^src/features/" },
      to: { path: "^src/(widgets|views|app)/" },
    },
    // widgets can only import from features, entities, and shared
    {
      name: "fsd-widgets-no-upward",
      severity: "error",
      comment: "widgets/ can only import from features/, entities/, and shared/",
      from: { path: "^src/widgets/" },
      to: { path: "^src/(views|app)/" },
    },
    // views can only import from widgets, features, entities, and shared
    {
      name: "fsd-views-no-upward",
      severity: "error",
      comment: "views/ can only import from widgets/, features/, entities/, and shared/",
      from: { path: "^src/views/" },
      to: { path: "^src/app/" },
    },
    // No cross-slice imports within the same layer
    {
      name: "fsd-no-cross-slice-entities",
      severity: "error",
      comment: "Entity slices must not import from sibling slices",
      from: { path: "^src/entities/([^/]+)/" },
      to: { path: "^src/entities/([^/]+)/", pathNot: "^src/entities/$1/" },
    },
    {
      name: "fsd-no-cross-slice-features",
      severity: "error",
      comment: "Feature slices must not import from sibling slices",
      from: { path: "^src/features/([^/]+)/" },
      to: { path: "^src/features/([^/]+)/", pathNot: "^src/features/$1/" },
    },
    {
      name: "fsd-no-cross-slice-widgets",
      severity: "error",
      comment: "Widget slices must not import from sibling slices",
      from: { path: "^src/widgets/([^/]+)/" },
      to: { path: "^src/widgets/([^/]+)/", pathNot: "^src/widgets/$1/" },
    },
    {
      name: "fsd-no-cross-slice-views",
      severity: "error",
      comment: "View slices must not import from sibling slices",
      from: { path: "^src/views/([^/]+)/" },
      to: { path: "^src/views/([^/]+)/", pathNot: "^src/views/$1/" },
    },
  ],
  options: {
    doNotFollow: { path: "node_modules" },
    tsPreCompilationDeps: true,
    tsConfig: { fileName: "tsconfig.json" },
    enhancedResolveOptions: {
      exportsFields: ["exports"],
      conditionNames: ["import", "require", "node", "default"],
    },
  },
};
