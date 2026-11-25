# Fases

Alcance: llevar ailoop a konor-flujo, resolver los FR-15 a FR-18 que pidió TECgpt y lo que falta del FR 5 de konor-kit. El FR-9 de TECgpt (interrumpir al modelo y seguir en el mismo run) queda para 1.x.

## Ya resuelto en `main`

De los FR de konor-kit, ya están mergeados en `origin/main` (`28a6535`):

- **FR 1**: `ProviderErrorKind` y `ProviderError::kind` en `ailoop-core/src/provider_error.rs`, con el mapeo en `ailoop-anthropic` y `ailoop-openai`, y `EngineError::provider_kind` (PR #68 y #69).
- **FR 2**: los marcadores de `ScriptedError` para cada `ProviderErrorKind`, `ScriptedError::of` y un `Retryable` que concuerda con ellos (`ailoop-core/src/testing.rs`).
- **FR 3**: adaptive thinking con `output_config.effort` en los modelos actuales y `AnthropicChatModel::server_fallbacks` (PR #67 y #72).
- **FR 4**: `BoxedModel` y `BoxedError` en `ailoop-core/src/boxed.rs` (PR #73).
- **FR 5, parte de OpenAI**: `OpenAIClient::list_models` y `OpenAIClient::model_capabilities` (`adapters/openai/src/models.rs`). El resto es la Fase 6.

## Medición

2026-11-25, sobre `origin/main` (`28a6535`), con `<skill>` = la carpeta de konor-flujo:

- `node <skill>/base/rust/comentarios.mjs --medir .`: 9023 líneas de comentario en 115 archivos. De esas, unas 8121 son rustdoc (`///`, `//!`) y unas 942 son `//`. Por carpeta: `ailoop/src` 2720, `ailoop-core/src` 2286, `ailoop/tests` 940, `ailoop-history/src` 763, `adapters/anthropic` 657, `adapters/openai` 622, `ailoop-tools/src` 515, `adapters/mcp` 200, el resto menos de 110 cada una.
- `node <skill>/base/rust/avisos.mjs --medir .`: 673 avisos en 66 archivos, de ellos 653 `clippy::unwrap_used` (333 en `ailoop/tests`), 10 `let_underscore_must_use`, 5 `allow_attributes_without_reason`, 3 `allow_attributes` y 2 `collapsible_match`.
- Supresiones: `#![allow(dead_code)]` en `adapters/openai/src/events.rs` y `adapters/openai/tests/support/mod.rs`; `#[allow(unused_variables)]` en `ailoop-core/src/middleware.rs:87` y en el código que genera `ailoop-derive/src/lib.rs:708`; `#[allow(clippy::too_many_arguments)]` en `ailoop/src/engine.rs:333` y `:841`.

## Fase 1: Base de konor-flujo y CLAUDE.md

Estado: pendiente
Depende de: nada

**Objetivo.** ailoop tiene su `CLAUDE.md` y la base de Rust con trinquete: el código nuevo cumple los lints y las reglas de comentarios, y lo que ya existe queda registrado y solo puede bajar.

**Hecha cuando.**
- El CI corre exactamente la verificación del `CLAUDE.md` y pasa en `main`.
- Un `.unwrap()` nuevo en un `src/` hace fallar `node herramientas/avisos.mjs`. Uno dentro de `#[test]` o en `tests/` no.
- Un comentario `//` nuevo hace fallar `node herramientas/comentarios.mjs`. Una línea `///` o `//!` nueva no.
- `node herramientas/rodeos.mjs` pasa sin lista de excepciones.

**Qué leer.** `<skill>/base/aplicar.md` (secciones «Rust» y «Proyecto existente»), `<skill>/SKILL.md` («Lo de cada proyecto»), `.github/workflows/ci.yml`, el `Cargo.toml` raíz y el de cada crate.

**Decisiones tomadas.**
- El rustdoc es la API pública de una biblioteca y no cuenta como comentario. El contador cuenta `//` y no `///`, `//!` ni `#[doc = "…"]`. Hoy `comentarios.mjs` no puede hacer eso: el cambio va en la skill (`base/rust/comentarios.mjs`, una opción que deje fuera el rustdoc, con su prueba en `base/pruebas/`), no en la copia de `herramientas/`, que no se edita. Es el primer encargo de la fase y es en el repo de konor-flujo.
- Sin interfaz: no hay recorridos ni script de capturas. La prueba de un cambio es la salida de `cargo test`.
- La verificación completa: `cargo fmt --all --check`, `node herramientas/avisos.mjs` (con y sin `--all-features`, como hoy el CI), `node herramientas/comentarios.mjs`, `node herramientas/rodeos.mjs`, `cargo test --workspace` y con `--all-features`, `cargo doc --workspace --no-deps --all-features` con `RUSTDOCFLAGS=-D warnings`, y el `cargo check` con 1.88 (MSRV). Por crate: lo mismo con `-p <crate>`.
- `live-tests` son pruebas `#[ignore]` que llaman a APIs reales o a un servidor local: el `CLAUDE.md` dice cómo correrlas y que no forman parte de la verificación.
- Los ataques que más importan, para el `CLAUDE.md`: cancelación y aborto a media llamada, el orden de los resultados de tools frente al de las llamadas, que cada middleware reciba exactamente un hook de cierre, las llamadas movidas a segundo plano, reanudar una suspensión y los límites (`max_iterations`, timeout).
- Pool de worktrees: tres, en `C:\Users\emili\Documents\Github\ailoop-wt\1` a `3`, creados con `git worktree add --detach`. `[profile.dev]` lleva `debug = "line-tables-only"`.
- Los `#[allow]` existentes pasan a `#[expect(lint, reason = "…")]`. El `#[allow(unused_variables)]` que genera el derive también: lo que emite el macro sale en el código de quien lo usa.
- El archivo de fases es este, `docs/fases.md`.

**Fuera de alcance.** Borrar los `//` que ya existen (Fase 5) y corregir los `unwrap` registrados.

**Encargos previstos.**
- En konor-flujo: opción de `comentarios.mjs` que no cuenta el rustdoc, con su prueba.
- `lints.toml`, `[lints] workspace = true` en cada crate, `clippy.toml`, `#![cfg(test)]` en los archivos de `tests/` y `debug = "line-tables-only"`.
- `rodeos.mjs` limpio: `#[allow]` a `#[expect]`, los `_ =`, los `#![allow(dead_code)]` (las ayudas de prueba compartidas van a un crate de apoyo si hace falta).
- `comentarios.mjs --inicial`, `avisos.mjs --inicial` y el CI pasa a correr la verificación del `CLAUDE.md`.
- `CLAUDE.md` con todo lo de «Lo de cada proyecto» y el pool de worktrees creado.

## Fase 2: Política de cancelación por tool (FR-15)

Estado: pendiente
Depende de: Fase 1

**Objetivo.** Una tool puede pedir que el motor espere su llamada al cancelarla en vez de soltarla, y el modelo lee lo que la tool devolvió. Cuando el motor sí suelta una llamada que estaba corriendo, el texto ya no dice que no corrió.

**Hecha cuando.** Pruebas en `ailoop/tests` que demuestran, con una tool de prueba que termina su trabajo después de que se dispara su token:
- Con `Race` (el valor por defecto), todo se comporta como hoy, salvo el bug del aborto.
- Con `Await`, `RunControl::cancel_tool` en la ruta en línea y en la ruta por task registra lo que devolvió la tool, y el evento lo marca como terminado después de la cancelación.
- Con `Await`, `RunControl::cancel_background_tool` entrega el resultado real con `cancelled = true`.
- Con `Await`, cancelar el run (`RunConfig::cancellation`) espera la llamada y la registra antes de cerrar el run como abortado.
- Con `Await`, el timeout del run no espera: suelta la llamada.
- Una llamada que corría cuando el run abortó (por cancelación o por timeout, con `Race`) ya no recibe «Tool not run: the run was aborted».
- Una edición no idempotente cancelada a media escritura con `Await` se registra como aplicada, y el modelo no la repite (prueba con un modelo de guion).

**Qué leer.** `ailoop/src/engine.rs`: `race_abort` (L70), `abort_step` (desde L333), las funciones de texto (L495 a L520), la ruta por task (cerca de L1590 a L1625), la ruta en línea (cerca de L1626 a L1645), la lectura de los resultados (cerca de L1712 a L1745), `BackgroundTask::run` (cerca de L2093) y `Slot::into_unanswered` (L2226). `ailoop-core/src/control.rs` (`cancel_tool`, `cancel_background_tool`). `ailoop-tools/src/registry.rs` (`Tool`, `ToolDyn`) y los atributos de `ailoop-derive`.

**Decisiones tomadas.**
- API: `ToolDyn::cancel_policy(&self) -> CancelPolicy`, con `pub enum CancelPolicy { Race, Await }` y `Race` por defecto. `Tool` lleva `const CANCEL_POLICY: CancelPolicy = CancelPolicy::Race`, y `#[ailoop_tool]` acepta un atributo para `Await`, igual que `concurrency_safe` y `backgroundable`.
- Con `Await`, el motor dispara `ctx.cancellation()` para que la tool pare si puede, espera la llamada y registra lo que devuelva.
- Qué cortes esperan a una tool `Await`: `cancel_tool`, `cancel_background_tool` y `RunConfig::cancellation`. El timeout del run mantiene su plazo y la suelta. Un stream soltado no puede esperar nada: una tool que no puede quedar a medias tiene que lanzar su propia task, y el rustdoc de `CancelPolicy` lo dice.
- Una llamada ya movida a segundo plano sigue su política: con `Await`, `cancel_background_tool` espera y entrega el resultado real con `cancelled = true`.
- El resultado de una llamada que terminó después de pedir su cancelación lleva una marca en su evento. El nombre lo elige la sesión principal, siguiendo `BackgroundToolResult::cancelled`.
- Bug encontrado al revisar el FR: si el run aborta con una llamada corriendo, `Slot::into_unanswered` la vuelve `PendingCall::Run` y se registra `not_run_result`. Se corrige con un texto propio («the run was aborted ({reason}) while it ran; it may have partly completed») y una variante de `PendingCall` que distinga a las llamadas que corrían. Su prueba se escribe primero, fallando contra el código de hoy.
- Los textos nuevos todavía son fijos. Personalizarlos es la Fase 3.

**Fuera de alcance.** Los textos por tool o por conversación (Fase 3). Un plazo de gracia para el timeout. Interrumpir al modelo (FR-9).

**Encargos previstos.**
- Prueba que falla y arreglo del aborto que marca como «not run» una llamada que corría.
- `CancelPolicy` en `ToolDyn`, `Tool` y el derive, sin cambiar el motor.
- `Await` en la ruta en línea y en `cancel_tool`, con la marca en el evento.
- `Await` en la ruta por task y en `cancel_background_tool`.
- `Await` al cancelar el run, y el timeout que la suelta.
- Rustdoc, README de `ailoop-tools` y CHANGELOG.

## Fase 3: Textos del motor al cortar una llamada (FR-16)

Estado: pendiente
Depende de: Fase 2

**Objetivo.** Lo que el modelo lee cuando el motor corta o no corre una llamada se puede cambiar por tool y por conversación, incluido el idioma.

**Hecha cuando.** Pruebas que demuestran, para cada motivo, que manda el texto de la tool, si no el de la conversación, y si no el de hoy. También cubren el aborto del run, donde hoy los resultados solo pasan por `on_chunk_mut`.

**Qué leer.** Las funciones de texto de `ailoop/src/engine.rs` (`cancelled_before_run`, `cancelled_while_running`, `not_run_result`, `interrupted_on_resume`, `cancelled_in_background` y el texto nuevo de la Fase 2), `abort_step` y `ConversationBuilder` en `ailoop/src/conversation.rs`.

**Decisiones tomadas.**
- `pub enum EngineStop { CancelledBeforeRun, CancelledWhileRunning, NotRun(AbortReason), AbortedWhileRunning(AbortReason), InterruptedOnResume, CancelledInBackground }`, `#[non_exhaustive]`. La sesión principal ajusta las variantes a los textos que existan al cerrar la Fase 2.
- Por tool: `ToolDyn::stopped_result(&self, why: &EngineStop) -> Option<ToolResultContent>`, `None` por defecto.
- Por conversación: un respaldo en `ConversationBuilder` que recibe el nombre de la tool y el motivo, y devuelve `Option<ToolResultContent>`. Gana el de la tool. Si los dos devuelven `None`, queda el texto de hoy.
- Con `Await` la tool escribe su propio resultado: este hook solo cubre las llamadas que el motor no corre o suelta.

**Fuera de alcance.** Cambiar los textos por defecto.

**Encargos previstos.**
- `EngineStop` y `ToolDyn::stopped_result`.
- El respaldo en `ConversationBuilder`.
- Usarlos en cada sitio donde el motor escribe uno de esos textos, el aborto incluido.
- Rustdoc y CHANGELOG.

## Fase 4: Esquema de enteros y errores del Tool genérico (FR-17 y FR-18)

Estado: pendiente
Depende de: Fase 1

**Objetivo.** `#[ailoop_tool]` emite `integer` para los enteros, y una tool hecha con `Tool` elige el texto de sus argumentos inválidos y de sus errores. Con eso TECgpt puede dejar su `NativeTool` propio.

**Hecha cuando.**
- Una prueba de esquema del derive: `i8` a `i128` e `isize` dan `{"type": "integer"}`; `u8` a `u128` y `usize`, `{"type": "integer", "minimum": 0}`; `f32` y `f64` siguen en `number`; dentro de `Option` y `Vec` también.
- Pruebas de `Tool`: sin implementar nada, los textos son los de hoy («Invalid args: {e}» y el `Display` del error); con los métodos implementados, el modelo lee lo que devuelven.

**Qué leer.** `ailoop-derive/src/lib.rs` (`get_json_type`, cerca de L425 a L450), `ailoop-tools/src/registry.rs` (`Tool` y la impl general de `ToolDyn`, L48 a L195).

**Decisiones tomadas.**
- FR-17: solo `integer` y `minimum: 0` para los sin signo, como lo genera schemars. Sin opción `schema = "schemars"` ni dependencia nueva. Es un cambio del esquema para quien ya usa el macro, y va como tal en el CHANGELOG.
- FR-18: `fn invalid_args(&self, err: serde_json::Error) -> ToolResultContent` y `fn render_error(&self, err: &Self::Error) -> ToolResultContent` en `Tool`, con los textos de hoy por defecto.

**Fuera de alcance.** Cambiar cómo se serializa la salida correcta de una tool.

**Encargos previstos.**
- FR-17 en el derive, con sus pruebas.
- FR-18 en `Tool` y la impl general, con sus pruebas.
- Rustdoc, README de `ailoop-derive` y `ailoop-tools`, y CHANGELOG.

## Fase 5: Limpiar los comentarios `//`

Estado: pendiente
Depende de: Fase 1

**Objetivo.** Los cerca de 940 comentarios `//` que registró la Fase 1 bajan a los que deja la regla de «Comentarios» de konor-flujo. El rustdoc no se toca.

**Hecha cuando.** `.comentarios.json` queda en lo que permiten la regla y los `// SAFETY:`, `cargo test --workspace` pasa sin cambios de comportamiento, y cada porqué que importaba quedó en el informe para ir a un tipo, un `assert` o una prueba.

**Qué leer.** «Comentarios» en `<skill>/SKILL.md` y «Limpiar lo mecánico» en `<skill>/planificador.md`.

**Decisiones tomadas.** Por carpeta y con un modelo más barato. Va después de las fases 2 a 4, que tocan `engine.rs`, para no pelear con ellas por los mismos archivos.

**Fuera de alcance.** Los `unwrap` registrados y el rustdoc.

**Encargos previstos.** Uno por carpeta: `ailoop/src`, `ailoop-core/src`, `ailoop/tests`, `ailoop-history/src`, `ailoop-tools/src`, los adaptadores, y el resto junto.

## Fase 6: Qué puede hacer un modelo y qué modelos ofrece una clave (FR 5 de konor-kit)

Estado: pendiente
Depende de: Fase 1

**Objetivo.** Una app pregunta a cualquier `CompletionModel`, también a un `BoxedModel`, si acepta imágenes y tools sin conocer el adaptador. `ailoop-anthropic` lista los modelos de una clave, como ya lo hace `ailoop-openai`.

**Hecha cuando.**
- `CompletionModel::supports` devuelve `None` en un modelo que no lo implementa, y `BoxedModel` reenvía lo que dice el modelo que envuelve (pruebas en `ailoop-core`).
- `AnthropicChatModel` responde según su tabla de modelos, y `None` para un id que no conoce.
- `ailoop-openai` responde lo que le dieron (con `model_capabilities` o a mano), y `None` si nadie se lo dijo; nunca lo adivina por el nombre.
- `AnthropicClient::list_models` contra un servidor falso, con sus errores clasificados con `ProviderErrorKind`, igual que `adapters/openai/tests/fake_server.rs`.

**Qué leer.** `ailoop-core/src/traits.rs`, `ailoop-core/src/boxed.rs`, `Capability` en `ailoop-core/src/provider_error.rs`, `adapters/openai/src/models.rs` y la tabla de modelos de `adapters/anthropic` (la de adaptive thinking, en `thinking.rs` y `request.rs`).

**Decisiones tomadas.**
- `CompletionModel::supports(&self, capability: Capability) -> Option<bool>`, con `None` por defecto. Reusa el `Capability` que ya existe (`Images`, `Tools`) en vez de un tipo nuevo. Es síncrono: responde lo que el adaptador ya sabe, sin ir a la red.
- `ailoop-openai`: el modelo guarda las capacidades que le pasen (por ejemplo, las que devolvió `model_capabilities`) con un método del builder, y `supports` las devuelve.
- `ailoop-anthropic`: `supports` sale de la tabla de modelos que vive junto a los ids. `list_models` llama al endpoint de modelos de Anthropic. El primer encargo confirma qué devuelve hoy ese endpoint y si dice capacidades; si no las dice, `list_models` devuelve ids y nombres, y las capacidades salen de la tabla.

**Fuera de alcance.** Pedir las capacidades a la red desde `supports`. Capacidades más allá de imágenes y tools.

**Encargos previstos.**
- Confirmar el endpoint de modelos de Anthropic y anotar su forma en el informe.
- `CompletionModel::supports` y su reenvío en `BoxedModel`.
- `supports` en `ailoop-openai`, con el método del builder.
- `supports` en `ailoop-anthropic` desde la tabla, y `AnthropicClient::list_models` con su servidor falso.
- Rustdoc, README de los adaptadores y CHANGELOG.

## Fuera de alcance de todo el plan

- FR-9 de TECgpt, interrumpir al modelo y seguir en el mismo run: pospuesto a 1.x.
- Lo que TECgpt resuelve de su lado: el trace sin `run_finished` al detener, usar `concurrency_safe` y `ctx.progress`, y lo que su lista anota como «No son FR».
- Preparar y publicar la siguiente rc.
